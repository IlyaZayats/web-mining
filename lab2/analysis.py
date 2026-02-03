from __future__ import annotations

import inspect
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from collections import Counter

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


# --------------------------
# Config
# --------------------------
@dataclass(frozen=True)
class AppConfig:
    api_version: str = "5.199"
    vk_base_url: str = "https://api.vk.com/method/"
    batch_size: int = 1000
    sleep_sec: float = 0.34  # пауза между запросами, чтобы не упираться в лимиты
    fields: str = "online,sex,city,country,universities,bdate"
    output_dir: str = "output"
    default_json: str = "members.json"


# --------------------------
# VK Client
# --------------------------
class VKAPIError(RuntimeError):
    pass


class VKClient:
    """
    Клиент для скачивания участников группы VK по batch'ам.
    Токен должен иметь доступ к groups.getMembers.
    """

    def __init__(self, token: str, cfg: AppConfig = AppConfig(), verbose: bool = True):
        self.token = token
        self.cfg = cfg
        self.verbose = verbose
        self._session = requests.Session()

    def close(self) -> None:
        self._session.close()

    def _call(self, method: str, params: Dict[str, Any]) -> Dict[str, Any]:
        url = self.cfg.vk_base_url + method
        payload = {
            "access_token": self.token,
            "v": self.cfg.api_version,
            **params,
        }
        resp = self._session.post(url, data=payload, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        if "error" in data:
            err = data["error"]
            raise VKAPIError(f"VK API error {err.get('error_code')}: {err.get('error_msg')}")

        if "response" not in data:
            raise VKAPIError(f"Unexpected response: {data}")

        return data["response"]

    def get_group_member_count(self, group_id: str) -> int:
        response = self._call("groups.getMembers", {"group_id": group_id, "count": 1})
        return int(response["count"])

    def get_all_members(
        self,
        group_id: str,
        fields: Optional[str] = None,
        batch_size: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Возвращает список словарей — всех участников группы.
        batch_size <= 1000
        """
        fields = fields or self.cfg.fields
        batch_size = batch_size or self.cfg.batch_size

        total = self.get_group_member_count(group_id)
        if self.verbose:
            print(f"Всего участников: {total}")

        offset = 0
        items: List[Dict[str, Any]] = []

        while offset < total:
            try:
                response = self._call(
                    "groups.getMembers",
                    {
                        "group_id": group_id,
                        "count": batch_size,
                        "offset": offset,
                        "fields": fields,
                    },
                )
            except VKAPIError as e:
                print("Ошибка при получении партии:", e)
                break

            batch = response.get("items", [])
            items.extend(batch)

            offset += batch_size
            if self.verbose:
                print(f"Загружено {len(items)}/{total}")

            time.sleep(self.cfg.sleep_sec)

        return items


# --------------------------
# DataLoader
# --------------------------
class DataLoader:
    @staticmethod
    def save_json(path: str | Path, data: Any) -> None:
        path = Path(path)
        with path.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @staticmethod
    def load_json(path: str | Path) -> Any:
        path = Path(path)
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def normalize_members_payload(raw: Any) -> List[Dict[str, Any]]:
        """
        Поддерживаем 2 формата:
          1) список участников
          2) { "response": { "items": [...] } }
        """
        if isinstance(raw, dict) and "response" in raw and isinstance(raw["response"], dict) and "items" in raw["response"]:
            items = raw["response"]["items"]
            if isinstance(items, list):
                return items
        if isinstance(raw, list):
            return raw
        raise ValueError("Не распознан формат JSON. Ожидается список или {response: {items: [...]}}.")

    @staticmethod
    def to_dataframe(members: List[Dict[str, Any]]) -> pd.DataFrame:
        rows: List[Dict[str, Any]] = []
        for m in members:
            row = {
                "id": m.get("id"),
                "first_name": m.get("first_name"),
                "last_name": m.get("last_name"),
                "sex": m.get("sex"),  # 1 - female, 2 - male, 0 - unknown
                "online": m.get("online", 0),
                "is_closed": m.get("is_closed", False),
                "can_access_closed": m.get("can_access_closed", False),
                "city_id": None,
                "city_title": None,
                "country_id": None,
                "country_title": None,
                "universities": None,
                "bdate": m.get("bdate"),
            }

            city = m.get("city")
            if isinstance(city, dict):
                row["city_id"] = city.get("id")
                row["city_title"] = city.get("title")

            country = m.get("country")
            if isinstance(country, dict):
                row["country_id"] = country.get("id")
                row["country_title"] = country.get("title")

            unis = m.get("universities")
            if isinstance(unis, list):
                row["universities"] = unis

            rows.append(row)

        return pd.DataFrame(rows)


# --------------------------
# Preprocessor
# --------------------------
class Preprocessor:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

    @staticmethod
    def _extract_age(bdate: Any, reference_year: Optional[int] = None) -> Optional[int]:
        """
        bdate: 'DD.MM.YYYY' или 'DD.MM'. Если нет года — None.
        """
        if not isinstance(bdate, str):
            return None
        parts = bdate.split(".")
        if len(parts) != 3:
            return None
        try:
            year = int(parts[2])
        except ValueError:
            return None
        if reference_year is None:
            reference_year = pd.Timestamp.now().year
        age = reference_year - year
        return age if 4 <= age <= 120 else None

    def add_age(self) -> Preprocessor:
        self.df["age"] = self.df["bdate"].apply(self._extract_age)
        return self

    def unify_city(self) -> Preprocessor:
        self.df["city_title"] = self.df["city_title"].fillna("Не указано")
        return self

    def extract_university_features(self) -> Preprocessor:
        def extract(unis: Any) -> Dict[str, Any]:
            if not isinstance(unis, list) or len(unis) == 0:
                return {
                    "primary_university": "Не указано",
                    "n_universities": 0,
                    "graduation_year": None,
                    "faculty_name": "Не указано",
                    "education_status": "Не указано",
                }
            u = unis[0] if isinstance(unis[0], dict) else {}
            return {
                "primary_university": u.get("name") or u.get("faculty_name") or u.get("chair_name") or "Не указано",
                "n_universities": len(unis),
                "graduation_year": u.get("graduation"),
                "faculty_name": u.get("faculty_name", "Не указано"),
                "education_status": u.get("education_status", "Не указано"),
            }

        feats = self.df["universities"].apply(extract)
        for col in ["primary_university", "n_universities", "graduation_year", "faculty_name", "education_status"]:
            self.df[col] = feats.apply(lambda x: x[col])
        return self

    def fill_na_for_analysis(self) -> Preprocessor:
        self.df["sex"] = self.df["sex"].fillna(0).astype(int)
        self.df["online"] = self.df["online"].fillna(0).astype(int)
        for col in ["city_title", "primary_university", "faculty_name", "education_status"]:
            self.df[col] = self.df[col].fillna("Не указано")
        return self

    def run_all(self) -> pd.DataFrame:
        return (
            self.add_age()
            .unify_city()
            .extract_university_features()
            .fill_na_for_analysis()
            .df
        )


# --------------------------
# Analyzer
# --------------------------
class Analyzer:
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def gender_counts(self) -> Dict[str, int]:
        c = Counter(self.df["sex"])
        return {"female": c.get(1, 0), "male": c.get(2, 0), "unknown": c.get(0, 0)}

    def online_count(self) -> int:
        return int(self.df["online"].sum())

    def top_cities(self, n: int = 10) -> List[Tuple[str, int]]:
        return Counter(self.df["city_title"]).most_common(n)

    def top_universities(self, n: int = 10) -> List[Tuple[str, int]]:
        return Counter(self.df["primary_university"]).most_common(n)

    def age_stats(self) -> Dict[str, float]:
        ages = self.df["age"].dropna()
        if len(ages) == 0:
            return {}
        return {
            "count": int(ages.count()),
            "mean": float(ages.mean()),
            "median": float(ages.median()),
            "std": float(ages.std()),
        }

    def summary_table(self) -> pd.DataFrame:
        table = self.df.pivot_table(
            index="city_title", columns="sex", values="id", aggfunc="count", fill_value=0
        ).rename(columns={0: "unknown", 1: "female", 2: "male"})
        table["total"] = table.sum(axis=1)
        return table.sort_values("total", ascending=False)


# --------------------------
# Classifier
# --------------------------
class ClassifierModel:
    """
    Pipeline для классификации пола (1/2) на расширенных признаках.
    """

    def __init__(self, df: pd.DataFrame, target_col: str = "sex"):
        self.df = df.copy()
        self.target_col = target_col
        self.pipeline: Optional[Pipeline] = None

    def prepare_X_y(self) -> Tuple[pd.DataFrame, pd.Series]:
        features = [
            "city_title",
            "primary_university",
            "faculty_name",
            "education_status",
            "graduation_year",
            "n_universities",
            "online",
            "age",
        ]
        X = self.df[features].copy()
        y = self.df[self.target_col].copy()

        mask = y.isin([1, 2])
        return X[mask], y[mask]

    def build_pipeline(self) -> None:
        numeric_features = ["age", "n_universities", "graduation_year", "online"]
        categorical_features = ["city_title", "primary_university", "faculty_name", "education_status"]

        numeric_transformer = Pipeline(
            steps=[("imputer", SimpleImputer(strategy="median"))]
        )

        encoder_kwargs = {"handle_unknown": "ignore"}
        sig = inspect.signature(OneHotEncoder)
        if "sparse_output" in sig.parameters:
            encoder_kwargs["sparse_output"] = False
        else:
            encoder_kwargs["sparse"] = False

        categorical_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="constant", fill_value="Не указано")),
                ("ohe", OneHotEncoder(**encoder_kwargs)),
            ]
        )

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, numeric_features),
                ("cat", categorical_transformer, categorical_features),
            ]
        )

        clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        self.pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("clf", clf)])

    def train_and_evaluate(self, test_size: float = 0.25, random_state: int = 42) -> Dict[str, Any]:
        X, y = self.prepare_X_y()
        if len(X) < 50:
            raise ValueError("Недостаточно данных для обучения (требуется >=50 примеров с меткой пола).")

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        if self.pipeline is None:
            self.build_pipeline()

        assert self.pipeline is not None
        self.pipeline.fit(X_train, y_train)
        y_pred = self.pipeline.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        report = classification_report(
            y_test,
            y_pred,
            labels=[1, 2],
            target_names=["female(1)", "male(2)"],
            zero_division=0,
        )
        cm = confusion_matrix(y_test, y_pred, labels=[1, 2])

        return {
            "accuracy": acc,
            "report_text": report,
            "confusion_matrix": cm,
            "y_test": y_test,
            "y_pred": y_pred,
        }


# --------------------------
# Visualizer
# --------------------------
class Visualizer:
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def plot_bar(
        self,
        items: List[Tuple[str, int]],
        title: str,
        filename: str,
        top_n: Optional[int] = None,
    ) -> str:
        if top_n is not None:
            items = items[:top_n]
        labels, counts = zip(*items) if items else ([], [])

        plt.figure(figsize=(10, 6))
        plt.bar(range(len(labels)), counts)
        plt.xticks(range(len(labels)), labels, rotation=45, ha="right")
        plt.title(title)
        plt.tight_layout()

        path = self.output_dir / filename
        plt.savefig(path)
        plt.close()
        return str(path)

    def plot_pie_gender(self, gender_counts: Dict[str, int], filename: str) -> str:
        labels = list(gender_counts.keys())
        sizes = list(gender_counts.values())

        plt.figure(figsize=(6, 6))
        plt.pie(sizes, labels=labels, autopct="%1.1f%%", startangle=90)
        plt.title("Gender distribution")
        plt.tight_layout()

        path = self.output_dir / filename
        plt.savefig(path)
        plt.close()
        return str(path)

    def save_table(self, df: pd.DataFrame, filename: str) -> str:
        path = self.output_dir / filename
        df.to_excel(path, index=True)
        return str(path)

    def plot_confusion_matrix(self, cm: np.ndarray, labels: List[str], filename: str) -> str:
        plt.figure(figsize=(5, 4))
        plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
        plt.title("Confusion matrix")
        plt.colorbar()

        tick_marks = np.arange(len(labels))
        plt.xticks(tick_marks, labels)
        plt.yticks(tick_marks, labels)

        thresh = cm.max() / 2.0 if cm.size else 0
        for i, j in np.ndindex(cm.shape):
            plt.text(
                j,
                i,
                f"{cm[i, j]:d}",
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black",
            )

        plt.ylabel("True label")
        plt.xlabel("Predicted label")
        plt.tight_layout()

        path = self.output_dir / filename
        plt.savefig(path)
        plt.close()
        return str(path)


# --------------------------
# End-to-end pipeline
# --------------------------
def run_pipeline_from_file(json_path: str | Path, cfg: AppConfig = AppConfig(), save_output: bool = True) -> Dict[str, Any]:
    raw = DataLoader.load_json(json_path)
    members = DataLoader.normalize_members_payload(raw)

    df = DataLoader.to_dataframe(members)
    df_processed = Preprocessor(df).run_all()

    analyzer = Analyzer(df_processed)
    gender = analyzer.gender_counts()
    online = analyzer.online_count()
    top_cities = analyzer.top_cities(20)
    top_unis = analyzer.top_universities(20)
    age_stats = analyzer.age_stats()
    summary_table = analyzer.summary_table()

    # классификация (если есть данные)
    try:
        clf = ClassifierModel(df_processed)
        classifier_results: Dict[str, Any] = clf.train_and_evaluate()
    except Exception as e:
        classifier_results = {"error": str(e)}

    vis = Visualizer(cfg.output_dir)
    gender_path = vis.plot_pie_gender(gender, "gender_pie.png")
    cities_path = vis.plot_bar(top_cities, "Top cities", "top_cities.png", top_n=20)
    unis_path = vis.plot_bar(top_unis, "Top universities", "top_universities.png", top_n=20)
    table_path = vis.save_table(summary_table, "summary_by_city.xlsx")

    cm_path = None
    if "confusion_matrix" in classifier_results and not classifier_results.get("error"):
        cm = classifier_results["confusion_matrix"]
        cm_path = vis.plot_confusion_matrix(cm, ["female(1)", "male(2)"], "confusion_matrix.png")

    out = {
        "gender_counts": gender,
        "online_count": online,
        "top_cities": top_cities,
        "top_universities": top_unis,
        "age_stats": age_stats,
        "summary_table_path": table_path,
        "gender_chart": gender_path,
        "cities_chart": cities_path,
        "universities_chart": unis_path,
        "classifier": classifier_results,
        "confusion_matrix_chart": cm_path,
    }

    if save_output:
        DataLoader.save_json(vis.output_dir / "members_processed.json", df_processed.to_dict(orient="records"))
        df_processed.to_excel(vis.output_dir / "members_processed.xlsx", index=False)

    return out


# --------------------------
# CLI / пример использования
# --------------------------
def main() -> None:
    cfg = AppConfig()
    json_file = Path(cfg.default_json)

    if json_file.exists():
        print("Запуск pipeline по файлу:", str(json_file))
        results = run_pipeline_from_file(json_file, cfg=cfg, save_output=True)

        print("Результаты анализа (кратко):")
        print("Gender:", results["gender_counts"])
        print("Online:", results["online_count"])
        print("Top cities (top5):", results["top_cities"][:5])
        print("Top unis (top5):", results["top_universities"][:5])
        print("Age stats:", results["age_stats"])

        classifier = results["classifier"]
        if isinstance(classifier, dict) and classifier.get("error"):
            print("Классификация не выполнена:", classifier["error"])
        else:
            print("Accuracy:", classifier["accuracy"])
            print("Classification report:\n", classifier["report_text"])
        return

    # Если файла нет — пробуем скачать из VK (как и в исходнике: token нужно указать)
    token = "fef5fbcafef5fbcafef5fbcab5fdcb49b7ffef5fef5fbca9777fdf34d762500eb280e2e"  # заменить на свой реальный токен
    group_id = "samosbor_original"

    if token.startswith("<") or len(token) < 20:
        print("Файл members.json не найден. Чтобы скачать из VK, укажите реальный token в коде.")
        return

    vk = VKClient(token, cfg=cfg, verbose=True)
    try:
        members = vk.get_all_members(group_id, fields=cfg.fields)
    finally:
        vk.close()

    DataLoader.save_json(json_file, {"response": {"count": len(members), "items": members}})
    print(f"Сохранено {json_file}. Запустите скрипт снова для анализа.")


if __name__ == "__main__":
    main()
