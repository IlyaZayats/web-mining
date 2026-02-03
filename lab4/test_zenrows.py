import json
import os
import time
from typing import Dict, List, Any, Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import psutil

# pip install zenrows
from zenrows import ZenRowsClient


# ---------- HH PARSER (ZenRows) ----------

def _pct_encode(value: str) -> str:
    """
    Percent-encode (RFC 3986)
    Кодируем в UTF-8 и оставляем только unreserved: A-Z a-z 0-9 - . _ ~
    """
    unreserved = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-._~"
    out = []
    for b in value.encode("utf-8"):
        if b in unreserved:
            out.append(chr(b))
        else:
            out.append(f"%{b:02X}")
    return "".join(out)


def _build_query(params: Dict[str, Any]) -> str:
    """Собирает query string, корректно кодируя ключи/значения."""
    pairs = []
    for k, v in params.items():
        if v is None:
            continue
        pairs.append(f"{_pct_encode(str(k))}={_pct_encode(str(v))}")
    return "&".join(pairs)


class HHParserZenRows:
    """
    Аналог HHParser, но запросы идут через ZenRows SDK.

    Требования:
      - pip install zenrows
      - переменная окружения ZENROWS_API_KEY (или передайте api_key)
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        hh_base_url: str = "https://api.hh.ru/vacancies",
        retries: int = 1,
        concurrency: int = 1,
        zenrows_params: Optional[Dict[str, Any]] = None,
        user_agent: str = (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/122.0.0.0 Safari/537.36"
        ),
    ):
        self.api_key = api_key or os.getenv("ZENROWS_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Не найден ключ ZenRows. Укажите api_key в конструкторе "
                "или задайте переменную окружения ZENROWS_API_KEY."
            )

        self.hh_base_url = hh_base_url
        self.client = ZenRowsClient(self.api_key, retries=retries, concurrency=concurrency)

        self.headers = {
            "User-Agent": user_agent,
            "Accept": "application/json, text/plain, */*",
        }

        # Параметры ZenRows (для HH API чаще всего не нужны)
        self.zenrows_params = zenrows_params or {}

    def search_vacancies(
        self,
        text: str,
        area: int = 1,
        per_page: int = 100,
        page: int = 0,
        only_with_salary: bool = True,
    ) -> Optional[Dict[str, Any]]:
        params = {
            "text": text,
            "area": area,
            "per_page": per_page,
            "page": page,
            "only_with_salary": str(only_with_salary).lower(),
        }

        target_url = f"{self.hh_base_url}?{_build_query(params)}"

        try:
            resp = self.client.get(target_url, params=self.zenrows_params, headers=self.headers)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            print(f"Ошибка при запросе: {e}")
            return None


# ---------- UTILS ----------

def create_directories() -> None:
    """Создает директории для графиков."""
    for d in ("graphs/load_testing", "graphs/exploratory_analysis", "graphs/technologies"):
        os.makedirs(d, exist_ok=True)


def get_memory_usage() -> float:
    """Получает использование памяти текущим процессом в МБ."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024


# ---------- LOAD TESTING ----------

def load_testing() -> pd.DataFrame:
    """
    Нагрузочное тестирование: измерение времени ответа, RPS и использования памяти.
    (Логика и метрики идентичны исходнику, только парсер ZenRows)
    """
    parser = HHParserZenRows(
        api_key="b93e3a059bf38d2d27db5727152acaf81b9c6b01",
    )
    requests_count = [5, 10, 20, 30, 50]

    avg_response_times: List[float] = []
    rps_list: List[float] = []
    memory_usage_list: List[float] = []
    detailed_metrics: List[Dict[str, Any]] = []

    for count in requests_count:
        print(f"\n--- Тест с {count} запросами ---")

        start_time = time.time()
        _ = get_memory_usage()  # start_memory не использовался и в исходнике
        response_times: List[float] = []
        successful_requests = 0

        for _i in range(count):
            request_start = time.time()

            data = parser.search_vacancies("Python", page=0)

            request_end = time.time()
            request_time_ms = (request_end - request_start) * 1000

            if data is not None:
                successful_requests += 1
                response_times.append(request_time_ms)

            time.sleep(0.1)

        end_time = time.time()
        end_memory = get_memory_usage()

        total_time = end_time - start_time
        avg_response_time = sum(response_times) / len(response_times) if response_times else 0.0
        rps = successful_requests / total_time if total_time > 0 else 0.0
        memory_used = end_memory

        avg_response_times.append(avg_response_time)
        rps_list.append(rps)
        memory_usage_list.append(memory_used)

        detailed_metrics.append(
            {
                "requests_count": count,
                "successful_requests": successful_requests,
                "avg_response_time_ms": avg_response_time,
                "rps": rps,
                "memory_used_mb": memory_used,
                "total_time_sec": total_time,
            }
        )

        print(f"Успешных запросов: {successful_requests}/{count}")
        print(f"Среднее время ответа: {avg_response_time:.2f} мс")
        print(f"RPS: {rps:.2f} запросов/сек")
        print(f"Использование памяти: {memory_used:.2f} МБ")
        print(f"Общее время теста: {total_time:.2f} сек")

        time.sleep(2)

    # График 1: Среднее время ответа
    plt.figure(figsize=(10, 6))
    plt.plot(requests_count, avg_response_times, "bo-", linewidth=2, markersize=8)
    plt.title("Среднее время ответа API HH.ru")
    plt.xlabel("Количество страниц")
    plt.ylabel("Время ответа (мс)")
    plt.grid(True, alpha=0.3)
    plt.savefig("graphs/load_testing/response_time.png", dpi=300, bbox_inches="tight")
    plt.close()

    # График 2: RPS
    plt.figure(figsize=(10, 6))
    plt.plot(requests_count, rps_list, "go-", linewidth=2, markersize=8)
    plt.title("Производительность API HH.ru (RPS)")
    plt.xlabel("Количество страниц")
    plt.ylabel("Запросов в секунду")
    plt.grid(True, alpha=0.3)
    plt.savefig("graphs/load_testing/rps_performance.png", dpi=300, bbox_inches="tight")
    plt.close()

    # График 3: Использование памяти
    plt.figure(figsize=(10, 6))
    plt.plot(requests_count, memory_usage_list, "ro-", linewidth=2, markersize=8)
    plt.title("Использование памяти при нагрузочном тестировании")
    plt.xlabel("Количество страниц")
    plt.ylabel("Память (МБ)")
    plt.grid(True, alpha=0.3)
    plt.savefig("graphs/load_testing/memory_usage.png", dpi=300, bbox_inches="tight")
    plt.close()

    df_metrics = pd.DataFrame(detailed_metrics)
    df_metrics.to_csv("load_testing_metrics.csv", index=False, encoding="utf-8-sig")
    print("\nМетрики нагрузочного тестирования сохранены в 'load_testing_metrics.csv'")

    return df_metrics


# ---------- TECHNOLOGIES EXTRACTION ----------

def extract_technologies(vacancy_name: str, vacancy_snippet: str) -> List[str]:
    """Извлекает технологии из названия вакансии и описания."""
    tech_keywords = {
        "Python": ["python"],
        "Django и др.": ["django", "flask", "fastapi"],
        "JavaScript": ["javascript", "js", "node.js", "nodejs"],
        "React и др.": ["react", "vue", "angular"],
        "Java": ["java", "hibernate"],
        "Spring": ["spring"],
        "C++": ["c++", "cpp"],
        "C# & .NET": ["c#", "csharp", ".net"],
        "PHP": ["php", "laravel", "symfony"],
        "Go": ["go", "golang"],
        "Ruby": ["ruby", "rails"],
        "SQL": ["sql", "mysql", "postgresql", "oracle"],
        "NoSQL": ["mongodb", "redis", "cassandra"],
        "Docker": ["docker", "container"],
        "Kubernetes": ["kubernetes", "k8s"],
        "AWS": ["aws", "amazon web services"],
        "Azure": ["azure"],
        "Git": ["git", "github", "gitlab"],
        "Linux": ["linux", "unix"],
    }

    found = set()
    text = f"{vacancy_name} {vacancy_snippet}".lower()

    for tech, keywords in tech_keywords.items():
        for kw in keywords:
            if kw in text:
                found.add(tech)
                break

    return list(found)


# ---------- EXPLORATORY ANALYSIS ----------

def exploratory_analysis(filename: str = "programming_vacancies.json") -> Tuple[pd.DataFrame, pd.Series]:
    """
    Исследовательский анализ собранных данных с анализом технологий.
    """
    with open(filename, "r", encoding="utf-8") as f:
        vacancies = json.load(f)

    print(f"Всего вакансий для анализа: {len(vacancies)}")

    df_rows: List[Dict[str, Any]] = []
    all_technologies: List[str] = []

    for vac in vacancies:
        salary = vac.get("salary")
        if salary:
            salary_from = salary.get("from")
            salary_to = salary.get("to")
            if salary_from and salary_to:
                avg_salary = (salary_from + salary_to) / 2
            elif salary_from:
                avg_salary = salary_from
            elif salary_to:
                avg_salary = salary_to
            else:
                avg_salary = None
        else:
            avg_salary = None

        experience = vac.get("experience", {}).get("name", "Не указан")
        employment = vac.get("employment", {}).get("name", "Не указана")

        snippet = vac.get("snippet", {}) or {}
        requirement = snippet.get("requirement", "") or ""
        responsibility = snippet.get("responsibility", "") or ""
        vacancy_snippet = f"{requirement} {responsibility}"

        technologies = extract_technologies(vac.get("name", ""), vacancy_snippet)
        all_technologies.extend(technologies)

        df_rows.append(
            {
                "id": vac.get("id"),
                "name": vac.get("name"),
                "salary_avg": avg_salary,
                "experience": experience,
                "employment": employment,
                "area": (vac.get("area") or {}).get("name"),
                "published_at": vac.get("published_at"),
                "technologies": technologies,
            }
        )

    df = pd.DataFrame(df_rows)

    tech_counts = pd.Series(all_technologies).value_counts()

    # 1) Распределение зарплат
    plt.figure(figsize=(12, 8))
    salaries = df["salary_avg"].dropna()
    if len(salaries) > 0:
        plt.hist(salaries, bins=20, alpha=0.7, color="skyblue", edgecolor="black")
        plt.title("Распределение средних зарплат программистов")
        plt.xlabel("Зарплата (руб)")
        plt.ylabel("Количество вакансий")
        plt.grid(True, alpha=0.3)
        plt.savefig("graphs/exploratory_analysis/salary_distribution.png", dpi=300, bbox_inches="tight")
        plt.close()
        print("Сохранен график: Распределение зарплат")
    else:
        plt.close()

    # 2) Зарплаты по опыту
    plt.figure(figsize=(12, 8))
    exp_salary = df.groupby("experience")["salary_avg"].mean().dropna()
    if len(exp_salary) > 0:
        colors = ["#ff9999", "#66b3ff", "#99ff99", "#ffcc99"]
        exp_salary.plot(kind="bar", color=colors)
        plt.title("Средняя зарплата по опыту работы")
        plt.xlabel("Опыт работы")
        plt.ylabel("Средняя зарплата (руб)")
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)

        for i, v in enumerate(exp_salary):
            plt.text(i, v + 5000, f"{v:.0f}", ha="center", va="bottom")

        plt.savefig("graphs/exploratory_analysis/salary_by_experience.png", dpi=300, bbox_inches="tight")
        plt.close()
        print("Сохранен график: Зарплаты по опыту работы")
    else:
        plt.close()

    # 3) Типы занятости
    plt.figure(figsize=(12, 8))
    employment_counts = df["employment"].value_counts()
    if len(employment_counts) > 0:
        colors = ["#ff9999", "#66b3ff", "#99ff99", "#ffcc99", "#c2c2f0"]
        plt.pie(
            employment_counts.values,
            labels=employment_counts.index,
            autopct="%1.1f%%",
            colors=colors,
            startangle=90,
        )
        plt.title("Распределение вакансий по типам занятости")
        plt.savefig("graphs/exploratory_analysis/employment_types.png", dpi=300, bbox_inches="tight")
        plt.close()
        print("Сохранен график: Типы занятости")
    else:
        plt.close()

    # 4) Топ-15 технологий (горизонтальный)
    plt.figure(figsize=(12, 8))
    if len(tech_counts) > 0:
        top_tech = tech_counts.head(15)
        colors = plt.cm.Set3(range(len(top_tech)))
        bars = plt.barh(range(len(top_tech)), top_tech.values, color=colors)
        plt.yticks(range(len(top_tech)), top_tech.index)
        plt.title("Топ-15 востребованных технологий")
        plt.xlabel("Количество упоминаний в вакансиях")
        plt.gca().invert_yaxis()
        plt.grid(True, alpha=0.3)

        for i, bar in enumerate(bars):
            width = bar.get_width()
            plt.text(width + 1, bar.get_y() + bar.get_height() / 2, f"{int(width)}", ha="left", va="center")

        plt.savefig("graphs/technologies/top_15_technologies.png", dpi=300, bbox_inches="tight")
        plt.close()
        print("Сохранен график: Топ-15 технологий")
    else:
        plt.close()

    # 5) Топ-20 технологий (вертикальный)
    plt.figure(figsize=(14, 8))
    if len(tech_counts) > 0:
        top_tech_all = tech_counts.head(20)
        colors = plt.cm.viridis(range(len(top_tech_all)))
        bars = plt.bar(range(len(top_tech_all)), top_tech_all.values, color=colors)
        plt.xticks(range(len(top_tech_all)), top_tech_all.index, rotation=45, ha="right")
        plt.title("Топ-20 востребованных технологий в программировании")
        plt.xlabel("Технологии")
        plt.ylabel("Количество упоминаний в вакансиях")
        plt.grid(True, alpha=0.3)

        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2.0, height + 1, f"{int(height)}", ha="center", va="bottom")

        plt.tight_layout()
        plt.savefig("graphs/technologies/top_20_technologies.png", dpi=300, bbox_inches="tight")
        plt.close()
        print("Сохранен график: Топ-20 технологий")
    else:
        plt.close()

    # 6) Круговой топ-10 технологий
    plt.figure(figsize=(10, 8))
    if len(tech_counts) > 0:
        top_tech_pie = tech_counts.head(10)
        colors = plt.cm.Pastel1(range(len(top_tech_pie)))
        plt.pie(
            top_tech_pie.values,
            labels=top_tech_pie.index,
            autopct="%1.1f%%",
            colors=colors,
            startangle=90,
        )
        plt.title("Распределение топ-10 технологий")
        plt.savefig("graphs/technologies/technologies_pie_chart.png", dpi=300, bbox_inches="tight")
        plt.close()
        print("Сохранен график: Круговой график технологий")
    else:
        plt.close()

    return df, tech_counts


def print_detailed_analysis(df: pd.DataFrame, tech_counts: pd.Series) -> None:
    """Выводит детальный анализ данных."""
    print("\n" + "=" * 50)
    print("ДЕТАЛЬНЫЙ АНАЛИЗ РЕЗУЛЬТАТОВ")
    print("=" * 50)

    print("\nОБЩАЯ СТАТИСТИКА:")
    print(f"   Всего вакансий: {len(df)}")
    print(f"   Вакансий с указанной зарплатой: {df['salary_avg'].notna().sum()}")
    print(f"   Средняя зарплата: {df['salary_avg'].mean():.2f} руб.")
    print(f"   Медианная зарплата: {df['salary_avg'].median():.2f} руб.")
    print(f"   Максимальная зарплата: {df['salary_avg'].max():.2f} руб.")
    print(f"   Минимальная зарплата: {df['salary_avg'].min():.2f} руб.")

    print("\nТОП-10 ТЕХНОЛОГИЙ:")
    for i, (tech, count) in enumerate(tech_counts.head(10).items(), 1):
        percentage = (count / len(df)) * 100 if len(df) else 0
        print(f"   {i:2d}. {tech:<15} {count:>3} упоминаний ({percentage:.1f}%)")

    print("\nРАСПРЕДЕЛЕНИЕ ПО ОПЫТУ:")
    exp_counts = df["experience"].value_counts()
    for exp, count in exp_counts.items():
        percentage = (count / len(df)) * 100 if len(df) else 0
        print(f"   {exp:<20} {count:>3} вакансий ({percentage:.1f}%)")

    print("\nТИПЫ ЗАНЯТОСТИ:")
    emp_counts = df["employment"].value_counts()
    for emp, count in emp_counts.items():
        percentage = (count / len(df)) * 100 if len(df) else 0
        print(f"   {emp:<20} {count:>3} вакансий ({percentage:.1f}%)")


if __name__ == "__main__":
    create_directories()

    print("=== НАГРУЗОЧНОЕ ТЕСТИРОВАНИЕ ===")
    _load_metrics = load_testing()

    print("\n=== ИССЛЕДОВАТЕЛЬСКИЙ АНАЛИЗ ===")
    df, tech_counts = exploratory_analysis()

    print_detailed_analysis(df, tech_counts)

    df.to_csv("vacancies_detailed_analysis.csv", index=False, encoding="utf-8-sig")

    tech_df = tech_counts.reset_index()
    tech_df.columns = ["Technology", "Count"]
    tech_df["Percentage"] = (tech_df["Count"] / len(df)) * 100 if len(df) else 0
    tech_df.to_csv("technologies_stats.csv", index=False, encoding="utf-8-sig")

    print("\nДанные сохранены:")
    print("   - 'vacancies_detailed_analysis.csv' - детальная информация по вакансиям")
    print("   - 'technologies_stats.csv' - статистика по технологиям")
    print("   - 'load_testing_metrics.csv' - метрики нагрузочного тестирования")
