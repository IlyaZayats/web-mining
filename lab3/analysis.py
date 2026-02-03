import json
import os
from dataclasses import dataclass
from typing import Dict, List

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


Skills = Dict[str, List[str]]  # {"soft_skills": [...], "hard_skills": [...]}


def _unique_clean(items: List[str], min_len: int = 2) -> List[str]:
    """Чистит список строк и убирает дубликаты."""
    cleaned = []
    for x in items:
        if not x:
            continue
        x = x.strip()
        if not x or x == "br" or len(x) < min_len:
            continue
        cleaned.append(x)
    return list(set(cleaned))


def _split_composite(skill_text: str) -> List[str]:
    """
    Разбивает составной навык вида 'Категория: a, b, c' на ['a','b','c'].
    Если двоеточия нет — возвращает исходный навык.
    """
    if ":" not in skill_text:
        return [skill_text.strip()]

    _, values = skill_text.split(":", 1)
    return [v.strip() for v in values.split(",") if v.strip()]


def extract_skills_from_resume(html_file: str) -> Skills:
    """
    Извлекает soft/hard skills из HTML-резюме.
    Логика соответствует исходной: ищем таблицу, категории по c1, значения по c2.
    """
    with open(html_file, "r", encoding="utf-8") as f:
        content = f.read()

    soup = BeautifulSoup(content, "html.parser")

    skills: Skills = {"soft_skills": [], "hard_skills": []}

    # Ищем все строки таблицы с class odd/even (как в исходнике)
    for tr in soup.find_all("tr", class_=["odd", "even"]):
        td_cat = tr.find("td", class_="c1")
        td_val = tr.find("td", class_="c2")

        if not td_cat or not td_val:
            continue

        category = td_cat.get_text(" ", strip=True)
        # get_text(" ", ...) сохраняет переносы как пробелы; <br> будет разделяться пробелами,
        # а для некоторых кейсов нам удобнее также иметь разделение на строки:
        raw_html_val = str(td_val)
        value_text = td_val.get_text("\n", strip=True)

        if not value_text:
            continue

        # --- Soft skills ---
        if category == "Личные качества":
            skills["soft_skills"].extend([x.strip() for x in value_text.split(",") if x.strip()])

        elif category == "Увлечения":
            # В исходнике <br> превращали в ';' и делили по ';'
            # Здесь делим по строкам (get_text("\n"))
            parts = [p.strip() for p in value_text.split("\n") if p.strip()]
            for part in parts:
                if ":" not in part:
                    skills["soft_skills"].append(part)
                else:
                    hobby_type, hobby_list = part.split(":", 1)
                    hobby_type = hobby_type.strip()
                    skills["soft_skills"].append(hobby_type)

                    sub = [h.strip() for h in hobby_list.split(",") if h.strip()]
                    skills["soft_skills"].extend(sub)

        elif category == "Личные достижения":
            skills["soft_skills"].append(value_text)

        # --- Hard skills ---
        elif category == "Владение языками":
            # В исходнике делили по ';'
            for lang in [x.strip() for x in value_text.split(";") if x.strip()]:
                if "-" in lang:
                    lang_name, level = lang.split("-", 1)
                    skills["hard_skills"].append(f"{lang_name.strip()} ({level.strip()})")
                else:
                    skills["hard_skills"].append(lang)

        elif category == "Профессиональная специализация и владение компьютером":
            # В исходнике собирали несколько строк и делили по <br>.
            # Здесь берём HTML, делим логически по <br>, затем очищаем.
            # (BeautifulSoup может свести всё в '\n', но HTML-деление ближе к исходному.)
            val_soup = BeautifulSoup(raw_html_val, "html.parser")
            # Заменим <br> на '\n' явным образом
            for br in val_soup.find_all("br"):
                br.replace_with("\n")

            full_text = val_soup.get_text("\n", strip=True)
            for section in [s.strip() for s in full_text.split("\n") if s.strip()]:
                # Удаляем ведущую нумерацию "1. ..."
                if section and section[0].isdigit() and "." in section:
                    section = section.split(".", 1)[1].strip()

                skills["hard_skills"].extend(_split_composite(section))

    # Финальная очистка (как в исходнике: soft > 1 символ, hard > 2 символов)
    skills["soft_skills"] = _unique_clean(skills["soft_skills"], min_len=2)
    skills["hard_skills"] = _unique_clean(skills["hard_skills"], min_len=3)

    return skills


def calculate_semantic_similarity(text1: str, text2: str) -> float:
    """TF-IDF + cosine similarity в процентах."""
    if not text1 or not text2:
        return 0.0

    vectorizer = TfidfVectorizer(stop_words="english", min_df=1)
    try:
        tfidf = vectorizer.fit_transform([text1, text2])
        sim = cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]
        return float(sim * 100)
    except Exception:
        return 0.0


def analyze_vacancies_similarity(vacancies_folder: str, resume_skills: Skills) -> pd.DataFrame:
    """Читает json вакансий из папки и считает сходства."""
    results = []

    soft_text = " ".join(resume_skills["soft_skills"])
    hard_text = " ".join(resume_skills["hard_skills"])

    for fname in os.listdir(vacancies_folder):
        if not fname.endswith(".json"):
            continue

        path = os.path.join(vacancies_folder, fname)
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)

            desc_html = data.get("description", "") or ""
            if not desc_html:
                continue

            vacancy_name = data.get("name", "")
            vacancy_id = data.get("id", "")

            clean_desc = BeautifulSoup(desc_html, "html.parser").get_text(" ", strip=True)

            soft_sim = calculate_semantic_similarity(clean_desc, soft_text)
            hard_sim = calculate_semantic_similarity(clean_desc, hard_text)
            overall = (soft_sim + hard_sim) / 2

            results.append(
                {
                    "vacancy_id": vacancy_id,
                    "vacancy_name": vacancy_name,
                    "soft_similarity": soft_sim,
                    "hard_similarity": hard_sim,
                    "overall_similarity": overall,
                    "description_length": len(clean_desc),
                }
            )
        except Exception as e:
            print(f"Ошибка при обработке вакансии {fname}: {e}")

    return pd.DataFrame(results)


def visualize_results(df: pd.DataFrame, resume_skills: Skills) -> None:
    """Строит те же графики и сохраняет PNG (как в исходнике)."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # 1. Распределение сходства
    axes[0, 0].hist(df["soft_similarity"], bins=20, alpha=0.7, label="Soft Skills")
    axes[0, 0].hist(df["hard_similarity"], bins=20, alpha=0.7, label="Hard Skills")
    axes[0, 0].set_title("Распределение семантического сходства")
    axes[0, 0].set_xlabel("Сходство")
    axes[0, 0].set_ylabel("Количество вакансий")
    axes[0, 0].legend()

    # 2. Soft vs Hard
    axes[0, 1].scatter(df["soft_similarity"], df["hard_similarity"], alpha=0.6)
    axes[0, 1].set_title("Soft Skills vs Hard Skills сходство")
    axes[0, 1].set_xlabel("Soft Skills сходство")
    axes[0, 1].set_ylabel("Hard Skills сходство")

    # 3. Топ-10 по общему сходству
    top = df.nlargest(10, "overall_similarity")
    axes[0, 2].barh(range(len(top)), top["overall_similarity"])
    axes[0, 2].set_yticks(range(len(top)))
    axes[0, 2].set_yticklabels([name[:30] + "..." for name in top["vacancy_name"]], fontsize=8)
    axes[0, 2].set_title("Топ-10 вакансий по сходству")

    # 4. Длина описания vs сходство
    axes[1, 0].scatter(df["description_length"], df["overall_similarity"], alpha=0.6)
    axes[1, 0].set_title("Зависимость сходства от длины описания")
    axes[1, 0].set_xlabel("Длина описания")
    axes[1, 0].set_ylabel("Общее сходство")

    # 5. Boxplot
    axes[1, 1].boxplot([df["soft_similarity"], df["hard_similarity"]], labels=["Soft Skills", "Hard Skills"])
    axes[1, 1].set_title("Распределение сходства по типам навыков")

    # 6. Корреляция
    corr = df[["soft_similarity", "hard_similarity", "overall_similarity", "description_length"]].corr()
    sns.heatmap(corr, annot=True, ax=axes[1, 2], cmap="coolwarm")
    axes[1, 2].set_title("Матрица корреляции")

    plt.tight_layout()
    plt.savefig("semantic_similarity_analysis.png", dpi=300, bbox_inches="tight")
    plt.show()

    print("\n=== СТАТИСТИКА АНАЛИЗА СЕМАНТИЧЕСКОГО СХОДСТВА ===")
    print(f"Всего проанализировано вакансий: {len(df)}")
    print(f"\nSoft Skills из резюме: {resume_skills['soft_skills']}")
    print(f"\nHard Skills из резюме: {resume_skills['hard_skills']}")
    print("\nСтатистика сходства:")
    print(
        f"Soft Skills - Среднее: {df['soft_similarity'].mean():.3f}%, "
        f"Макс: {df['soft_similarity'].max():.3f}%"
    )
    print(
        f"Hard Skills - Среднее: {df['hard_similarity'].mean():.3f}%, "
        f"Макс: {df['hard_similarity'].max():.3f}%"
    )
    print(
        f"Общее - Среднее: {df['overall_similarity'].mean():.3f}%, "
        f"Макс: {df['overall_similarity'].max():.3f}%"
    )


def main() -> None:
    print("Извлечение навыков из резюме...")
    resume_skills = extract_skills_from_resume("index.html")

    print("Soft Skills:", resume_skills["soft_skills"])
    print("Hard Skills:", resume_skills["hard_skills"])

    folder = "vacancies_desc"
    if not os.path.isdir(folder):
        print("Папка 'vacancies' не найдена. Сначала выполните сбор вакансий.")
        return

    print("\nАнализ семантического сходства...")
    df = analyze_vacancies_similarity(folder, resume_skills)

    if df.empty:
        print("Нет данных для анализа. Убедитесь, что вакансии были собраны.")
        return

    df.to_excel("semantic_similarity_results.xlsx", index=False)

    df_sorted = df.sort_values("overall_similarity", ascending=False)
    df_sorted.head(20).to_excel("top_similar_vacancies.xlsx", index=False)

    visualize_results(df_sorted, resume_skills)

    print("\nРезультаты сохранены в файлы:")
    print("- semantic_similarity_results.xlsx (все вакансии)")
    print("- top_similar_vacancies.xlsx (топ-20 вакансий)")
    print("- semantic_similarity_analysis.png (графики)")


if __name__ == "__main__":
    main()
