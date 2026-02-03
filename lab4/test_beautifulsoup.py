import json
import os
import re
import time
from typing import Dict, List, Any, Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import psutil
import requests
from bs4 import BeautifulSoup


# ---------- HH PARSER (BeautifulSoup) ----------

class HHParserBS4:
    """
    HHParserBS4 - парсер BeautifulSoup
      - search_vacancies(text, area, per_page, page, only_with_salary=True) -> dict{items,pages,page,per_page}
      - parse_vacancies(search_query, pages_to_parse) -> List[Dict]
      - save_to_json(data, filename)

    ВАЖНО:
      - HH может показать капчу/антибот — тогда вернётся None.
      - HTML-разметка HH может меняться.
    """

    def __init__(
        self,
        base_url: str = "https://hh.ru/search/vacancy",
        user_agent: str = (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/122.0.0.0 Safari/537.36"
        ),
        timeout: int = 30,
    ):
        self.base_url = base_url
        self.timeout = timeout

        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": user_agent,
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "ru-RU,ru;q=0.9,en;q=0.8",
                "Connection": "keep-alive",
            }
        )

    @staticmethod
    def _clean_text(s: Optional[str]) -> str:
        return re.sub(r"\s+", " ", (s or "")).strip()

    @staticmethod
    def _detect_blocked(html: str) -> bool:
        low = html.lower()
        return any(
            k in low
            for k in (
                "captcha",
                "похоже, вы робот",
                "подтвердите, что вы не робот",
                "access denied",
                "forbidden",
            )
        )

    @staticmethod
    def _extract_vacancy_id(url: str) -> str:
        m = re.search(r"/vacancy/(\d+)", url)
        return m.group(1) if m else ""

    @staticmethod
    def _parse_salary(salary_text: str) -> Optional[Dict[str, Any]]:
        """
        Парсит строку зарплаты (HTML SERP) в формат, похожий на HH API salary.
        """
        text = (salary_text or "").replace("\u202f", " ").replace("\xa0", " ").strip()
        text_low = text.lower()
        if not text:
            return None

        currency = None
        if "руб" in text_low or "₽" in text_low:
            currency = "RUR"
        elif "$" in text or "usd" in text_low:
            currency = "USD"
        elif "€" in text or "eur" in text_low:
            currency = "EUR"
        elif "₸" in text or "kzt" in text_low:
            currency = "KZT"

        nums = [int(n.replace(" ", "")) for n in re.findall(r"(\d[\d ]*\d|\d)", text)]
        if not nums:
            return {"from": None, "to": None, "currency": currency, "gross": None}

        salary_from = None
        salary_to = None

        if "от" in text_low and "до" not in text_low:
            salary_from = nums[0]
        elif "до" in text_low and "от" not in text_low:
            salary_to = nums[0]
        elif len(nums) >= 2:
            salary_from, salary_to = nums[0], nums[1]
        else:
            salary_from = nums[0]

        return {"from": salary_from, "to": salary_to, "currency": currency, "gross": None}

    def _get_total_pages(self, soup: BeautifulSoup) -> int:
        """
        Определяет число страниц выдачи.
        """
        pages = []
        for a in soup.select('[data-qa="pager-page"]'):
            t = self._clean_text(a.get_text())
            if t.isdigit():
                pages.append(int(t))
        return max(pages) if pages else 1

    def _parse_items(self, soup: BeautifulSoup) -> List[Dict[str, Any]]:
        """
        Извлекает карточки вакансий из HTML выдачи.
        Возвращает items в структуре, близкой к HH API.
        """
        items: List[Dict[str, Any]] = []

        cards = soup.select('[data-qa="vacancy-serp__vacancy"]')
        if not cards:
            cards = soup.select('[data-qa="serp-item"]')

        for card in cards:
            title_a = (
                card.select_one('a[data-qa="serp-item__title"]')
                or card.select_one('a[data-qa="vacancy-serp__vacancy-title"]')
            )
            if not title_a:
                continue

            name = self._clean_text(title_a.get_text())
            href = title_a.get("href", "") or ""
            vid = self._extract_vacancy_id(href)

            salary_el = card.select_one('[data-qa="vacancy-serp__vacancy-compensation"]')
            salary_text = self._clean_text(salary_el.get_text()) if salary_el else ""
            salary = self._parse_salary(salary_text) if salary_text else None

            area_el = card.select_one('[data-qa="vacancy-serp__vacancy-address"]')
            area_name = self._clean_text(area_el.get_text()) if area_el else None

            req_el = card.select_one('[data-qa="vacancy-serp__vacancy_snippet_requirement"]')
            resp_el = card.select_one('[data-qa="vacancy-serp__vacancy_snippet_responsibility"]')
            requirement = self._clean_text(req_el.get_text()) if req_el else None
            responsibility = self._clean_text(resp_el.get_text()) if resp_el else None

            date_el = card.select_one('[data-qa="vacancy-serp__vacancy-date"]')
            published_at = self._clean_text(date_el.get_text()) if date_el else None

            items.append(
                {
                    "id": vid or None,
                    "name": name or None,
                    "salary": salary,
                    "experience": {"name": "Не указан"},   # на SERP часто нет
                    "employment": {"name": "Не указана"},  # на SERP часто нет
                    "area": {"name": area_name},
                    "published_at": published_at,
                    "snippet": {"requirement": requirement, "responsibility": responsibility},
                    "alternate_url": href or None,
                }
            )

        return items

    def search_vacancies(
        self,
        text: str,
        area: int = 1,
        per_page: int = 100,
        page: int = 0,
        only_with_salary: bool = True,
    ) -> Optional[Dict[str, Any]]:
        """
        Возвращает структуру как у HH API (минимально нужные поля).
        """
        params = {
            "text": text,
            "area": area,
            "page": page,
            # HH может не строго соблюдать, но это "аналог" per_page
            "items_on_page": min(max(per_page, 1), 100),
        }
        if only_with_salary:
            params["only_with_salary"] = "true"

        try:
            resp = self.session.get(self.base_url, params=params, timeout=self.timeout)
            resp.raise_for_status()

            html = resp.text
            # if self._detect_blocked(html):
            #     print("Похоже, HH показал капчу/антибот. Запрос вернул None.")
            #     return None

            soup = BeautifulSoup(html, "html.parser")
            items = self._parse_items(soup)
            pages = self._get_total_pages(soup)

            return {"items": items, "pages": pages, "page": page, "per_page": params["items_on_page"]}
        except requests.RequestException as e:
            print(f"Ошибка при запросе: {e}")
            return None

    def parse_vacancies(self, search_query: str, pages_to_parse: int = 5) -> List[Dict[str, Any]]:
        """
        Парсит несколько страниц выдачи (как у вас раньше).
        """
        all_vacancies: List[Dict[str, Any]] = []

        for page in range(pages_to_parse):
            print(f"Парсинг страницы {page + 1}...")

            data = self.search_vacancies(search_query, page=page)

            if data is None:
                print(f"Не удалось получить данные для страницы {page}. Прерывание.")
                break

            vacancies = data.get("items", [])
            if not vacancies:
                print("Больше вакансий нет. Прерывание.")
                break

            all_vacancies.extend(vacancies)

            total_pages = data.get("pages")
            if isinstance(total_pages, int) and page >= total_pages - 1:
                break

            time.sleep(0.5)

        print(f"Всего спарсено вакансий: {len(all_vacancies)}")
        return all_vacancies

    def save_to_json(self, data: List[Dict[str, Any]], filename: str = "hh_vacancies.json") -> None:
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"Данные сохранены в {filename}")


# ---------- UTILS ----------

def create_directories() -> None:
    for d in ("graphs/load_testing", "graphs/exploratory_analysis", "graphs/technologies"):
        os.makedirs(d, exist_ok=True)


def get_memory_usage() -> float:
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024


# ---------- LOAD TESTING ----------

def load_testing() -> pd.DataFrame:
    """
    Нагрузочное тестирование: измерение времени ответа, RPS и использования памяти.
    Логика/метрики сохранены, заменён только источник данных (BS4).
    """
    parser = HHParserBS4()
    requests_count = [5, 10, 20, 30, 50]

    avg_response_times: List[float] = []
    rps_list: List[float] = []
    memory_usage_list: List[float] = []
    detailed_metrics: List[Dict[str, Any]] = []

    for count in requests_count:
        print(f"\n--- Тест с {count} запросами ---")

        start_time = time.time()
        _ = get_memory_usage()
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

    plt.figure(figsize=(10, 6))
    plt.plot(requests_count, avg_response_times, "bo-", linewidth=2, markersize=8)
    plt.title("Среднее время ответа HH.ru (HTML+BS4)")
    plt.xlabel("Количество запросов")
    plt.ylabel("Время ответа (мс)")
    plt.grid(True, alpha=0.3)
    plt.savefig("graphs/load_testing/response_time.png", dpi=300, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(requests_count, rps_list, "go-", linewidth=2, markersize=8)
    plt.title("Производительность HH.ru (RPS) (HTML+BS4)")
    plt.xlabel("Количество запросов")
    plt.ylabel("Запросов в секунду")
    plt.grid(True, alpha=0.3)
    plt.savefig("graphs/load_testing/rps_performance.png", dpi=300, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(requests_count, memory_usage_list, "ro-", linewidth=2, markersize=8)
    plt.title("Использование памяти при нагрузочном тестировании (HTML+BS4)")
    plt.xlabel("Количество запросов")
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
        if any(kw in text for kw in keywords):
            found.add(tech)
    return list(found)


# ---------- EXPLORATORY ANALYSIS ----------

def exploratory_analysis(filename: str = "programming_vacancies.json") -> Tuple[pd.DataFrame, pd.Series]:
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

        experience = (vac.get("experience") or {}).get("name", "Не указан")
        employment = (vac.get("employment") or {}).get("name", "Не указана")

        snippet = vac.get("snippet") or {}
        requirement = snippet.get("requirement", "") or ""
        responsibility = snippet.get("responsibility", "") or ""
        vacancy_snippet = f"{requirement} {responsibility}"

        technologies = extract_technologies(vac.get("name", "") or "", vacancy_snippet)
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

    # Графики — без изменений (как у вас)
    plt.figure(figsize=(12, 8))
    salaries = df["salary_avg"].dropna()
    if len(salaries) > 0:
        plt.hist(salaries, bins=20, alpha=0.7, color="skyblue", edgecolor="black")
        plt.title("Распределение средних зарплат программистов")
        plt.xlabel("Зарплата")
        plt.ylabel("Количество вакансий")
        plt.grid(True, alpha=0.3)
        plt.savefig("graphs/exploratory_analysis/salary_distribution.png", dpi=300, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(12, 8))
    exp_salary = df.groupby("experience")["salary_avg"].mean().dropna()
    if len(exp_salary) > 0:
        colors = ["#ff9999", "#66b3ff", "#99ff99", "#ffcc99"]
        exp_salary.plot(kind="bar", color=colors)
        plt.title("Средняя зарплата по опыту работы")
        plt.xlabel("Опыт работы")
        plt.ylabel("Средняя зарплата")
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        for i, v in enumerate(exp_salary):
            plt.text(i, v + 5000, f"{v:.0f}", ha="center", va="bottom")
        plt.savefig("graphs/exploratory_analysis/salary_by_experience.png", dpi=300, bbox_inches="tight")
    plt.close()

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

    plt.figure(figsize=(12, 8))
    if len(tech_counts) > 0:
        top_tech = tech_counts.head(15)
        colors = plt.cm.Set3(range(len(top_tech)))
        bars = plt.barh(range(len(top_tech)), top_tech.values, color=colors)
        plt.yticks(range(len(top_tech)), top_tech.index)
        plt.title("Топ-15 востребованных технологий")
        plt.xlabel("Количество упоминаний")
        plt.gca().invert_yaxis()
        plt.grid(True, alpha=0.3)
        for bar in bars:
            w = bar.get_width()
            plt.text(w + 1, bar.get_y() + bar.get_height() / 2, f"{int(w)}", ha="left", va="center")
        plt.savefig("graphs/technologies/top_15_technologies.png", dpi=300, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(14, 8))
    if len(tech_counts) > 0:
        top_tech_all = tech_counts.head(20)
        colors = plt.cm.viridis(range(len(top_tech_all)))
        bars = plt.bar(range(len(top_tech_all)), top_tech_all.values, color=colors)
        plt.xticks(range(len(top_tech_all)), top_tech_all.index, rotation=45, ha="right")
        plt.title("Топ-20 востребованных технологий в программировании")
        plt.xlabel("Технологии")
        plt.ylabel("Количество упоминаний")
        plt.grid(True, alpha=0.3)
        for bar in bars:
            h = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2.0, h + 1, f"{int(h)}", ha="center", va="bottom")
        plt.tight_layout()
        plt.savefig("graphs/technologies/top_20_technologies.png", dpi=300, bbox_inches="tight")
    plt.close()

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

    return df, tech_counts


def print_detailed_analysis(df: pd.DataFrame, tech_counts: pd.Series) -> None:
    print("\n" + "=" * 50)
    print("ДЕТАЛЬНЫЙ АНАЛИЗ РЕЗУЛЬТАТОВ")
    print("=" * 50)

    print("\nОБЩАЯ СТАТИСТИКА:")
    print(f"   Всего вакансий: {len(df)}")
    print(f"   Вакансий с указанной зарплатой: {df['salary_avg'].notna().sum()}")
    print(f"   Средняя зарплата: {df['salary_avg'].mean():.2f}")
    print(f"   Медианная зарплата: {df['salary_avg'].median():.2f}")
    print(f"   Максимальная зарплата: {df['salary_avg'].max():.2f}")
    print(f"   Минимальная зарплата: {df['salary_avg'].min():.2f}")

    print("\nТОП-10 ТЕХНОЛОГИЙ:")
    for i, (tech, count) in enumerate(tech_counts.head(10).items(), 1):
        percentage = (count / len(df)) * 100 if len(df) else 0
        print(f"   {i:2d}. {tech:<15} {count:>3} упоминаний ({percentage:.1f}%)")

    print("\nРАСПРЕДЕЛЕНИЕ ПО ОПЫТУ:")
    for exp, count in df["experience"].value_counts().items():
        percentage = (count / len(df)) * 100 if len(df) else 0
        print(f"   {exp:<20} {count:>3} вакансий ({percentage:.1f}%)")

    print("\nТИПЫ ЗАНЯТОСТИ:")
    for emp, count in df["employment"].value_counts().items():
        percentage = (count / len(df)) * 100 if len(df) else 0
        print(f"   {emp:<20} {count:>3} вакансий ({percentage:.1f}%)")


if __name__ == "__main__":
    create_directories()

    # --- Сбор вакансий (если нужно)
    parser = HHParserBS4()
    search_queries = ["Golang", "ML Engineer", "PostgreSQL", "Программист", "Разработчик"]
    all_vacancies: List[Dict[str, Any]] = []
    for q in search_queries:
        print(f"\n=== Поиск вакансий для: '{q}' ===")
        all_vacancies.extend(parser.parse_vacancies(search_query=q, pages_to_parse=3))
    parser.save_to_json(all_vacancies, "programming_vacancies.json")

    # --- Нагрузочное тестирование
    print("\n=== НАГРУЗОЧНОЕ ТЕСТИРОВАНИЕ ===")
    _load_metrics = load_testing()

    # --- Аналитика
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
