import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests


# --- Конфигурация ---
HH_SEARCH_URL = "https://api.hh.ru/vacancies"

SEARCH_TEXT = "NAME:Программист"
AREA_ID = 1
PER_PAGE = 100

PAGES_TO_FETCH = 4

SLEEP_BETWEEN_PAGES = 0.2
SLEEP_BETWEEN_VACANCIES = 0.25

PAGINATION_DIR = Path("pagination_desc")
VACANCIES_DIR = Path("vacancies_desc")

RESUME_KEYWORDS: List[str] = [
    "Golang-разработчик", "Golang-developer",
    "машинное обучение", "machine learning",
    "PostgreSQL", "Backend"
]


# --- Утилиты ---
_TAG_RE = re.compile(r"<[^<]+?>")


def ensure_dirs() -> None:
    """Создает папки для сохранения данных (если их нет)."""
    PAGINATION_DIR.mkdir(parents=True, exist_ok=True)
    VACANCIES_DIR.mkdir(parents=True, exist_ok=True)


def get_search_page(session: requests.Session, page: int = 0) -> Dict[str, Any]:
    """
    Получает страницу поиска вакансий (JSON -> dict).
    page: индекс страницы (с 0).
    """
    params = {
        "text": SEARCH_TEXT,
        "area": AREA_ID,
        "page": page,
        "per_page": PER_PAGE,
    }

    resp = session.get(HH_SEARCH_URL, params=params, timeout=30)
    resp.raise_for_status()
    return resp.json()


def save_json(path: Path, obj: Any) -> None:
    """Сохраняет объект как JSON (UTF-8)."""
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False)


def load_json(path: Path) -> Dict[str, Any]:
    """Читает JSON из файла."""
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def contains_resume_keywords(description: Optional[str]) -> bool:
    """Проверяет, содержит ли описание вакансии ключевые слова из резюме."""
    if not description:
        return False

    clean = _TAG_RE.sub(" ", description).lower()

    found = [kw for kw in RESUME_KEYWORDS if kw.lower() in clean]
    if found:
        print(f"Найдены ключевые слова: {found}")
        return True
    return False


def fetch_and_save_search_pages(session: requests.Session) -> None:
    """Скачивает страницы поиска, сохраняет их и соблюдает остановку на последней странице."""
    for page in range(PAGES_TO_FETCH):
        try:
            js_obj = get_search_page(session, page=page)

            out_file = PAGINATION_DIR / f"page_{page}.json"
            save_json(out_file, js_obj)
            print(f"Сохранена страница {page}")

            # Проверка на последнюю страницу (логика сохранена)
            if (js_obj["pages"] - page) <= 1:
                print(f'Достигнута последняя страница. Всего страниц: {js_obj["pages"]}')
                break

            time.sleep(SLEEP_BETWEEN_PAGES)

        except Exception as e:
            print(f"Ошибка при обработке страницы {page}: {e}")
            break


def process_vacancies(session: requests.Session) -> None:
    """Читает сохраненные страницы, тянет детали по вакансиям, фильтрует и сохраняет подходящие."""
    total_vacancies = 0
    matched_vacancies = 0

    for page_file in os.listdir(PAGINATION_DIR):
        try:
            json_obj = load_json(PAGINATION_DIR / page_file)

            for v in json_obj.get("items", []):
                total_vacancies += 1

                # Получаем детали вакансии
                resp = session.get(v["url"], timeout=30)
                resp.raise_for_status()
                vacancy_data = resp.json()

                description = vacancy_data.get("description", "")

                if contains_resume_keywords(description):
                    matched_vacancies += 1

                    # Сохраняем ТОЛЬКО подходящие вакансии (как и было)
                    out_file = VACANCIES_DIR / f'{v["id"]}.json'
                    # сохраняем исходный JSON-объект, а не строку (функционал тот же: файл с данными вакансии)
                    save_json(out_file, vacancy_data)

                    print(f'Сохранена подходящая вакансия: {v["name"]}')

                time.sleep(SLEEP_BETWEEN_VACANCIES)

        except Exception as e:
            print(f"Ошибка при обработке файла {page_file}: {e}")
            continue

    print("Сбор вакансий завершен!")
    print(f"Всего обработано вакансий: {total_vacancies}")
    print(f"Сохранено подходящих вакансий: {matched_vacancies}")


def main() -> None:
    ensure_dirs()

    with requests.Session() as session:
        fetch_and_save_search_pages(session)

        print("Страницы поиска собраны. Далее получаем список вакансий...")

        process_vacancies(session)


if __name__ == "__main__":
    main()
