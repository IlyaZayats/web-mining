import json
import os
import time
from typing import Dict, List, Any, Optional

from zenrows import ZenRowsClient


def _pct_encode(value: str) -> str:
    """
    Percent-encode (RFC 3986).
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
        key = _pct_encode(str(k))
        val = _pct_encode(str(v))
        pairs.append(f"{key}={val}")
    return "&".join(pairs)


class HHParserZenRows:
    """
    Парсер для API HeadHunter для поиска вакансий, связанных с программированием,
    но запросы выполняются через ZenRows SDK.

    Требования:
      - pip install zenrows
      - ZENROWS_API_KEY в окружении или api_key в конструкторе
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        retries: int = 1,
        concurrency: int = 1,
        hh_base_url: str = "https://api.hh.ru/vacancies",
        user_agent: str = (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/122.0.0.0 Safari/537.36"
        ),
        zenrows_params: Optional[Dict[str, Any]] = None,
    ):
        self.api_key = api_key or os.getenv("ZENROWS_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Не найден ключ ZenRows. Укажите api_key в конструкторе "
                "или задайте переменную окружения ZENROWS_API_KEY."
            )

        self.hh_base_url = hh_base_url

        # Заголовки, которые ZenRows передаст целевому сайту (HH API)
        self.headers = {
            "User-Agent": user_agent,
            "Accept": "application/json, text/plain, */*",
        }

        # Клиент ZenRows
        self.client = ZenRowsClient(self.api_key, retries=retries, concurrency=concurrency)

        # Параметры ZenRows (для API HH обычно НЕ нужно js_render/premium_proxy,
        # но оставим возможность включить при необходимости)
        self.zenrows_params = {}
        if zenrows_params:
            self.zenrows_params.update(zenrows_params)

    def search_vacancies(
        self,
        text: str,
        area: int = 1,
        per_page: int = 100,
        page: int = 0
    ) -> Optional[Dict[str, Any]]:
        """
        Поиск вакансий по текстовому запросу.

        Args:
            text (str): Текст для поиска (напр., "Python разработчик").
            area (int): ID региона (1 - Москва, 2 - СПб, 113 - Россия).
            per_page (int): Количество вакансий на странице (макс. 100).
            page (int): Номер страницы (начинается с 0).

        Returns:
            Dict[str, Any]: JSON-ответ от API или None в случае ошибки.
        """
        hh_params = {
            "text": text,
            "area": area,
            "per_page": per_page,
            "page": page,
            "only_with_salary": True,
        }

        query = _build_query(hh_params)
        target_url = f"{self.hh_base_url}?{query}"

        try:
            # params здесь — параметры ZenRows (js_render, premium_proxy и т.д.), а не HH
            response = self.client.get(target_url, params=self.zenrows_params, headers=self.headers)

            # ZenRowsClient возвращает requests.Response -> можно .json()
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"Ошибка при запросе: {e}")
            return None

    def parse_vacancies(self, search_query: str, pages_to_parse: int = 5) -> List[Dict[str, Any]]:
        """
        Парсит несколько страниц с вакансиями.

        Args:
            search_query (str): Запрос для поиска.
            pages_to_parse (int): Количество страниц для парсинга.

        Returns:
            List[Dict[str, Any]]: Список спарсенных вакансий.
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
        """Сохраняет данные в JSON файл."""
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"Данные сохранены в {filename}")


if __name__ == "__main__":
    parser = HHParserZenRows(
        api_key="b93e3a059bf38d2d27db5727152acaf81b9c6b01",
        retries=1,
        concurrency=1,
    )

    search_queries = [
        "Golang",
        "ML Engineer",
        "PostgreSQL",
        "Программист",
        "Разработчик",
    ]

    all_vacancies_data: List[Dict[str, Any]] = []

    for query in search_queries:
        print(f"\n=== Поиск вакансий для: '{query}' ===")
        vacancies = parser.parse_vacancies(search_query=query, pages_to_parse=3)
        all_vacancies_data.extend(vacancies)

    parser.save_to_json(all_vacancies_data, "programming_vacancies.json")
