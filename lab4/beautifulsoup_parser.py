import json
import os
import re
import time
from typing import Dict, List, Any, Optional, Tuple

import requests
from bs4 import BeautifulSoup


class HHParserBS4:
    """
    Парсер HeadHunter через HTML (hh.ru/search/vacancy) + BeautifulSoup.
      - search_vacancies(text, area, per_page, page) -> dict c ключами items/pages
      - parse_vacancies(search_query, pages_to_parse) -> list вакансий
      - save_to_json(data, filename)

    ВАЖНО:
      - Это не HH API, а парсинг HTML. Структура страницы может меняться.
      - Иногда HH может показывать капчу/антибот — в таком случае вернётся None.
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

    # ---------- helpers ----------

    @staticmethod
    def _clean_text(s: Optional[str]) -> str:
        return re.sub(r"\s+", " ", (s or "")).strip()

    @staticmethod
    def _extract_vacancy_id(url: str) -> str:
        # Примеры: https://hh.ru/vacancy/12345678?... или /vacancy/12345678
        m = re.search(r"/vacancy/(\d+)", url)
        return m.group(1) if m else ""

    @staticmethod
    def _parse_salary(salary_text: str) -> Optional[Dict[str, Any]]:
        """
        Преобразует строку зарплаты HH в формат, близкий к API:
        {'from': int|None, 'to': int|None, 'currency': 'RUR'|'USD'|..., 'gross': None}
        """
        text = salary_text.lower()
        text = text.replace("\u202f", " ").replace("\xa0", " ")
        text = re.sub(r"\s+", " ", text).strip()

        if not text:
            return None

        # Валюта (очень грубо, но достаточно для аналитики)
        currency = None
        if "руб" in text or "₽" in text:
            currency = "RUR"
        elif "$" in text or "usd" in text:
            currency = "USD"
        elif "€" in text or "eur" in text:
            currency = "EUR"
        elif "kzt" in text or "₸" in text:
            currency = "KZT"

        # Вытаскиваем все числа
        nums = [int(n.replace(" ", "")) for n in re.findall(r"(\d[\d ]*\d|\d)", text)]
        if not nums:
            return {"from": None, "to": None, "currency": currency, "gross": None}

        salary_from = None
        salary_to = None

        if "от" in text and "до" not in text:
            salary_from = nums[0]
        elif "до" in text and "от" not in text:
            salary_to = nums[0]
        elif len(nums) >= 2:
            salary_from, salary_to = nums[0], nums[1]
        else:
            # если непонятно, положим в from
            salary_from = nums[0]

        return {"from": salary_from, "to": salary_to, "currency": currency, "gross": None}

    @staticmethod
    def _detect_blocked(html: str) -> bool:
        """Простая эвристика: капча/антибот/блокировка."""
        low = html.lower()
        return any(
            key in low
            for key in (
                "captcha",
                "похоже, вы робот",
                "подтвердите, что вы не робот",
                "access denied",
                "forbidden",
            )
        )

    def _get_total_pages(self, soup: BeautifulSoup) -> int:
        """
        Определяет число страниц.
        HH обычно имеет пагинацию с data-qa='pager-page' (может меняться).
        """
        # Пробуем через data-qa
        page_links = soup.select('[data-qa="pager-page"]')
        pages = []
        for a in page_links:
            t = self._clean_text(a.get_text())
            if t.isdigit():
                pages.append(int(t))
        if pages:
            return max(pages)

        # Фолбэк: найти любые ссылки пагинации с цифрами
        for a in soup.find_all("a"):
            t = self._clean_text(a.get_text())
            if t.isdigit():
                pages.append(int(t))
        return max(pages) if pages else 1

    def _parse_serp_items(self, soup: BeautifulSoup) -> List[Dict[str, Any]]:
        """
        Парсит карточки вакансий со страницы поиска.
        Возвращает список "вакансий" в формате, похожем на API items.
        """
        items: List[Dict[str, Any]] = []

        # Карточки часто имеют data-qa="vacancy-serp__vacancy"
        cards = soup.select('[data-qa="vacancy-serp__vacancy"]')
        if not cards:
            # запасной вариант: иногда карточка — это "serp-item"
            cards = soup.select('[data-qa="serp-item"]')

        for card in cards:
            # Заголовок / ссылка
            title_a = (
                card.select_one('a[data-qa="serp-item__title"]')
                or card.select_one('a[data-qa="vacancy-serp__vacancy-title"]')
                or card.find("a", href=True)
            )
            if not title_a:
                continue

            name = self._clean_text(title_a.get_text())
            href = title_a.get("href", "")
            vid = self._extract_vacancy_id(href)

            # Зарплата
            salary_el = card.select_one('[data-qa="vacancy-serp__vacancy-compensation"]')
            salary_text = self._clean_text(salary_el.get_text()) if salary_el else ""
            salary = self._parse_salary(salary_text) if salary_text else None

            # Город/регион
            area_el = card.select_one('[data-qa="vacancy-serp__vacancy-address"]')
            area_name = self._clean_text(area_el.get_text()) if area_el else ""
            area = {"name": area_name} if area_name else {"name": None}

            # Сниппеты
            req_el = card.select_one('[data-qa="vacancy-serp__vacancy_snippet_requirement"]')
            resp_el = card.select_one('[data-qa="vacancy-serp__vacancy_snippet_responsibility"]')
            requirement = self._clean_text(req_el.get_text()) if req_el else ""
            responsibility = self._clean_text(resp_el.get_text()) if resp_el else ""
            snippet = {"requirement": requirement or None, "responsibility": responsibility or None}

            # Дата публикации (на SERP часто относительная)
            date_el = card.select_one('[data-qa="vacancy-serp__vacancy-date"]')
            published_at = self._clean_text(date_el.get_text()) if date_el else None

            # Опыт/занятость на SERP могут отсутствовать — оставим как "не указано"
            experience = {"name": "Не указан"}
            employment = {"name": "Не указана"}

            items.append(
                {
                    "id": vid or None,
                    "name": name or None,
                    "salary": salary,  # None или dict как в API
                    "experience": experience,
                    "employment": employment,
                    "area": area,
                    "published_at": published_at,
                    "snippet": snippet,
                    # Дополнительно (не мешает): ссылка на вакансию
                    "alternate_url": href or None,
                }
            )

        return items

    # ---------- public API ----------

    def search_vacancies(
        self,
        text: str,
        area: int = 1,
        per_page: int = 100,
        page: int = 0,
    ) -> Optional[Dict[str, Any]]:
        """
        Ищет вакансии через HTML поиск HH и возвращает структуру, похожую на API:
          {
            "items": [...],
            "page": page,
            "pages": total_pages,
            "per_page": per_page,
          }
        """
        # Параметры HH для поиска:
        # text — запрос, area — регион, page — страница (0..),
        # items_on_page — сколько на странице (не всегда строго выполняется),
        # only_with_salary — аналог фильтра зарплаты.
        params = {
            "text": text,
            "area": area,
            "page": page,
            "items_on_page": min(max(per_page, 1), 100),
            "only_with_salary": "true",
        }

        try:
            resp = self.session.get(self.base_url, params=params, timeout=self.timeout)
            resp.raise_for_status()
            html = resp.text

            if self._detect_blocked(html):
                print("Похоже, HH показал капчу/антибот. Попробуйте позже или используйте прокси/ZenRows.")
                return None

            soup = BeautifulSoup(html, "html.parser")

            items = self._parse_serp_items(soup)
            total_pages = self._get_total_pages(soup)

            return {
                "items": items,
                "page": page,
                "pages": total_pages,
                "per_page": params["items_on_page"],
            }
        except requests.RequestException as e:
            print(f"Ошибка при запросе: {e}")
            return None

    def parse_vacancies(self, search_query: str, pages_to_parse: int = 5) -> List[Dict[str, Any]]:
        """Парсит несколько страниц с вакансиями (как в исходнике)."""
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
    parser = HHParserBS4()

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
