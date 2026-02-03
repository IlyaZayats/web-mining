import json
import time
from collections import Counter
from typing import Any, Dict, List

import requests


TOKEN = "fef5fbcafef5fbcafef5fbcab5fdcb49b7ffef5fef5fbca9777fdf34d762500eb280e2e"
BASE_URL = "https://api.vk.com/method/"
API_VERSION = "5.199"
GROUP_ID = "samosbor_original"

BATCH_SIZE = 1000  # максимум за один запрос (VK groups.getMembers)


class VKAPIError(RuntimeError):
    pass


def vk_call(session: requests.Session, method: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """Единая точка вызова VK API + обработка ошибок."""
    payload = {
        "v": API_VERSION,
        "access_token": TOKEN,
        **params,
    }

    resp = session.post(BASE_URL + method, data=payload, timeout=30)
    data = resp.json()

    if "error" in data:
        err = data["error"]
        code = err.get("error_code")
        msg = err.get("error_msg")
        raise VKAPIError(f"VK API error {code}: {msg}")

    if "response" not in data:
        raise VKAPIError(f"Unexpected response: {data}")

    return data["response"]


def get_total_members(session: requests.Session, group_id: str) -> int:
    """Получаем общее число участников (count)."""
    response = vk_call(session, "groups.getMembers", {"group_id": group_id, "count": 1})
    return int(response["count"])


def fetch_all_members(
    session: requests.Session,
    group_id: str,
    batch_size: int = BATCH_SIZE,
    sleep_sec: float = 0.34,  # небольшая пауза для снижения риска rate limit
) -> List[Dict[str, Any]]:
    """Скачиваем всех участников пачками."""
    total = get_total_members(session, group_id)
    print("Всего участников:", total)

    members: List[Dict[str, Any]] = []
    offset = 0

    while offset < total:
        response = vk_call(
            session,
            "groups.getMembers",
            {
                "group_id": group_id,
                "lang": 0,
                "count": batch_size,
                "offset": offset,
                "fields": "online,sex,city,universities",
            },
        )

        items = response.get("items", [])
        members.extend(items)

        offset += batch_size
        print(f"Загружено {len(members)} участников...")

        if sleep_sec:
            time.sleep(sleep_sec)

    return members


def analyze_members(members: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Подсчёты: пол, онлайн, города, университеты."""
    male_count = sum(1 for m in members if m.get("sex") == 2)
    female_count = sum(1 for m in members if m.get("sex") == 1)
    no_count = sum(1 for m in members if m.get("sex") == 0)

    online_count = sum(1 for m in members if m.get("online") == 1)

    cities = Counter(m["city"]["title"] for m in members if isinstance(m.get("city"), dict) and "title" in m["city"])

    universities = Counter()
    for m in members:
        unis = m.get("universities")
        if isinstance(unis, list) and unis:
            for uni in unis:
                if isinstance(uni, dict):
                    name = uni.get("name")
                    if name:
                        universities[name] += 1
        else:
            universities["Не указано"] += 1

    return {
        "female": female_count,
        "male": male_count,
        "no_sex": no_count,
        "online": online_count,
        "cities": cities,
        "universities": universities,
    }


def main() -> None:
    with requests.Session() as session:
        try:
            all_members = fetch_all_members(session, GROUP_ID, batch_size=BATCH_SIZE)
        except VKAPIError as e:
            print("Ошибка VK:", e)
            return
        except requests.RequestException as e:
            print("Ошибка сети:", e)
            return

    stats = analyze_members(all_members)

    print("\nПервые 5 участников:")
    print(json.dumps(all_members[:5], ensure_ascii=False, indent=2))

    print("\nУчастники (женщины):", stats["female"])
    print("Участники (мужчины):", stats["male"])
    print("Участники (пол не указан):", stats["no_sex"])

    print("\nОнлайн:", stats["online"])

    print("\nРаспределение по городам:")
    for city, count in stats["cities"].most_common(10):  # топ-10
        print(f"  {city}: {count}")

    print("\nРаспределение по университетам:")
    for uni, count in stats["universities"].most_common(10):  # топ-10
        print(f"  {uni}: {count}")


if __name__ == "__main__":
    main()
