from __future__ import annotations

import json

import requests

from app.core.config import RAW_DIR, settings

BASE_URL = "https://ecos.bok.or.kr/api/StatisticSearch"

# 통계표코드/통계항목코드 참고: https://ecos.bok.or.kr/api/#/
BASE_RATE_STAT_CODE = "722Y001"
BASE_RATE_ITEM_CODE = "0101000"  # 한국은행 기준금리


def fetch_series(
    stat_code: str, item_code: str, start: str, end: str, cycle: str = "M"
) -> list[dict]:
    """ECOS 통계 시계열 조회. start/end는 cycle에 맞는 포맷(예: 월배열 YYYYMM)."""
    cache_path = RAW_DIR / "ecos" / f"{stat_code}_{item_code}_{cycle}_{start}_{end}.json"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if cache_path.exists():
        return json.loads(cache_path.read_text(encoding="utf-8"))

    url = (
        f"{BASE_URL}/{settings.ecos_api_key}/json/kr/1/1000/"
        f"{stat_code}/{cycle}/{start}/{end}/{item_code}"
    )
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    payload = resp.json()

    rows = payload.get("StatisticSearch", {}).get("row", [])
    if not rows and "RESULT" in payload:
        raise RuntimeError(f"ECOS API error: {payload['RESULT']}")

    cache_path.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
    return rows


def fetch_base_rate(start_year: int, end_year: int) -> list[dict]:
    """기준금리 연간 시계열: [{year, base_rate}]. 월별 데이터를 연평균으로 집계."""
    rows = fetch_series(
        BASE_RATE_STAT_CODE, BASE_RATE_ITEM_CODE, f"{start_year}01", f"{end_year}12", cycle="M"
    )

    yearly: dict[int, list[float]] = {}
    for row in rows:
        time_val = row.get("TIME", "")
        value = row.get("DATA_VALUE")
        if not time_val or value in (None, ""):
            continue
        year = int(time_val[:4])
        yearly.setdefault(year, []).append(float(value))

    return [
        {"year": year, "base_rate": sum(values) / len(values)}
        for year, values in sorted(yearly.items())
    ]
