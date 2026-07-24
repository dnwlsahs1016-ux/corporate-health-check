from __future__ import annotations

import io
import json
import xml.etree.ElementTree as ET
import zipfile
from functools import lru_cache

import requests

from app.core.config import RAW_DIR, settings

BASE_URL = "https://opendart.fss.or.kr/api"
CORP_CODE_CACHE = RAW_DIR / "corp_code.json"


class DartQuotaExceeded(Exception):
    """일일 API 호출 한도 초과(status=020). 지금까지 캐싱된 결과는 유지하고
    호출부에서 전체 수집을 중단, 다음 날(한도 리셋 후) 재실행하면 캐시된 것은
    건너뛰고 나머지만 이어서 받는다."""


class DartAPIError(Exception):
    """그 외 일시적/예상치 못한 DART API 오류. 캐싱하지 않아 재실행 시 재시도된다."""


def _download_corp_code_map() -> dict[str, dict]:
    """DART 전체 기업 corp_code 목록을 받아 회사명 -> {corp_code, stock_code} 매핑을 만든다."""
    resp = requests.get(
        f"{BASE_URL}/corpCode.xml", params={"crtfc_key": settings.dart_api_key}, timeout=30
    )
    resp.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        xml_bytes = zf.read(zf.namelist()[0])

    root = ET.fromstring(xml_bytes)
    mapping: dict[str, dict] = {}
    for node in root.findall("list"):
        corp_name = (node.findtext("corp_name") or "").strip()
        corp_code = (node.findtext("corp_code") or "").strip()
        stock_code = (node.findtext("stock_code") or "").strip()
        if not corp_name or not corp_code:
            continue
        mapping[corp_name] = {"corp_code": corp_code, "stock_code": stock_code}
    return mapping


@lru_cache(maxsize=1)
def get_corp_code_map() -> dict[str, dict]:
    if CORP_CODE_CACHE.exists():
        return json.loads(CORP_CODE_CACHE.read_text(encoding="utf-8"))
    mapping = _download_corp_code_map()
    CORP_CODE_CACHE.write_text(json.dumps(mapping, ensure_ascii=False), encoding="utf-8")
    return mapping


def resolve_corp_code(corp_name: str, stock_code: str | None = None) -> str | None:
    """회사명(정확히 일치) 또는 종목코드로 DART corp_code를 찾는다."""
    mapping = get_corp_code_map()

    if corp_name in mapping:
        return mapping[corp_name]["corp_code"]

    if stock_code:
        for info in mapping.values():
            if info.get("stock_code") == stock_code:
                return info["corp_code"]

    # 부분 일치 fallback (사명 변경/공백 차이 대응)
    for name, info in mapping.items():
        if corp_name in name or name in corp_name:
            return info["corp_code"]

    return None


def fetch_financial_statement(
    corp_code: str, year: int, reprt_code: str = "11011"
) -> list[dict]:
    """단일회사 전체 재무제표(fnlttSinglAcntAll). 연결(CFS) 우선, 없으면 개별(OFS)로 재시도.
    응답 원자료는 raw/dart 아래 캐싱한다."""
    cache_path = RAW_DIR / "dart" / f"{corp_code}_{year}_{reprt_code}.json"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if cache_path.exists():
        return json.loads(cache_path.read_text(encoding="utf-8"))

    records: list[dict] = []
    got_definitive_answer = False
    for fs_div in ("CFS", "OFS"):
        resp = requests.get(
            f"{BASE_URL}/fnlttSinglAcntAll.json",
            params={
                "crtfc_key": settings.dart_api_key,
                "corp_code": corp_code,
                "bsns_year": str(year),
                "reprt_code": reprt_code,
                "fs_div": fs_div,
            },
            timeout=30,
        )
        resp.raise_for_status()
        payload = resp.json()
        status = payload.get("status")

        if status == "000":
            records = payload.get("list", [])
            got_definitive_answer = True
            break
        if status == "013":  # 조회된 데이터 없음 -> 이 회사/연도는 정말로 데이터가 없는 것
            got_definitive_answer = True
            continue
        if status == "020":
            raise DartQuotaExceeded(payload.get("message", "일일 호출 한도를 초과했습니다."))
        raise DartAPIError(f"DART API 오류(status={status}): {payload.get('message')}")

    if not got_definitive_answer:
        raise DartAPIError(f"corp_code={corp_code} year={year}: 알 수 없는 응답")

    cache_path.write_text(json.dumps(records, ensure_ascii=False), encoding="utf-8")
    return records


def fetch_company_profile(corp_code: str) -> dict:
    """회사개황(company.json). 업종코드(induty_code) 등 포함."""
    cache_path = RAW_DIR / "dart" / f"profile_{corp_code}.json"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if cache_path.exists():
        return json.loads(cache_path.read_text(encoding="utf-8"))

    resp = requests.get(
        f"{BASE_URL}/company.json",
        params={"crtfc_key": settings.dart_api_key, "corp_code": corp_code},
        timeout=30,
    )
    resp.raise_for_status()
    payload = resp.json()
    cache_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return payload
