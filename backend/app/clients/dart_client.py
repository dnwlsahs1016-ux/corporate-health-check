from __future__ import annotations

import io
import json
import re
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


# 회사명 표기 차이(공백, 법인형태 표기)만 제거하기 위한 패턴. 여기 없는 글자는 절대
# 제거하지 않는다 — "한국전자"처럼 짧은 이름이 "한국전자통신"(완전히 다른 회사)과
# 섞이는 걸 막으려면, 정규화는 반드시 "의미 없는 표기 차이"로만 한정해야 한다.
_CORP_SUFFIX_PATTERN = re.compile(r"(\(주\)|㈜|주식회사|홀딩스|지주회사|지주|\s+)")


def _normalize_corp_name(name: str) -> str:
    return _CORP_SUFFIX_PATTERN.sub("", name)


def resolve_corp_code(corp_name: str, stock_code: str | None = None) -> str | None:
    """회사명(정확히 일치) 또는 종목코드로 DART corp_code를 찾는다."""
    mapping = get_corp_code_map()

    if corp_name in mapping:
        return mapping[corp_name]["corp_code"]

    if stock_code:
        for info in mapping.values():
            if info.get("stock_code") == stock_code:
                return info["corp_code"]

    # 부분 일치(substring) fallback은 쓰지 않는다. dict 순회 순서에 의존해 "먼저 걸리는
    # 아무 회사"를 채택하면 조용한 오매칭(silent mismatch)이 생긴다 — 예: "한국전자"를
    # 찾다가 완전히 무관한 "한국전자통신"이 먼저 걸려 그 회사 재무데이터가 섞여 들어갈 수
    # 있다. 대신 공백/법인형태 표기((주), 홀딩스 등) 차이만 제거한 뒤 "완전히 같아지는"
    # 경우만 매칭으로 인정한다. 그마저도 후보가 여러 개면(모호하면) 매칭 실패로 남겨
    # 수동 검토 대상이 되게 한다 — 틀린 데이터가 섞이는 것보다 실패가 눈에 보이는 편이
    # 훨씬 안전하다.
    normalized_target = _normalize_corp_name(corp_name)
    normalized_candidates = [
        (name, info) for name, info in mapping.items() if _normalize_corp_name(name) == normalized_target
    ]

    if len(normalized_candidates) == 1:
        name, info = normalized_candidates[0]
        print(f"[dart_client] '{corp_name}' -> '{name}' 표기차이 정규화로 매칭")
        return info["corp_code"]

    if len(normalized_candidates) > 1:
        print(
            f"[dart_client] '{corp_name}' 정규화 후에도 후보 {len(normalized_candidates)}개로 "
            f"모호해 매칭 보류: {[name for name, _ in normalized_candidates[:5]]}"
        )

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
