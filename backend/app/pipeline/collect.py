from __future__ import annotations

import csv

import pandas as pd

from app.clients import dart_client
from app.clients.dart_client import DartAPIError, DartQuotaExceeded
from app.core.config import DATA_DIR, PROCESSED_DIR, settings
from app.pipeline.features import compute_altman_zscore, compute_ratios, extract_line_items

SEED_CSV = DATA_DIR / "delisted_companies.csv"


def load_seed_companies() -> list[dict]:
    with open(SEED_CSV, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def collect_company_panel(corp_name: str, stock_code: str | None) -> pd.DataFrame:
    """한 기업의 연도별 원자료 line item + 재무비율 패널을 만든다."""
    corp_code = dart_client.resolve_corp_code(corp_name, stock_code or None)
    if corp_code is None:
        print(f"[collect] corp_code를 찾지 못함: {corp_name}")
        return pd.DataFrame()

    try:
        profile = dart_client.fetch_company_profile(corp_code)
        industry_code = (profile.get("induty_code") or "unknown")[:2]
    except DartAPIError as exc:
        print(f"[collect] {corp_name} 회사개황 조회 실패({exc}), industry_code=unknown 처리")
        industry_code = "unknown"

    yearly_items: dict[int, dict] = {}
    for year in range(settings.start_year, settings.end_year + 1):
        try:
            records = dart_client.fetch_financial_statement(corp_code, year)
        except DartAPIError as exc:
            print(f"[collect] {corp_name} {year}년 재무제표 조회 실패({exc}), 건너뜀")
            continue
        if not records:
            continue
        yearly_items[year] = extract_line_items(records)

    rows = []
    sorted_years = sorted(yearly_items)
    for i, year in enumerate(sorted_years):
        curr = yearly_items[year]
        prev = yearly_items.get(sorted_years[i - 1]) if i > 0 else None
        ratios = compute_ratios(curr, prev)
        row = {
            "corp_name": corp_name,
            "corp_code": corp_code,
            "stock_code": stock_code,
            "industry_code": industry_code,
            "year": year,
            **{f"raw_{k}": v for k, v in curr.items()},
            **ratios,
            "altman_zscore": compute_altman_zscore(curr),
        }
        rows.append(row)

    return pd.DataFrame(rows)


def collect_all() -> pd.DataFrame:
    companies = load_seed_companies()
    frames = []
    for i, company in enumerate(companies):
        try:
            df = collect_company_panel(company["corp_name"], company.get("stock_code") or None)
        except DartQuotaExceeded as exc:
            remaining = len(companies) - i
            print(
                f"[collect] DART 일일 API 한도 초과({exc}). "
                f"{i}/{len(companies)}개 기업 수집 완료, {remaining}개 남음. "
                "지금까지 결과를 저장하고 중단합니다. 내일 다시 run_pipeline.py를 실행하면 "
                "이미 받은 데이터는 캐시에서 건너뛰고 이어서 받습니다."
            )
            break
        if not df.empty:
            df["category"] = company["category"]
            df["delisting_date"] = company.get("delisting_date") or None
            frames.append(df)

    panel = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(PROCESSED_DIR / "raw_panel.parquet", index=False)
    return panel
