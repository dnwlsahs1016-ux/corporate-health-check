"""코스닥 전체 유니버스 시드(backend/data/delisted_companies.csv) 생성.

KRX KIND에서 (1) 현재 코스닥 상장사 전체 목록과 (2) 2015년 이후 상장폐지 이력 전체를
받아온 뒤, 폐지 사유 텍스트를 키워드로 분류해 재무적 사유(financial_distress)만 골라내고
현재 상장사 중 일부를 건전기업(healthy_benchmark) 샘플로 뽑아 하나의 CSV로 합친다.

사용법 (backend 디렉터리에서):
    ./.venv/Scripts/python.exe scripts/build_universe.py
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.config import DATA_DIR  # noqa: E402

HEALTHY_SAMPLE_SIZE = 500
RANDOM_SEED = 42

# 재무적 사유가 아닌(M&A/SPAC합병/자진상장폐지/이전상장 등) 것으로 판단되는 키워드.
# 이 키워드가 하나라도 포함되면 제외하고, 나머지는 전부 financial_distress로 분류한다
# (감사의견거절/자본잠식/부도/실질심사 등은 모두 재무위험과 직접 연관).
EXCLUDE_KEYWORDS = ["합병", "유가증권시장 상장", "완전자회사", "상장폐지신청", "상장폐지 신청", "상장예비심사"]


def fetch_listed_companies() -> pd.DataFrame:
    resp = requests.get(
        "https://kind.krx.co.kr/corpgeneral/corpList.do",
        params={"method": "download", "searchType": "13"},
        timeout=30,
    )
    resp.encoding = "euc-kr"
    df = pd.read_html(io.StringIO(resp.text))[0]
    df.columns = [
        "corp_name",
        "market",
        "stock_code",
        "industry",
        "main_product",
        "listing_date",
        "settlement_month",
        "ceo",
        "homepage",
        "region",
    ]
    df["stock_code"] = df["stock_code"].astype(str).str.zfill(6)
    return df[df["market"] == "코스닥"].reset_index(drop=True)


def fetch_delisted_companies(from_date: str, to_date: str) -> pd.DataFrame:
    resp = requests.post(
        "https://kind.krx.co.kr/investwarn/delcompany.do",
        data={
            "method": "searchDelCompanySub",
            "currentPageSize": "3000",
            "pageIndex": "1",
            "marketType": "2",  # 코스닥
            "fromDate": from_date,
            "toDate": to_date,
            "searchCodeType": "",
            "forward": "delcompany_sub",
        },
        timeout=30,
    )
    resp.encoding = "utf-8"
    df = pd.read_html(io.StringIO(resp.text))[0]
    df.columns = ["no", "corp_name", "delisting_date", "reason", "note"]
    return df


def classify_reason(reason: str) -> str:
    reason = str(reason)
    if any(kw in reason for kw in EXCLUDE_KEYWORDS):
        return "exclude"
    return "financial_distress"


def main() -> None:
    print("[1/3] 코스닥 상장사 전체 목록 수집 중...")
    listed = fetch_listed_companies()
    print(f"  -> {len(listed)}개사")

    print("[2/3] 2015년 이후 상장폐지 이력 수집 및 분류 중...")
    delisted = fetch_delisted_companies("2015-01-01", "2026-12-31")
    delisted["label_category"] = delisted["reason"].apply(classify_reason)
    distress = delisted[delisted["label_category"] == "financial_distress"].copy()
    print(f"  -> 전체 {len(delisted)}건 중 재무적 사유 {len(distress)}건")

    print(f"[3/3] 건전기업 샘플 {HEALTHY_SAMPLE_SIZE}개 추출 및 CSV 저장 중...")
    healthy = listed.sample(n=HEALTHY_SAMPLE_SIZE, random_state=RANDOM_SEED)

    rows = []
    for _, r in distress.iterrows():
        rows.append(
            {
                "corp_name": r["corp_name"],
                "stock_code": "",
                "category": "financial_distress",
                "delisting_date": r["delisting_date"],
                "reason": r["reason"],
            }
        )
    for _, r in healthy.iterrows():
        rows.append(
            {
                "corp_name": r["corp_name"],
                "stock_code": r["stock_code"],
                "category": "healthy_benchmark",
                "delisting_date": "",
                "reason": "",
            }
        )

    out = pd.DataFrame(rows).drop_duplicates(subset=["corp_name"], keep="first")
    out_path = DATA_DIR / "delisted_companies.csv"
    out.to_csv(out_path, index=False, encoding="utf-8")
    print(f"  -> {out_path} 에 {len(out)}개 기업 저장 완료 (재무적사유 {len(distress)} + 건전기업 {len(healthy)})")


if __name__ == "__main__":
    main()
