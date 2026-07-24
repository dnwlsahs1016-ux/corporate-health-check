"""코스닥 전체 종목 시가총액 순위 수집 (네이버 금융).

목록 화면에서 "시총 순위" 정렬을 위해, 회사명 -> {rank, market_cap(백만원)} 매핑을
data/processed/market_cap.csv 로 저장한다. DART API와 무관하며 호출 한도가 없다.

사용법 (backend 디렉터리에서):
    ./.venv/Scripts/python.exe scripts/fetch_market_cap.py
"""

from __future__ import annotations

import io
import sys
import time
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.config import PROCESSED_DIR  # noqa: E402

HEADERS = {"User-Agent": "Mozilla/5.0"}
MAX_PAGES = 40  # 코스닥 전체 상장사(~1840개) 커버하기에 충분한 여유


def fetch_all_market_cap() -> pd.DataFrame:
    frames = []
    for page in range(1, MAX_PAGES + 1):
        resp = requests.get(
            "https://finance.naver.com/sise/sise_market_sum.naver",
            params={"sosok": "1", "page": str(page)},
            headers=HEADERS,
            timeout=20,
        )
        resp.encoding = "euc-kr"
        table = pd.read_html(io.StringIO(resp.text))[1].dropna(how="all").dropna(subset=["N"])
        if table.empty:
            break
        frames.append(table[["N", "종목명", "시가총액"]])
        time.sleep(0.2)  # 예의상 살짝 텀

    df = pd.concat(frames, ignore_index=True)
    df.columns = ["market_cap_rank", "corp_name", "market_cap"]
    df["market_cap_rank"] = df["market_cap_rank"].astype(int)
    return df.drop_duplicates(subset=["corp_name"])


def main() -> None:
    print("코스닥 전체 시가총액 순위 수집 중...")
    df = fetch_all_market_cap()
    out_path = PROCESSED_DIR / "market_cap.csv"
    df.to_csv(out_path, index=False, encoding="utf-8")
    print(f"-> {out_path} 에 {len(df)}개 기업 저장 완료")


if __name__ == "__main__":
    main()
