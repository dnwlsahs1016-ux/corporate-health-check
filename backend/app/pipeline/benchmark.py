from __future__ import annotations

import pandas as pd

from app.pipeline.features import RATIO_META


def format_value(value: float | None, fmt: str) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "N/A"
    if fmt == "percent":
        return f"{value * 100:.1f}%"
    if fmt == "multiple":
        return f"{value:.1f}배"
    return f"{value:.2f}"


SIZE_BAND_LOW = 1 / 3  # 총자산이 대상 기업의 1/3 ~ 3배인 기업까지를 "유사 규모"로 본다
SIZE_BAND_HIGH = 3


def peer_average(panel: pd.DataFrame, corp_name: str, year: int, ratio_key: str) -> dict:
    """비교군을 동일 연도 -> 동종업계 -> 유사 규모(총자산 1/3~3배) 순으로 좁혀서 평균을 낸다.
    한 단계라도 표본이 없으면 그 앞 단계(더 넓은 비교군)로 되돌아간다(cascading fallback)."""
    company_row = panel[(panel["corp_name"] == corp_name) & (panel["year"] == year)]
    if company_row.empty:
        return {"peer_avg": None, "n_peers": 0, "scope": "none"}

    industry_code = company_row.iloc[0]["industry_code"]
    assets = company_row.iloc[0].get("raw_assets")
    same_year = panel[(panel["year"] == year) & (panel["corp_name"] != corp_name)]
    industry_peers = same_year[same_year["industry_code"] == industry_code]

    size_peers = pd.DataFrame()
    if assets is not None and pd.notna(assets) and assets > 0 and not industry_peers.empty:
        size_peers = industry_peers[
            industry_peers["raw_assets"].between(assets * SIZE_BAND_LOW, assets * SIZE_BAND_HIGH)
        ]

    if len(size_peers) >= 1 and size_peers[ratio_key].notna().any():
        peers, scope = size_peers, "industry_size"
    elif len(industry_peers) >= 1 and industry_peers[ratio_key].notna().any():
        peers, scope = industry_peers, "industry"
    else:
        peers, scope = same_year, "market"

    values = peers[ratio_key].dropna()
    if values.empty:
        return {"peer_avg": None, "n_peers": 0, "scope": "none"}

    return {"peer_avg": float(values.mean()), "n_peers": int(len(values)), "scope": scope}


def build_commentary(ratio_key: str, company_value: float | None, peer_info: dict) -> str:
    meta = RATIO_META[ratio_key]
    peer_avg = peer_info.get("peer_avg")
    n_peers = peer_info.get("n_peers", 0)
    scope = peer_info.get("scope")

    if company_value is None or peer_avg is None:
        return "비교할 수 있는 데이터가 충분하지 않습니다."

    is_higher = company_value >= peer_avg
    is_better = is_higher == meta["higher_is_better"]
    verdict = "양호한" if is_better else "우려되는"
    scope_label = {
        "industry_size": "동종업계·유사규모",
        "industry": "동종업계",
    }.get(scope, "비교기업 전체")

    company_fmt = format_value(company_value, meta["format"])
    peer_fmt = format_value(peer_avg, meta["format"])

    return (
        f"{meta['label']}은 {company_fmt}로, {scope_label} 평균({peer_fmt}, {n_peers}개사) 대비 "
        f"{verdict} 수준입니다."
    )
