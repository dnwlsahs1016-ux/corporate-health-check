from __future__ import annotations

import math

import pandas as pd
from fastapi import APIRouter, HTTPException

from app.core.config import PROCESSED_DIR
from app.pipeline.benchmark import build_commentary, peer_average
from app.pipeline.features import RATIO_COLUMNS
from app.pipeline.model import MODEL_FEATURES, explain_score, load_model, prepare_model_frame

router = APIRouter(prefix="/companies", tags=["companies"])

PANEL_PATH = PROCESSED_DIR / "panel.parquet"


def _load_panel() -> pd.DataFrame:
    if not PANEL_PATH.exists():
        raise HTTPException(
            status_code=503,
            detail="파이프라인 결과가 없습니다. scripts/run_pipeline.py를 먼저 실행하세요.",
        )
    return pd.read_parquet(PANEL_PATH)


def _clean(value):
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


@router.get("")
def list_companies():
    panel = _load_panel()
    latest = panel.sort_values("year").groupby("corp_name").tail(1)
    rows = latest.sort_values("risk_score", ascending=False)
    return [
        {
            "corp_name": row["corp_name"],
            "stock_code": _clean(row["stock_code"]),
            "category": row["category"],
            "year": int(row["year"]),
            "risk_score": _clean(row["risk_score"]),
            "label": int(row["label"]),
            "capital_impairment": bool(row["capital_impairment"]),
        }
        for _, row in rows.iterrows()
    ]


@router.get("/{corp_name}")
def get_company_detail(corp_name: str):
    panel = _load_panel()
    company_rows = panel[panel["corp_name"] == corp_name].sort_values("year")
    if company_rows.empty:
        raise HTTPException(status_code=404, detail=f"기업을 찾을 수 없습니다: {corp_name}")

    timeline = []
    for _, row in company_rows.iterrows():
        entry = {
            "year": int(row["year"]),
            "risk_score": _clean(row["risk_score"]),
            "label": int(row["label"]),
            "ratios": {r: _clean(row[r]) for r in RATIO_COLUMNS},
            "ratios_idiosyncratic": {r: _clean(row[f"{r}_idio"]) for r in RATIO_COLUMNS},
        }
        timeline.append(entry)

    latest_row = company_rows.iloc[-1]
    latest_year = int(latest_row["year"])

    ratio_commentary = {}
    for ratio_key in RATIO_COLUMNS:
        peer_info = peer_average(panel, corp_name, latest_year, ratio_key)
        ratio_commentary[ratio_key] = {
            "peer_avg": _clean(peer_info["peer_avg"]),
            "n_peers": peer_info["n_peers"],
            "scope": peer_info["scope"],
            "text": build_commentary(ratio_key, _clean(latest_row[ratio_key]), peer_info),
        }

    bundle = load_model()
    # 결측치를 전체 패널 기준 중앙값으로 채우기 위해 회사 1건이 아닌 전체 패널을 기준으로 준비한다.
    prepared_panel = prepare_model_frame(panel)
    prepared_row = prepared_panel[
        (prepared_panel["corp_name"] == corp_name) & (prepared_panel["year"] == latest_year)
    ].iloc[-1]
    feature_values = prepared_row[MODEL_FEATURES].to_dict()
    risk_explanation = explain_score(bundle, feature_values)
    risk_explanation["rule_adjusted"] = bool(latest_row.get("rule_adjusted", False))
    risk_explanation["final_risk_score"] = _clean(latest_row["risk_score"])
    if risk_explanation["rule_adjusted"]:
        risk_explanation["rule_note"] = (
            "완전자본잠식 상태로 판단되어, 모델 예측 점수"
            f"({risk_explanation['model_score']}점)와 무관하게 위험점수 하한선(70점)을 적용했습니다."
        )

    first = company_rows.iloc[0]
    return {
        "corp_name": corp_name,
        "stock_code": _clean(first["stock_code"]),
        "category": first["category"],
        "delisting_date": _clean(first["delisting_date"]),
        "capital_impairment": bool(latest_row["capital_impairment"]),
        "timeline": timeline,
        "ratio_commentary": ratio_commentary,
        "risk_explanation": risk_explanation,
    }
