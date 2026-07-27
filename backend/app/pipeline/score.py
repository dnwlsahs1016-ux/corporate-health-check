from __future__ import annotations

import json

import pandas as pd

from app.core.config import PROCESSED_DIR
from app.pipeline.model import (
    MODEL_FEATURES,
    fit_altman_calibrator,
    load_model,
    prepare_model_frame,
    score_altman,
)

SCORES_PATH = PROCESSED_DIR / "scores.json"

# 완전자본잠식은 상장폐지의 직접·즉시 사유 중 하나로 규정상 비중이 매우 크다.
# 표본이 작아 로지스틱회귀 계수만으로는 이 위험을 과소평가할 수 있어,
# 자본잠식 상태인 기업-연도는 모델 점수와 무관하게 위험점수 하한선을 규칙으로 강제한다.
CAPITAL_IMPAIRMENT_SCORE_FLOOR = 70.0


def score_panel(panel: pd.DataFrame) -> pd.DataFrame:
    bundle = load_model()
    model = bundle["model"]
    frame = prepare_model_frame(panel)
    proba = model.predict_proba(frame[MODEL_FEATURES])[:, 1]
    panel = panel.copy()
    panel["delisting_probability"] = proba
    model_score = pd.Series((proba * 100).round(1), index=panel.index)

    is_impaired = frame["capital_impairment"] == 1
    panel["rule_adjusted"] = is_impaired & (model_score < CAPITAL_IMPAIRMENT_SCORE_FLOOR)
    panel["risk_score"] = model_score.where(
        ~is_impaired, model_score.clip(lower=CAPITAL_IMPAIRMENT_SCORE_FLOOR)
    )

    altman_calibrator = fit_altman_calibrator(panel)
    panel["altman_risk_score"] = score_altman(panel, altman_calibrator)

    return panel


def build_company_summary(scored_panel: pd.DataFrame) -> list[dict]:
    """기업별 최신 연도 기준 요약(목록 페이지용)."""
    latest = scored_panel.sort_values("year").groupby("corp_name").tail(1)
    records = latest[
        [
            "corp_name",
            "stock_code",
            "category",
            "year",
            "risk_score",
            "delisting_probability",
            "label",
            "capital_impairment",
            "rule_adjusted",
        ]
    ].to_dict(orient="records")

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    SCORES_PATH.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    return records
