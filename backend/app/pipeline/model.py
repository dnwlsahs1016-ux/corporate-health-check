from __future__ import annotations

import math

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from app.core.config import PROCESSED_DIR
from app.pipeline.features import RATIO_COLUMNS, RATIO_META

IDIO_COLUMNS = [f"{r}_idio" for r in RATIO_COLUMNS]
# capital_impairment(완전자본잠식 여부)는 거시요인과 무관한 구조적 신호이므로
# 거시조정을 거치지 않고 원본 그대로 모델에 넣는다.
MODEL_FEATURES = IDIO_COLUMNS + ["capital_impairment"]
MODEL_PATH = PROCESSED_DIR / "model.joblib"


def prepare_model_frame(panel: pd.DataFrame) -> pd.DataFrame:
    frame = panel.copy()
    for col in IDIO_COLUMNS:
        frame[col] = frame[col].fillna(frame[col].median())
    frame["capital_impairment"] = frame["capital_impairment"].fillna(0)
    return frame


def train_model(panel: pd.DataFrame) -> dict:
    """소표본(프로토타입) class-weighted 로지스틱 회귀.
    표본이 매우 작아(회사 수 한 자릿수) 아래 AUC는 통계적으로 신뢰할 수 있는 일반화 성능이
    아니라 파이프라인이 정상 동작하는지 보는 참고용 수치임을 명시한다."""
    frame = prepare_model_frame(panel)
    X = frame[MODEL_FEATURES]
    y = frame["label"]

    # 표본(n=100대) 대비 피처(10개)가 많아 규제 없이는 과적합되기 쉬움 -> L2 규제를 강하게(C=0.2) 적용
    model = LogisticRegression(class_weight="balanced", max_iter=2000, C=0.2)
    model.fit(X, y)

    metrics = {"n_samples": len(frame), "n_positive": int(y.sum())}
    if y.nunique() > 1:
        proba = model.predict_proba(X)[:, 1]
        metrics["in_sample_auc"] = float(roc_auc_score(y, proba))
    else:
        metrics["in_sample_auc"] = None

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "features": MODEL_FEATURES}, MODEL_PATH)

    return metrics


def load_model():
    return joblib.load(MODEL_PATH)


def _feature_label(feature: str) -> str:
    if feature == "capital_impairment":
        return "완전자본잠식 여부"
    ratio_key = feature.removesuffix("_idio")
    return RATIO_META.get(ratio_key, {}).get("label", ratio_key) + " (거시조정 고유위험)"


def explain_score(bundle: dict, feature_values: dict[str, float]) -> dict:
    """로지스틱회귀 계수 * 지표값 = 각 지표가 로그오즈(위험도)에 기여한 크기.
    intercept + 모든 contribution의 합이 logit이고, sigmoid(logit)이 모델 확률이다."""
    model: LogisticRegression = bundle["model"]
    features: list[str] = bundle["features"]
    coefs = model.coef_[0]
    intercept = float(model.intercept_[0])

    contributions = []
    logit = intercept
    for feature, coef in zip(features, coefs):
        value = feature_values.get(feature) or 0.0
        contribution = float(coef) * float(value)
        logit += contribution
        contributions.append(
            {
                "feature": feature,
                "label": _feature_label(feature),
                "value": float(value),
                "coefficient": float(coef),
                "contribution": contribution,
            }
        )

    contributions.sort(key=lambda c: -abs(c["contribution"]))
    probability = 1 / (1 + math.exp(-logit))

    return {
        "intercept": intercept,
        "contributions": contributions,
        "logit": logit,
        "model_probability": probability,
        "model_score": round(probability * 100, 1),
    }
