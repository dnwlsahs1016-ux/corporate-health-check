from __future__ import annotations

import math

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, precision_score, recall_score, roc_auc_score

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


def _fit_logreg(X: pd.DataFrame, y: pd.Series) -> LogisticRegression:
    # 표본(n=100대~1000대) 대비 피처(10개)가 많아 규제 없이는 과적합되기 쉬움 -> L2 규제를 강하게(C=0.2) 적용
    model = LogisticRegression(class_weight="balanced", max_iter=2000, C=0.2)
    model.fit(X, y)
    return model


def train_model(panel: pd.DataFrame) -> dict:
    """실서비스에 쓰는 모델은 전체 기간(2015~2025) 데이터로 학습한다(가장 최신 데이터까지
    반영해야 실제 위험점수가 정확해지므로). 이 함수가 보고하는 AUC는 in-sample이라
    일반화 성능 지표로 쓸 수 없다 — 진짜 검증은 evaluate_temporal_holdout()이 담당한다."""
    frame = prepare_model_frame(panel)
    X = frame[MODEL_FEATURES]
    y = frame["label"]

    model = _fit_logreg(X, y)

    metrics = {"n_samples": len(frame), "n_positive": int(y.sum())}
    if y.nunique() > 1:
        proba = model.predict_proba(X)[:, 1]
        metrics["in_sample_auc"] = float(roc_auc_score(y, proba))
    else:
        metrics["in_sample_auc"] = None

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "features": MODEL_FEATURES}, MODEL_PATH)

    return metrics


def evaluate_temporal_holdout(panel: pd.DataFrame, split_year: int = 2022) -> dict:
    """split_year까지의 데이터로만 별도로 학습한 뒤, 그 이후 연도(실제 상장폐지 사례 포함)로
    검증한다. 미래 시점 데이터를 전혀 보지 않고 학습했을 때도 부실기업을 구분할 수 있는지
    확인하는 것이 목적이며, 여기서 나온 모델은 저장하지 않고(서빙용 모델과 별개) 평가에만 쓴다."""
    frame = prepare_model_frame(panel)
    train = frame[frame["year"] <= split_year]
    test = frame[frame["year"] > split_year]

    result = {
        "split_year": split_year,
        "n_train": int(len(train)),
        "n_train_positive": int(train["label"].sum()),
        "n_test": int(len(test)),
        "n_test_positive": int(test["label"].sum()),
    }

    if train["label"].nunique() < 2 or test["label"].nunique() < 2:
        result["error"] = "학습 또는 검증 구간에 양성(폐지) 사례가 부족해 평가할 수 없습니다."
        return result

    model = _fit_logreg(train[MODEL_FEATURES], train["label"])
    proba = model.predict_proba(test[MODEL_FEATURES])[:, 1]
    pred = model.predict(test[MODEL_FEATURES])  # 기본 임계값 0.5
    y_test = test["label"]

    tn, fp, fn, tp = confusion_matrix(y_test, pred, labels=[0, 1]).ravel()
    result.update(
        {
            "roc_auc": float(roc_auc_score(y_test, proba)),
            "precision": float(precision_score(y_test, pred, zero_division=0)),
            "recall": float(recall_score(y_test, pred, zero_division=0)),
            "confusion_matrix": {
                "true_negative": int(tn),
                "false_positive": int(fp),
                "false_negative": int(fn),
                "true_positive": int(tp),
            },
        }
    )
    return result


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
