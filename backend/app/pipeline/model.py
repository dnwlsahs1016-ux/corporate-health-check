from __future__ import annotations

import math

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

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


# 후보 정규화 강도. 표본(양성 사례 수십~백여 건) 대비 피처가 10개가 넘어 EPV(변수당 이벤트 수)가
# 낮으므로, 임의로 하나 고정하는 대신 후보군 중 교차검증 성능이 가장 좋은 값을 고른다.
C_CANDIDATES = [0.01, 0.03, 0.1, 0.2, 0.3, 0.5, 1.0]


def _select_best_C(X: pd.DataFrame, y: pd.Series) -> float:
    """StratifiedKFold 교차검증(ROC-AUC 기준)으로 최적 정규화 강도를 고른다. 표본이 매우 작아
    (한 자릿수 양성 사례) fold를 나눌 수 없는 경우에는 기본값으로 후퇴한다."""
    n_positive = int(y.sum())
    n_splits = min(5, n_positive, int((y == 0).sum()))
    if n_splits < 2:
        return 0.2  # 표본이 너무 작아 교차검증이 불가능할 때의 안전한 기본값

    pipeline = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(class_weight="balanced", max_iter=2000))])
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    grid = GridSearchCV(pipeline, {"clf__C": C_CANDIDATES}, scoring="roc_auc", cv=cv)
    grid.fit(X, y)
    return float(grid.best_params_["clf__C"])


def _fit_logreg(X: pd.DataFrame, y: pd.Series) -> Pipeline:
    """표준화(StandardScaler) + 로지스틱회귀 파이프라인.
    지표마다 값의 스케일이 크게 다른데(예: 이자보상배율은 -50~50, 영업이익률은 -1~1),
    L2 정규화는 계수 크기에 페널티를 주므로 표준화 없이 학습하면 스케일이 작은 지표가
    부당하게 억눌린다. 표준화로 이 편향을 없앤다.
    정규화 강도(C)는 주어진 X, y로 교차검증해 고른다(하드코딩하지 않음)."""
    best_C = _select_best_C(X, y)
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(class_weight="balanced", max_iter=2000, C=best_C)),
        ]
    )
    pipeline.fit(X, y)
    return pipeline


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


def evaluate_random_split(panel: pd.DataFrame, test_size: float = 0.2, random_state: int = 42) -> dict:
    """시간 순서를 무시하고 전체 기업-연도 로우를 라벨 비율 유지(stratify)한 채 무작위로
    80:20 분할해 평가한다. temporal_holdout(2022/2023 시계열 분리)과 별개의 참고용 실험이다.
    무작위 분할은 미래 시점의 정보가 학습 데이터에 섞여 들어갈 수 있어(예: 2024년 로우가 학습에,
    2020년 로우가 검증에 들어가는 식) 시계열 분리보다 낙관적인(더 높은) 성능이 나오는 경향이 있다
    — '내년도 위험을 예측'한다는 이 서비스의 실제 사용 시나리오와는 맞지 않는 평가 방식이므로
    참고용으로만 병기한다."""
    from sklearn.model_selection import train_test_split

    frame = prepare_model_frame(panel)
    result = {"test_size": test_size}

    if frame["label"].nunique() < 2:
        result["error"] = "양성(폐지) 사례가 없어 평가할 수 없습니다."
        return result

    train, test = train_test_split(
        frame, test_size=test_size, random_state=random_state, stratify=frame["label"]
    )
    result.update(
        {
            "n_train": int(len(train)),
            "n_train_positive": int(train["label"].sum()),
            "n_test": int(len(test)),
            "n_test_positive": int(test["label"].sum()),
        }
    )

    model = _fit_logreg(train[MODEL_FEATURES], train["label"])
    proba = model.predict_proba(test[MODEL_FEATURES])[:, 1]
    pred = model.predict(test[MODEL_FEATURES])
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


def evaluate_altman_benchmark(panel: pd.DataFrame, split_year: int = 2022) -> dict:
    """같은 검증 구간(split_year 이후)에서 전통적 Altman Z'-Score를 위험 순위로 썼을 때의
    판별력(AUC)을 계산해, 로지스틱회귀 모델과 나란히 비교한다. Z-Score가 낮을수록 위험하므로
    부호를 뒤집어(-Z) 위험 점수로 사용한다. Z-Score 계산에 필요한 이익잉여금 등이 없는
    기업-연도는 비교에서 제외한다."""
    test = panel[panel["year"] > split_year].dropna(subset=["altman_zscore", "label"])
    result = {"split_year": split_year, "n_test": int(len(test))}

    if test.empty or test["label"].nunique() < 2:
        result["error"] = "검증 구간에 Z-Score를 계산할 수 있는 양성/음성 사례가 부족합니다."
        return result

    result["n_test_positive"] = int(test["label"].sum())
    result["roc_auc"] = float(roc_auc_score(test["label"], -test["altman_zscore"]))
    return result


def evaluate_ensemble_benchmark(panel: pd.DataFrame, split_year: int = 2022) -> dict:
    """로지스틱회귀 모델과 Altman Z'-Score를 같은 검증 구간에서 각각, 그리고 평균낸
    앙상블로 평가해 세 방식의 AUC를 나란히 비교한다. 데이터 누수를 피하기 위해 두 모델 모두
    train(<=split_year) 구간에서만 새로 학습하고(서빙용 모델/보정기와는 별개), Z-Score가
    없는 기업-연도는 공정한 비교를 위해 두 모델 평가에서 함께 제외한다."""
    frame = prepare_model_frame(panel)
    train = frame[frame["year"] <= split_year]
    test = frame[frame["year"] > split_year]

    train_z = train.dropna(subset=["altman_zscore"])
    test_z = test.dropna(subset=["altman_zscore", "label"])

    result = {"split_year": split_year, "n_test": int(len(test_z))}
    if (
        train["label"].nunique() < 2
        or train_z["label"].nunique() < 2
        or test_z.empty
        or test_z["label"].nunique() < 2
    ):
        result["error"] = "학습 또는 검증 구간에 양성/음성 사례가 부족해 평가할 수 없습니다."
        return result

    logreg = _fit_logreg(train[MODEL_FEATURES], train["label"])
    p_logistic = logreg.predict_proba(test_z[MODEL_FEATURES])[:, 1]

    altman_clf = LogisticRegression(class_weight="balanced", max_iter=2000)
    altman_clf.fit(train_z[["altman_zscore"]], train_z["label"])
    p_altman = altman_clf.predict_proba(test_z[["altman_zscore"]])[:, 1]

    p_ensemble = (p_logistic + p_altman) / 2
    y_test = test_z["label"]

    result.update(
        {
            "n_test_positive": int(y_test.sum()),
            "logistic_only_auc": float(roc_auc_score(y_test, p_logistic)),
            "altman_only_auc": float(roc_auc_score(y_test, p_altman)),
            "ensemble_auc": float(roc_auc_score(y_test, p_ensemble)),
        }
    )
    return result


ALTMAN_MODEL_PATH = PROCESSED_DIR / "altman_calibrator.joblib"


def fit_altman_calibrator(panel: pd.DataFrame) -> LogisticRegression | None:
    """Altman Z'-Score는 그 자체로는 0~100 확률이 아니라 임의 스케일의 점수라, 우리 모델의
    위험점수와 나란히 비교하려면 같은 척도로 환산해야 한다. Z-Score 하나만 입력으로 쓰는
    1변수 로지스틱회귀를 별도로 학습해서(Platt scaling과 같은 방식) '이 Z-Score를 가진 기업이
    내년 상장폐지될 확률'로 보정한다. 전체 기간 데이터로 학습해 서빙용 모델과 동일한 철학을
    따른다."""
    frame = panel.dropna(subset=["altman_zscore", "label"])
    if frame["label"].nunique() < 2:
        return None
    X = frame[["altman_zscore"]]
    y = frame["label"]
    calibrator = LogisticRegression(class_weight="balanced", max_iter=2000)
    calibrator.fit(X, y)
    joblib.dump(calibrator, ALTMAN_MODEL_PATH)
    return calibrator


def score_altman(panel: pd.DataFrame, calibrator: LogisticRegression) -> pd.Series:
    """Z-Score를 0~100 위험점수로 환산. calibrator가 없거나 Z-Score가 없는 행은 NaN."""
    scores = pd.Series(index=panel.index, dtype=float)
    valid = panel["altman_zscore"].notna()
    if calibrator is not None and valid.any():
        proba = calibrator.predict_proba(panel.loc[valid, ["altman_zscore"]])[:, 1]
        scores.loc[valid] = (proba * 100).round(1)
    return scores


def load_model():
    return joblib.load(MODEL_PATH)


def _feature_label(feature: str) -> str:
    if feature == "capital_impairment":
        return "완전자본잠식 여부"
    ratio_key = feature.removesuffix("_idio")
    return RATIO_META.get(ratio_key, {}).get("label", ratio_key) + " (거시조정 고유위험)"


def explain_score(bundle: dict, feature_values: dict[str, float]) -> dict:
    """로지스틱회귀 계수 * (표준화된) 지표값 = 각 지표가 로그오즈(위험도)에 기여한 크기.
    intercept + 모든 contribution의 합이 logit이고, sigmoid(logit)이 모델 확률이다.
    모델이 StandardScaler를 거치므로, 계수는 원래 단위가 아니라 '표준편차 단위'에 곱해진다 —
    화면에는 원래 값(value)을 보여주되, 기여도(contribution) 계산은 표준화된 값으로 한다."""
    pipeline: Pipeline = bundle["model"]
    features: list[str] = bundle["features"]
    scaler: StandardScaler = pipeline.named_steps["scaler"]
    clf: LogisticRegression = pipeline.named_steps["clf"]
    coefs = clf.coef_[0]
    intercept = float(clf.intercept_[0])

    raw_values = [float(feature_values.get(f) or 0.0) for f in features]
    scaled_values = scaler.transform([raw_values])[0]

    contributions = []
    logit = intercept
    for feature, coef, raw_value, scaled_value in zip(features, coefs, raw_values, scaled_values):
        contribution = float(coef) * float(scaled_value)
        logit += contribution
        contributions.append(
            {
                "feature": feature,
                "label": _feature_label(feature),
                "value": raw_value,
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
