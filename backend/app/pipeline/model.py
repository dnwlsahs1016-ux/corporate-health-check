from __future__ import annotations

import math

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from app.core.config import PROCESSED_DIR
from app.pipeline.features import RATIO_COLUMNS, RATIO_META
from app.pipeline.trend_features import SIZE_COLUMNS, TREND_COLUMNS

IDIO_COLUMNS = [f"{r}_idio" for r in RATIO_COLUMNS]
# capital_impairment(완전자본잠식 여부)는 거시요인과 무관한 구조적 신호이므로
# 거시조정을 거치지 않고 원본 그대로 모델에 넣는다.
MODEL_FEATURES = IDIO_COLUMNS + TREND_COLUMNS + SIZE_COLUMNS + ["capital_impairment"]
MODEL_PATH = PROCESSED_DIR / "model.joblib"


def prepare_model_frame(panel: pd.DataFrame, reference: pd.DataFrame | None = None) -> pd.DataFrame:
    """결측치를 채운다. reference를 주면 그 구간(보통 학습구간)에서만 계산한 중앙값을
    채우는 값으로 쓴다 — 안 주면 panel 자기 자신의 중앙값을 쓰는데, 검증 단계에서
    이렇게 하면 검증구간 값이 "학습 데이터 전처리"에 섞여 들어가는 약한 데이터 누수가
    생긴다(StandardScaler를 학습구간에만 fit하는 것과 같은 원칙)."""
    ref = reference if reference is not None else panel
    frame = panel.copy()
    for col in MODEL_FEATURES:
        if col == "capital_impairment":
            frame[col] = frame[col].fillna(0)
            continue
        frame[col] = frame[col].fillna(ref[col].median())
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


def _select_f2_threshold(X: pd.DataFrame, y: pd.Series, beta: float = 2.0) -> float:
    """분류 임계값을 고른다. class_weight="balanced"를 쓴 로지스틱회귀에서는 0.5가
    "실제 위험확률 50%"를 뜻하지 않는다(소수 클래스를 인위적으로 키웠기 때문) — 그래서
    기본값 0.5를 그대로 쓰면 오탐(false positive)이 불필요하게 많아진다.

    학습구간 내부 교차검증(out-of-fold) 예측 확률로 F-beta(재현율에 beta배 더 가중한
    F-score, 기본 beta=2라 재현율을 더 중시)가 최대인 임계값을 고른다. 테스트 구간
    라벨은 여기서 전혀 보지 않으므로 임계값 선택 자체가 검증 결과를 오염시키지 않는다.
    표본이 너무 작아 교차검증이 불가능하면 기본값 0.5로 후퇴한다."""
    n_positive = int(y.sum())
    n_splits = min(5, n_positive, int((y == 0).sum()))
    if n_splits < 2:
        return 0.5

    best_C = _select_best_C(X, y)
    pipeline = Pipeline(
        [("scaler", StandardScaler()), ("clf", LogisticRegression(class_weight="balanced", max_iter=2000, C=best_C))]
    )
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    oof_proba = cross_val_predict(pipeline, X, y, cv=cv, method="predict_proba")[:, 1]

    precision, recall, thresholds = precision_recall_curve(y, oof_proba)
    with np.errstate(divide="ignore", invalid="ignore"):
        f_beta = (1 + beta**2) * precision * recall / (beta**2 * precision + recall)
    f_beta = np.nan_to_num(f_beta, nan=0.0)
    if len(thresholds) == 0:
        return 0.5
    best_idx = int(np.argmax(f_beta[:-1]))
    return float(thresholds[best_idx])


def _bootstrap_auc_ci(y_true: pd.Series, y_score: np.ndarray, n_boot: int = 1000, ci: float = 0.95) -> dict | None:
    """부트스트랩 재추출로 ROC-AUC의 신뢰구간을 계산한다. 양성 사례가 한 자릿수~수십 건뿐인
    검증구간에서는 AUC 점추정치 하나만 보고하면, 실제로는 표본 재추출에 따라 크게 흔들리는
    잡음 수준의 변화를 "성능이 좋아졌다/나빠졌다"로 착각하기 쉽다."""
    rng = np.random.RandomState(42)
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    n = len(y_true)

    scores = []
    for _ in range(n_boot):
        idx = rng.randint(0, n, n)
        y_b, s_b = y_true[idx], y_score[idx]
        if len(np.unique(y_b)) < 2:
            continue
        scores.append(roc_auc_score(y_b, s_b))

    if len(scores) < n_boot // 2:
        return None

    lower_pct = (1 - ci) / 2 * 100
    upper_pct = (1 + ci) / 2 * 100
    return {
        "lower": float(np.percentile(scores, lower_pct)),
        "upper": float(np.percentile(scores, upper_pct)),
        "n_boot": len(scores),
    }


def _precision_recall_at_k(y_true: pd.Series, y_score: np.ndarray, k_values=(10, 20, 30, 50)) -> dict:
    """이 서비스는 실제로는 "임계값 하나로 자동 분류"가 아니라 "위험점수 상위 몇 개 기업을
    사람이 심층검토"하는 방식으로 쓰일 가능성이 높다. 그런 실무 시나리오에 더 맞는 지표가
    Precision@K/Recall@K다 — 예: "위험점수 상위 20개를 봤을 때 그중 몇 개가 실제 폐지
    기업이었는가"."""
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    order = np.argsort(-y_score)
    y_sorted = y_true[order]
    total_positive = int(y_sorted.sum())

    result = {}
    for k in k_values:
        k_eff = min(k, len(y_sorted))
        top_k = y_sorted[:k_eff]
        tp = int(top_k.sum())
        result[f"top_{k}"] = {
            "k": k_eff,
            "true_positive": tp,
            "precision_at_k": tp / k_eff if k_eff else None,
            "recall_at_k": tp / total_positive if total_positive else None,
        }
    return result


def train_production_model(panel: pd.DataFrame, up_to_year: int) -> dict:
    """실서비스 모델은 라벨이 확정된 구간(up_to_year까지)의 데이터로만 학습한다.
    가장 최근 사업연도(예: 2025)는 아직 상장폐지 여부가 확정되지 않은 우측절단
    구간(labels.latest_label_confirmed_year() 참고)이라 label=0으로 취급하면 안 되므로
    학습에서 제외한다.

    이 함수가 보고하는 AUC는 in-sample이라 일반화 성능 지표로 쓸 수 없다 — 진짜 검증은
    evaluate_temporal_holdout()이 담당한다. 다만 검증에 썼던 구간(예: 2023~up_to_year)도
    확정된 정보인 이상 버리지 않고 이 최종 모델 학습에 포함시킨다 — 그래서 holdout AUC가
    이 최종 모델의 실제 성능을 정확히 대변하지는 않는다(최종 모델이 조금 더 나을 수 있다).
    """
    frame = prepare_model_frame(panel[panel["year"] <= up_to_year])
    X = frame[MODEL_FEATURES]
    y = frame["label"]

    model = _fit_logreg(X, y)

    metrics = {"n_samples": len(frame), "n_positive": int(y.sum()), "up_to_year": up_to_year}
    if y.nunique() > 1:
        proba = model.predict_proba(X)[:, 1]
        metrics["in_sample_auc"] = float(roc_auc_score(y, proba))
    else:
        metrics["in_sample_auc"] = None

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "features": MODEL_FEATURES}, MODEL_PATH)

    return metrics


def evaluate_temporal_holdout(panel: pd.DataFrame, split_year: int = 2022, end_year: int | None = None) -> dict:
    """split_year까지의 데이터로만 별도로 학습한 뒤, 그 이후 연도(실제 상장폐지 사례 포함)로
    검증한다. 미래 시점 데이터를 전혀 보지 않고 학습했을 때도 부실기업을 구분할 수 있는지
    확인하는 것이 목적이며, 여기서 나온 모델은 저장하지 않고(서빙용 모델과 별개) 평가에만 쓴다.

    end_year를 주면 검증 구간을 (split_year, end_year]로 제한한다 — 라벨이 아직 확정되지
    않은 최신 사업연도(우측절단 구간, labels.latest_label_confirmed_year() 참고)가 검증에
    잘못 섞여 들어가는 것을 막기 위함이다.
    """
    if end_year is not None:
        panel = panel[panel["year"] <= end_year]
    train_raw = panel[panel["year"] <= split_year]
    test_raw = panel[panel["year"] > split_year]

    result = {
        "split_year": split_year,
        "n_train": int(len(train_raw)),
        "n_train_positive": int(train_raw["label"].sum()),
        "n_test": int(len(test_raw)),
        "n_test_positive": int(test_raw["label"].sum()),
    }

    if train_raw["label"].nunique() < 2 or test_raw["label"].nunique() < 2:
        result["error"] = "학습 또는 검증 구간에 양성(폐지) 사례가 부족해 평가할 수 없습니다."
        return result

    # 결측치는 학습구간 중앙값만으로 채운다(reference=train_raw) — 검증구간 값이
    # 결측치 대체에 섞여 들어가는 누수를 막기 위함.
    train = prepare_model_frame(train_raw, reference=train_raw)
    test = prepare_model_frame(test_raw, reference=train_raw)

    model = _fit_logreg(train[MODEL_FEATURES], train["label"])
    proba = model.predict_proba(test[MODEL_FEATURES])[:, 1]
    y_test = test["label"]

    best_threshold = _select_f2_threshold(train[MODEL_FEATURES], train["label"])
    pred_default = (proba >= 0.5).astype(int)
    pred_tuned = (proba >= best_threshold).astype(int)

    def _confusion(pred: np.ndarray) -> dict:
        tn, fp, fn, tp = confusion_matrix(y_test, pred, labels=[0, 1]).ravel()
        return {
            "precision": float(precision_score(y_test, pred, zero_division=0)),
            "recall": float(recall_score(y_test, pred, zero_division=0)),
            "confusion_matrix": {
                "true_negative": int(tn),
                "false_positive": int(fp),
                "false_negative": int(fn),
                "true_positive": int(tp),
            },
        }

    result.update(
        {
            "roc_auc": float(roc_auc_score(y_test, proba)),
            "roc_auc_ci95": _bootstrap_auc_ci(y_test, proba),
            # PR-AUC(average precision): 양성(폐지) 사례가 희귀할 때 ROC-AUC는 압도적으로
            # 많은 음성 사례 덕에 낙관적으로 보일 수 있다. PR-AUC는 "찾아낸 것 중 얼마나
            # 맞았는지"에 더 민감해 이런 왜곡을 보완한다. 무작위 분류기의 PR-AUC 기대값은
            # 양성 비율(pr_auc_baseline)이므로, PR-AUC는 항상 이 기준선과 같이 봐야 의미가
            # 있다(ROC-AUC의 0.5 같은 고정 기준선이 없다).
            "pr_auc": float(average_precision_score(y_test, proba)),
            "pr_auc_baseline": float(y_test.mean()),
            # threshold_default(0.5)와 threshold_tuned(학습구간 내부 F2 최적값)를 나란히
            # 보여준다 — 0.5는 balanced 로지스틱회귀에서 임의의 기준일 뿐이라, 실제로 오탐을
            # 줄이려면 tuned 쪽을 참고해야 한다.
            "threshold_default": {"threshold": 0.5, **_confusion(pred_default)},
            "threshold_tuned": {"threshold": best_threshold, **_confusion(pred_tuned)},
            "precision_recall_at_k": _precision_recall_at_k(y_test, proba),
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
            "pr_auc": float(average_precision_score(y_test, proba)),
            "pr_auc_baseline": float(y_test.mean()),
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


def evaluate_altman_benchmark(panel: pd.DataFrame, split_year: int = 2022, end_year: int | None = None) -> dict:
    """같은 검증 구간(split_year 이후)에서 전통적 Altman Z'-Score를 위험 순위로 썼을 때의
    판별력(AUC)을 계산해, 로지스틱회귀 모델과 나란히 비교한다. Z-Score가 낮을수록 위험하므로
    부호를 뒤집어(-Z) 위험 점수로 사용한다. Z-Score 계산에 필요한 이익잉여금 등이 없는
    기업-연도는 비교에서 제외한다.

    end_year를 주면 검증 구간을 (split_year, end_year]로 제한한다 — 라벨이 아직 확정되지
    않은 최신 사업연도(우측절단 구간)가 검증에 섞이는 것을 막기 위함이다."""
    if end_year is not None:
        panel = panel[panel["year"] <= end_year]
    test = panel[panel["year"] > split_year].dropna(subset=["altman_zscore", "label"])
    result = {"split_year": split_year, "n_test": int(len(test))}

    if test.empty or test["label"].nunique() < 2:
        result["error"] = "검증 구간에 Z-Score를 계산할 수 있는 양성/음성 사례가 부족합니다."
        return result

    result["n_test_positive"] = int(test["label"].sum())
    result["roc_auc"] = float(roc_auc_score(test["label"], -test["altman_zscore"]))
    result["pr_auc"] = float(average_precision_score(test["label"], -test["altman_zscore"]))
    result["pr_auc_baseline"] = float(test["label"].mean())
    return result


def evaluate_ensemble_benchmark(panel: pd.DataFrame, split_year: int = 2022, end_year: int | None = None) -> dict:
    """로지스틱회귀 모델과 Altman Z'-Score를 같은 검증 구간에서 각각, 그리고 평균낸
    앙상블로 평가해 세 방식의 AUC를 나란히 비교한다. 데이터 누수를 피하기 위해 두 모델 모두
    train(<=split_year) 구간에서만 새로 학습하고(서빙용 모델/보정기와는 별개), Z-Score가
    없는 기업-연도는 공정한 비교를 위해 두 모델 평가에서 함께 제외한다.

    end_year를 주면 검증 구간을 (split_year, end_year]로 제한한다 — 라벨이 아직 확정되지
    않은 최신 사업연도(우측절단 구간)가 검증에 섞이는 것을 막기 위함이다."""
    if end_year is not None:
        panel = panel[panel["year"] <= end_year]
    train_raw = panel[panel["year"] <= split_year]
    test_raw = panel[panel["year"] > split_year]
    # 결측치는 학습구간 중앙값만으로 채운다 - evaluate_temporal_holdout과 같은 이유.
    train = prepare_model_frame(train_raw, reference=train_raw)
    test = prepare_model_frame(test_raw, reference=train_raw)

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
            "pr_auc_baseline": float(y_test.mean()),
            "logistic_only_auc": float(roc_auc_score(y_test, p_logistic)),
            "logistic_only_pr_auc": float(average_precision_score(y_test, p_logistic)),
            "altman_only_auc": float(roc_auc_score(y_test, p_altman)),
            "altman_only_pr_auc": float(average_precision_score(y_test, p_altman)),
            "ensemble_auc": float(roc_auc_score(y_test, p_ensemble)),
            "ensemble_pr_auc": float(average_precision_score(y_test, p_ensemble)),
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
    if feature == "log_assets":
        return "기업 규모(총자산, log)"
    if feature == "operating_loss_2y":
        return "최근 2년 영업손실 지속"
    if feature == "negative_ocf_2y":
        return "최근 2년 영업현금흐름 마이너스 지속"
    if feature.endswith("_idio_change_1y"):
        ratio_key = feature.removesuffix("_idio_change_1y")
        return RATIO_META.get(ratio_key, {}).get("label", ratio_key) + " (전년 대비 변화, 고유위험 기준)"
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
