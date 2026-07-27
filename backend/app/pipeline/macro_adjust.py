from __future__ import annotations

import warnings

import pandas as pd
import statsmodels.formula.api as smf

from app.clients import ecos_client
from app.core.config import settings
from app.pipeline.features import RATIO_COLUMNS


def attach_macro_variables(panel: pd.DataFrame) -> pd.DataFrame:
    base_rate = ecos_client.fetch_base_rate(settings.start_year, settings.end_year)
    macro_df = pd.DataFrame(base_rate)
    panel = panel.merge(macro_df, on="year", how="left")
    # pandas 3.x는 문자열 컬럼을 기본적으로 Arrow 기반 StringDtype으로 만드는데, patsy가
    # 이 dtype을 다룰 때 간헐적으로 문제를 일으킨 적이 있어 순수 object dtype으로 되돌린다.
    panel["industry_code"] = panel["industry_code"].astype(object)
    # 일부 회사는 특정 지표(예: 차입금이 없어 interest_coverage, 매출 이력이 없어
    # revenue_growth 등)가 보유한 전체 연도에 걸쳐 전부 None이다. 회사별로 따로 만든
    # DataFrame들을 concat할 때 그런 컬럼만 (전부 None이라 float으로 추론되지 못하고)
    # object dtype으로 섞여 들어갈 수 있는데, patsy가 이런 "숫자인데 object dtype인" 컬럼을
    # 범주형으로 오인해 값마다 더미 컬럼을 만들어버려 회귀가 깨지는 사고가 있었다
    # ("endog에 컬럼이 수천 개" 에러). 회귀 직전에 명시적으로 숫자형으로 되돌려 차단한다.
    for col in [*RATIO_COLUMNS, "base_rate"]:
        panel[col] = pd.to_numeric(panel[col], errors="coerce")
    return panel


def fit_macro_models(train_panel: pd.DataFrame) -> dict:
    """비율 = a + b*base_rate + 산업더미 + 잔차(고유위험) 회귀를 학습 구간에만 적합(fit)한다.

    검증/예측 구간 데이터는 여기서 절대 보지 않는다. 전체 패널(미래 구간 포함)에 한 번에
    회귀를 돌리면, 검증/예측 구간의 정보가 회귀계수에 섞여 과거 구간의 잔차 계산에까지
    새어 들어가는 데이터 누수가 생긴다 — apply_macro_models()에서 이 계수를 그대로
    "적용(transform)"만 해야 시계열 검증이 깨끗해진다.

    표본이 작아 회귀가 불안정/특이(singular)하면 산업더미를 빼거나, 그래도 안되면
    평균차감(demean)으로 대체한다.
    """
    train_panel = attach_macro_variables(train_panel)
    n_industries = train_panel["industry_code"].nunique()

    fitted: dict[str, dict] = {}
    for ratio in RATIO_COLUMNS:
        sub = train_panel[["year", "industry_code", "base_rate", ratio]].dropna()
        mean_fallback = train_panel[ratio].mean()

        if len(sub) < 5:
            fitted[ratio] = {"model": None, "mean_fallback": mean_fallback, "known_industries": None}
            continue

        formula = f"{ratio} ~ base_rate"
        known_industries = None
        if n_industries > 1 and sub["industry_code"].nunique() > 1:
            formula += " + C(industry_code)"
            known_industries = set(sub["industry_code"].unique())

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                ols_model = smf.ols(formula, data=sub).fit()
            fitted[ratio] = {
                "model": ols_model,
                "mean_fallback": mean_fallback,
                "known_industries": known_industries,
            }
        except Exception as exc:  # noqa: BLE001 - 소표본 회귀 실패는 흔하므로 fallback 처리
            print(f"[macro_adjust] {ratio} 학습구간 회귀 실패({exc}), 평균차감으로 대체")
            fitted[ratio] = {"model": None, "mean_fallback": mean_fallback, "known_industries": None}

    return fitted


def apply_macro_models(panel: pd.DataFrame, fitted: dict) -> pd.DataFrame:
    """fit_macro_models()가 학습 구간에서 적합한 계수를 검증/예측 구간을 포함한 모든
    데이터에 적용(transform)해 잔차(고유위험)를 계산한다. 여기서는 어떤 회귀도 다시
    적합하지 않는다 — StandardScaler의 fit/transform과 같은 원칙이다.

    학습 구간에 없던 산업코드가 검증/예측 구간에 나타나면 그 행에서만 예측이 실패하므로
    (patsy가 처음 보는 카테고리를 거부한다), 그런 행만 평균차감으로 대체하고 나머지 행은
    정상적으로 회귀 예측을 적용한다 — 카테고리 하나가 새로 나타났다고 그 지표 전체를
    통째로 평균차감으로 버리면 불필요하게 정밀도를 잃는다.
    """
    panel = attach_macro_variables(panel)
    panel = panel.copy()

    for ratio in RATIO_COLUMNS:
        idio_col = f"{ratio}_idio"
        spec = fitted[ratio]
        mean_fallback = spec["mean_fallback"]
        model = spec["model"]
        known_industries = spec["known_industries"]

        if model is None:
            panel[idio_col] = panel[ratio] - mean_fallback
            continue

        if known_industries is not None:
            predictable = panel["industry_code"].isin(known_industries)
        else:
            predictable = pd.Series(True, index=panel.index)

        idio = pd.Series(index=panel.index, dtype=float)
        if predictable.any():
            try:
                predicted = model.predict(panel.loc[predictable])
                idio.loc[predictable] = panel.loc[predictable, ratio] - predicted
            except Exception as exc:  # noqa: BLE001 - 그래도 실패하면 전체 평균차감으로 대체
                print(f"[macro_adjust] {ratio} 적용(predict) 실패({exc}), 평균차감으로 대체")
                predictable = pd.Series(False, index=panel.index)

        if not predictable.all():
            idio.loc[~predictable] = panel.loc[~predictable, ratio] - mean_fallback

        panel[idio_col] = idio.fillna(panel[ratio] - mean_fallback)

    return panel
