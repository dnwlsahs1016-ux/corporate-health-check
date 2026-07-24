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
    return panel.merge(macro_df, on="year", how="left")


def remove_macro_effect(panel: pd.DataFrame) -> pd.DataFrame:
    """비율 = a + b*base_rate + 산업더미 + 잔차(고유위험). 표본이 작아 회귀가
    불안정/특이(singular)하면 산업더미를 빼거나, 그래도 안되면 평균차감(demean)으로 대체한다."""
    panel = attach_macro_variables(panel)
    panel = panel.copy()

    n_industries = panel["industry_code"].nunique()

    for ratio in RATIO_COLUMNS:
        idio_col = f"{ratio}_idio"
        sub = panel[["year", "industry_code", "base_rate", ratio]].dropna()

        if len(sub) < 5:
            panel[idio_col] = panel[ratio] - panel[ratio].mean()
            continue

        formula = f"{ratio} ~ base_rate"
        if n_industries > 1 and sub["industry_code"].nunique() > 1:
            formula += " + C(industry_code)"

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = smf.ols(formula, data=sub).fit()
            resid = model.resid
            panel.loc[resid.index, idio_col] = resid
            panel[idio_col] = panel[idio_col].fillna(panel[ratio] - panel[ratio].mean())
        except Exception as exc:  # noqa: BLE001 - 소표본 회귀 실패는 흔하므로 fallback 처리
            print(f"[macro_adjust] {ratio} 회귀 실패({exc}), 평균차감으로 대체")
            panel[idio_col] = panel[ratio] - panel[ratio].mean()

    return panel
