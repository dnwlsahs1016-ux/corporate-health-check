from __future__ import annotations

import numpy as np
import pandas as pd

from app.pipeline.features import RATIO_COLUMNS

IDIO_COLUMNS = [f"{r}_idio" for r in RATIO_COLUMNS]

# 상장폐지는 한 시점의 수준보다 악화 속도·지속 기간이 더 중요한 신호인 경우가 많다.
# 여기서 만드는 피처는 macro_adjust와 달리 회귀를 "적합"하지 않는 결정적(deterministic)
# 시계열 변환(diff, rolling)이라 fit/transform을 분리할 필요가 없다 — 같은 기업의 과거
# 연도만 참조하므로 미래 정보가 섞여 들어가지 않는다.
TREND_COLUMNS = [f"{r}_idio_change_1y" for r in RATIO_COLUMNS] + [
    "operating_loss_2y",
    "negative_ocf_2y",
]
SIZE_COLUMNS = ["log_assets"]


def add_trend_features(panel: pd.DataFrame) -> pd.DataFrame:
    """전년 대비 변화량(고유위험 기준), 연속 영업손실/음의 영업현금흐름 여부, 기업 규모를
    피처로 추가한다."""
    panel = panel.sort_values(["corp_name", "year"]).copy()
    grouped = panel.groupby("corp_name")

    for ratio_col in IDIO_COLUMNS:
        change_col = f"{ratio_col}_change_1y"
        # 첫 관측 연도는 비교할 전년도가 없다 - "변화 없음(0)"으로 본다. 여기서 회사 전체
        # 평균으로 채우면 오히려 "관측되지 않은 변화"에 임의의 방향성을 부여하게 된다.
        panel[change_col] = grouped[ratio_col].diff(1).fillna(0.0)

    panel["operating_loss"] = (panel["operating_margin"] < 0).astype(int)
    panel["negative_ocf"] = (panel["ocf_to_assets"] < 0).astype(int)

    grouped = panel.groupby("corp_name")
    panel["operating_loss_2y"] = (
        grouped["operating_loss"].rolling(2, min_periods=1).sum().reset_index(level=0, drop=True)
    )
    panel["negative_ocf_2y"] = (
        grouped["negative_ocf"].rolling(2, min_periods=1).sum().reset_index(level=0, drop=True)
    )

    panel["log_assets"] = np.log1p(panel["raw_assets"].fillna(0).clip(lower=0))

    return panel
