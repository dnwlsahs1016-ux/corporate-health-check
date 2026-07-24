from __future__ import annotations

import pandas as pd


def add_labels(panel: pd.DataFrame) -> pd.DataFrame:
    """'내년도 상장폐지' 라벨을 만든다.
    폐지연도-1 회계연도 데이터에 label=1 (그 해 재무제표를 보고 다음 해 폐지를 맞추는 문제 설정).
    그 외 연도(건전기업 포함)는 label=0.
    """
    panel = panel.copy()
    panel["label"] = 0

    for corp_name, group in panel.groupby("corp_name"):
        delisting_date = group["delisting_date"].iloc[0]
        if not delisting_date:
            continue
        delisting_year = int(str(delisting_date)[:4])
        target_year = delisting_year - 1
        mask = (panel["corp_name"] == corp_name) & (panel["year"] == target_year)
        panel.loc[mask, "label"] = 1

    return panel
