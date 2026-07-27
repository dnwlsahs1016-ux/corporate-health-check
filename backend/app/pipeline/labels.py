from __future__ import annotations

from datetime import date

import pandas as pd


def add_labels(panel: pd.DataFrame, horizon_years: int = 1) -> pd.DataFrame:
    """'향후 horizon_years년 내 상장폐지' 라벨을 만든다.
    폐지연도-1 ~ 폐지연도-horizon_years 회계연도 데이터에 label=1
    (그 해 재무제표를 보고 향후 horizon_years년 내 폐지를 맞추는 문제 설정).
    그 외 연도(건전기업 포함)는 label=0.

    horizon_years=1(기본값)이면 기존과 동일하게 "내년도 폐지"만 라벨링한다. 값을 늘리면
    양성 표본이 늘어나지만("향후 2년 내 폐지"처럼 과제 정의 자체가 바뀌는 것), 폐지
    1~2년 전에는 아직 재무지표가 정상처럼 보이는 기업도 섞여 들어가 라벨 노이즈가
    커질 수 있다 — 실제로 도움이 되는지는 검증해봐야 한다.
    """
    panel = panel.copy()
    panel["label"] = 0

    for corp_name, group in panel.groupby("corp_name"):
        delisting_date = group["delisting_date"].iloc[0]
        if not delisting_date:
            continue
        delisting_year = int(str(delisting_date)[:4])
        target_years = {delisting_year - h for h in range(1, horizon_years + 1)}
        mask = (panel["corp_name"] == corp_name) & (panel["year"].isin(target_years))
        panel.loc[mask, "label"] = 1

    return panel


def latest_label_confirmed_year(horizon_years: int = 1) -> int:
    """라벨이 확정된(우측절단이 아닌) 마지막 사업연도를 계산한다.

    사업연도 Y의 라벨은 "Y+1~Y+horizon_years년 내 폐지됐는가"를 뜻하므로, Y+horizon_years년이
    완전히 지나야 확정된다. 아직 진행 중인 연도에는, 지금은 label=0(건전)으로 보이지만 그 해가
    끝나기 전에 폐지되어 사실은 1이었어야 할 기업이 섞여 있을 수 있다(우측절단, right-censoring).
    보수적으로 "현재 연도가 Y+horizon_years년보다 커야 확정"으로 본다
    -> Y <= 현재 연도 - horizon_years - 1.
    이 경계보다 최신인 사업연도는 학습·검증에 쓰지 않고 예측(스코어링) 전용으로만 쓴다.
    """
    return date.today().year - horizon_years - 1
