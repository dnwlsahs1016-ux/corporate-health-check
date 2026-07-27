"""전체 파이프라인 실행: 수집 -> 라벨링 -> 시계열 검증 -> 배포용 모델 재학습 -> 위험점수.

라벨 시점 처리 원칙 (labels.latest_label_confirmed_year 참고):
    사업연도 Y의 라벨은 "Y+1년에 폐지됐는가"를 뜻하므로, Y+1년이 완전히 지나야 확정된다.
    가장 최근 사업연도(예: 2025)는 아직 우측절단(right-censoring) 상태라 학습·검증에 쓰지
    않고, 확정된 구간(~2024)만으로 검증/재학습한 뒤 최신 연도는 예측(스코어링) 전용으로 쓴다.

거시요인 제거도 fit(학습 구간에만 회귀 적합)과 apply(전체에 계수 적용)를 분리한다.
전체 패널에 한 번에 회귀를 돌리면 검증 구간의 정보가 학습 구간 잔차 계산에 새어 들어가는
데이터 누수가 생기기 때문이다.

사용법 (backend 디렉터리에서):
    ./.venv/Scripts/python.exe scripts/run_pipeline.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.config import PROCESSED_DIR  # noqa: E402
from app.pipeline import collect, labels, macro_adjust, model, score, trend_features  # noqa: E402

SPLIT_YEAR = 2022


def main() -> None:
    print("[1/10] DART/ECOS 원자료 수집 및 재무비율 계산 중...")
    panel = collect.collect_all()
    if panel.empty:
        print("수집된 데이터가 없습니다. delisted_companies.csv와 API 키를 확인하세요.")
        return
    print(f"  -> {panel['corp_name'].nunique()}개 기업, {len(panel)}개 기업-연도 로우 수집")

    print("[2/10] 상장폐지 라벨 부여 중...")
    panel = labels.add_labels(panel)
    confirmed_max_year = labels.latest_label_confirmed_year()
    print(
        f"  -> label=1 로우 수: {int(panel['label'].sum())} / "
        f"라벨 확정된 최신 사업연도: {confirmed_max_year}년 "
        f"(그 이후는 우측절단 구간이라 학습·검증에서 제외하고 예측 전용으로만 사용)"
    )

    print(f"[3/10] 거시요인 제거 회귀를 학습구간(~{SPLIT_YEAR}년)에만 적합 중...")
    macro_fit_holdout = macro_adjust.fit_macro_models(panel[panel["year"] <= SPLIT_YEAR])
    panel_confirmed = macro_adjust.apply_macro_models(
        panel[panel["year"] <= confirmed_max_year], macro_fit_holdout
    )
    # 전년대비 변화량/연속악화 여부/기업규모 피처는 같은 기업의 과거 연도만 참조하는
    # 결정적 변환이라(회귀 fit이 아니므로) 미래 정보 누수 없이 그냥 적용하면 된다.
    panel_confirmed = trend_features.add_trend_features(panel_confirmed)

    print(
        f"[4/10] 시계열 학습/검증 분리 평가 중 "
        f"({SPLIT_YEAR}년까지 학습 -> {SPLIT_YEAR + 1}~{confirmed_max_year}년 검증)..."
    )
    holdout = model.evaluate_temporal_holdout(panel_confirmed, split_year=SPLIT_YEAR, end_year=confirmed_max_year)
    print(f"  -> {holdout}")

    print("[5/10] Altman Z'-Score 벤치마크 비교 평가 중 (동일 검증 구간)...")
    altman = model.evaluate_altman_benchmark(panel_confirmed, split_year=SPLIT_YEAR, end_year=confirmed_max_year)
    print(f"  -> {altman}")

    print("[6/10] 로지스틱회귀+Altman 앙상블 평가 중 (동일 검증 구간)...")
    ensemble = model.evaluate_ensemble_benchmark(panel_confirmed, split_year=SPLIT_YEAR, end_year=confirmed_max_year)
    print(f"  -> {ensemble}")

    print("[7/10] 참고용 무작위 8:2 분할 평가 중 (시간 순서 무시, 라벨 확정 구간 내에서만)...")
    random_split = model.evaluate_random_split(panel_confirmed, test_size=0.2)
    print(f"  -> {random_split}")

    print(f"[8/10] 배포용 거시요인 제거 회귀를 라벨 확정 전체(~{confirmed_max_year}년)로 재적합 중...")
    macro_fit_prod = macro_adjust.fit_macro_models(panel[panel["year"] <= confirmed_max_year])
    panel_final = macro_adjust.apply_macro_models(panel, macro_fit_prod)
    panel_final = trend_features.add_trend_features(panel_final)

    print(f"[9/10] 배포용 모델 학습 중 (라벨 확정 전체, ~{confirmed_max_year}년 데이터)...")
    metrics = model.train_production_model(panel_final, up_to_year=confirmed_max_year)
    print(f"  -> {metrics}")

    print("[10/10] 위험점수 계산 및 저장 중 (라벨 미확정 최신 연도 포함 전체 예측)...")
    scored = score.score_panel(panel_final)
    summary = score.build_company_summary(scored)

    scored.to_parquet(PROCESSED_DIR / "panel.parquet", index=False)
    (PROCESSED_DIR / "pipeline_metrics.json").write_text(
        json.dumps(
            {
                "label_confirmed_through_year": confirmed_max_year,
                "in_sample": metrics,
                "temporal_holdout": holdout,
                "altman_benchmark": altman,
                "ensemble_benchmark": ensemble,
                "random_split_benchmark": random_split,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\n=== 기업별 최신 위험점수 ===")
    print(
        f"(주의: 사업연도 {confirmed_max_year + 1}년 이후 로우의 label은 아직 확정되지 않은 "
        f"우측절단 값입니다 — 예측 결과일 뿐 정답이 아닙니다)"
    )
    for row in sorted(summary, key=lambda r: -r["risk_score"]):
        print(
            f"  {row['corp_name']:<10} ({row['year']}) "
            f"risk_score={row['risk_score']:>5} label={row['label']} category={row['category']}"
        )


if __name__ == "__main__":
    main()
