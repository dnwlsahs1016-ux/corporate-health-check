"""전체 파이프라인 실행: 수집 -> 지표계산 -> 거시조정 -> 라벨링 -> 모델학습 -> 위험점수.

사용법 (backend 디렉터리에서):
    ./.venv/Scripts/python.exe scripts/run_pipeline.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.config import PROCESSED_DIR  # noqa: E402
from app.pipeline import collect, labels, macro_adjust, model, score  # noqa: E402


def main() -> None:
    print("[1/5] DART/ECOS 원자료 수집 및 재무비율 계산 중...")
    panel = collect.collect_all()
    if panel.empty:
        print("수집된 데이터가 없습니다. delisted_companies.csv와 API 키를 확인하세요.")
        return
    print(f"  -> {panel['corp_name'].nunique()}개 기업, {len(panel)}개 기업-연도 로우 수집")

    print("[2/5] 거시요인(기준금리) 제거 중...")
    panel = macro_adjust.remove_macro_effect(panel)

    print("[3/5] 상장폐지 라벨 부여 중...")
    panel = labels.add_labels(panel)
    print(f"  -> label=1 로우 수: {int(panel['label'].sum())}")

    print("[4/5] 모델 학습 중...")
    metrics = model.train_model(panel)
    print(f"  -> {metrics}")

    print("[5/5] 위험점수 계산 및 저장 중...")
    scored = score.score_panel(panel)
    summary = score.build_company_summary(scored)

    scored.to_parquet(PROCESSED_DIR / "panel.parquet", index=False)
    (PROCESSED_DIR / "pipeline_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("\n=== 기업별 최신 위험점수 ===")
    for row in sorted(summary, key=lambda r: -r["risk_score"]):
        print(
            f"  {row['corp_name']:<10} ({row['year']}) "
            f"risk_score={row['risk_score']:>5} label={row['label']} category={row['category']}"
        )


if __name__ == "__main__":
    main()
