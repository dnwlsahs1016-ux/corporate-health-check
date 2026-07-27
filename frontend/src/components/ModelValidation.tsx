import { useState } from "react";
import type { ModelMetrics } from "../types";

export default function ModelValidation({ metrics }: { metrics: ModelMetrics }) {
  const [open, setOpen] = useState(false);
  const h = metrics.temporal_holdout;
  const a = metrics.altman_benchmark;
  const e = metrics.ensemble_benchmark;
  const r = metrics.random_split_benchmark;

  if (h.error || h.roc_auc == null) return null;
  const tuned = h.threshold_tuned;
  const shown = tuned ?? h.threshold_default;

  return (
    <div
      style={{
        border: "1px solid var(--color-border)",
        borderRadius: "var(--radius)",
        marginBottom: 20,
        overflow: "hidden",
      }}
    >
      <button
        onClick={() => setOpen(!open)}
        style={{
          width: "100%",
          textAlign: "left",
          padding: "12px 16px",
          background: "var(--color-surface)",
          border: "none",
          cursor: "pointer",
          fontSize: 14,
          fontWeight: 700,
        }}
      >
        📊 모델 검증 결과 (시계열 학습/검증 분리) {open ? "▲" : "▼"}
      </button>
      {open && (
        <div style={{ padding: "4px 16px 16px" }}>
          <p style={{ fontSize: 12.5, color: "var(--color-text-muted)", lineHeight: 1.6 }}>
            {h.split_year}년까지의 데이터로만 학습한 뒤, {h.split_year + 1}년 이후(실제
            상장폐지 사례 포함)로 검증했습니다. 학습 시점에 미래 정보를 전혀 사용하지 않았을
            때도 부실기업을 구분할 수 있는지 확인하기 위한 것으로, 실제 서비스에 쓰이는
            모델(전체 기간 학습)과는 별개의 평가용 모델입니다.
          </p>
          <div style={{ display: "flex", gap: 24, flexWrap: "wrap", margin: "8px 0" }}>
            <Stat
              label="ROC-AUC"
              value={
                h.roc_auc_ci95
                  ? `${h.roc_auc!.toFixed(3)} (95% CI ${h.roc_auc_ci95.lower.toFixed(2)}~${h.roc_auc_ci95.upper.toFixed(2)})`
                  : h.roc_auc!.toFixed(3)
              }
            />
            {h.pr_auc != null && (
              <Stat
                label="PR-AUC"
                value={`${h.pr_auc.toFixed(3)} (기준선 ${(h.pr_auc_baseline! * 100).toFixed(1)}%)`}
              />
            )}
            {shown && <Stat label="Recall(재현율)" value={`${(shown.recall * 100).toFixed(1)}%`} />}
            {shown && <Stat label="Precision(정밀도)" value={`${(shown.precision * 100).toFixed(1)}%`} />}
            <Stat label="검증 구간 폐지 사례" value={`${h.n_test_positive}건`} />
          </div>
          {shown && (
            <p style={{ fontSize: 12.5, color: "var(--color-text-muted)", lineHeight: 1.6 }}>
              혼동행렬(임계값 {shown.threshold.toFixed(2)} 기준): 실제 폐지 {h.n_test_positive}건 중{" "}
              {shown.confusion_matrix.true_positive}건 적중(재현율{" "}
              {(shown.recall * 100).toFixed(1)}%), {shown.confusion_matrix.false_negative}건 놓침.
              위험 예측{" "}
              {shown.confusion_matrix.true_positive + shown.confusion_matrix.false_positive}건 중
              실제로 맞은 건 {shown.confusion_matrix.true_positive}건(정밀도{" "}
              {(shown.precision * 100).toFixed(1)}%).
            </p>
          )}
          {tuned && h.threshold_default && (
            <p style={{ fontSize: 11.5, color: "var(--color-text-muted)", lineHeight: 1.6 }}>
              ※ 기본 임계값 0.5는 class_weight='balanced'를 쓴 로지스틱회귀에서 "실제 위험확률
              50%"를 뜻하지 않습니다. 학습구간 내부 교차검증만으로 고른 임계값
              {" "}{tuned.threshold.toFixed(2)}을 대신 적용하면, 오탐(위험 예측{" "}
              {h.threshold_default.confusion_matrix.true_positive +
                h.threshold_default.confusion_matrix.false_positive}
              건 → {tuned.confusion_matrix.true_positive + tuned.confusion_matrix.false_positive}
              건)이 달라집니다. 재현율을 더 중시하는 F2 기준으로 골랐습니다.
            </p>
          )}
          <p style={{ fontSize: 11.5, color: "var(--color-text-muted)", lineHeight: 1.6, margin: "8px 0 0" }}>
            ※ 정밀도가 낮은 건 예상된 트레이드오프입니다: 부실기업이 전체의 2%도 안 되는 극단적
            불균형 데이터에서 재현율을 우선했기 때문에, 위험하다고 예측한 기업 중 실제로 폐지된
            비율은 낮지만 실제 폐지 사례의 상당수는 놓치지 않습니다.
          </p>

          {a && !a.error && a.roc_auc != null && (
            <>
              <hr style={{ border: "none", borderTop: "1px solid var(--color-border)", margin: "14px 0" }} />
              <p style={{ fontSize: 13, fontWeight: 700, margin: "0 0 6px" }}>
                전통적 Altman Z'-Score 대비
              </p>
              <p style={{ fontSize: 12.5, color: "var(--color-text-muted)", lineHeight: 1.6 }}>
                같은 검증 구간({h.split_year + 1}년 이후)에서, 감사·신용평가 실무에서 흔히 쓰이는
                전통적 부실예측 공식인 Altman Z'-Score를 위험 순위로 사용했을 때의 판별력과
                비교했습니다.
              </p>
              <div style={{ display: "flex", gap: 24, flexWrap: "wrap", margin: "8px 0" }}>
                <Stat label="로지스틱회귀 모델 AUC" value={h.roc_auc!.toFixed(3)} highlight />
                <Stat label="Altman Z'-Score AUC" value={a.roc_auc!.toFixed(3)} />
              </div>
              <p style={{ fontSize: 11.5, color: "var(--color-text-muted)", lineHeight: 1.6 }}>
                고정된 계수식인 Z-Score와 달리, 로지스틱 회귀는 거시조정된 지표로 데이터에 맞춰
                계수를 학습해 이 검증 구간에서 더 높은 판별력을 보였습니다.
              </p>
            </>
          )}

          {e && !e.error && e.ensemble_auc != null && (
            <>
              <hr style={{ border: "none", borderTop: "1px solid var(--color-border)", margin: "14px 0" }} />
              <p style={{ fontSize: 13, fontWeight: 700, margin: "0 0 6px" }}>
                로지스틱회귀 + Altman Z-Score 앙상블 실험
              </p>
              <p style={{ fontSize: 12.5, color: "var(--color-text-muted)", lineHeight: 1.6 }}>
                두 모델의 예측 확률을 단순평균해서 결합하면 더 나아지는지도 같은 검증 구간에서
                테스트했습니다.
              </p>
              <div style={{ display: "flex", gap: 24, flexWrap: "wrap", margin: "8px 0" }}>
                <Stat label="로지스틱 단독" value={e.logistic_only_auc!.toFixed(3)} highlight />
                <Stat label="Altman 단독" value={e.altman_only_auc!.toFixed(3)} />
                <Stat label="단순평균 앙상블" value={e.ensemble_auc!.toFixed(3)} />
              </div>
              <p style={{ fontSize: 11.5, color: "var(--color-text-muted)", lineHeight: 1.6 }}>
                ※ 정직하게 보고하면, 이 앙상블은 로지스틱 단독보다 오히려 낮은 AUC가 나왔습니다.
                Altman 단독 성능이 상대적으로 약해서 평균을 내면 로지스틱의 신호를 끌어내리는
                효과가 난 것으로 보입니다. 그래서 서비스에는 앙상블 대신 로지스틱회귀 단독
                점수를 사용합니다.
              </p>
            </>
          )}

          {r && !r.error && r.roc_auc != null && (
            <>
              <hr style={{ border: "none", borderTop: "1px solid var(--color-border)", margin: "14px 0" }} />
              <p style={{ fontSize: 13, fontWeight: 700, margin: "0 0 6px" }}>
                (참고) 시간 순서를 무시한 무작위 {Math.round((1 - r.test_size) * 100)}:
                {Math.round(r.test_size * 100)} 분할 결과
              </p>
              <p style={{ fontSize: 12.5, color: "var(--color-text-muted)", lineHeight: 1.6 }}>
                기업-연도 로우를 연도 상관없이 라벨 비율을 유지한 채 무작위로 나눠 평가하면
                어떻게 달라지는지도 확인했습니다.
              </p>
              <div style={{ display: "flex", gap: 24, flexWrap: "wrap", margin: "8px 0" }}>
                <Stat label="무작위 분할 AUC" value={r.roc_auc!.toFixed(3)} />
                <Stat label="시계열 분할 AUC(위 기준)" value={h.roc_auc!.toFixed(3)} />
              </div>
              <p style={{ fontSize: 11.5, color: "var(--color-text-muted)", lineHeight: 1.6 }}>
                ※ 무작위 분할이 시계열 분할보다 수치가 더 높게 나옵니다. 무작위 분할은 2024년
                데이터가 학습에, 2020년 데이터가 검증에 들어가는 식으로 미래 정보가 은연중에
                섞일 수 있어 실제보다 낙관적인 성능으로 보이기 쉽습니다. "내년도 위험을
                예측한다"는 이 서비스의 실제 사용 시나리오와는 시계열 분할이 더 정직한
                검증이라고 판단해, 대표 검증 지표로는 시계열 분할(위) 결과를 사용합니다.
              </p>
            </>
          )}
        </div>
      )}
    </div>
  );
}

function Stat({ label, value, highlight }: { label: string; value: string; highlight?: boolean }) {
  return (
    <div>
      <div style={{ fontSize: 11, color: "var(--color-text-muted)" }}>{label}</div>
      <div style={{ fontSize: 20, fontWeight: 700, color: highlight ? "var(--color-primary)" : undefined }}>
        {value}
      </div>
    </div>
  );
}
