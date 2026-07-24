import type { RiskExplanation as RiskExplanationType } from "../types";

export default function RiskExplanation({ explanation }: { explanation: RiskExplanationType }) {
  const maxAbs = Math.max(...explanation.contributions.map((c) => Math.abs(c.contribution)), 0.01);

  return (
    <div
      style={{
        border: "1px solid var(--color-border)",
        borderRadius: "var(--radius)",
        padding: "20px 24px",
        background: "var(--color-surface)",
      }}
    >
      <h2 style={{ fontSize: 16, margin: "0 0 4px" }}>위험도 산출 근거</h2>
      <p style={{ margin: "0 0 16px", fontSize: 13, color: "var(--color-text-muted)" }}>
        모델 예측 확률 {(explanation.model_probability * 100).toFixed(1)}% (모델 점수{" "}
        {explanation.model_score}점)
        {explanation.rule_adjusted && explanation.final_risk_score != null && (
          <> → 규칙 조정 후 최종 {explanation.final_risk_score}점</>
        )}
        . 아래는 각 재무지표(거시조정 고유위험)가 위험도를 얼마나 끌어올리거나 낮췄는지
        보여줍니다.
      </p>

      {explanation.rule_note && (
        <div
          style={{
            background: "#fbeae7",
            border: "1px solid #f0c4bc",
            borderRadius: 8,
            padding: "10px 14px",
            fontSize: 13,
            color: "var(--color-danger)",
            marginBottom: 16,
          }}
        >
          ⚠ {explanation.rule_note}
        </div>
      )}

      <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
        {explanation.contributions.map((c) => {
          const isRisk = c.contribution > 0;
          const widthPct = (Math.abs(c.contribution) / maxAbs) * 100;
          return (
            <div key={c.feature}>
              <div
                style={{
                  display: "flex",
                  justifyContent: "space-between",
                  fontSize: 12.5,
                  marginBottom: 3,
                }}
              >
                <span style={{ color: "var(--color-text)" }}>{c.label}</span>
                <span style={{ color: isRisk ? "var(--color-danger)" : "var(--color-safe)" }}>
                  {isRisk ? "위험 ↑" : "위험 ↓"} {c.contribution >= 0 ? "+" : ""}
                  {c.contribution.toFixed(3)}
                </span>
              </div>
              <div
                style={{
                  height: 6,
                  borderRadius: 4,
                  background: "var(--color-border)",
                  overflow: "hidden",
                }}
              >
                <div
                  style={{
                    height: "100%",
                    width: `${widthPct}%`,
                    background: isRisk ? "var(--color-danger)" : "var(--color-safe)",
                  }}
                />
              </div>
            </div>
          );
        })}
      </div>

      <p style={{ margin: "16px 0 0", fontSize: 11.5, color: "var(--color-text-muted)" }}>
        ※ 현재 학습 표본이 매우 작아(상장폐지 확정 사례 소수) 일부 지표의 기여 방향이 재무이론과
        다르게 나타날 수 있습니다. 완전자본잠식처럼 규정상 명확한 위험 요인은 위 규칙으로 별도
        보정했지만, 나머지 지표별 가중치는 코스닥 전체 유니버스로 학습 데이터를 확장하면 더
        신뢰도가 높아집니다.
      </p>
    </div>
  );
}
