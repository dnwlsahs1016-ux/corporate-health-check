import { useEffect, useState } from "react";
import { Link, useParams } from "react-router-dom";
import { fetchCompanyDetail } from "../api";
import IndicatorChart from "../components/IndicatorChart";
import RiskExplanation from "../components/RiskExplanation";
import RiskGauge from "../components/RiskGauge";
import { RATIO_LABELS, type CompanyDetail as CompanyDetailType } from "../types";

export default function CompanyDetail() {
  const { corpName } = useParams<{ corpName: string }>();
  const [detail, setDetail] = useState<CompanyDetailType | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!corpName) return;
    fetchCompanyDetail(corpName)
      .then(setDetail)
      .catch((err) => setError(err.message));
  }, [corpName]);

  if (error) return <p style={{ padding: 48, color: "var(--color-danger)" }}>{error}</p>;
  if (!detail) return <p style={{ padding: 48, color: "var(--color-text-muted)" }}>불러오는 중...</p>;

  const latest = detail.timeline[detail.timeline.length - 1];

  return (
    <div style={{ maxWidth: 880, margin: "0 auto", padding: "48px 24px" }}>
      <Link to="/" style={{ color: "var(--color-primary)", fontSize: 13, fontWeight: 600 }}>
        ← 목록으로
      </Link>

      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
          margin: "16px 0 24px",
          padding: "24px 28px",
          borderRadius: "var(--radius)",
          border: "1px solid var(--color-border)",
          background: "var(--color-surface)",
        }}
      >
        <div>
          <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
            <h1 style={{ margin: 0, fontSize: 26 }}>{detail.corp_name}</h1>
            {detail.capital_impairment && (
              <span
                style={{
                  fontSize: 12,
                  fontWeight: 700,
                  color: "var(--color-danger)",
                  background: "#fbeae7",
                  borderRadius: 6,
                  padding: "3px 8px",
                }}
              >
                ⚠ 완전자본잠식
              </span>
            )}
          </div>
          <p style={{ margin: "6px 0 0", color: "var(--color-text-muted)", fontSize: 13 }}>
            {detail.stock_code ? `종목코드 ${detail.stock_code}` : "종목코드 미상"}
            {detail.delisting_date && ` · 상장폐지일 ${detail.delisting_date}`}
          </p>
        </div>
        {latest && <RiskGauge score={latest.risk_score ?? 0} size={110} />}
      </div>

      <div style={{ marginBottom: 32 }}>
        <RiskExplanation explanation={detail.risk_explanation} />
      </div>

      <h2 style={{ fontSize: 18, marginBottom: 4 }}>재무지표 추이 (원지표 vs 거시조정 고유위험)</h2>
      <p style={{ fontSize: 12.5, color: "var(--color-text-muted)", margin: "0 0 16px" }}>
        각 그래프 아래 문구는 최신 연도 기준으로 동종업계·유사규모(총자산 1/3~3배, 부족하면 동종업계
        전체 → 비교기업 전체 순으로 범위를 넓힘) 평균과 비교한 해설입니다.
      </p>
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 24 }}>
        {Object.entries(RATIO_LABELS).map(([key, label]) => (
          <IndicatorChart
            key={key}
            timeline={detail.timeline}
            ratioKey={key}
            label={label}
            commentary={detail.ratio_commentary[key]?.text}
          />
        ))}
      </div>
    </div>
  );
}
