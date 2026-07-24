import { Link } from "react-router-dom";
import type { CompanySummary } from "../types";
import RiskGauge from "./RiskGauge";

export default function CompanyCard({ company }: { company: CompanySummary }) {
  return (
    <Link
      to={`/companies/${encodeURIComponent(company.corp_name)}`}
      style={{
        display: "flex",
        alignItems: "center",
        justifyContent: "space-between",
        gap: 20,
        padding: "20px 24px",
        borderRadius: "var(--radius)",
        border: "1px solid var(--color-border)",
        background: "var(--color-surface)",
        boxShadow: "var(--shadow-sm)",
        transition: "transform 0.15s ease, border-color 0.15s ease",
      }}
      onMouseEnter={(e) => (e.currentTarget.style.borderColor = "var(--color-primary)")}
      onMouseLeave={(e) => (e.currentTarget.style.borderColor = "var(--color-border)")}
    >
      <div>
        <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
          <h3 style={{ margin: 0, fontSize: 18 }}>{company.corp_name}</h3>
          {company.stock_code && (
            <span style={{ color: "var(--color-text-muted)", fontSize: 13 }}>
              {company.stock_code}
            </span>
          )}
          {company.capital_impairment && (
            <span
              style={{
                fontSize: 11,
                fontWeight: 700,
                color: "var(--color-danger)",
                background: "#fbeae7",
                borderRadius: 6,
                padding: "2px 6px",
              }}
            >
              완전자본잠식
            </span>
          )}
        </div>
        <p style={{ margin: "6px 0 0", color: "var(--color-text-muted)", fontSize: 13 }}>
          {company.category === "financial_distress" ? "상장폐지 사례 (검증용)" : "건전기업 벤치마크"}
          {" · "}
          기준연도 {company.year}
        </p>
      </div>
      <RiskGauge score={company.risk_score ?? 0} size={72} />
    </Link>
  );
}
