import { useEffect, useState } from "react";
import { fetchCompanies } from "../api";
import CompanyCard from "../components/CompanyCard";
import type { CompanySummary } from "../types";

export default function Home() {
  const [companies, setCompanies] = useState<CompanySummary[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [query, setQuery] = useState("");

  useEffect(() => {
    fetchCompanies()
      .then(setCompanies)
      .catch((err) => setError(err.message));
  }, []);

  const filtered = companies?.filter((c) => c.corp_name.includes(query.trim()));

  return (
    <div style={{ maxWidth: 880, margin: "0 auto", padding: "48px 24px" }}>
      <section style={{ marginBottom: 40 }}>
        <h1 style={{ fontSize: 30, margin: "0 0 12px", letterSpacing: -0.5 }}>
          AI 기반 코스닥 상장사 고유 재무위험 진단
        </h1>
        <p style={{ color: "var(--color-text-muted)", fontSize: 15, lineHeight: 1.6, margin: 0 }}>
          class-weighted 로지스틱 회귀 AI 모델이 <strong>DART 전자공시 재무제표(2015~2025년,
          코스닥 상장사 600여 곳)</strong>, <strong>한국은행 ECOS 기준금리</strong>,{" "}
          <strong>KRX 상장폐지 이력(2015년 이후 재무적 사유로 폐지된 100여 건)</strong> 데이터를
          학습했습니다. 거시경제(기준금리) 요인을 제거한 기업 고유의 재무위험을 계산해, 내년도
          상장폐지 위험점수를 보여줍니다.
        </p>
      </section>

      <div
        style={{
          background: "#fdf3e0",
          border: "1px solid #f0dcae",
          borderRadius: "var(--radius)",
          padding: "12px 16px",
          fontSize: 13,
          lineHeight: 1.6,
          color: "#6b5b1f",
          marginBottom: 24,
        }}
      >
        ⚠️ 소규모 표본으로 학습된 실험적 모델의 결과이며 통계적으로 검증되지 않았습니다.{" "}
        <strong>실제 투자 판단, 신용평가, 거래 의사결정에 사용하지 마세요.</strong> 표시된
        기업명은 모델 검증을 위한 예시일 뿐, 해당 기업의 재무 건전성을 공식적으로 나타내지
        않습니다.
      </div>

      <input
        value={query}
        onChange={(e) => setQuery(e.target.value)}
        placeholder="기업명 검색"
        style={{
          width: "100%",
          padding: "12px 16px",
          borderRadius: "var(--radius)",
          border: "1px solid var(--color-border)",
          fontSize: 14,
          marginBottom: 20,
          outline: "none",
        }}
      />

      {error && <p style={{ color: "var(--color-danger)" }}>{error}</p>}
      {!companies && !error && <p style={{ color: "var(--color-text-muted)" }}>불러오는 중...</p>}

      <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
        {filtered?.map((company) => (
          <CompanyCard key={company.corp_name} company={company} />
        ))}
      </div>
    </div>
  );
}
