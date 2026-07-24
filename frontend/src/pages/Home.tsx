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
          코스닥 상장사 고유 재무위험 진단
        </h1>
        <p style={{ color: "var(--color-text-muted)", fontSize: 15, lineHeight: 1.6, margin: 0 }}>
          한국은행 ECOS·DART 재무제표 데이터를 기반으로 거시경제 요인을 제거한 기업 고유의
          재무위험을 계산해, 내년도 상장폐지 위험점수를 보여줍니다.
        </p>
      </section>

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
