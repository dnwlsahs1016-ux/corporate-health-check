import type { CompanyDetail, CompanySummary } from "./types";

const API_BASE = "http://localhost:8000";

export async function fetchCompanies(): Promise<CompanySummary[]> {
  const res = await fetch(`${API_BASE}/companies`);
  if (!res.ok) throw new Error("기업 목록을 불러오지 못했습니다.");
  return res.json();
}

export async function fetchCompanyDetail(corpName: string): Promise<CompanyDetail> {
  const res = await fetch(`${API_BASE}/companies/${encodeURIComponent(corpName)}`);
  if (!res.ok) throw new Error("기업 상세 정보를 불러오지 못했습니다.");
  return res.json();
}
