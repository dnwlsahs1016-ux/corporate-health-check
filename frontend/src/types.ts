export interface CompanySummary {
  corp_name: string;
  stock_code: string | null;
  category: "financial_distress" | "healthy_benchmark";
  year: number;
  risk_score: number | null;
  label: number;
  capital_impairment: boolean;
}

export interface TimelineEntry {
  year: number;
  risk_score: number | null;
  label: number;
  ratios: Record<string, number | null>;
  ratios_idiosyncratic: Record<string, number | null>;
}

export interface RatioCommentary {
  peer_avg: number | null;
  n_peers: number;
  scope: "industry" | "market" | "none";
  text: string;
}

export interface RiskContribution {
  feature: string;
  label: string;
  value: number;
  coefficient: number;
  contribution: number;
}

export interface RiskExplanation {
  intercept: number;
  contributions: RiskContribution[];
  logit: number;
  model_probability: number;
  model_score: number;
  rule_adjusted: boolean;
  final_risk_score: number | null;
  rule_note?: string;
}

export interface CompanyDetail {
  corp_name: string;
  stock_code: string | null;
  category: "financial_distress" | "healthy_benchmark";
  delisting_date: string | null;
  capital_impairment: boolean;
  timeline: TimelineEntry[];
  ratio_commentary: Record<string, RatioCommentary>;
  risk_explanation: RiskExplanation;
}

export const RATIO_LABELS: Record<string, string> = {
  roa: "ROA (총자산순이익률)",
  roe: "ROE (자기자본순이익률)",
  operating_margin: "영업이익률",
  debt_ratio: "부채비율",
  current_ratio: "유동비율",
  equity_ratio: "자기자본비율",
  interest_coverage: "이자보상배율",
  ocf_to_assets: "영업현금흐름/총자산",
  revenue_growth: "매출액증가율",
};
