import { Link } from "react-router-dom";

export default function Header() {
  return (
    <header
      style={{
        borderBottom: "1px solid var(--color-border)",
        padding: "20px 24px",
      }}
    >
      <div style={{ maxWidth: 880, margin: "0 auto" }}>
        <Link to="/" style={{ display: "inline-flex", alignItems: "baseline", gap: 10 }}>
          <span
            style={{
              fontSize: 30,
              fontWeight: 800,
              letterSpacing: -0.8,
              color: "var(--color-text)",
            }}
          >
            기업건강검진
          </span>
          <span style={{ fontSize: 13, color: "var(--color-primary)", fontWeight: 600 }}>
            코스닥 재무위험 진단
          </span>
        </Link>
      </div>
    </header>
  );
}
