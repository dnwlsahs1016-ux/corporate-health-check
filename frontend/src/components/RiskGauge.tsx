function riskColor(score: number): string {
  if (score >= 60) return "var(--color-danger)";
  if (score >= 30) return "var(--color-caution)";
  return "var(--color-safe)";
}

function riskLabel(score: number): string {
  if (score >= 60) return "고위험";
  if (score >= 30) return "주의";
  return "안전";
}

export default function RiskGauge({ score, size = 96 }: { score: number; size?: number }) {
  const radius = (size - 12) / 2;
  const circumference = 2 * Math.PI * radius;
  const progress = Math.max(0, Math.min(100, score)) / 100;
  const color = riskColor(score);

  return (
    <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 6 }}>
      <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`}>
        <circle
          cx={size / 2}
          cy={size / 2}
          r={radius}
          fill="none"
          stroke="var(--color-border)"
          strokeWidth={8}
        />
        <circle
          cx={size / 2}
          cy={size / 2}
          r={radius}
          fill="none"
          stroke={color}
          strokeWidth={8}
          strokeLinecap="round"
          strokeDasharray={circumference}
          strokeDashoffset={circumference * (1 - progress)}
          transform={`rotate(-90 ${size / 2} ${size / 2})`}
        />
        <text
          x="50%"
          y="48%"
          textAnchor="middle"
          dominantBaseline="middle"
          fontSize={size * 0.24}
          fontWeight={700}
          fill="var(--color-text)"
        >
          {score.toFixed(1)}
        </text>
        <text
          x="50%"
          y="68%"
          textAnchor="middle"
          dominantBaseline="middle"
          fontSize={size * 0.12}
          fill="var(--color-text-muted)"
        >
          / 100
        </text>
      </svg>
      <span style={{ color, fontWeight: 600, fontSize: 13 }}>{riskLabel(score)}</span>
    </div>
  );
}
