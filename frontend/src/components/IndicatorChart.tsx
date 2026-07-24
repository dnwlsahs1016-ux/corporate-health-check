import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { TimelineEntry } from "../types";

export default function IndicatorChart({
  timeline,
  ratioKey,
  label,
  commentary,
}: {
  timeline: TimelineEntry[];
  ratioKey: string;
  label: string;
  commentary?: string;
}) {
  const data = timeline.map((entry) => ({
    year: entry.year,
    원지표: entry.ratios[ratioKey] ?? null,
    고유위험: entry.ratios_idiosyncratic[ratioKey] ?? null,
  }));

  return (
    <div>
      <h4 style={{ margin: "0 0 8px", fontSize: 14, color: "var(--color-text-muted)" }}>
        {label}
      </h4>
      <ResponsiveContainer width="100%" height={220}>
        <LineChart data={data} margin={{ top: 4, right: 12, left: 0, bottom: 4 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="var(--color-border)" />
          <XAxis dataKey="year" tick={{ fontSize: 12 }} />
          <YAxis tick={{ fontSize: 12 }} width={50} />
          <Tooltip />
          <Legend wrapperStyle={{ fontSize: 12 }} />
          <Line type="monotone" dataKey="원지표" stroke="var(--color-text-muted)" strokeWidth={2} dot={false} />
          <Line type="monotone" dataKey="고유위험" stroke="var(--color-primary)" strokeWidth={2} dot={false} />
        </LineChart>
      </ResponsiveContainer>
      {commentary && (
        <p
          style={{
            margin: "4px 0 0",
            fontSize: 12.5,
            color: "var(--color-text-muted)",
            lineHeight: 1.5,
          }}
        >
          {commentary}
        </p>
      )}
    </div>
  );
}
