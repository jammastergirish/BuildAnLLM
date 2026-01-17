"use client";

import {
  Bar,
  BarChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
  Cell,
} from "recharts";

type GradientData = {
  layer: string;
  norm: number;
};

export default function GradientChart({ data }: { data: GradientData[] }) {
  // Simple color scale based on magnitude could be cool, but fixed color is safer for now.
  // We can use a blue-ish gradient.
  
  return (
    <div style={{ width: "100%", height: 260 }}>
      <ResponsiveContainer>
        <BarChart data={data} margin={{ top: 8, right: 16, left: 0, bottom: 40 }}>
          <XAxis
            dataKey="layer"
            stroke="var(--chart-axis)"
            tick={{
              fontSize: 10,
              fill: "var(--chart-text)",
              fontFamily: "var(--font-mono)",
            }}
            angle={-45}
            textAnchor="end"
            interval={0}
            tickLine={{ stroke: "var(--chart-axis)" }}
            axisLine={{ stroke: "var(--chart-axis)" }}
          />
          <YAxis
            stroke="var(--chart-axis)"
            tick={{
              fontSize: 10,
              fill: "var(--chart-text)",
              fontFamily: "var(--font-mono)",
            }}
            tickLine={{ stroke: "var(--chart-axis)" }}
            axisLine={{ stroke: "var(--chart-axis)" }}
          />
          <Tooltip
            cursor={{ fill: "var(--surface-hover)" }}
            contentStyle={{
              background: "var(--tooltip-bg)",
              border: "1px solid var(--stroke-strong)",
              color: "var(--ink-1)",
              fontSize: 12,
              fontFamily: "var(--font-mono)",
            }}
            labelStyle={{ color: "var(--ink-1)" }}
            itemStyle={{ color: "var(--ink-1)" }}
            formatter={(value: number) => [value.toFixed(4), "Gradient Norm"]}
          />
          <Bar dataKey="norm" fill="var(--accent)" radius={[2, 2, 0, 0]}>
            {data.map((entry, index) => (
              <Cell key={`cell-${index}`} fill={entry.layer === "Embedding" ? "var(--accent-2)" : "var(--accent)"} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
