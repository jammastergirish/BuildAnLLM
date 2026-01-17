"use client";

import {
  ScatterChart,
  Scatter,
  XAxis,
  YAxis,
  ZAxis,
  Tooltip,
  ResponsiveContainer,
} from "recharts";

type TrajectoryPoint = {
  x: number;
  y: number;
  iter: number;
  loss?: number;
};

export default function LossLandscape({ data }: { data: TrajectoryPoint[] }) {
  // We want to show the path. Recharts Scatter can do line if we provide sorting.
  // But let's just show points for now, maybe connected.
  // Recharts Scatter 'line' prop connects points.
  
  return (
    <div style={{ width: "100%", height: 260 }}>
      <ResponsiveContainer>
        <ScatterChart margin={{ top: 20, right: 20, bottom: 20, left: 0 }}>
          <XAxis
            type="number"
            dataKey="x"
            name="proj_u"
            stroke="var(--chart-axis)"
            tick={{ fontSize: 10, fill: "var(--chart-text)" }}
            tickLine={false}
            axisLine={{ stroke: "var(--chart-axis)" }}
          />
          <YAxis
            type="number"
            dataKey="y"
            name="proj_v"
            stroke="var(--chart-axis)"
            tick={{ fontSize: 10, fill: "var(--chart-text)" }}
            tickLine={false}
            axisLine={{ stroke: "var(--chart-axis)" }}
          />
          <ZAxis type="number" dataKey="iter" range={[20, 20]} />
          <Tooltip
            cursor={{ strokeDasharray: "3 3" }}
            contentStyle={{
              background: "var(--tooltip-bg)",
              border: "1px solid var(--stroke-strong)",
              color: "var(--ink-1)",
              fontSize: 12,
              fontFamily: "var(--font-mono)",
            }}
            labelStyle={{ color: "var(--ink-1)" }}
            itemStyle={{ color: "var(--accent)" }}
            formatter={(value: any, name: any, props: any) => {
              if (name === "proj_u" || name === "proj_v") return value.toFixed(4);
              if (name === "iter") return value; 
              return value;
            }}
          />
          <Scatter
            name="Trajectory"
            data={data}
            fill="var(--accent)"
            line={{ stroke: "var(--accent)", strokeWidth: 1 }}
            lineType="joint"
          />
        </ScatterChart>
      </ResponsiveContainer>
    </div>
  );
}
