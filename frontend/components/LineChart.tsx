"use client";

import {
  Line,
  LineChart as RechartsLineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
  Legend,
} from "recharts";

export type LineSeries = {
  dataKey: string;
  name: string;
  color: string;
};

export default function LineChart({
  data,
  lines,
  xKey,
}: {
  data: Array<Record<string, number>>;
  lines: LineSeries[];
  xKey: string;
}) {
  return (
    <div style={{ width: "100%", height: 260 }}>
      <ResponsiveContainer>
        <RechartsLineChart data={data} margin={{ top: 8, right: 16, left: 0, bottom: 0 }}>
          <XAxis dataKey={xKey} tick={{ fontSize: 12 }} />
          <YAxis tick={{ fontSize: 12 }} />
          <Tooltip />
          <Legend />
          {lines.map((line) => (
            <Line
              key={line.dataKey}
              type="monotone"
              dataKey={line.dataKey}
              name={line.name}
              stroke={line.color}
              strokeWidth={2}
              dot={false}
            />
          ))}
        </RechartsLineChart>
      </ResponsiveContainer>
    </div>
  );
}
