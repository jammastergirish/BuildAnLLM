"use client";

import {
  Line,
  LineChart as RechartsLineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
  Legend,
  Label,
} from "recharts";

export type LineSeries = {
  dataKey: string;
  name: string;
  color: string;
};

type AxisDomainValue = number | "dataMin" | "dataMax" | "auto";
type AxisDomain = [AxisDomainValue, AxisDomainValue];

export default function LineChart({
  data,
  lines,
  xKey,
  xLabel,
  yLabel,
  xDomain,
  yDomain,
}: {
  data: Array<Record<string, number>>;
  lines: LineSeries[];
  xKey: string;
  xLabel?: string;
  yLabel?: string;
  xDomain?: AxisDomain;
  yDomain?: AxisDomain;
}) {
  const legendAtTop = Boolean(xLabel);
  const margin = {
    top: legendAtTop ? 28 : 8,
    right: 16,
    left: yLabel ? 24 : 0,
    bottom: xLabel ? 24 : 0,
  };

  return (
    <div style={{ width: "100%", height: 260 }}>
      <ResponsiveContainer>
        <RechartsLineChart data={data} margin={margin}>
          <XAxis
            dataKey={xKey}
            type="number"
            domain={xDomain}
            allowDataOverflow={Boolean(xDomain)}
            stroke="var(--chart-axis)"
            tick={{
              fontSize: 12,
              fill: "var(--chart-text)",
              fontFamily: "var(--font-mono), IBM Plex Mono, SFMono-Regular, Menlo, monospace",
            }}
            tickLine={{ stroke: "var(--chart-axis)" }}
            axisLine={{ stroke: "var(--chart-axis)" }}
          >
            {xLabel && (
              <Label value={xLabel} position="insideBottom" offset={-6} fill="var(--chart-text)" />
            )}
          </XAxis>
          <YAxis
            type="number"
            domain={yDomain}
            allowDataOverflow={Boolean(yDomain)}
            stroke="var(--chart-axis)"
            tick={{
              fontSize: 12,
              fill: "var(--chart-text)",
              fontFamily: "var(--font-mono), IBM Plex Mono, SFMono-Regular, Menlo, monospace",
            }}
            tickLine={{ stroke: "var(--chart-axis)" }}
            axisLine={{ stroke: "var(--chart-axis)" }}
          >
            {yLabel && (
              <Label
                value={yLabel}
                angle={-90}
                position="insideLeft"
                offset={-8}
                fill="var(--chart-text)"
              />
            )}
          </YAxis>
          <Tooltip
            contentStyle={{
              background: "var(--tooltip-bg)",
              border: "1px solid var(--stroke-strong)",
              color: "var(--ink-1)",
              fontSize: 12,
              fontFamily: "var(--font-mono), IBM Plex Mono, SFMono-Regular, Menlo, monospace",
            }}
            labelStyle={{ color: "var(--ink-1)" }}
            itemStyle={{ color: "var(--ink-1)" }}
          />
          <Legend
            verticalAlign={legendAtTop ? "top" : "bottom"}
            wrapperStyle={{
              color: "var(--chart-text)",
              fontSize: 12,
              fontFamily: "var(--font-mono), IBM Plex Mono, SFMono-Regular, Menlo, monospace",
              paddingTop: legendAtTop ? 4 : 0,
            }}
          />
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
