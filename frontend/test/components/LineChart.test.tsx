import React from "react";
import { render, screen } from "@testing-library/react";

vi.mock("recharts", () => ({
  ResponsiveContainer: ({ children }: { children: React.ReactNode }) => (
    <div data-testid="responsive">{children}</div>
  ),
  LineChart: ({ children }: { children: React.ReactNode }) => (
    <div data-testid="line-chart">{children}</div>
  ),
  Line: ({ dataKey }: { dataKey: string }) => <div data-testid={`line-${dataKey}`} />,
  XAxis: ({ children, domain }: { children?: React.ReactNode; domain?: unknown }) => (
    <div data-testid="x-axis" data-domain={JSON.stringify(domain)}>
      {children}
    </div>
  ),
  YAxis: ({ children }: { children?: React.ReactNode }) => (
    <div data-testid="y-axis">{children}</div>
  ),
  Tooltip: () => <div data-testid="tooltip" />,
  Legend: () => <div data-testid="legend" />,
  Label: ({ value }: { value?: string }) => <span>{value}</span>,
}));

import LineChart from "../../components/LineChart";

describe("LineChart", () => {
  const chartProps = {
    data: [{ step: 1, loss: 0.2 }],
    lines: [{ dataKey: "loss", name: "Loss", color: "#f00" }],
    xKey: "step",
  };

  it("ends the x-axis at the last data point by default", () => {
    render(<LineChart {...chartProps} />);
    expect(screen.getByTestId("x-axis")).toHaveAttribute(
      "data-domain",
      JSON.stringify([0, "dataMax"])
    );
  });

  it("uses the x-axis domain the caller passes", () => {
    render(<LineChart {...chartProps} xDomain={[0, 100]} />);
    expect(screen.getByTestId("x-axis")).toHaveAttribute(
      "data-domain",
      JSON.stringify([0, 100])
    );
  });

  it("renders line series and labels", () => {
    render(
      <LineChart
        data={[{ step: 1, loss: 0.2 }]}
        lines={[{ dataKey: "loss", name: "Loss", color: "#f00" }]}
        xKey="step"
        xLabel="Step"
        yLabel="Loss"
      />
    );

    expect(screen.getByTestId("line-loss")).toBeInTheDocument();
    expect(screen.getByText("Step")).toBeInTheDocument();
    expect(screen.getByText("Loss")).toBeInTheDocument();
  });
});
