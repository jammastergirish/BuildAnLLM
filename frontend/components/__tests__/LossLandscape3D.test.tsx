import { describe, it, expect, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import LossLandscape3D from "../../components/LossLandscape3D";

vi.mock("next/dynamic", () => ({
  default: () => {
      const MockPlot = () => <div data-testid="mock-plotly">Plotly Chart</div>;
      return MockPlot;
  }
}));

describe("LossLandscape3D", () => {
  it("renders empty state message when no data", () => {
    render(<LossLandscape3D data={[]} />);
    expect(screen.getByText("Waiting for training data...")).toBeInTheDocument();
  });

  it("renders plotly chart when data is provided", () => {
    const mockData = [
        { x: 0, y: 0, loss: 1.0 },
        { x: 0.1, y: 0.1, loss: 0.9 }
    ];
    render(<LossLandscape3D data={mockData} />);
    expect(screen.getByTestId("mock-plotly")).toBeInTheDocument();
  });
});
