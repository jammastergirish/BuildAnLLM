import { render, screen } from "@testing-library/react";

import DemoBanner from "../../components/DemoBanner";
import { useDemoMode } from "../../lib/demo";

vi.mock("../../lib/demo", async () => {
  const actual = await vi.importActual<typeof import("../../lib/demo")>("../../lib/demo");
  return {
    ...actual,
    useDemoMode: vi.fn(),
  };
});

const useDemoModeMock = vi.mocked(useDemoMode);

describe("DemoBanner", () => {
  beforeEach(() => {
    useDemoModeMock.mockReturnValue(false);
  });

  it("does not render when demo mode is off", () => {
    render(<DemoBanner />);
    expect(screen.queryByText("Demo mode")).not.toBeInTheDocument();
  });

  it("renders the repo link when demo mode is on", () => {
    useDemoModeMock.mockReturnValue(true);
    render(<DemoBanner />);

    expect(screen.getByText("Demo mode")).toBeInTheDocument();
    const link = screen.getByRole("link", { name: /clone the repo/i });
    expect(link).toHaveAttribute("href", "https://github.com/jammastergirish/buildanllm");
  });
});
