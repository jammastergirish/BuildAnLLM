import { act, fireEvent, render, screen, waitFor } from "@testing-library/react";

import PretrainPage from "../../app/pretrain/page";
import { fetchJson } from "../../lib/api";

vi.mock("../../lib/useSse", () => ({
  useSse: () => ({ lastEvent: null, error: null }),
}));

vi.mock("../../lib/useScrollSpy", () => ({
  useScrollSpy: () => ({ activeSection: "", setActiveSection: vi.fn() }),
}));

vi.mock("../../components/GraphvizDiagram", () => ({
  default: () => <div data-testid="graphviz" />,
}));

vi.mock("../../components/LineChart", () => ({
  default: () => <div data-testid="linechart" />,
}));

vi.mock("../../components/Heatmap", () => ({
  default: () => <div data-testid="heatmap" />,
}));

vi.mock("../../components/MarkdownBlock", () => ({
  default: () => <div data-testid="markdown" />,
}));

vi.mock("../../components/CodePanel", () => ({
  default: ({ snippet }: { snippet: { title: string } }) => <div>{snippet.title}</div>,
}));

vi.mock("../../lib/api", async () => {
  const actual = await vi.importActual<typeof import("../../lib/api")>("../../lib/api");
  return {
    ...actual,
    fetchJson: vi.fn(),
  };
});

const fetchJsonMock = vi.mocked(fetchJson);

describe("PretrainPage", () => {
  beforeEach(() => {
    fetchJsonMock.mockReset();
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  it("switches tokenizer when preset changes", () => {
    fetchJsonMock.mockResolvedValue({ snippets: [] });
    render(<PretrainPage />);

    fireEvent.click(screen.getByRole("button", { name: "LLaMA 4" }));
    const tokenizerButton = screen.getByRole("button", { name: "sentencepiece" });
    expect(tokenizerButton).toHaveAttribute("aria-pressed", "true");
  });

  it("loads code snippets after debounce", async () => {
    vi.useFakeTimers();
    fetchJsonMock.mockImplementation(async (path) => {
      if (path === "/api/docs/model-code") {
        return { snippets: [{ title: "Snippet A" }] };
      }
      return {};
    });

    render(<PretrainPage />);
    await act(async () => {
      vi.advanceTimersByTime(500);
      await Promise.resolve();
    });

    expect(screen.getByText("Snippet A")).toBeInTheDocument();
  });

  it("starts a training job with form data payload", async () => {
    fetchJsonMock.mockImplementation(async (path) => {
      if (path === "/api/pretrain/jobs") {
        return {
          job_id: "job-1",
          kind: "pretrain",
          status: "paused",
          iter: 0,
          max_iters: 10,
          created_at: 0,
        };
      }
      return { snippets: [] };
    });

    render(<PretrainPage />);
    fireEvent.click(screen.getByRole("button", { name: "Start Training" }));

    await waitFor(() => {
      expect(fetchJsonMock).toHaveBeenCalledWith(
        "/api/pretrain/jobs",
        expect.objectContaining({ method: "POST" })
      );
    });

    const call = fetchJsonMock.mock.calls.find(([path]) => path === "/api/pretrain/jobs");
    const form = call?.[1]?.body as FormData;
    const payload = JSON.parse(String(form.get("payload")));
    expect(payload.tokenizer_type).toBe("bpe-tiktoken");
    expect(payload.use_einops).toBe(true);
  });
});
