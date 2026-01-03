import { fireEvent, render, screen, waitFor } from "@testing-library/react";

import InferencePage from "../../app/inference/page";
import { fetchJson } from "../../lib/api";
import { useDemoMode } from "../../lib/demo";

vi.mock("../../lib/useScrollSpy", () => ({
  useScrollSpy: () => ({ activeSection: "", setActiveSection: vi.fn() }),
}));

vi.mock("../../components/Heatmap", () => ({
  default: () => <div data-testid="heatmap" />,
}));

vi.mock("../../components/LineChart", () => ({
  default: () => <div data-testid="linechart" />,
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

vi.mock("../../lib/demo", async () => {
  const actual = await vi.importActual<typeof import("../../lib/demo")>("../../lib/demo");
  return {
    ...actual,
    useDemoMode: vi.fn(),
  };
});

const fetchJsonMock = vi.mocked(fetchJson);
const useDemoModeMock = vi.mocked(useDemoMode);

describe("InferencePage", () => {
  beforeEach(() => {
    fetchJsonMock.mockReset();
    useDemoModeMock.mockReturnValue(false);
  });

  it("loads session and diagnostics data", async () => {
    fetchJsonMock.mockImplementation(async (path) => {
      if (path === "/api/checkpoints") {
        return {
          checkpoints: [{ id: "ckpt-1", name: "checkpoint_1.pt", mtime: 100 }],
        };
      }
      if (path === "/api/inference/sessions") {
        return {
          session_id: "session-1",
          tokenizer_type: "character",
          param_count_m: 1.2,
          cfg: { d_model: 256, n_layers: 2, n_heads: 4, n_ctx: 128 },
        };
      }
      if (String(path).includes("/attention")) {
        return { attention: [[1, 0], [0, 1]] };
      }
      if (String(path).includes("/logit-lens")) {
        return { layers: [{ layer: 0, predictions: [{ rank: 1, token: "A", prob: 0.5 }] }] };
      }
      if (String(path).includes("/layer-norms")) {
        return { layers: [{ layer: 0, avg_norm: 1.23 }] };
      }
      if (String(path).includes("/diagnostics")) {
        return {
          diagnostic_id: "diag-1",
          token_ids: [1, 2],
          token_labels: ["A", "B"],
        };
      }
      if (path === "/api/docs/inference-code") {
        return { snippets: [] };
      }
      return {};
    });

    render(<InferencePage />);

    await waitFor(() => {
      expect(screen.getByText("Tokenizer")).toBeInTheDocument();
      expect(screen.getByText("character")).toBeInTheDocument();
    });

    await waitFor(() => {
      expect(screen.getByText(/1\. A/)).toBeInTheDocument();
    });
  });

  it("streams generated text from the API", async () => {
    fetchJsonMock.mockImplementation(async (path) => {
      if (path === "/api/checkpoints") {
        return { checkpoints: [{ id: "ckpt-1", name: "checkpoint_1.pt", mtime: 100 }] };
      }
      if (path === "/api/inference/sessions") {
        return {
          session_id: "session-1",
          tokenizer_type: "character",
          param_count_m: 1.2,
          cfg: { d_model: 256, n_layers: 2, n_heads: 4, n_ctx: 128 },
        };
      }
      if (String(path).includes("/attention")) {
        return { attention: [[1, 0], [0, 1]] };
      }
      if (String(path).includes("/logit-lens")) {
        return { layers: [] };
      }
      if (String(path).includes("/layer-norms")) {
        return { layers: [] };
      }
      if (String(path).includes("/diagnostics")) {
        return {
          diagnostic_id: "diag-1",
          token_ids: [1, 2],
          token_labels: ["A", "B"],
        };
      }
      if (path === "/api/docs/inference-code") {
        return { snippets: [] };
      }
      return {};
    });

    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      text: async () => "",
      body: {
        getReader: () => {
          let done = false;
          return {
            read: async () => {
              if (done) {
                return { done: true, value: undefined };
              }
              done = true;
              const chunk = new TextEncoder().encode(
                'event: token\ndata: {"token":"X"}\n\n'
              );
              return { done: false, value: chunk };
            },
          };
        },
      },
    });
    vi.stubGlobal("fetch", fetchMock);

    render(<InferencePage />);

    await waitFor(() => {
      expect(screen.getByRole("button", { name: "Generate" })).toBeEnabled();
    });

    fireEvent.click(screen.getByRole("button", { name: "Generate" }));

    await waitFor(() => {
      const textareas = screen.getAllByRole("textbox");
      const output = textareas.find((node) => (node as HTMLTextAreaElement).readOnly);
      expect(output).toBeDefined();
      expect((output as HTMLTextAreaElement).value).toContain("X");
    });
  });

  it("disables generation controls in demo mode", async () => {
    useDemoModeMock.mockReturnValue(true);
    fetchJsonMock.mockImplementation(async (path) => {
      if (path === "/api/checkpoints") {
        return { checkpoints: [] };
      }
      if (path === "/api/docs/inference-code") {
        return { snippets: [] };
      }
      return {};
    });

    render(<InferencePage />);

    await waitFor(() => {
      expect(screen.getByRole("button", { name: "Generate" })).toBeDisabled();
    });
    expect(screen.getByText("Demo mode: inference disabled.")).toBeInTheDocument();
  });
});
