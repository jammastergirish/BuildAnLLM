import { act, fireEvent, render, screen, waitFor } from "@testing-library/react";

import FinetunePage from "../../app/finetune/page";
import { fetchJson } from "../../lib/api";
import { useDemoMode } from "../../lib/demo";

vi.mock("../../lib/useSse", () => ({
  useSse: () => ({ lastEvent: null, error: null }),
}));

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

class MockFileReader {
  result: string | ArrayBuffer | null = null;
  onload: (() => void) | null = null;
  readAsText() {
    this.result = "prompt,response\nHello,World\n";
    this.onload?.();
  }
}

describe("FinetunePage", () => {
  beforeEach(() => {
    fetchJsonMock.mockReset();
    vi.stubGlobal("FileReader", MockFileReader);
    useDemoModeMock.mockReturnValue(false);
  });

  it("loads checkpoints and selects the latest non-finetuned", async () => {
    fetchJsonMock.mockImplementation(async (path) => {
      if (path === "/api/checkpoints") {
        return {
          checkpoints: [
            { id: "ckpt-1", name: "checkpoint_1.pt", mtime: 100, is_finetuned: false },
            { id: "ckpt-2", name: "checkpoint_2.pt", mtime: 200, is_finetuned: true },
            { id: "ckpt-3", name: "checkpoint_3.pt", mtime: 300, is_finetuned: false },
          ],
        };
      }
      if (path === "/api/docs/finetuning-code") {
        return { snippets: [] };
      }
      if (String(path).startsWith("/api/checkpoints/")) {
        return { cfg: { d_model: 256, n_layers: 2, n_heads: 4 } };
      }
      return {};
    });

    render(<FinetunePage />);

    await waitFor(() => {
      const option = screen.getByRole("option", { name: "Select checkpoint" });
      const select = option.parentElement as HTMLSelectElement;
      expect(select.value).toBe("ckpt-3");
    });

    expect(fetchJsonMock).toHaveBeenCalledWith("/api/checkpoints");
    expect(fetchJsonMock.mock.calls.some(([path]) => String(path).startsWith("/api/checkpoints/"))).toBe(true);
  });

  it("shows lora controls when selected", async () => {
    fetchJsonMock.mockImplementation(async (path) => {
      if (path === "/api/checkpoints") {
        return { checkpoints: [] };
      }
      if (path === "/api/docs/finetuning-code") {
        return { snippets: [] };
      }
      return {};
    });
    render(<FinetunePage />);
    await waitFor(() => {
      expect(fetchJsonMock).toHaveBeenCalled();
    });

    fireEvent.click(screen.getByRole("button", { name: "LoRA" }));
    expect(screen.getByText(/LoRA enabled/)).toBeInTheDocument();
    expect(screen.getByText("Rank")).toBeInTheDocument();
  });

  it("previews CSV content when a file is chosen", async () => {
    fetchJsonMock.mockImplementation(async (path) => {
      if (path === "/api/checkpoints") {
        return { checkpoints: [] };
      }
      if (path === "/api/docs/finetuning-code") {
        return { snippets: [] };
      }
      return {};
    });
    const { container } = render(<FinetunePage />);
    await waitFor(() => {
      expect(fetchJsonMock).toHaveBeenCalled();
    });

    const fileInput = container.querySelector('input[type="file"]') as HTMLInputElement;
    const file = new File(["prompt,response\nHello,World\n"], "data.csv", { type: "text/csv" });
    await act(async () => {
      fireEvent.change(fileInput, { target: { files: [file] } });
    });

    await waitFor(() => {
      expect(screen.getByText("prompt")).toBeInTheDocument();
      expect(screen.getByText("Hello")).toBeInTheDocument();
    });
  });

  it("starts a fine-tune job with lora payload", async () => {
    fetchJsonMock.mockImplementation(async (path) => {
      if (path === "/api/checkpoints") {
        return { checkpoints: [{ id: "ckpt-1", name: "checkpoint_1.pt", mtime: 100, is_finetuned: false }] };
      }
      if (path === "/api/docs/finetuning-code") {
        return { snippets: [] };
      }
      if (path === "/api/finetune/jobs") {
        return {
          job_id: "job-1",
          kind: "finetune",
          status: "paused",
          iter: 0,
          max_iters: 10,
          created_at: 0,
        };
      }
      if (String(path).startsWith("/api/checkpoints/")) {
        return { cfg: { d_model: 256, n_layers: 2, n_heads: 4 } };
      }
      return {};
    });

    render(<FinetunePage />);

    await waitFor(() => {
      const option = screen.getByRole("option", { name: "Select checkpoint" });
      const select = option.parentElement as HTMLSelectElement;
      expect(select.value).toBe("ckpt-1");
    });

    fireEvent.click(screen.getByRole("button", { name: "LoRA" }));
    fireEvent.click(screen.getByRole("button", { name: "Start Fine-Tuning" }));

    await waitFor(() => {
      expect(fetchJsonMock).toHaveBeenCalledWith(
        "/api/finetune/jobs",
        expect.objectContaining({ method: "POST" })
      );
    });

    const call = fetchJsonMock.mock.calls.find(([path]) => path === "/api/finetune/jobs");
    const form = call?.[1]?.body as FormData;
    const payload = JSON.parse(String(form.get("payload")));
    expect(payload.use_lora).toBe(true);
    expect(payload.lora_rank).toBe(8);
  });

  it("disables training controls in demo mode", () => {
    useDemoModeMock.mockReturnValue(true);
    fetchJsonMock.mockImplementation(async (path) => {
      if (path === "/api/checkpoints") {
        return { checkpoints: [] };
      }
      if (path === "/api/docs/finetuning-code") {
        return { snippets: [] };
      }
      return {};
    });

    render(<FinetunePage />);

    expect(screen.getByRole("button", { name: "Start Fine-Tuning" })).toBeDisabled();
    expect(screen.getByRole("button", { name: "Step" })).toBeDisabled();
    expect(screen.getByText("Demo mode: fine-tuning disabled.")).toBeInTheDocument();
  });
});
