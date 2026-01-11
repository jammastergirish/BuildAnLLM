import { act, fireEvent, render, screen, waitFor } from "@testing-library/react";

import PretrainPage from "../../app/pretrain/page";
import { fetchJson } from "../../lib/api";
import { useDemoMode } from "../../lib/demo";

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

vi.mock("../../lib/demo", async () => {
  const actual = await vi.importActual<typeof import("../../lib/demo")>("../../lib/demo");
  return {
    ...actual,
    useDemoMode: vi.fn(),
  };
});

const fetchJsonMock = vi.mocked(fetchJson);
const useDemoModeMock = vi.mocked(useDemoMode);

describe("PretrainPage", () => {
  const defaultDataSources = {
    sources: [
      { name: "Test Author", filename: "test.txt", language: "English", script: "Latin", words: 100, chars: 500 },
    ],
  };

  beforeEach(() => {
    fetchJsonMock.mockReset();
    useDemoModeMock.mockReturnValue(false);
    // Default mock that handles common endpoints
    fetchJsonMock.mockImplementation(async (path) => {
      if (path === "/api/pretrain/data-sources") {
        return defaultDataSources;
      }
      if (path === "/api/docs/model-code") {
        return { snippets: [] };
      }
      return {};
    });
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  it("switches tokenizer when preset changes", () => {
    render(<PretrainPage />);

    fireEvent.click(screen.getByRole("button", { name: "LLaMA 4" }));
    const tokenizerButton = screen.getByRole("button", { name: "sentencepiece" });
    expect(tokenizerButton).toHaveAttribute("aria-pressed", "true");
  });

  it("loads code snippets after debounce", async () => {
    vi.useFakeTimers();
    fetchJsonMock.mockImplementation(async (path) => {
      if (path === "/api/pretrain/data-sources") {
        return defaultDataSources;
      }
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
      if (path === "/api/pretrain/data-sources") {
        return defaultDataSources;
      }
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

    // Wait for data sources to load first
    await waitFor(() => {
      expect(screen.getByText("Test Author")).toBeInTheDocument();
    });

    // Click Start Training wrapped in act
    await act(async () => {
      fireEvent.click(screen.getByRole("button", { name: "Start Training" }));
    });

    await waitFor(
      () => {
        expect(fetchJsonMock).toHaveBeenCalledWith(
          "/api/pretrain/jobs",
          expect.objectContaining({ method: "POST" })
        );
      },
      { timeout: 3000 }
    );

    const call = fetchJsonMock.mock.calls.find(([path]) => path === "/api/pretrain/jobs");
    const form = call?.[1]?.body as FormData;
    const payload = JSON.parse(String(form.get("payload")));
    expect(payload.tokenizer_type).toBe("bpe-tiktoken");
    expect(payload.use_einops).toBe(true);
  });

  it("disables training controls in demo mode", () => {
    useDemoModeMock.mockReturnValue(true);
    fetchJsonMock.mockImplementation(async (path) => {
      if (path === "/api/pretrain/data-sources") {
        return defaultDataSources;
      }
      if (path === "/api/docs/model-code") {
        return { snippets: [] };
      }
      return {};
    });

    render(<PretrainPage />);

    expect(screen.getByRole("button", { name: "Start Training" })).toBeDisabled();
    expect(screen.getByRole("button", { name: "Step" })).toBeDisabled();
    expect(screen.getByText("Demo mode: pre-training disabled.")).toBeInTheDocument();
  });

  describe("data source selection", () => {
    const mockDataSources = [
      {
        name: "George Orwell",
        filename: "input_data/pretraining/orwell.txt",
        language: "English",
        script: "Latin",
        words: 1000,
        chars: 5000,
      },
      {
        name: "Muhammad al-Khwarizmi",
        filename: "input_data/pretraining/aljbr.txt",
        language: "Arabic",
        script: "Arabic",
        words: 500,
        chars: 2500,
      },
    ];

    it("renders data sources table with language and script columns", async () => {
      fetchJsonMock.mockImplementation(async (path) => {
        if (path === "/api/pretrain/data-sources") {
          return { sources: mockDataSources };
        }
        if (path === "/api/docs/model-code") {
          return { snippets: [] };
        }
        return {};
      });

      render(<PretrainPage />);

      await waitFor(() => {
        expect(screen.getByText("George Orwell")).toBeInTheDocument();
      });

      // Check for language/script columns (Arabic appears twice - language and script)
      expect(screen.getByText("English")).toBeInTheDocument();
      expect(screen.getAllByText("Arabic").length).toBeGreaterThanOrEqual(1);
      expect(screen.getByText("Muhammad al-Khwarizmi")).toBeInTheDocument();
    });

    it("includes training_text_paths in job creation payload when sources selected", async () => {
      fetchJsonMock.mockImplementation(async (path) => {
        if (path === "/api/pretrain/data-sources") {
          return { sources: mockDataSources };
        }
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

      // Wait for data sources to load
      await waitFor(() => {
        expect(screen.getByText("George Orwell")).toBeInTheDocument();
      });

      // Start training using act
      await act(async () => {
        fireEvent.click(screen.getByRole("button", { name: "Start Training" }));
      });

      // Allow async operations to complete
      await waitFor(
        () => {
          const jobsCalls = fetchJsonMock.mock.calls.filter(([p]) => p === "/api/pretrain/jobs");
          expect(jobsCalls.length).toBeGreaterThan(0);
        },
        { timeout: 3000 }
      );

      // Find the job creation call and verify training_text_paths is included
      const call = fetchJsonMock.mock.calls.find(([path]) => path === "/api/pretrain/jobs");
      const form = call?.[1]?.body as FormData;
      const payload = JSON.parse(String(form.get("payload")));
      
      // Should include training_text_paths with selected sources
      expect(payload).toHaveProperty("training_text_paths");
      expect(Array.isArray(payload.training_text_paths)).toBe(true);
    });

    it("shows word and character counts for data sources", async () => {
      fetchJsonMock.mockImplementation(async (path) => {
        if (path === "/api/pretrain/data-sources") {
          return { sources: mockDataSources };
        }
        return { snippets: [] };
      });

      render(<PretrainPage />);

      await waitFor(() => {
        expect(screen.getByText("George Orwell")).toBeInTheDocument();
      });

      // Check for word counts (may appear multiple times in different rows)
      expect(screen.getAllByText("1,000").length).toBeGreaterThanOrEqual(1);
      expect(screen.getAllByText("5,000").length).toBeGreaterThanOrEqual(1);
    });
  });

  describe("multi-file upload", () => {
    beforeEach(() => {
      fetchJsonMock.mockImplementation(async (path) => {
        if (path === "/api/pretrain/data-sources") {
          return { sources: [] };
        }
        if (path === "/api/docs/model-code") {
          return { snippets: [] };
        }
        return {};
      });
    });

    it("shows file upload input that accepts multiple files", async () => {
      render(<PretrainPage />);

      // Find file input with multiple attribute
      const fileInput = document.querySelector('input[type="file"][multiple]');
      expect(fileInput).toBeInTheDocument();
      expect(fileInput).toHaveAttribute("accept", ".txt");
    });

    it("displays uploaded files in the table", async () => {
      render(<PretrainPage />);

      const fileInput = document.querySelector('input[type="file"]') as HTMLInputElement;
      expect(fileInput).toBeInTheDocument();

      // Create mock files
      const file1 = new File(["Hello world content"], "testfile1.txt", { type: "text/plain" });
      const file2 = new File(["More content here"], "testfile2.txt", { type: "text/plain" });

      // Simulate file upload
      await act(async () => {
        Object.defineProperty(fileInput, "files", {
          value: [file1, file2],
          writable: false,
        });
        fireEvent.change(fileInput);
      });

      // Wait for files to appear in the table
      await waitFor(() => {
        expect(screen.getByText(/testfile1.txt/)).toBeInTheDocument();
      });
      expect(screen.getByText(/testfile2.txt/)).toBeInTheDocument();
    });

    it("uses FormData when starting training job", async () => {
      // This test verifies that the component uses FormData format for API calls
      // which is required for file uploads
      fetchJsonMock.mockImplementation(async (path) => {
        if (path === "/api/pretrain/data-sources") {
          return {
            sources: [
              { name: "Test Source", filename: "test.txt", language: "English", script: "Latin", words: 100, chars: 500 },
            ],
          };
        }
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

      // Wait for data sources to load
      await waitFor(() => {
        expect(screen.getByText("Test Source")).toBeInTheDocument();
      });

      // Start training
      await act(async () => {
        fireEvent.click(screen.getByRole("button", { name: "Start Training" }));
      });

      // Wait for API call
      await waitFor(
        () => {
          const calls = fetchJsonMock.mock.calls.filter(
            ([p]: [string]) => p === "/api/pretrain/jobs"
          );
          expect(calls.length).toBeGreaterThan(0);
        },
        { timeout: 3000 }
      );

      // Check that FormData was used
      const call = fetchJsonMock.mock.calls.find(([p]: [string]) => p === "/api/pretrain/jobs");
      expect(call).toBeDefined();
      const form = call?.[1]?.body as FormData;
      expect(form).toBeInstanceOf(FormData);
    });

    it("shows remove button for uploaded files", async () => {
      render(<PretrainPage />);

      const fileInput = document.querySelector('input[type="file"]') as HTMLInputElement;

      const file = new File(["Test content"], "removable.txt", { type: "text/plain" });

      await act(async () => {
        Object.defineProperty(fileInput, "files", {
          value: [file],
          writable: false,
        });
        fireEvent.change(fileInput);
      });

      await waitFor(() => {
        expect(screen.getByText(/removable.txt/)).toBeInTheDocument();
      });

      // Find and click remove button (×)
      const removeButton = screen.getByRole("button", { name: "×" });
      expect(removeButton).toBeInTheDocument();
    });
  });
});
