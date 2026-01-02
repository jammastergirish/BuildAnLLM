import { render, screen, waitFor } from "@testing-library/react";

import PlaygroundPage from "../../app/playground/page";
import { fetchJson } from "../../lib/api";

vi.mock("../../lib/api", async () => {
  const actual = await vi.importActual<typeof import("../../lib/api")>("../../lib/api");
  return {
    ...actual,
    fetchJson: vi.fn(),
  };
});

const fetchJsonMock = vi.mocked(fetchJson);

describe("PlaygroundPage", () => {
  beforeEach(() => {
    fetchJsonMock.mockReset();
  });

  it("loads models and tokenizes input", async () => {
    fetchJsonMock.mockImplementation(async (path) => {
      if (path === "/api/tokenizers/tiktoken/models") {
        return { models: ["gpt-4", "gpt-3.5"] };
      }
      if (path === "/api/tokenizers/tiktoken/encode") {
        return {
          tokens: [1, 2],
          decoded_tokens: ["Hi", "!"],
          token_count: 2,
          char_count: 3,
          chars_per_token: 1.5,
        };
      }
      return {};
    });

    render(<PlaygroundPage />);

    await waitFor(() => {
      expect(screen.getByText("gpt-4")).toBeInTheDocument();
    });

    await waitFor(() => {
      expect(screen.getByText("Hi")).toBeInTheDocument();
      expect(screen.getByText("!")).toBeInTheDocument();
    });

    const tokensHeading = screen.getByRole("heading", { name: "Tokens" });
    const tokensStat = tokensHeading.closest(".stat");
    expect(tokensStat).toHaveTextContent("2");
  });
});
