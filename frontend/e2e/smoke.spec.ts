import { test, expect, type Page } from "@playwright/test";

const apiBaseUrl = process.env.NEXT_PUBLIC_API_BASE_URL || "http://127.0.0.1:8000";
const apiOrigin = new URL(apiBaseUrl).origin;
const corsHeaders = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Methods": "GET,POST,OPTIONS",
  "Access-Control-Allow-Headers": "Content-Type",
};

const checkpoints = [
  {
    id: "ckpt-1",
    name: "checkpoint_1.pt",
    mtime: 1710000000,
    is_finetuned: false,
  },
];

async function mockApi(page: Page) {
  await page.route(`${apiOrigin}/api/**`, async (route) => {
    const request = route.request();
    if (request.method() === "OPTIONS") {
      await route.fulfill({ status: 204, headers: corsHeaders, body: "" });
      return;
    }

    const url = new URL(request.url());
    const path = url.pathname;

    if (path === "/api/checkpoints") {
      await route.fulfill({
        status: 200,
        headers: corsHeaders,
        contentType: "application/json",
        body: JSON.stringify({ checkpoints }),
      });
      return;
    }

    if (path.startsWith("/api/checkpoints/")) {
      await route.fulfill({
        status: 200,
        headers: corsHeaders,
        contentType: "application/json",
        body: JSON.stringify({ cfg: { d_model: 256, n_layers: 2, n_heads: 4 } }),
      });
      return;
    }

    if (path === "/api/docs/model-code" || path === "/api/docs/finetuning-code" || path === "/api/docs/inference-code") {
      await route.fulfill({
        status: 200,
        headers: corsHeaders,
        contentType: "application/json",
        body: JSON.stringify({ snippets: [] }),
      });
      return;
    }

    if (path === "/api/inference/sessions") {
      await route.fulfill({
        status: 200,
        headers: corsHeaders,
        contentType: "application/json",
        body: JSON.stringify({
          session_id: "session-1",
          tokenizer_type: "character",
          param_count_m: 1.2,
          cfg: { d_model: 256, n_layers: 2, n_heads: 4, n_ctx: 128 },
        }),
      });
      return;
    }

    if (path.endsWith("/diagnostics")) {
      await route.fulfill({
        status: 200,
        headers: corsHeaders,
        contentType: "application/json",
        body: JSON.stringify({
          diagnostic_id: "diag-1",
          token_ids: [1, 2],
          token_labels: ["A", "B"],
        }),
      });
      return;
    }

    if (path.endsWith("/attention")) {
      await route.fulfill({
        status: 200,
        headers: corsHeaders,
        contentType: "application/json",
        body: JSON.stringify({ attention: [[1, 0], [0, 1]] }),
      });
      return;
    }

    if (path.endsWith("/logit-lens")) {
      await route.fulfill({
        status: 200,
        headers: corsHeaders,
        contentType: "application/json",
        body: JSON.stringify({
          layers: [{ layer: 0, predictions: [{ rank: 1, token: "A", prob: 0.5 }] }],
        }),
      });
      return;
    }

    if (path.endsWith("/layer-norms")) {
      await route.fulfill({
        status: 200,
        headers: corsHeaders,
        contentType: "application/json",
        body: JSON.stringify({ layers: [{ layer: 0, avg_norm: 1.23 }] }),
      });
      return;
    }

    if (path === "/api/tokenizers/tiktoken/models") {
      await route.fulfill({
        status: 200,
        headers: corsHeaders,
        contentType: "application/json",
        body: JSON.stringify({ models: ["gpt-4"] }),
      });
      return;
    }

    if (path === "/api/tokenizers/tiktoken/encode") {
      await route.fulfill({
        status: 200,
        headers: corsHeaders,
        contentType: "application/json",
        body: JSON.stringify({
          tokens: [1, 2],
          decoded_tokens: ["Hi", "!"],
          token_count: 2,
          char_count: 3,
          chars_per_token: 1.5,
        }),
      });
      return;
    }

    await route.fulfill({
      status: 200,
      headers: corsHeaders,
      contentType: "application/json",
      body: JSON.stringify({}),
    });
  });
}

test.beforeEach(async ({ page }) => {
  await mockApi(page);
});

test("pretrain page renders base sections", async ({ page }) => {
  await page.goto("/pretrain");
  await expect(page.getByRole("heading", { name: "Training Data" })).toBeVisible();
  await expect(page.getByRole("button", { name: "GPT-2" })).toBeVisible();
});

test("finetune page loads latest checkpoint", async ({ page }) => {
  await page.goto("/finetune");
  const checkpointSelect = page.locator("section#checkpoint select");
  await expect(checkpointSelect).toHaveValue("ckpt-1");
  await expect(page.getByRole("button", { name: "LoRA" })).toBeVisible();
});

test("inference page loads diagnostics data", async ({ page }) => {
  await page.goto("/inference");
  await expect(page.getByRole("button", { name: "Generate" })).toBeEnabled();
  await expect(page.getByText(/1\. A/)).toBeVisible();
});

test("playground page shows tokenized output", async ({ page }) => {
  await page.goto("/playground");
  await expect(page.getByRole("heading", { name: "Tokenizer" })).toBeVisible();
  await expect(page.getByText("Hi", { exact: true })).toBeVisible();
});
