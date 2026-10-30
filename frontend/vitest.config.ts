import { fileURLToPath } from "node:url";
import { defineConfig, configDefaults } from "vitest/config";

const plausibleTrackerStub = fileURLToPath(
  new URL("./test/stubs/plausible-tracker.ts", import.meta.url)
);

export default defineConfig({
  resolve: {
    alias: {
      "@plausible-analytics/tracker": plausibleTrackerStub,
    },
  },
  // Vite 8 compiles with oxc instead of esbuild. tsconfig says jsx "preserve"
  // for Next.js, so tests must ask for the automatic JSX runtime explicitly.
  oxc: {
    jsx: { runtime: "automatic" },
  },
  test: {
    environment: "jsdom",
    setupFiles: ["./test/setup.ts"],
    globals: true,
    clearMocks: true,
    exclude: [...configDefaults.exclude, "e2e/**"],
  },
});
