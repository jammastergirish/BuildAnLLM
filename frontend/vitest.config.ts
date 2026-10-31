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
  test: {
    environment: "jsdom",
    setupFiles: ["./test/setup.ts"],
    globals: true,
    clearMocks: true,
    exclude: [...configDefaults.exclude, "e2e/**"],
  },
});
