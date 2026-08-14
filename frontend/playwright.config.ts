import { defineConfig, devices } from "@playwright/test";

/**
 * Real-browser E2E.
 *
 * Requires browser binaries: `npx playwright install chromium`.
 * By default this boots ONLY the frontend and stubs the API at the network layer
 * (see e2e/fixtures.ts), so the suite is hermetic and needs no database or LLM
 * key. Point E2E_API_URL at a running backend to exercise the real stack instead.
 */
const PORT = Number(process.env.E2E_PORT || 3100);

export default defineConfig({
  testDir: "./e2e",
  fullyParallel: true,
  forbidOnly: !!process.env.CI,
  retries: process.env.CI ? 2 : 0,
  workers: process.env.CI ? 1 : undefined,
  reporter: process.env.CI ? "list" : [["list"], ["html", { open: "never" }]],
  timeout: 30_000,
  expect: { timeout: 10_000 },
  use: {
    baseURL: `http://localhost:${PORT}`,
    trace: "on-first-retry",
    screenshot: "only-on-failure",
    video: "retain-on-failure",
  },
  projects: [
    { name: "chromium", use: { ...devices["Desktop Chrome"] } },
    // Verifies the layout survives a narrow viewport; the app is a 3-pane desktop
    // shell, so this is where responsive breakage shows up first.
    { name: "mobile", use: { ...devices["Pixel 5"] } },
  ],
  webServer: {
    command: `npm run build && npx next start -p ${PORT}`,
    url: `http://localhost:${PORT}`,
    reuseExistingServer: !process.env.CI,
    timeout: 180_000,
    env: {
      NEXT_PUBLIC_API_URL: process.env.E2E_API_URL || "http://localhost:8000",
    },
  },
});
