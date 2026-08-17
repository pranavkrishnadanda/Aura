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
/** Real FastAPI instance for the integrated specs. Runs with LLM_PROVIDER=mock and
 *  an unreachable database, so it needs no API key and no Postgres -- the point is
 *  to prove the two halves agree on the wire, not to exercise the providers. */
const API_PORT = Number(process.env.E2E_API_PORT || 8100);

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
    {
      name: "chromium",
      testIgnore: /(integrated|drawer)\.spec\.ts/,
      use: { ...devices["Desktop Chrome"] },
    },
    // Verifies the layout survives a narrow viewport; this is where responsive
    // breakage shows up first.
    {
      name: "mobile",
      testIgnore: /integrated\.spec\.ts/,
      use: { ...devices["Pixel 5"] },
    },
    // No stubs. Drives the real FastAPI backend through a real browser, so a
    // frontend/backend contract mismatch fails here instead of in production.
    // Every other project intercepts the API, which means they verify the UI
    // behaves given well-formed responses -- not that the two halves agree.
    {
      name: "integrated",
      testMatch: /integrated\.spec\.ts/,
      use: { ...devices["Desktop Chrome"], baseURL: `http://localhost:${PORT}` },
    },
  ],
  webServer: [
    {
      command: `../backend/.venv/bin/python -m uvicorn app.main:app --host 127.0.0.1 --port ${API_PORT}`,
      cwd: "../backend",
      url: `http://127.0.0.1:${API_PORT}/health`,
      reuseExistingServer: !process.env.CI,
      timeout: 60_000,
      env: {
        LLM_PROVIDER: "mock",
        // Unreachable on purpose: the backend falls back to its in-memory store
        // immediately rather than waiting on a connect timeout.
        DATABASE_URL: "postgresql://nobody@127.0.0.1:1/nothing",
        CORS_ORIGINS: `http://localhost:${PORT}`,
        GEMINI_API_KEY: "",
        GROQ_API_KEY: "",
      },
    },
    {
      command: `npm run build && npx next start -p ${PORT}`,
      url: `http://localhost:${PORT}`,
      reuseExistingServer: !process.env.CI,
      timeout: 180_000,
      env: {
        NEXT_PUBLIC_API_URL: process.env.E2E_API_URL || `http://127.0.0.1:${API_PORT}`,
      },
    },
  ],
});
