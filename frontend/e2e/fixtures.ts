import { test as base, expect, type Page } from "@playwright/test";

/**
 * Network stubs for hermetic E2E.
 *
 * The suite runs against a real browser and a real Next.js build, but intercepts
 * the backend so it needs no Postgres, no LLM key, and no quota. Set
 * E2E_LIVE_API=1 to skip the stubs and hit whatever NEXT_PUBLIC_API_URL points at.
 */
const LIVE = !!process.env.E2E_LIVE_API;

export const SEED_CITATION = {
  id: "chk_001",
  doc_id: "doc_fda_2024",
  doc_title: "FDA Hypertension Guideline 2024",
  page: 12,
  chunk_text:
    "For adults with hypertension and chronic kidney disease, first-line therapy includes ACE inhibitors (e.g., lisinopril 10mg daily) or ARBs.",
  score: 0.91,
  idx: 1,
};

/** Encode one SSE frame. */
function frame(event: string, data: unknown): string {
  return `event: ${event}\ndata: ${JSON.stringify(data)}\n\n`;
}

/** Build a complete SSE body for a grounded, cited answer. */
export function groundedStream(tokens: string[], citations = [SEED_CITATION]): string {
  const full = tokens.join("");
  return (
    frame("meta", { citations, is_refusal: false, thread_id: "e2e" }) +
    tokens.map((t) => frame("token", { token: t })).join("") +
    frame("done", { full_text: full, citations }) +
    frame("heartbeat", { ts: 1 })
  );
}

export async function stubApi(
  page: Page,
  opts: { chat?: string; threads?: unknown[]; health?: Record<string, unknown> } = {}
) {
  if (LIVE) return;

  // The rail reports the live retrieval mode, so /health is part of the backend
  // surface the suite must stand in for. Leaving it unstubbed let a real request
  // escape to a backend that is not running, which surfaces as a console error.
  await page.route("**/health", (route) =>
    route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify(
        opts.health ?? {
          status: "ok",
          retrieval_mode: "pgvector",
          threshold: 0.85,
          storage_mode: "postgres",
          max_pdf_mb: 50,
        }
      ),
    })
  );

  await page.route("**/api/v1/threads", async (route) => {
    if (route.request().method() === "POST") {
      return route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify({
          id: "thr_new1",
          title: "New consultation",
          created_at: "",
          user_id: "anonymous",
        }),
      });
    }
    return route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify(opts.threads ?? []),
    });
  });

  await page.route("**/api/v1/threads/*/messages", (route) =>
    route.fulfill({ status: 200, contentType: "application/json", body: "[]" })
  );

  await page.route("**/api/v1/chat/stream", (route) =>
    route.fulfill({
      status: 200,
      headers: { "Content-Type": "text/event-stream", "Cache-Control": "no-cache" },
      body:
        opts.chat ??
        groundedStream([
          "For adults with hypertension and chronic kidney disease, ",
          "first-line therapy includes ACE inhibitors ",
          "[1]",
        ]),
    })
  );
}

export const test = base.extend<{ stubbed: undefined }>({
  stubbed: [
    async ({ page }, use) => {
      await stubApi(page);
      // The fixture carries no value; it exists for its setup side-effect.
      await use(undefined);
    },
    { auto: true },
  ],
});

export { expect };
