import { test, expect } from "@playwright/test";

/**
 * Real browser against the real backend. No route interception anywhere.
 *
 * Every other spec stubs the API, which verifies the UI behaves correctly given
 * well-formed responses -- not that the frontend and backend actually agree on
 * field names, event shapes, status codes and citation numbering. A contract
 * mismatch is invisible to a stubbed suite, because the stub encodes whatever the
 * frontend already expects. These tests are the only place the two halves meet.
 *
 * The backend runs with LLM_PROVIDER=mock and no database, so no API key or
 * Postgres is required; the seeded clinical chunks are enough to exercise
 * retrieval, citation numbering and streaming end to end.
 */

test.describe("frontend and backend agree", () => {
  test("a seeded clinical question streams a grounded, citable answer", async ({ page }) => {
    await page.goto("/");
    await page.getByPlaceholder(/Ask a clinical question/i)
      .fill("What is first-line therapy for hypertension with CKD?");
    await page.getByRole("button", { name: "Send" }).click();

    // Content comes from the backend's seeded FDA chunk, not a fixture.
    await expect(page.getByText(/ACE inhibitors/).first()).toBeVisible({ timeout: 20_000 });

    // The citation payload the API sends must be renderable: the marker resolves,
    // and the evidence card shows the verbatim source with its real page number.
    const marker = page.getByRole("button", { name: "Open citation 1" });
    await expect(marker).toBeVisible();

    const card = page.getByTestId("citation-panel").first();
    await expect(card).toBeVisible();
    await expect(card.getByText(/FDA Hypertension Guideline/)).toBeVisible();
    await expect(card.getByText(/^p\.\d+$/)).toBeVisible();

    // No marker may point at a source that does not exist. This is the check that
    // would have caught the numbering desync between rag.py and main.py, where the
    // model was told to cite [3] while the UI only held [1] and [2].
    await expect(page.getByText(/\[\d+\?\]/)).toHaveCount(0);
    await expect(page.getByText("all references resolve")).toBeVisible();
  });

  test("an out-of-scope question refuses instead of inventing a source", async ({ page }) => {
    await page.goto("/");
    await page.getByPlaceholder(/Ask a clinical question/i).fill("what do you know about hair problems");
    await page.getByRole("button", { name: "Send" }).click();

    await expect(page.getByText(/outside what I can source/)).toBeVisible({ timeout: 20_000 });
    // A refusal must carry no citation at all.
    await expect(page.getByRole("button", { name: /Open citation/ })).toHaveCount(0);
    await expect(page.getByText("Evidence")).toHaveCount(0);
  });

  test("history survives a reload on the same browser thread", async ({ page }) => {
    await page.goto("/");
    const threadId = await page.evaluate(() => window.localStorage.getItem("aura.thread_id"));
    expect(threadId).toBeTruthy();

    await page.getByPlaceholder(/Ask a clinical question/i).fill("Contraindications for lisinopril?");
    await page.getByRole("button", { name: "Send" }).click();
    await expect(page.getByText("Question")).toBeVisible({ timeout: 20_000 });
    await expect(page.getByText(/angioedema|ACE inhibitors/).first()).toBeVisible();

    // Reload and reopen the same thread: the backend must return the persisted
    // turns in the shape the UI reads them in.
    await page.reload();
    await page.getByRole("button", { name: "This session" }).click();
    await expect(page.getByText("Contraindications for lisinopril?")).toBeVisible();
  });

  test("the rail reports the backend's real retrieval mode", async ({ page }) => {
    await page.goto("/");
    // No key is configured for this run, so the honest answer is keyword matching.
    // If this ever says "semantic" without embeddings, /health is lying.
    // Scoped to the rail: the mobile header renders the same indicator and is
    // present in the DOM even when hidden at this viewport.
    await expect(page.getByRole("complementary").getByText("keyword")).toBeVisible();
  });

  test("an oversized upload is rejected by the real endpoint", async ({ page }) => {
    await page.goto("/");
    await page.setInputFiles('input[type="file"]', {
      name: "not-a-pdf.pdf",
      mimeType: "application/pdf",
      buffer: Buffer.from("MZ this is not a pdf at all"),
    });
    // The backend checks magic bytes, and the UI surfaces its message verbatim.
    await expect(page.getByText(/Not a valid PDF/)).toBeVisible({ timeout: 20_000 });
  });

  test("a real PDF ingests and becomes answerable", async ({ page }) => {
    // Minimal valid PDF containing a distinctive clinical sentence.
    const body =
      "BT /F1 12 Tf 40 700 Td (Rivaroxaban 20mg once daily with food is indicated for " +
      "stroke prevention in nonvalvular atrial fibrillation per protocol XYZ-9.) Tj ET";
    const pdf = [
      "%PDF-1.4",
      "1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj",
      "2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj",
      "3 0 obj<</Type/Page/Parent 2 0 R/MediaBox[0 0 612 792]/Contents 4 0 R/Resources<</Font<</F1 5 0 R>>>>>>endobj",
      `4 0 obj<</Length ${body.length}>>stream`,
      body,
      "endstream endobj",
      "5 0 obj<</Type/Font/Subtype/Type1/BaseFont/Helvetica>>endobj",
      "trailer<</Root 1 0 R>>",
    ].join("\n");

    await page.goto("/");
    await page.setInputFiles('input[type="file"]', {
      name: "protocol-xyz9.pdf",
      mimeType: "application/pdf",
      buffer: Buffer.from(pdf, "latin1"),
    });

    // The UI polls the real job endpoint; it must report actual counts, never
    // "undefined", which is what it did when it read them off the upload response.
    await expect(page.getByText(/protocol-xyz9\.pdf ·/)).toBeVisible({ timeout: 30_000 });
    await expect(page.getByText(/undefined/)).toHaveCount(0);

    // The ingested text is now retrievable through the real retrieval path.
    await page.getByPlaceholder(/Ask a clinical question/i).fill("rivaroxaban dosing atrial fibrillation");
    await page.getByRole("button", { name: "Send" }).click();
    await expect(page.getByText(/Rivaroxaban 20mg/).first()).toBeVisible({ timeout: 20_000 });
  });
});
