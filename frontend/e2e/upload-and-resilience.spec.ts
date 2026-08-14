import { test, expect } from "./fixtures";

test.describe("document ingest", () => {
  test("reports real counts from the job, never 'undefined'", async ({ page }) => {
    // Regression: the UI read doc_title/pages/chunks straight off the upload
    // response, which only returns {job_id, status, filename, bytes} -- so every
    // successful upload rendered "undefined - undefined pages - undefined chunks".
    await page.route("**/api/v1/documents/upload", (route) =>
      route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify({ job_id: "job_abc", status: "queued", filename: "protocol.pdf", bytes: 2048 }),
      })
    );
    await page.route("**/api/v1/documents/jobs/*", (route) =>
      route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify({
          job_id: "job_abc", status: "completed", doc_title: "protocol.pdf",
          pages: 12, chunks: 34, stored: 34, embedded: 34,
        }),
      })
    );

    await page.goto("/");
    await page.setInputFiles('input[type="file"]', {
      name: "protocol.pdf",
      mimeType: "application/pdf",
      buffer: Buffer.from("%PDF-1.4 fake"),
    });

    await expect(page.getByText(/protocol\.pdf · 12 pages · 34 chunks/)).toBeVisible();
    await expect(page.getByText(/undefined/)).toHaveCount(0);
  });

  test("surfaces a failed background ingest", async ({ page }) => {
    await page.route("**/api/v1/documents/upload", (route) =>
      route.fulfill({
        status: 200, contentType: "application/json",
        body: JSON.stringify({ job_id: "job_bad", status: "queued", filename: "broken.pdf", bytes: 10 }),
      })
    );
    await page.route("**/api/v1/documents/jobs/*", (route) =>
      route.fulfill({
        status: 200, contentType: "application/json",
        body: JSON.stringify({ job_id: "job_bad", status: "failed", error: "no chunks could be stored" }),
      })
    );

    await page.goto("/");
    await page.setInputFiles('input[type="file"]', {
      name: "broken.pdf", mimeType: "application/pdf", buffer: Buffer.from("%PDF-1.4"),
    });

    await expect(page.getByText(/Ingest failed: no chunks could be stored/)).toBeVisible();
  });

  test("shows the server's message when upload is rejected", async ({ page }) => {
    await page.route("**/api/v1/documents/upload", (route) =>
      route.fulfill({ status: 400, contentType: "application/json", body: JSON.stringify({ detail: "Not a valid PDF" }) })
    );
    await page.goto("/");
    await page.setInputFiles('input[type="file"]', {
      name: "evil.pdf", mimeType: "application/pdf", buffer: Buffer.from("MZnotapdf"),
    });
    await expect(page.getByText(/Not a valid PDF/)).toBeVisible();
  });
});

test.describe("resilience", () => {
  test("survives a backend that is entirely down", async ({ page }) => {
    await page.route("**/api/v1/**", (route) => route.abort("failed"));
    await page.goto("/");

    // The shell still renders rather than throwing on a failed thread fetch.
    await expect(page.getByText("Aura")).toBeVisible();
    await expect(page.getByPlaceholder(/Ask a clinical question/i)).toBeVisible();
  });

  test("does not crash when /threads returns a non-array error body", async ({ page }) => {
    // Regression: the sidebar called .map() on whatever came back.
    await page.route("**/api/v1/threads", (route) =>
      route.fulfill({ status: 500, contentType: "application/json", body: JSON.stringify({ detail: "boom" }) })
    );
    await page.goto("/");
    await expect(page.getByText("Aura")).toBeVisible();
    await expect(page.getByRole("button", { name: "New consultation" })).toBeVisible();
  });

  test("has no console errors on a clean load", async ({ page }) => {
    const errors: string[] = [];
    page.on("console", (m) => m.type() === "error" && errors.push(m.text()));
    page.on("pageerror", (e) => errors.push(e.message));

    await page.goto("/");
    await expect(page.getByText("Start a clinical query")).toBeVisible();

    expect(errors, `console errors on load:\n${errors.join("\n")}`).toEqual([]);
  });
});
