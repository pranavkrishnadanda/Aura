import { test, expect, groundedStream, SEED_CITATION } from "./fixtures";

test.describe("clinical chat", () => {
  test("streams a grounded answer and opens the evidence panel", async ({ page }) => {
    await page.goto("/");

    // Empty state offers the canned clinical prompts.
    await expect(page.getByText("Start a clinical query")).toBeVisible();
    await page.getByRole("button", { name: "First-line therapy for hypertension with CKD?" }).click();

    const input = page.getByPlaceholder(/Ask a clinical question/i);
    await expect(input).toHaveValue("First-line therapy for hypertension with CKD?");

    await page.getByRole("button", { name: "Send" }).click();

    // The answer streams in and is attributed.
    await expect(page.getByText(/ACE inhibitors/)).toBeVisible();
    await expect(page.getByText("You")).toBeVisible();

    // The inline [1] marker is clickable and opens the verbatim source.
    await page.getByRole("button", { name: "Open citation 1" }).click();
    await expect(page.getByText("Evidence")).toBeVisible();
    await expect(page.getByText(SEED_CITATION.doc_title)).toBeVisible();
    await expect(page.getByText(`p.${SEED_CITATION.page}`).first()).toBeVisible();
    await expect(page.getByText("Verified chunk — rendered verbatim. No LLM rewrite.")).toBeVisible();

    await page.getByRole("button", { name: "Close" }).click();
    await expect(page.getByText("Evidence")).toHaveCount(0);
  });

  test("Enter sends, Shift+Enter does not", async ({ page }) => {
    await page.goto("/");
    const input = page.getByPlaceholder(/Ask a clinical question/i);

    await input.fill("Contraindications for lisinopril?");
    await input.press("Shift+Enter");
    await expect(page.getByText(/ACE inhibitors/)).toHaveCount(0);

    await input.press("Enter");
    await expect(page.getByText(/ACE inhibitors/)).toBeVisible();
  });

  test("composer is re-enabled after a mid-stream error", async ({ page }) => {
    // Regression: `streaming` was only cleared inside onDone/onError, so any
    // throw left the Send button disabled for the rest of the session.
    await page.route("**/api/v1/chat/stream", (route) => route.abort("failed"));
    await page.goto("/");

    await page.getByPlaceholder(/Ask a clinical question/i).fill("does this recover?");
    await page.getByRole("button", { name: "Send" }).click();

    await expect(page.getByText(/Error:/)).toBeVisible();
    await expect(page.getByRole("button", { name: "Send" })).toBeEnabled();
  });

  test("each browser profile gets its own thread, not a shared one", async ({ page }) => {
    // Regression: threadId was the literal "default" for every visitor, so all
    // users shared one conversation.
    await page.goto("/");
    const id = await page.evaluate(() => window.localStorage.getItem("aura.thread_id"));
    expect(id).toBeTruthy();
    expect(id).not.toBe("default");
  });

  test("a refusal renders without fabricated citations", async ({ page }) => {
    await page.route("**/api/v1/chat/stream", (route) =>
      route.fulfill({
        status: 200,
        headers: { "Content-Type": "text/event-stream" },
        body: groundedStream(
          ["That's outside my current clinical intelligence scope — I can only cite verified guidelines."],
          []
        ),
      })
    );
    await page.goto("/");
    await page.getByPlaceholder(/Ask a clinical question/i).fill("what do you know about hair problems");
    await page.getByRole("button", { name: "Send" }).click();

    await expect(page.getByText(/outside my current clinical intelligence scope/)).toBeVisible();
    await expect(page.getByRole("button", { name: /Open citation/ })).toHaveCount(0);
  });

  test("renders untrusted chunk text as text, never as markup", async ({ page }) => {
    const hostile = '<img src=x onerror="window.__pwned=1"> <script>window.__pwned=1</script>';
    await page.route("**/api/v1/chat/stream", (route) =>
      route.fulfill({
        status: 200,
        headers: { "Content-Type": "text/event-stream" },
        body: groundedStream(["See source [1]"], [{ ...SEED_CITATION, chunk_text: hostile }]),
      })
    );
    await page.goto("/");
    await page.getByPlaceholder(/Ask a clinical question/i).fill("show me the source");
    await page.getByRole("button", { name: "Send" }).click();
    await page.getByRole("button", { name: "Open citation 1" }).click();

    // Rendered verbatim as text, and no injected node executed.
    await expect(page.getByText(hostile)).toBeVisible();
    expect(await page.evaluate(() => (window as any).__pwned)).toBeUndefined();
    expect(await page.locator("script:not([src])").filter({ hasText: "__pwned" }).count()).toBe(0);
  });
});
