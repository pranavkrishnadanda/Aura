import { expect, test } from "./fixtures";

/**
 * The drawer is the one genuinely gesture-driven surface in the app, so it is the
 * one place the motion primitives have to hold up against a real pointer.
 *
 * Mobile viewport only: at md and up the rail is a static column.
 */

const WIDTH = 236;

/** Current translateX of the panel, in px. */
async function panelX(page: import("@playwright/test").Page): Promise<number> {
  return page.evaluate(() => {
    const el = document.querySelector("aside") as HTMLElement;
    const m = new DOMMatrixReadOnly(getComputedStyle(el).transform);
    return m.m41;
  });
}

test.describe("drawer gesture", () => {
  test.beforeEach(async ({ page }) => {
    await page.goto("/");
    await page.getByRole("button", { name: "Open menu" }).click();
    // Let the opening spring settle before measuring.
    await expect.poll(() => panelX(page), { timeout: 3000 }).toBeCloseTo(0, 0);
  });

  test("tracks the pointer 1:1 while dragging", async ({ page }) => {
    await page.mouse.move(200, 400);
    await page.mouse.down();
    await page.mouse.move(120, 400, { steps: 8 });

    // Dragged 80px left, so the panel sits ~80px left. Content must move with the
    // finger, not jump when the gesture ends.
    await expect.poll(() => panelX(page)).toBeLessThan(-60);
    await expect.poll(() => panelX(page)).toBeGreaterThan(-100);
    await page.mouse.up();
  });

  test("resists instead of stopping dead when dragged past fully open", async ({ page }) => {
    await page.mouse.move(100, 400);
    await page.mouse.down();
    await page.mouse.move(300, 400, { steps: 10 }); // 200px past the open bound

    const x = await panelX(page);
    // It moves -- a frozen panel reads as a hung interface -- but far less than
    // the 200px dragged.
    expect(x).toBeGreaterThan(0);
    expect(x).toBeLessThan(80);
    await page.mouse.up();
  });

  test("a small drag, held still then released, springs back open", async ({ page }) => {
    await page.mouse.move(200, 400);
    await page.mouse.down();
    await page.mouse.move(180, 400, { steps: 4 }); // only 20px
    // Hold before lifting. This is positioning, not a flick, so it must not be
    // thrown -- velocity has to decay while the finger is stationary.
    await page.waitForTimeout(200);
    await page.mouse.up();

    await expect.poll(() => panelX(page), { timeout: 3000 }).toBeCloseTo(0, 0);
  });

  test("a fast flick closes it even from near fully open", async ({ page }) => {
    // Barely past the open position, but thrown hard. Deciding from the release
    // point alone would keep this open; projecting the momentum closes it, which
    // is what makes a flick feel thrown rather than dragged.
    await page.mouse.move(220, 400);
    await page.mouse.down();
    await page.mouse.move(200, 400, { steps: 1 });
    await page.mouse.move(150, 400, { steps: 1 });
    await page.mouse.move(80, 400, { steps: 1 });
    await page.mouse.up();

    await expect.poll(() => panelX(page), { timeout: 3000 }).toBeLessThan(-WIDTH + 5);
  });

  test("can be caught mid-flight and pulled back", async ({ page }) => {
    // Start it closing, then grab it before it settles. A CSS transition would
    // run to completion first; the spring hands over from its live position.
    await page.mouse.move(200, 400);
    await page.mouse.down();
    await page.mouse.move(60, 400, { steps: 3 });
    await page.mouse.up();

    // Catch it while it is still travelling. The grab must land on the panel,
    // which by now has moved left -- grabbing where it used to be would hit the
    // scrim instead.
    await page.mouse.move(30, 400);
    await page.mouse.down();
    await page.mouse.move(260, 400, { steps: 6 });
    await page.mouse.up();

    await expect.poll(() => panelX(page), { timeout: 3000 }).toBeCloseTo(0, 0);
  });

  test("the scrim dims in step with the panel, not at the end", async ({ page }) => {
    const scrimOpacity = () =>
      page.evaluate(() =>
        Number(getComputedStyle(document.querySelector("[aria-hidden]")! as HTMLElement).opacity)
      );

    expect(await scrimOpacity()).toBeGreaterThan(0.8);

    await page.mouse.move(200, 400);
    await page.mouse.down();
    await page.mouse.move(80, 400, { steps: 8 });

    // Roughly half closed, so roughly half dimmed -- the scrim follows the finger.
    const mid = await scrimOpacity();
    expect(mid).toBeGreaterThan(0.2);
    expect(mid).toBeLessThan(0.8);
    await page.mouse.up();
  });

  test("tapping the scrim closes it", async ({ page }) => {
    await page.mouse.click(340, 400);
    await expect.poll(() => panelX(page), { timeout: 3000 }).toBeLessThan(-WIDTH + 5);
  });
});
