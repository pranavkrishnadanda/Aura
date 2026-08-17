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

/** Wait until the panel stops moving.
 *
 * Polling for a value can succeed on a transient read while the spring is still
 * travelling, which made every measurement downstream race the animation. Two
 * identical consecutive samples mean it is genuinely at rest.
 */
async function settled(page: import("@playwright/test").Page): Promise<number> {
  let last = Number.NaN;
  await expect
    .poll(
      async () => {
        const x = await panelX(page);
        const stable = x === last;
        last = x;
        return stable;
      },
      { timeout: 5000, intervals: [50] }
    )
    .toBe(true);
  return last;
}

test.describe("drawer gesture", () => {
  test.beforeEach(async ({ page }) => {
    await page.goto("/");
    await page.getByRole("button", { name: "Open menu" }).click();
    expect(await settled(page)).toBeCloseTo(0, 0);
  });

  test("tracks the pointer 1:1 while dragging", async ({ page }) => {
    await page.mouse.move(200, 400);
    await page.mouse.down();
    await page.mouse.move(120, 400, { steps: 8 });

    // Dragged 80px left, so the panel sits ~80px left. Content must move with the
    // finger, not jump when the gesture ends.
    await expect.poll(() => panelX(page)).toBeLessThan(-60);
    expect(await panelX(page)).toBeGreaterThan(-100);
    await page.mouse.up();
  });

  test("resists instead of stopping dead when dragged past fully open", async ({ page }) => {
    await page.mouse.move(100, 400);
    await page.mouse.down();
    await page.mouse.move(300, 400, { steps: 10 }); // 200px past the open bound

    // Poll rather than read once: the transform is applied through React's event
    // handling, so a single read can land before the move has been painted.
    // It moves -- a frozen panel reads as a hung interface -- but far less than
    // the 200px dragged.
    await expect.poll(() => panelX(page)).toBeGreaterThan(0);
    expect(await panelX(page)).toBeLessThan(80);
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

    expect(await settled(page)).toBeCloseTo(0, 0);
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

    expect(await settled(page)).toBeLessThan(-WIDTH + 5);
  });

  test("can be caught mid-flight and pulled back", async ({ page }) => {
    // Start it closing from fully open with no flick, so the spring has the whole
    // width to travel and the catch window is as wide as the interaction allows.
    // A hard flick closes in a fraction of that, which made this race the animation
    // under parallel load rather than testing interruption.
    await page.mouse.click(340, 400); // scrim: closes from x=0, zero velocity

    // Catch it while it is still travelling. The grab must land on the panel,
    // which by now has moved left -- grabbing where it used to be would hit the
    // scrim instead.
    await page.mouse.move(30, 400);
    await page.mouse.down();
    await page.mouse.move(260, 400, { steps: 6 });
    await page.mouse.up();

    expect(await settled(page)).toBeCloseTo(0, 0);
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
    await expect.poll(scrimOpacity).toBeLessThan(0.8);
    expect(await scrimOpacity()).toBeGreaterThan(0.2);
    await page.mouse.up();
  });

  test("tapping the scrim closes it", async ({ page }) => {
    await page.mouse.click(340, 400);
    expect(await settled(page)).toBeLessThan(-WIDTH + 5);
  });
});
