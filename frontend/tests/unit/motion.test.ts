import { describe, it, expect, vi } from "vitest";
import { project, rubberband, velocityFrom, spring } from "@/lib/motion";

describe("project — momentum landing point", () => {
  it("returns zero for a release with no velocity", () => {
    expect(project(0)).toBe(0);
  });

  it("projects further the harder the flick", () => {
    expect(project(2000)).toBeGreaterThan(project(500));
  });

  it("carries the sign of the gesture", () => {
    expect(project(-800)).toBeLessThan(0);
    expect(project(800)).toBeGreaterThan(0);
  });

  it("uses exponential decay, not the textbook v^2 form", () => {
    // (v/1000) * d / (1 - d), d = 0.998  ->  1000px/s lands ~499px away.
    expect(project(1000)).toBeCloseTo(499, 0);
  });

  it("a lower deceleration rate lands sooner", () => {
    expect(project(1000, 0.99)).toBeLessThan(project(1000, 0.998));
  });
});

describe("rubberband — boundary resistance", () => {
  it("resists rather than hard-stopping, so the surface never reads as frozen", () => {
    const r = rubberband(50, 400);
    expect(r).toBeGreaterThan(0);
    expect(r).toBeLessThan(50);
  });

  it("resists progressively — each extra pixel dragged moves it less", () => {
    const first = rubberband(50, 400);
    const second = rubberband(100, 400) - first;
    expect(second).toBeLessThan(first);
  });

  it("is symmetric about the boundary", () => {
    expect(rubberband(-60, 400)).toBeCloseTo(-rubberband(60, 400), 6);
  });

  it("is zero at the boundary and safe on a zero dimension", () => {
    expect(rubberband(0, 400)).toBe(0);
    expect(rubberband(50, 0)).toBe(0);
  });
});

describe("velocityFrom", () => {
  it("returns zero without enough samples", () => {
    expect(velocityFrom([])).toBe(0);
    expect(velocityFrom([{ pos: 0, t: 0 }])).toBe(0);
  });

  it("computes px/s over the sample window", () => {
    // 100px in 100ms = 1000px/s
    expect(velocityFrom([{ pos: 0, t: 0 }, { pos: 100, t: 100 }])).toBeCloseTo(1000, 0);
  });

  it("ignores samples older than the window, so it reflects the recent gesture", () => {
    const samples = [
      { pos: 0, t: 0 },      // stale: outside the 100ms window
      { pos: 0, t: 900 },
      { pos: 50, t: 1000 },  // 50px in 100ms = 500px/s
    ];
    expect(velocityFrom(samples, 100)).toBeCloseTo(500, 0);
  });

  it("returns zero when the pointer was held still", () => {
    expect(velocityFrom([{ pos: 20, t: 0 }, { pos: 20, t: 100 }])).toBe(0);
  });
});

describe("spring", () => {
  it("settles at the target instantly when reduced motion is requested", () => {
    const frames: number[] = [];
    const onRest = vi.fn();
    spring(0, 100, (v) => frames.push(v), { instant: true }, onRest);
    expect(frames).toEqual([100]);
    expect(onRest).toHaveBeenCalled();
  });

  it("converges to the target and reports rest", async () => {
    const frames: number[] = [];
    await new Promise<void>((done) => {
      spring(0, 100, (v) => frames.push(v), { damping: 1, response: 0.15 }, done);
    });
    expect(frames.at(-1)).toBe(100);
    expect(frames.length).toBeGreaterThan(1);
  });

  it("critical damping does not overshoot", async () => {
    const frames: number[] = [];
    await new Promise<void>((done) => {
      spring(0, 100, (v) => frames.push(v), { damping: 1, response: 0.15 }, done);
    });
    expect(Math.max(...frames)).toBeLessThanOrEqual(100.001);
  });

  it("under-damping overshoots, which is only wanted after a momentum gesture", async () => {
    const frames: number[] = [];
    await new Promise<void>((done) => {
      spring(0, 100, (v) => frames.push(v), { damping: 0.5, response: 0.2 }, done);
    });
    expect(Math.max(...frames)).toBeGreaterThan(100);
  });

  it("honours initial velocity, so a release has no seam between drag and animation", async () => {
    const withKick: number[] = [];
    const without: number[] = [];
    await new Promise<void>((d) => spring(0, 100, (v) => withKick.push(v), { velocity: 800, response: 0.3 }, d));
    await new Promise<void>((d) => spring(0, 100, (v) => without.push(v), { velocity: 0, response: 0.3 }, d));
    // A spring launched with the finger's speed covers ground sooner.
    expect(withKick[1]).toBeGreaterThan(without[1]);
  });

  it("stop() halts it and exposes the live velocity for a clean hand-off", async () => {
    let handle!: ReturnType<typeof spring>;
    const frames: number[] = [];
    await new Promise<void>((done) => {
      handle = spring(0, 500, (v) => {
        frames.push(v);
        if (frames.length === 3) {
          handle.stop();
          done();
        }
      }, { response: 0.5 });
    });
    const atStop = frames.length;
    // Velocity is non-zero mid-flight; a re-target seeded with it avoids the
    // discontinuity that makes a reversal feel like hitting a wall.
    expect(Math.abs(handle.velocity())).toBeGreaterThan(0);
    await new Promise((r) => setTimeout(r, 60));
    expect(frames.length).toBe(atStop);
  });

  it("can be interrupted and re-targeted from its live value", async () => {
    let current = 0;
    let handle!: ReturnType<typeof spring>;
    await new Promise<void>((done) => {
      handle = spring(0, 500, (v) => {
        current = v;
        if (v > 50) {
          handle.stop();
          // Reverse from where it actually is, carrying its velocity.
          spring(current, 0, (v2) => { current = v2; }, { velocity: handle.velocity() }, done);
        }
      }, { response: 0.4 });
    });
    expect(current).toBe(0);
  });
});
