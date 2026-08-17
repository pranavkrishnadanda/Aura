/**
 * Motion primitives: momentum projection, boundary resistance, and an
 * interruptible spring.
 *
 * These exist because CSS transitions cannot be grabbed mid-flight. A transition
 * runs to completion on a fixed schedule from a fixed start value, so a user who
 * catches a closing drawer has to wait for it to finish before it responds. A
 * spring integrates from wherever the value currently is, at whatever velocity it
 * currently has, which is what makes interruption and reversal continuous.
 */

/** Where a flick would come to rest, given its release velocity.
 *
 * Snapping from the release *point* ignores how hard the user threw it, so a fast
 * flick and a slow drag ending in the same place behave identically. Projecting
 * forward is what makes a flick feel thrown.
 *
 * This is exponential scroll deceleration, not the textbook v^2/(2a): matching
 * the platform's own scroll feel matters more than physical purity.
 *
 * @param velocity px/s at release
 * @param decelerationRate 0.998 for normal scroll feel, 0.99 for snappier
 */
export function project(velocity: number, decelerationRate = 0.998): number {
  return ((velocity / 1000) * decelerationRate) / (1 - decelerationRate);
}

/** Progressive resistance past a boundary.
 *
 * A hard stop reads as frozen -- indistinguishable from a hung interface. Motion
 * that continues but resists reads as responsive with nothing further to reveal.
 *
 * @param overshoot how far past the bound the pointer has travelled
 * @param dimension the size of the dragged surface
 */
export function rubberband(overshoot: number, dimension: number, constant = 0.55): number {
  if (dimension <= 0) return 0;
  return (overshoot * dimension * constant) / (dimension + constant * Math.abs(overshoot));
}

export type SpringOptions = {
  /** 1 = critically damped (no overshoot). Below 1 overshoots and settles back. */
  damping?: number;
  /** Seconds to approach the target. Not a duration -- a spring has no fixed end. */
  response?: number;
  /** Initial velocity in px/s. Hand the release velocity here so there is no
   *  visible seam between dragging and animating. */
  velocity?: number;
  /** Skip the animation and settle immediately (reduced-motion). */
  instant?: boolean;
};

export type SpringHandle = { stop: () => void; velocity: () => number };

/**
 * Animate `from` -> `to`, calling `onFrame` each tick.
 *
 * Interruption is the caller's job and is trivial: call stop(), read the live
 * value and velocity, and start a new spring from them. Because the new spring
 * inherits the old velocity there is no discontinuity at the reversal -- hard
 * cutting velocity is what produces the "brick wall" feel.
 */
export function spring(
  from: number,
  to: number,
  onFrame: (value: number) => void,
  { damping = 1, response = 0.4, velocity = 0, instant = false }: SpringOptions = {},
  onRest?: () => void
): SpringHandle {
  let v = velocity;
  let x = from;

  if (instant || response <= 0) {
    onFrame(to);
    onRest?.();
    return { stop: () => {}, velocity: () => 0 };
  }

  const omega = (2 * Math.PI) / response;
  const k = omega * omega;
  const c = 2 * damping * omega;

  let raf = 0;
  let last = 0;
  let stopped = false;

  const step = (now: number) => {
    if (stopped) return;
    if (!last) last = now;
    // Clamp dt so a backgrounded tab does not integrate one enormous step and
    // fling the value somewhere absurd on return.
    const dt = Math.min((now - last) / 1000, 1 / 30);
    last = now;

    // Sub-step for stability at high stiffness; semi-implicit Euler.
    const steps = Math.max(1, Math.ceil(dt / (1 / 240)));
    const h = dt / steps;
    for (let i = 0; i < steps; i++) {
      const a = -k * (x - to) - c * v;
      v += a * h;
      x += v * h;
    }

    // Settled: close in both position and velocity, or it visibly creeps.
    if (Math.abs(x - to) < 0.05 && Math.abs(v) < 0.05) {
      x = to;
      v = 0;
      onFrame(x);
      stopped = true;
      onRest?.();
      return;
    }
    onFrame(x);
    raf = requestAnimationFrame(step);
  };

  raf = requestAnimationFrame(step);
  return {
    stop: () => {
      stopped = true;
      cancelAnimationFrame(raf);
    },
    velocity: () => v,
  };
}

/** Velocity in px/s from a short history of pointer samples.
 *
 * A single last-two-points delta is dominated by whatever jitter landed in the
 * final frame; a short window over recent samples is stable without lagging.
 */
export function velocityFrom(samples: { pos: number; t: number }[], window = 100): number {
  if (samples.length < 2) return 0;
  const newest = samples[samples.length - 1];
  let oldest = samples[0];
  for (let i = samples.length - 1; i >= 0; i--) {
    if (newest.t - samples[i].t > window) break;
    oldest = samples[i];
  }
  const dt = newest.t - oldest.t;
  if (dt <= 0) return 0;
  return ((newest.pos - oldest.pos) / dt) * 1000;
}

/** True when the user has asked for reduced motion. */
export function prefersReducedMotion(): boolean {
  return typeof window !== "undefined" &&
    !!window.matchMedia?.("(prefers-reduced-motion: reduce)").matches;
}
