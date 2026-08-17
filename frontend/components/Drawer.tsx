"use client";
import { useCallback, useEffect, useRef } from "react";
import {
  prefersReducedMotion,
  project,
  rubberband,
  type SpringHandle,
  spring,
  velocityFrom,
} from "@/lib/motion";

const WIDTH = 236;

/**
 * The mobile navigation drawer.
 *
 * Deliberately not a CSS transition. A transition runs on a fixed schedule from a
 * fixed start value, so a user who grabs the drawer while it is closing has to
 * wait for it to finish before it responds -- and if they reverse, the motion
 * hard-cuts. Here the pointer moves the panel 1:1, release projects the flick
 * forward to decide open or closed, and the spring launches at the finger's exact
 * speed so there is no seam between dragging and animating. Grabbing it mid-flight
 * stops the spring and continues from wherever it actually is.
 */
export default function Drawer({
  open,
  onOpenChange,
  children,
}: {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  children: React.ReactNode;
}) {
  const panel = useRef<HTMLElement | null>(null);
  const scrim = useRef<HTMLDivElement | null>(null);
  const anim = useRef<SpringHandle | null>(null);
  const x = useRef(open ? 0 : -WIDTH);
  const drag = useRef<{
    startX: number;
    grabbed: number;
    samples: { pos: number; t: number }[];
  } | null>(null);

  const paint = useCallback((value: number) => {
    x.current = value;
    if (panel.current) panel.current.style.transform = `translate3d(${value}px,0,0)`;
    if (scrim.current) {
      // The scrim tracks the panel, so the dimming follows the finger rather than
      // switching state at the end of the gesture.
      const p = 1 - Math.abs(value) / WIDTH;
      scrim.current.style.opacity = String(Math.max(0, Math.min(1, p)));
      scrim.current.style.pointerEvents = p > 0.1 ? "auto" : "none";
    }
  }, []);

  const settle = useCallback(
    (toOpen: boolean, velocity = 0) => {
      anim.current?.stop();
      anim.current = spring(
        x.current,
        toOpen ? 0 : -WIDTH,
        paint,
        {
          // Critically damped: this is a panel arriving, not an object thrown.
          // Overshoot here would read as decorative rather than physical.
          damping: 1,
          response: 0.35,
          velocity,
          instant: prefersReducedMotion(),
        },
        () => onOpenChange(toOpen)
      );
    },
    [paint, onOpenChange]
  );

  // Follow external open/close (the Menu button) unless a finger is on it.
  useEffect(() => {
    if (drag.current) return;
    const target = open ? 0 : -WIDTH;
    if (Math.abs(x.current - target) < 0.5) return;
    settle(open, anim.current?.velocity() ?? 0);
  }, [open, settle]);

  useEffect(() => {
    paint(open ? 0 : -WIDTH);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [paint, open]);

  const onPointerDown = (e: React.PointerEvent) => {
    if (e.pointerType === "mouse" && e.button !== 0) return;
    // Interrupt: take over from wherever the spring currently is, not from the
    // value it was heading toward.
    anim.current?.stop();
    (e.currentTarget as HTMLElement).setPointerCapture(e.pointerId);
    drag.current = {
      startX: e.clientX,
      grabbed: x.current,
      samples: [{ pos: e.clientX, t: performance.now() }],
    };
  };

  const onPointerMove = (e: React.PointerEvent) => {
    const d = drag.current;
    if (!d) return;
    d.samples.push({ pos: e.clientX, t: performance.now() });
    if (d.samples.length > 12) d.samples.shift();

    let next = d.grabbed + (e.clientX - d.startX);
    // Past fully-open, resist instead of stopping dead.
    if (next > 0) next = rubberband(next, WIDTH);
    if (next < -WIDTH) next = -WIDTH + rubberband(next + WIDTH, WIDTH);
    paint(next);
  };

  const endDrag = (e: React.PointerEvent) => {
    const d = drag.current;
    if (!d) return;
    drag.current = null;
    // Record where the pointer actually was at release. Without this, a user who
    // drags, holds still, then lifts gets the velocity of the last movement --
    // so a deliberate positioning gesture is treated as a flick and thrown.
    // Holding still must decay to zero.
    d.samples.push({ pos: e.clientX, t: performance.now() });
    const v = velocityFrom(d.samples);
    // Decide from where the flick would land, not from where the finger stopped:
    // a fast flick and a slow drag ending at the same point should not behave
    // identically.
    const landing = x.current + project(v);
    settle(landing > -WIDTH / 2, v);
  };

  return (
    <>
      <div
        ref={scrim}
        aria-hidden
        onClick={() => settle(false)}
        className="fixed inset-0 z-30 bg-black/25 md:hidden"
        style={{ opacity: 0, pointerEvents: "none" }}
      />
      <aside
        ref={panel as any}
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onPointerUp={endDrag}
        onPointerCancel={endDrag}
        className="fixed inset-y-0 left-0 z-40 flex w-[236px] shrink-0 touch-pan-y flex-col border-r md:static md:!transform-none"
        style={{
          borderColor: "var(--rule)",
          background: "var(--chrome)",
          backdropFilter: "var(--chrome-blur)",
        }}
      >
        {children}
      </aside>
    </>
  );
}
