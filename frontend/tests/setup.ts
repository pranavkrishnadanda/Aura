import "@testing-library/jest-dom/vitest";
import { cleanup } from "@testing-library/react";
import { afterEach, beforeEach, vi } from "vitest";

/**
 * Restore a working localStorage.
 *
 * Node >= 22 ships an experimental global `localStorage` that shadows the working
 * one jsdom installs on `window`. Without `--localstorage-file` it is inert, so
 * `window.localStorage.getItem` comes back undefined and anything reading it
 * throws -- which breaks every Chat mount, since Chat assigns the per-browser
 * thread id on mount. Reinstalling a plain in-memory Storage keeps the tests
 * independent of the host Node version.
 *
 * Applied in beforeEach because the afterEach below calls vi.unstubAllGlobals().
 */
function installLocalStorage() {
  let store: Record<string, string> = {};
  const mock: Storage = {
    get length() {
      return Object.keys(store).length;
    },
    key: (i: number) => Object.keys(store)[i] ?? null,
    getItem: (k: string) => (k in store ? store[k] : null),
    setItem: (k: string, v: string) => {
      store[k] = String(v);
    },
    removeItem: (k: string) => {
      delete store[k];
    },
    clear: () => {
      store = {};
    },
  };
  vi.stubGlobal("localStorage", mock);
  Object.defineProperty(window, "localStorage", {
    value: mock,
    configurable: true,
    writable: true,
  });
}

beforeEach(() => {
  installLocalStorage();
  // jsdom does not implement scrollTo; Chat autoscrolls on every message.
  if (!Element.prototype.scrollTo) {
    Element.prototype.scrollTo = vi.fn() as unknown as Element["scrollTo"];
  }
  // jsdom implements neither the Pointer Capture API nor scrollIntoView. The
  // drawer captures the pointer so a drag keeps tracking once it leaves the
  // panel's bounds, and citation markers scroll their evidence into view.
  if (!Element.prototype.setPointerCapture) {
    Element.prototype.setPointerCapture = vi.fn() as unknown as Element["setPointerCapture"];
    Element.prototype.releasePointerCapture =
      vi.fn() as unknown as Element["releasePointerCapture"];
    Element.prototype.hasPointerCapture = (() => false) as unknown as Element["hasPointerCapture"];
  }
  if (!Element.prototype.scrollIntoView) {
    Element.prototype.scrollIntoView = vi.fn() as unknown as Element["scrollIntoView"];
  }
});

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
  // Guarded: a test may have stubbed localStorage, and unstubAllGlobals can leave
  // it briefly absent.
  try {
    window.localStorage?.clear();
  } catch {
    /* not available in this environment */
  }
});

/**
 * Build a ReadableStream that yields the given strings as UTF-8 chunks, so tests
 * can drive streamChat exactly the way a real SSE response does -- including
 * splitting an event across chunk boundaries.
 */
export function sseStream(chunks: string[]): ReadableStream<Uint8Array> {
  const enc = new TextEncoder();
  let i = 0;
  return new ReadableStream({
    pull(controller) {
      if (i < chunks.length) controller.enqueue(enc.encode(chunks[i++]));
      else controller.close();
    },
  });
}

/** A fetch Response carrying an SSE body. */
export function sseResponse(chunks: string[], init: Partial<Response> = {}): Response {
  return {
    ok: true,
    status: 200,
    body: sseStream(chunks),
    headers: new Headers({ "Content-Type": "text/event-stream" }),
    json: async () => ({}),
    ...init,
  } as Response;
}

/** Format one SSE event frame. */
export function frame(event: string, data: unknown): string {
  return `event: ${event}\ndata: ${JSON.stringify(data)}\n\n`;
}
