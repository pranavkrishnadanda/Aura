import { describe, expect, it, vi } from "vitest";
import type { StreamCallbacks } from "@/lib/api";
import { authHeaders, streamChat } from "@/lib/api";
import { frame, sseResponse, sseStream } from "../setup";

/** Build a fresh set of spy callbacks for one streamChat() call. */
function makeCbs(): StreamCallbacks {
  return {
    onMeta: vi.fn(),
    onToken: vi.fn(),
    onDone: vi.fn(),
    onError: vi.fn(),
  };
}

describe("streamChat", () => {
  it("parses meta/token/done frames and invokes the right callbacks in order", async () => {
    const order: string[] = [];
    const cbs: StreamCallbacks = {
      onMeta: vi.fn((c) => order.push(`meta:${JSON.stringify(c)}`)),
      onToken: vi.fn((t) => order.push(`token:${t}`)),
      onDone: vi.fn((f) => order.push(`done:${f}`)),
      onError: vi.fn(() => order.push("error")),
    };
    vi.stubGlobal(
      "fetch",
      vi
        .fn()
        .mockResolvedValue(
          sseResponse([
            frame("meta", { citations: [{ idx: 1 }], is_refusal: false }),
            frame("token", { token: "Hello" }),
            frame("token", { token: " world" }),
            frame("done", { full_text: "Hello world" }),
          ])
        )
    );

    await streamChat("hi", "t1", cbs);

    expect(order).toEqual(['meta:[{"idx":1}]', "token:Hello", "token: world", "done:Hello world"]);
    expect(cbs.onMeta).toHaveBeenCalledWith([{ idx: 1 }], false);
    expect(cbs.onError).not.toHaveBeenCalled();
  });

  it("reassembles an SSE event split across two network chunks", async () => {
    // Regression guard: the parser buffers partial input and only processes
    // complete "\n\n"-delimited events, so a frame cut mid-way across two
    // reads must still be parsed as one event.
    const full = frame("token", { token: "hello" });
    const cut = full.indexOf("hello") + 2; // slice inside the JSON payload
    const cbs = makeCbs();
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue({
        ok: true,
        status: 200,
        body: sseStream([full.slice(0, cut), full.slice(cut)]),
        headers: new Headers({ "Content-Type": "text/event-stream" }),
        json: async () => ({}),
      } as Response)
    );

    await streamChat("hi", "t1", cbs);

    expect(cbs.onToken).toHaveBeenCalledTimes(1);
    expect(cbs.onToken).toHaveBeenCalledWith("hello");
  });

  it("does not corrupt a multi-byte UTF-8 character split across chunk boundaries", async () => {
    const enc = new TextEncoder();
    const payload = JSON.stringify({ token: "🚀ship" });
    const evt = `event: token\ndata: ${payload}\n\n`;
    const emojiIndex = evt.indexOf("🚀");
    const prefixBytes = enc.encode(evt.slice(0, emojiIndex)).length;
    const bytes = enc.encode(evt);
    // Split two bytes into the emoji's 4-byte UTF-8 sequence.
    const splitPoint = prefixBytes + 2;
    const chunk1 = bytes.slice(0, splitPoint);
    const chunk2 = bytes.slice(splitPoint);

    const parts = [chunk1, chunk2];
    let i = 0;
    const body = new ReadableStream<Uint8Array>({
      pull(controller) {
        if (i < parts.length) controller.enqueue(parts[i++]);
        else controller.close();
      },
    });

    const cbs = makeCbs();
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue({
        ok: true,
        status: 200,
        body,
        headers: new Headers({ "Content-Type": "text/event-stream" }),
        json: async () => ({}),
      } as Response)
    );

    await streamChat("hi", "t1", cbs);

    expect(cbs.onToken).toHaveBeenCalledWith("🚀ship");
  });

  it("concatenates repeated data: lines within one event per the SSE spec", async () => {
    // Regression: the parser used to keep only the last "data:" line, which
    // silently dropped/garbled multi-line payloads instead of joining them.
    const raw = 'event: token\ndata: {"token":\ndata: "AB"}\n\n';
    const cbs = makeCbs();
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(sseResponse([raw])));

    await streamChat("hi", "t1", cbs);

    expect(cbs.onToken).toHaveBeenCalledWith("AB");
  });

  it("invokes onError when an error event is received", async () => {
    // Regression: "error" events were previously ignored entirely, leaving
    // the UI stuck showing "streaming..." forever.
    const cbs = makeCbs();
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(sseResponse([frame("error", { detail: "generation failed" })]))
    );

    await streamChat("hi", "t1", cbs);

    expect(cbs.onError).toHaveBeenCalledWith("generation failed");
    expect(cbs.onDone).not.toHaveBeenCalled();
  });

  it("invokes onError with the server's detail on a non-ok HTTP response", async () => {
    const cbs = makeCbs();
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue({
        ok: false,
        status: 429,
        body: null,
        headers: new Headers(),
        json: async () => ({ detail: "Rate limited" }),
      } as unknown as Response)
    );

    await streamChat("hi", "t1", cbs);

    expect(cbs.onError).toHaveBeenCalledWith("Rate limited");
  });

  it("invokes onError with the HTTP status when no detail is present", async () => {
    const cbs = makeCbs();
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue({
        ok: false,
        status: 500,
        body: null,
        headers: new Headers(),
        json: async () => ({}),
      } as unknown as Response)
    );

    await streamChat("hi", "t1", cbs);

    expect(cbs.onError).toHaveBeenCalledWith("HTTP 500");
  });

  it("settles onDone with accumulated text when the stream ends without a done event", async () => {
    const cbs = makeCbs();
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(sseResponse([frame("token", { token: "partial answer" })]))
    );

    await streamChat("hi", "t1", cbs);

    expect(cbs.onDone).toHaveBeenCalledWith("partial answer");
    expect(cbs.onError).not.toHaveBeenCalled();
  });

  it("settles onError when the stream ends with no accumulated text and no done event", async () => {
    const cbs = makeCbs();
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(sseResponse([])));

    await streamChat("hi", "t1", cbs);

    expect(cbs.onError).toHaveBeenCalledWith("Connection closed before a response completed");
    expect(cbs.onDone).not.toHaveBeenCalled();
  });

  it("settles exactly once on the normal done-event path", async () => {
    const cbs = makeCbs();
    vi.stubGlobal(
      "fetch",
      vi
        .fn()
        .mockResolvedValue(
          sseResponse([frame("token", { token: "hi" }), frame("done", { full_text: "hi" })])
        )
    );

    await streamChat("hi", "t1", cbs);

    const settleCount =
      (cbs.onDone as any).mock.calls.length + (cbs.onError as any).mock.calls.length;
    expect(settleCount).toBe(1);
  });

  it("settles exactly once on the error-event path", async () => {
    const cbs = makeCbs();
    vi.stubGlobal(
      "fetch",
      vi
        .fn()
        .mockResolvedValue(
          sseResponse([frame("token", { token: "hi" }), frame("error", { detail: "boom" })])
        )
    );

    await streamChat("hi", "t1", cbs);

    const settleCount =
      (cbs.onDone as any).mock.calls.length + (cbs.onError as any).mock.calls.length;
    expect(settleCount).toBe(1);
  });

  it("settles exactly once on the abort path", async () => {
    const cbs = makeCbs();
    const ctrl = new AbortController();
    const fetchMock = vi.fn(
      (_url: string, init: any) =>
        new Promise((_resolve, reject) => {
          init.signal.addEventListener("abort", () => {
            reject(new DOMException("Aborted", "AbortError"));
          });
        })
    );
    vi.stubGlobal("fetch", fetchMock);

    const promise = streamChat("hi", "t1", cbs, ctrl.signal);
    ctrl.abort();
    await promise;

    const settleCount =
      (cbs.onDone as any).mock.calls.length + (cbs.onError as any).mock.calls.length;
    expect(settleCount).toBe(1);
  });

  it("settles via onDone('') without throwing when aborted via AbortSignal", async () => {
    const cbs = makeCbs();
    const ctrl = new AbortController();
    const fetchMock = vi.fn(
      (_url: string, init: any) =>
        new Promise((_resolve, reject) => {
          init.signal.addEventListener("abort", () => {
            reject(new DOMException("Aborted", "AbortError"));
          });
        })
    );
    vi.stubGlobal("fetch", fetchMock);

    const promise = streamChat("hi", "t1", cbs, ctrl.signal);
    ctrl.abort();

    await expect(promise).resolves.toBeUndefined();
    expect(cbs.onDone).toHaveBeenCalledWith("");
    expect(cbs.onError).not.toHaveBeenCalled();
  });

  it("invokes onError instead of rejecting when fetch throws a network error", async () => {
    const cbs = makeCbs();
    vi.stubGlobal("fetch", vi.fn().mockRejectedValue(new Error("network down")));

    await expect(streamChat("hi", "t1", cbs)).resolves.toBeUndefined();

    expect(cbs.onError).toHaveBeenCalledWith("network down");
    expect(cbs.onDone).not.toHaveBeenCalled();
  });

  it("skips malformed JSON in a frame without killing the stream", async () => {
    const cbs = makeCbs();
    vi.stubGlobal(
      "fetch",
      vi
        .fn()
        .mockResolvedValue(
          sseResponse(["event: token\ndata: {not valid json\n\n", frame("token", { token: "ok" })])
        )
    );

    await streamChat("hi", "t1", cbs);

    expect(cbs.onToken).toHaveBeenCalledTimes(1);
    expect(cbs.onToken).toHaveBeenCalledWith("ok");
  });
});

describe("authHeaders", () => {
  it("returns an empty object when no API key is configured", () => {
    expect(authHeaders()).toEqual({});
  });
});
