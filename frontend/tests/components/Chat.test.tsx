import { describe, it, expect, vi, beforeAll, beforeEach } from "vitest";
import { render, screen, fireEvent, act, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import Chat from "@/components/Chat";
import { sseResponse, frame } from "../setup";

// jsdom does not implement Element.scrollTo; Chat.tsx calls it to keep the
// transcript pinned to the bottom on every new message.
beforeAll(() => {
  Element.prototype.scrollTo = vi.fn();
});

/**
 * In this Node runtime, the built-in experimental global `localStorage`
 * (gated behind `--localstorage-file`) shadows jsdom's working
 * window.localStorage and leaves it with no .getItem/.setItem -- unrelated to
 * anything under test here, but it breaks Chat.tsx's per-browser thread id,
 * so each test gets a real in-memory Storage instead. setup.ts's global
 * afterEach calls vi.unstubAllGlobals(), so this must be re-applied per test.
 */
function installMemoryLocalStorage() {
  const store = new Map<string, string>();
  const storage: Storage = {
    get length() {
      return store.size;
    },
    clear: () => store.clear(),
    getItem: (key: string) => (store.has(key) ? store.get(key)! : null),
    key: (index: number) => Array.from(store.keys())[index] ?? null,
    removeItem: (key: string) => void store.delete(key),
    setItem: (key: string, value: string) => void store.set(key, String(value)),
  };
  vi.stubGlobal("localStorage", storage);
}

beforeEach(() => {
  installMemoryLocalStorage();
});

const API_URL = "http://localhost:8000";

type ChatStreamHandler = (body: any, signal?: AbortSignal) => Response | Promise<Response>;

/**
 * A minimal fetch dispatcher covering every endpoint Chat.tsx calls:
 * GET /threads (sidebar list), POST /threads (new thread), GET
 * /threads/:id/messages (switch thread), POST /chat/stream (send).
 */
function makeFetch(opts: {
  threads?: any;
  threadMessages?: Record<string, any[]>;
  chatStream?: ChatStreamHandler;
} = {}) {
  const threads = opts.threads ?? [];
  const threadMessages = opts.threadMessages ?? {};
  const chatStream: ChatStreamHandler =
    opts.chatStream ??
    (() =>
      sseResponse([
        frame("meta", { citations: [], is_refusal: false }),
        frame("token", { token: "ok" }),
        frame("done", { full_text: "ok" }),
      ]));

  return vi.fn(async (url: string, init?: any) => {
    const u = String(url);
    if (u.endsWith("/api/v1/threads") && (!init?.method || init.method === "GET")) {
      return { ok: true, status: 200, headers: new Headers(), json: async () => threads } as Response;
    }
    if (u.endsWith("/api/v1/threads") && init?.method === "POST") {
      return {
        ok: true,
        status: 200,
        headers: new Headers(),
        json: async () => ({ id: "thr_new", title: "New thread" }),
      } as Response;
    }
    const msgMatch = u.match(/\/api\/v1\/threads\/([^/]+)\/messages$/);
    if (msgMatch) {
      const id = decodeURIComponent(msgMatch[1]);
      return { ok: true, status: 200, headers: new Headers(), json: async () => threadMessages[id] ?? [] } as Response;
    }
    if (u.endsWith("/api/v1/chat/stream")) {
      const body = init?.body ? JSON.parse(init.body) : {};
      return chatStream(body, init?.signal);
    }
    throw new Error(`unexpected fetch: ${u}`);
  });
}

/** A ReadableStream whose chunks are pushed manually, to observe intermediate render states. */
function controlledSSE() {
  const enc = new TextEncoder();
  let controllerRef!: ReadableStreamDefaultController<Uint8Array>;
  const stream = new ReadableStream<Uint8Array>({
    start(controller) {
      controllerRef = controller;
    },
  });
  return {
    stream,
    push: (text: string) => controllerRef.enqueue(enc.encode(text)),
    close: () => controllerRef.close(),
  };
}

const PLACEHOLDER = /Ask a clinical question/i;

describe("Chat - empty state", () => {
  it("renders the empty state and the suggested clinical questions", async () => {
    const fetchMock = makeFetch();
    vi.stubGlobal("fetch", fetchMock);
    render(<Chat />);
    // Let the mount-time GET /threads effect settle before asserting, so its
    // state update isn't reported outside of act().
    await waitFor(() => expect(fetchMock).toHaveBeenCalled());

    expect(screen.getByText("Start a clinical query")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "First-line therapy for hypertension with CKD?" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Contraindications for lisinopril?" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Enoxaparin dosing for VTE prophylaxis?" })).toBeInTheDocument();
  });

  it("populates the input when a suggested question is clicked", async () => {
    vi.stubGlobal("fetch", makeFetch());
    const user = userEvent.setup();
    render(<Chat />);

    await user.click(screen.getByRole("button", { name: "Contraindications for lisinopril?" }));

    expect(screen.getByPlaceholderText(PLACEHOLDER)).toHaveValue("Contraindications for lisinopril?");
  });
});

describe("Chat - sending a message", () => {
  it("renders the user message, streams assistant tokens progressively, and clears streaming on completion", async () => {
    const ctrl = controlledSSE();
    vi.stubGlobal(
      "fetch",
      makeFetch({
        chatStream: () =>
          ({ ok: true, status: 200, body: ctrl.stream, headers: new Headers(), json: async () => ({}) } as Response),
      })
    );
    const user = userEvent.setup();
    render(<Chat />);

    await user.type(screen.getByPlaceholderText(PLACEHOLDER), "First-line therapy?");
    await user.click(screen.getByRole("button", { name: "Send" }));

    expect(await screen.findByText("First-line therapy?")).toBeInTheDocument();
    expect(screen.getByText("streaming…")).toBeInTheDocument();

    await act(async () => ctrl.push(frame("meta", { citations: [], is_refusal: false })));
    await act(async () => ctrl.push(frame("token", { token: "Hel" })));
    expect(await screen.findByText("Hel")).toBeInTheDocument();

    await act(async () => ctrl.push(frame("token", { token: "lo" })));
    expect(await screen.findByText("Hello")).toBeInTheDocument();

    await act(async () => {
      ctrl.push(frame("done", { full_text: "Hello" }));
      ctrl.close();
    });

    expect(await screen.findByText("ready")).toBeInTheDocument();
  });

  it("disables Send while input is empty or whitespace-only", async () => {
    const fetchMock = makeFetch();
    vi.stubGlobal("fetch", fetchMock);
    render(<Chat />);
    await waitFor(() => expect(fetchMock).toHaveBeenCalled());

    expect(screen.getByRole("button", { name: "Send" })).toBeDisabled();
    fireEvent.change(screen.getByPlaceholderText(PLACEHOLDER), { target: { value: "   " } });
    expect(screen.getByRole("button", { name: "Send" })).toBeDisabled();
  });

  it("disables Send while a response is streaming", async () => {
    const ctrl = controlledSSE();
    vi.stubGlobal(
      "fetch",
      makeFetch({
        chatStream: () =>
          ({ ok: true, status: 200, body: ctrl.stream, headers: new Headers(), json: async () => ({}) } as Response),
      })
    );
    const user = userEvent.setup();
    render(<Chat />);

    const input = screen.getByPlaceholderText(PLACEHOLDER);
    await user.type(input, "question");
    const sendBtn = screen.getByRole("button", { name: "Send" });
    expect(sendBtn).not.toBeDisabled();

    await user.click(sendBtn);
    expect(sendBtn).toBeDisabled();

    // Clean up the open stream so nothing is left pending.
    await act(async () => {
      ctrl.push(frame("done", { full_text: "" }));
      ctrl.close();
    });
  });

  it("sends the message on Enter", async () => {
    const fetchMock = makeFetch();
    vi.stubGlobal("fetch", fetchMock);
    const user = userEvent.setup();
    render(<Chat />);

    const input = screen.getByPlaceholderText(PLACEHOLDER);
    await user.type(input, "hello{Enter}");

    await waitFor(() => {
      expect(fetchMock.mock.calls.some((c) => String(c[0]).endsWith("/chat/stream"))).toBe(true);
    });
  });

  it("does not send the message on Shift+Enter", async () => {
    const fetchMock = makeFetch();
    vi.stubGlobal("fetch", fetchMock);
    render(<Chat />);
    await waitFor(() => expect(fetchMock).toHaveBeenCalled());

    const input = screen.getByPlaceholderText(PLACEHOLDER);
    fireEvent.change(input, { target: { value: "hello" } });
    fireEvent.keyDown(input, { key: "Enter", shiftKey: true });

    expect(fetchMock.mock.calls.some((c) => String(c[0]).endsWith("/chat/stream"))).toBe(false);
    expect(input).toHaveValue("hello");
  });

  it("surfaces an error mid-stream to the user and re-enables the composer", async () => {
    // Regression guard: `streaming` used to only clear inside onDone/onError's
    // own bodies, so an error path that didn't touch it left Send permanently
    // disabled for the rest of the session.
    vi.stubGlobal(
      "fetch",
      makeFetch({
        chatStream: () =>
          sseResponse([
            frame("meta", { citations: [], is_refusal: false }),
            frame("token", { token: "partial" }),
            frame("error", { detail: "model timeout" }),
          ]),
      })
    );
    const user = userEvent.setup();
    render(<Chat />);

    await user.type(screen.getByPlaceholderText(PLACEHOLDER), "question");
    await user.click(screen.getByRole("button", { name: "Send" }));

    expect(await screen.findByText(/Error: model timeout/)).toBeInTheDocument();
    expect(screen.getByText("ready")).toBeInTheDocument();

    await user.type(screen.getByPlaceholderText(PLACEHOLDER), "another question");
    expect(screen.getByRole("button", { name: "Send" })).not.toBeDisabled();
  });
});

describe("Chat - per-browser thread id", () => {
  it("persists a generated (non-'default') thread id under localStorage key aura.thread_id", async () => {
    // Regression guard: this used to be the literal string "default" for every
    // visitor, so all browsers shared one conversation.
    vi.stubGlobal("fetch", makeFetch());
    render(<Chat />);

    await waitFor(() => expect(window.localStorage.getItem("aura.thread_id")).toBeTruthy());
    expect(window.localStorage.getItem("aura.thread_id")).not.toBe("default");
  });

  it("assigns a different thread id on a second mount after the store is cleared", async () => {
    vi.stubGlobal("fetch", makeFetch());
    const first = render(<Chat />);
    await waitFor(() => expect(window.localStorage.getItem("aura.thread_id")).toBeTruthy());
    const firstId = window.localStorage.getItem("aura.thread_id");
    first.unmount();

    window.localStorage.clear();

    const second = render(<Chat />);
    await waitFor(() => expect(window.localStorage.getItem("aura.thread_id")).toBeTruthy());
    const secondId = window.localStorage.getItem("aura.thread_id");
    second.unmount();

    expect(secondId).not.toBe(firstId);
  });
});

describe("Chat - citations", () => {
  const citation = {
    id: "chunk_1",
    doc_id: "doc_1",
    doc_title: "ACC/AHA Guideline",
    page: 12,
    chunk_text: "Evidence text.",
    score: 0.9,
    idx: 1,
  };

  it("renders citations as clickable [n] buttons that open the CitationPanel", async () => {
    vi.stubGlobal(
      "fetch",
      makeFetch({
        chatStream: () =>
          sseResponse([
            frame("meta", { citations: [citation], is_refusal: false }),
            frame("token", { token: "See [1] for details." }),
            frame("done", { full_text: "See [1] for details." }),
          ]),
      })
    );
    const user = userEvent.setup();
    render(<Chat />);

    await user.type(screen.getByPlaceholderText(PLACEHOLDER), "question");
    await user.click(screen.getByRole("button", { name: "Send" }));

    const citeBtn = await screen.findByRole("button", { name: "Open citation 1" });
    expect(screen.queryByText("Evidence text.")).not.toBeInTheDocument();

    await user.click(citeBtn);

    // The panel renders the doc_title again alongside the source chunk text,
    // so the doc_title now appears twice (chip + panel) -- assert the panel's
    // own content, which is unique, opened.
    expect(screen.getByText("Evidence")).toBeInTheDocument();
    expect(screen.getByText("Evidence text.")).toBeInTheDocument();
  });
});

describe("Chat - threads sidebar resilience", () => {
  it("does not crash when GET /threads returns a non-array body", async () => {
    // Regression guard: the sidebar renders threads.map(...), so a non-array
    // error response (e.g. {"detail": "..."}) used to throw and blank the UI.
    vi.stubGlobal("fetch", makeFetch({ threads: { detail: "boom" } }));
    render(<Chat />);

    expect(await screen.findByText("Threads")).toBeInTheDocument();
    // Only the always-present "Default session" entry counts, so length is 1.
    expect(screen.getByText("1")).toBeInTheDocument();
  });
});

describe("Chat - switching threads", () => {
  it("aborts the in-flight stream when the user switches threads", async () => {
    const otherThread = { id: "thr_other", title: "Other consult" };
    const ctrl = controlledSSE();
    let capturedSignal: AbortSignal | undefined;
    vi.stubGlobal(
      "fetch",
      makeFetch({
        threads: [otherThread],
        threadMessages: { thr_other: [] },
        chatStream: (_body, signal) => {
          capturedSignal = signal;
          return { ok: true, status: 200, body: ctrl.stream, headers: new Headers(), json: async () => ({}) } as Response;
        },
      })
    );
    const user = userEvent.setup();
    render(<Chat />);

    await screen.findByText("Other consult");
    await user.type(screen.getByPlaceholderText(PLACEHOLDER), "question");
    await user.click(screen.getByRole("button", { name: "Send" }));

    await waitFor(() => expect(capturedSignal).toBeDefined());
    expect(capturedSignal!.aborted).toBe(false);

    await user.click(screen.getByText("Other consult"));

    expect(capturedSignal!.aborted).toBe(true);
    expect(await screen.findByText("ready")).toBeInTheDocument();
  });
});

describe("cold-start feedback", () => {
  it("explains the free-tier wake-up when the first byte is slow", async () => {
    // Free-tier hosting sleeps when idle, so the first request can take ~30s.
    // Without this the UI shows only "Receiving tokens…" and reads as broken.
    vi.useFakeTimers();
    try {
      // A response whose body never produces a chunk, i.e. still cold-starting.
      const hanging = new ReadableStream<Uint8Array>({ start() {} });
      vi.stubGlobal(
        "fetch",
        vi.fn(async (url: string) => {
          if (String(url).includes("/chat/stream")) {
            return {
              ok: true,
              status: 200,
              body: hanging,
              headers: new Headers({ "Content-Type": "text/event-stream" }),
            } as unknown as Response;
          }
          return { ok: true, status: 200, json: async () => [] } as unknown as Response;
        })
      );

      render(<Chat />);
      const input = screen.getByPlaceholderText(PLACEHOLDER);
      fireEvent.change(input, { target: { value: "first-line therapy?" } });
      fireEvent.click(screen.getByRole("button", { name: "Send" }));

      // Nothing alarming before the threshold.
      await act(async () => {
        await vi.advanceTimersByTimeAsync(1000);
      });
      expect(screen.queryByText(/Waking the server/i)).not.toBeInTheDocument();

      await act(async () => {
        await vi.advanceTimersByTimeAsync(3000);
      });
      expect(screen.getByText(/Waking the server/i)).toBeInTheDocument();
      expect(screen.getByText(/free\s*tier that sleeps when idle/i)).toBeInTheDocument();
    } finally {
      vi.useRealTimers();
    }
  });

  it("does not show the wake-up notice when tokens arrive promptly", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async (url: string) => {
        if (String(url).includes("/chat/stream")) {
          return sseResponse([
            frame("meta", { citations: [], is_refusal: false, thread_id: "t" }),
            frame("token", { token: "Answer" }),
            frame("done", { full_text: "Answer", citations: [] }),
          ]);
        }
        return { ok: true, status: 200, json: async () => [] } as unknown as Response;
      })
    );

    render(<Chat />);
    fireEvent.change(screen.getByPlaceholderText(PLACEHOLDER), { target: { value: "hi" } });
    fireEvent.click(screen.getByRole("button", { name: "Send" }));

    await waitFor(() => expect(screen.getByText("Answer")).toBeInTheDocument());
    expect(screen.queryByText(/Waking the server/i)).not.toBeInTheDocument();
  });
});
