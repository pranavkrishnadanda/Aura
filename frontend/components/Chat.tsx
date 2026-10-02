"use client";
import { useEffect, useRef, useState } from "react";
import { API_URL, authHeaders, streamChat } from "@/lib/api";
import type { Citation, Message } from "@/lib/types";
import AdminUpload from "./AdminUpload";
import { Prose, Provenance } from "./AnswerProse";
import CitationPanel from "./CitationPanel";
import Drawer from "./Drawer";

/** Per-browser thread id, persisted so a reload resumes the same conversation.
 *
 * The backend requires ChatRequest.thread_id and has no shared fallback, so each
 * browser must supply its own id; a common value would put every visitor in one
 * conversation and expose each user's clinical queries in the others' history.
 */
const THREAD_KEY = "aura.thread_id";

function makeThreadId(): string {
  return `thr_${Math.random().toString(16).slice(2, 10)}${Date.now().toString(16).slice(-4)}`;
}

function storeThreadId(id: string) {
  try {
    window.localStorage.setItem(THREAD_KEY, id);
  } catch {}
}

function initialThreadId(): string {
  if (typeof window === "undefined") return "default"; // SSR pass; replaced on mount
  let id: string | null = null;
  try {
    id = window.localStorage.getItem(THREAD_KEY);
  } catch {}
  if (!id) {
    id = makeThreadId();
    storeThreadId(id);
  }
  return id;
}

/** The backend answers "Thread not found" for a thread owned by another identity.
 * A stored id can outlive the identity that created it (site data partly cleared,
 * or a thread created before visitors had per-browser identities), so the chat
 * starts a fresh thread instead of failing every send until storage is wiped. */
const THREAD_NOT_FOUND = "Thread not found";

const SUGGESTIONS = [
  "First-line therapy for hypertension with CKD?",
  "Contraindications for lisinopril?",
  "Enoxaparin dosing for VTE prophylaxis?",
];

export default function Chat() {
  const [threadId, setThreadId] = useState("default");
  const [sessionThread, setSessionThread] = useState("default");
  const [threads, setThreads] = useState<any[]>([]);
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [streaming, setStreaming] = useState(false);
  const [linked, setLinked] = useState<number | null>(null);
  const [health, setHealth] = useState<any>(null);
  const scroller = useRef<HTMLDivElement>(null);
  const abortRef = useRef<AbortController | null>(null);
  // Free-tier hosting sleeps after ~15 minutes idle, so the first request of the
  // day spends ~30s waking the container before a single token arrives. Without
  // saying so the UI just sits there and reads as broken.
  const [waking, setWaking] = useState(false);
  const [railOpen, setRailOpen] = useState(false);
  const wakeTimer = useRef<ReturnType<typeof setTimeout> | null>(null);

  // Keep the newest message in view: re-run on every messages change, including
  // each streamed token, even though the body only reads the scroller ref.
  // biome-ignore lint/correctness/useExhaustiveDependencies: messages is the scroll trigger
  useEffect(() => {
    scroller.current?.scrollTo(0, scroller.current.scrollHeight);
  }, [messages]);

  useEffect(() => {
    const id = initialThreadId();
    setSessionThread(id);
    setThreadId(id);
  }, []);

  useEffect(() => {
    fetch(`${API_URL}/api/v1/threads`, { headers: authHeaders() })
      .then((r) => r.json())
      // The list is rendered with .map, so a non-array error body would crash the
      // rail rather than just leaving it empty.
      .then((d) => setThreads(Array.isArray(d) ? d : []))
      .catch(() => {});
    // Retrieval mode is reported rather than assumed: the app answers from
    // embeddings or from keyword matching depending on what is actually available,
    // and a demo should not imply the former while doing the latter.
    fetch(`${API_URL}/health`)
      .then((r) => r.json())
      .then(setHealth)
      .catch(() => {});
  }, []);

  useEffect(() => () => abortRef.current?.abort(), []);

  async function newThread() {
    const r = await fetch(`${API_URL}/api/v1/threads`, {
      method: "POST",
      headers: { "Content-Type": "application/json", ...authHeaders() },
      body: JSON.stringify({ title: `Consult ${new Date().toLocaleDateString()}` }),
    });
    const t = await r.json();
    setThreads((prev) => [t, ...prev]);
    setThreadId(t.id);
    setMessages([]);
  }

  async function openThread(id: string) {
    // Cancel any stream still writing into the thread we are leaving; without this
    // its tokens land in the newly opened conversation.
    abortRef.current?.abort();
    setStreaming(false);
    setThreadId(id);
    setRailOpen(false);
    try {
      const r = await fetch(`${API_URL}/api/v1/threads/${encodeURIComponent(id)}/messages`, {
        headers: authHeaders(),
      });
      const data = await r.json();
      setMessages(Array.isArray(data) ? data : []);
    } catch {}
  }

  async function send() {
    if (!input.trim() || streaming) return;
    const q = input.trim();
    setInput("");
    setMessages((m) => [...m, { role: "user", content: q }, { role: "assistant", content: "" }]);
    setStreaming(true);
    let acc = "";
    let metaCites: Citation[] = [];

    const ctrl = new AbortController();
    abortRef.current = ctrl;
    wakeTimer.current = setTimeout(() => setWaking(true), 3000);
    const stopWaking = () => {
      if (wakeTimer.current) clearTimeout(wakeTimer.current);
      wakeTimer.current = null;
      setWaking(false);
    };
    const replaceLast = (patch: Partial<Message>) =>
      setMessages((prev) => {
        const copy = [...prev];
        const prevLast = copy[copy.length - 1] ?? { role: "assistant" as const, content: "" };
        copy[copy.length - 1] = { ...prevLast, ...patch, role: "assistant" };
        return copy;
      });

    // A stale thread id is detected by the server's 404 and replaced once; a
    // second "not found" is shown to the user rather than retried in a loop.
    let staleThread = false;
    const attempt = (tid: string) =>
      streamChat(
        q,
        tid,
        {
          onMeta: (cits) => {
            stopWaking();
            metaCites = cits as Citation[];
          },
          onToken: (tok) => {
            stopWaking();
            acc += tok;
            replaceLast({ content: acc, citations: metaCites });
          },
          onDone: (full, check) =>
            replaceLast({ content: full || acc, citations: metaCites, check }),
          onError: (e) => {
            if (e === THREAD_NOT_FOUND && !staleThread) {
              staleThread = true;
              return;
            }
            replaceLast({ content: `Couldn't complete that: ${e}`, citations: [] });
          },
        },
        ctrl.signal
      );

    try {
      await attempt(threadId);
      if (staleThread && !ctrl.signal.aborted) {
        const fresh = makeThreadId();
        storeThreadId(fresh);
        setSessionThread(fresh);
        setThreadId(fresh);
        await attempt(fresh);
      }
    } catch (e: any) {
      replaceLast({ content: `Couldn't complete that: ${e?.message || e}`, citations: [] });
    } finally {
      // Cleared on every path, including a throw, so the composer can never be
      // left disabled.
      stopWaking();
      if (abortRef.current === ctrl) abortRef.current = null;
      setStreaming(false);
    }
  }

  const mode = health?.retrieval_mode as string | undefined;
  const degraded = health && health.status !== "ok";

  return (
    <div className="flex h-screen overflow-hidden">
      {/* Gesture-driven below md; a static column at md and up. The panel tracks
          the finger, a flick projects forward to decide open or closed, and it can
          be caught and reversed mid-flight -- none of which a CSS transition can do. */}
      <Drawer open={railOpen} onOpenChange={setRailOpen}>
        <div className="border-b px-4 py-4" style={{ borderColor: "var(--rule)" }}>
          <div className="text-[15px] font-semibold tracking-tight">Aura</div>
          <div className="mt-0.5 text-[11px]" style={{ color: "var(--ink-soft)" }}>
            Clinical reference
          </div>
        </div>

        <div className="px-3 pt-3">
          <button
            type="button"
            onClick={newThread}
            className="press w-full rounded-sm border px-3 py-2 text-[12px] font-medium hover:bg-[var(--paper)]"
            style={{ borderColor: "var(--rule)" }}
          >
            New consultation
          </button>
        </div>

        <div className="mt-4 px-4">
          <div className="label">Consultations</div>
        </div>
        <nav className="mt-2 flex-1 overflow-y-auto px-2 pb-3">
          <button
            type="button"
            onClick={() => openThread(sessionThread)}
            aria-current={threadId === sessionThread}
            className="press block w-full truncate rounded-sm px-2 py-1.5 text-left text-[12px]"
            style={{
              background: threadId === sessionThread ? "var(--paper)" : "transparent",
              color: threadId === sessionThread ? "var(--ink)" : "var(--ink-soft)",
            }}
          >
            This session
          </button>
          {threads.map((t) => (
            <button
              type="button"
              key={t.id}
              onClick={() => openThread(t.id)}
              aria-current={threadId === t.id}
              className="press block w-full truncate rounded-sm px-2 py-1.5 text-left text-[12px]"
              style={{
                background: threadId === t.id ? "var(--paper)" : "transparent",
                color: threadId === t.id ? "var(--ink)" : "var(--ink-soft)",
              }}
            >
              {t.title}
            </button>
          ))}
        </nav>

        {/* Honest system state. The app answers from embeddings or from keyword
            matching depending on what is configured; saying which is the point. */}
        <div className="border-t px-4 py-3" style={{ borderColor: "var(--rule)" }}>
          <div className="label">Retrieval</div>
          <div className="mt-1.5 text-[11px] tabular-nums" style={{ color: "var(--ink-soft)" }}>
            {mode ? (
              <>
                <span style={{ color: mode === "pgvector" ? "var(--source)" : "var(--flag)" }}>
                  {mode === "pgvector" ? "semantic" : "keyword"}
                </span>
                {" · cutoff "}
                {health?.threshold}
              </>
            ) : (
              "checking…"
            )}
          </div>
          {degraded ? (
            <div className="mt-1.5 text-[11px]" style={{ color: "var(--flag)" }}>
              Storage degraded — nothing is being saved.
            </div>
          ) : null}
        </div>

        <div className="border-t px-3 py-3" style={{ borderColor: "var(--rule)" }}>
          <AdminUpload compact maxPdfMb={health?.max_pdf_mb} />
        </div>
      </Drawer>

      {/* Consultation */}
      <main className="flex min-w-0 flex-1 flex-col">
        <div
          className="sticky top-0 z-20 flex items-center gap-3 border-b px-4 py-2.5 md:hidden"
          style={{
            borderColor: "var(--rule)",
            background: "var(--chrome)",
            backdropFilter: "var(--chrome-blur)",
          }}
        >
          <button
            type="button"
            onClick={() => setRailOpen(true)}
            aria-label="Open menu"
            aria-expanded={railOpen}
            className="press rounded-sm border px-2.5 py-1.5 text-[11px]"
            style={{ borderColor: "var(--rule)" }}
          >
            Menu
          </button>
          <span className="text-[13px] font-semibold tracking-tight">Aura</span>
          {mode ? (
            <span
              className="ml-auto text-[11px]"
              style={{ color: mode === "pgvector" ? "var(--source)" : "var(--flag)" }}
            >
              {mode === "pgvector" ? "semantic" : "keyword"}
            </span>
          ) : null}
        </div>
        <div ref={scroller} className="scroll-edge flex-1 overflow-y-auto">
          <div className="mx-auto max-w-[68ch] px-6 py-10">
            {messages.length === 0 && (
              <div>
                <h1 className="prose-clinical prose-display text-[38px] font-medium">
                  Ask a clinical question.
                  <br />
                  <span style={{ color: "var(--source)" }}>Read the source it came from.</span>
                </h1>
                <p className="prose-clinical mt-5 text-[16px]" style={{ color: "var(--ink-soft)" }}>
                  Every answer is assembled from indexed guidelines and protocols. Each reference
                  opens the exact passage and page it was drawn from, quoted without alteration. If
                  nothing in the corpus covers your question, Aura says so rather than guessing.
                </p>
                <div className="mt-7">
                  <div className="label">Try</div>
                  <div className="mt-2.5 flex flex-col items-start gap-1.5">
                    {SUGGESTIONS.map((ex) => (
                      <button
                        type="button"
                        key={ex}
                        onClick={() => setInput(ex)}
                        className="press text-left text-[12px] underline decoration-dotted underline-offset-4 hover:decoration-solid"
                        style={{ color: "var(--source)" }}
                      >
                        {ex}
                      </button>
                    ))}
                  </div>
                </div>
              </div>
            )}

            <div className="space-y-9">
              {messages.map((m, i) =>
                m.role === "user" ? (
                  <div key={i}>
                    <div className="label">Question</div>
                    <p className="prose-clinical prose-question mt-1.5 text-[19px] font-medium">
                      {m.content}
                    </p>
                  </div>
                ) : (
                  <div key={i}>
                    <div className="label">Answer</div>
                    <div className="prose-clinical mt-1.5">
                      {m.content ? (
                        <Prose
                          text={m.content}
                          citations={m.citations ?? []}
                          linked={linked}
                          onLink={setLinked}
                        />
                      ) : streaming && i === messages.length - 1 && !waking ? (
                        <span style={{ color: "var(--ink-soft)" }}>▍</span>
                      ) : null}
                    </div>

                    <Provenance citations={m.citations ?? []} check={m.check} />

                    {m.citations?.length ? (
                      <div className="mt-4">
                        <div className="label">Evidence</div>
                        <div className="mt-2 space-y-2">
                          {m.citations.map((c) => (
                            <div key={c.id} id={`evidence-${c.idx}`}>
                              <CitationPanel
                                citation={c}
                                linked={linked === c.idx}
                                onHover={setLinked}
                              />
                            </div>
                          ))}
                        </div>
                      </div>
                    ) : null}
                  </div>
                )
              )}
            </div>

            {waking && (
              <div
                className="mt-6 rounded-sm border px-4 py-3 text-[12px] leading-relaxed"
                style={{ borderColor: "var(--flag)", color: "var(--flag)" }}
              >
                <span className="font-medium">Waking the server.</span> This deployment runs on a
                free tier that sleeps when idle, so the first request can take around 30 seconds.
                Later questions respond immediately.
              </div>
            )}
          </div>
        </div>

        {/* Composer */}
        <div
          className="border-t"
          style={{
            borderColor: "var(--rule)",
            background: "var(--chrome)",
            backdropFilter: "var(--chrome-blur)",
          }}
        >
          <div className="mx-auto max-w-[68ch] px-6 py-4">
            <div className="flex items-end gap-2">
              <input
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={(e) => e.key === "Enter" && !e.shiftKey && (e.preventDefault(), send())}
                placeholder="Ask a clinical question…"
                aria-label="Ask a clinical question"
                className="flex-1 border-b bg-transparent px-1 py-2 text-[13px] outline-none placeholder:text-[var(--ink-soft)] focus:border-[var(--source)]"
                style={{ borderColor: "var(--rule)" }}
              />
              <button
                type="button"
                onClick={send}
                disabled={streaming || !input.trim()}
                className="press shrink-0 rounded-sm px-4 py-2 text-[12px] font-medium text-white disabled:opacity-30"
                style={{ background: "var(--ink)" }}
              >
                Send
              </button>
            </div>
            <p className="mt-2 text-[11px]" style={{ color: "var(--ink-soft)" }}>
              Reference tool for clinicians. Not a diagnosis, and not a substitute for clinical
              judgement.
            </p>
          </div>
        </div>
      </main>
    </div>
  );
}
