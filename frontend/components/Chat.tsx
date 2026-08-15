"use client";
import { useState, useRef, useEffect } from "react";
import { streamChat, API_URL, authHeaders } from "@/lib/api";
import { Citation, CitationCheck, Message } from "@/lib/types";
import CitationPanel from "./CitationPanel";
import AdminUpload from "./AdminUpload";

/** Per-browser thread id, persisted so a reload resumes the same conversation.
 *
 * This was the literal string "default" for every visitor, and the backend's
 * ChatRequest.thread_id defaults to "default" too -- so every browser shared one
 * conversation and each user saw the others' clinical queries in their history.
 */
function initialThreadId(): string {
  if (typeof window === "undefined") return "default"; // SSR pass; replaced on mount
  const KEY = "aura.thread_id";
  let id = window.localStorage.getItem(KEY);
  if (!id) {
    id = `thr_${Math.random().toString(16).slice(2, 10)}${Date.now().toString(16).slice(-4)}`;
    window.localStorage.setItem(KEY, id);
  }
  return id;
}

/** Splits the model's prose on citation markers and renders each as a control
 *  linked to its evidence card. A marker with no matching source is shown as
 *  flagged text rather than a link -- it points at nothing. */
function Prose({
  text,
  citations,
  linked,
  onLink,
}: {
  text: string;
  citations: Citation[];
  linked: number | null;
  onLink: (idx: number | null) => void;
}) {
  const parts = text.split(/(\[\d+\])/g);
  return (
    <>
      {parts.map((p, i) => {
        const m = p.match(/^\[(\d+)\]$/);
        if (!m) return <span key={i}>{p}</span>;
        const idx = parseInt(m[1], 10);
        const cite = citations.find((c) => c.idx === idx);
        if (!cite) {
          return (
            <sup
              key={i}
              title="This reference does not match any retrieved source"
              className="mx-0.5 font-medium"
              style={{ color: "var(--flag)" }}
            >
              [{idx}?]
            </sup>
          );
        }
        return (
          <button
            key={i}
            type="button"
            className="marker"
            data-linked={linked === idx ? "true" : "false"}
            aria-label={`Open citation ${idx}`}
            onMouseEnter={() => onLink(idx)}
            onMouseLeave={() => onLink(null)}
            onFocus={() => onLink(idx)}
            onBlur={() => onLink(null)}
            onClick={() => {
              document.getElementById(`evidence-${idx}`)?.scrollIntoView({ block: "nearest" });
              onLink(idx);
            }}
          >
            [{idx}]
          </button>
        );
      })}
    </>
  );
}

/** One line stating what the system can actually vouch for in this answer. */
function Provenance({ citations, check }: { citations: Citation[]; check?: CitationCheck | null }) {
  if (!citations.length) return null;
  const bad = check?.invalid_markers?.length ?? 0;
  const uncited = check && !check.cited;
  return (
    <div className="mt-2.5 flex flex-wrap items-center gap-x-3 gap-y-1 text-[11px] tabular-nums">
      <span style={{ color: "var(--ink-soft)" }}>
        {citations.length} {citations.length === 1 ? "source" : "sources"}
      </span>
      {bad > 0 ? (
        <span style={{ color: "var(--flag)" }}>
          {bad} reference{bad === 1 ? "" : "s"} unmatched
        </span>
      ) : uncited ? (
        <span style={{ color: "var(--flag)" }}>no reference given</span>
      ) : (
        <span style={{ color: "var(--source)" }}>all references resolve</span>
      )}
    </div>
  );
}

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

  useEffect(() => { scroller.current?.scrollTo(0, scroller.current.scrollHeight); }, [messages, streaming]);

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
    fetch(`${API_URL}/health`).then((r) => r.json()).then(setHealth).catch(() => {});
  }, []);

  useEffect(() => () => abortRef.current?.abort(), []);

  async function newThread() {
    const r = await fetch(`${API_URL}/api/v1/threads`, {
      method: "POST", headers: { "Content-Type": "application/json", ...authHeaders() },
      body: JSON.stringify({ title: `Consult ${new Date().toLocaleDateString()}` }),
    });
    const t = await r.json();
    setThreads((prev) => [t, ...prev]);
    setThreadId(t.id); setMessages([]);
  }

  async function openThread(id: string) {
    // Cancel any stream still writing into the thread we are leaving; without this
    // its tokens land in the newly opened conversation.
    abortRef.current?.abort();
    setStreaming(false);
    setThreadId(id);
    setRailOpen(false);
    try {
      const r = await fetch(`${API_URL}/api/v1/threads/${encodeURIComponent(id)}/messages`, { headers: authHeaders() });
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

    try {
      await streamChat(q, threadId, {
        onMeta: (cits) => { stopWaking(); metaCites = cits as Citation[]; },
        onToken: (tok) => { stopWaking(); acc += tok; replaceLast({ content: acc, citations: metaCites }); },
        onDone: (full, check) => replaceLast({ content: full || acc, citations: metaCites, check }),
        onError: (e) => replaceLast({ content: `Couldn't complete that: ${e}`, citations: [] }),
      }, ctrl.signal);
    } catch (e: any) {
      replaceLast({ content: `Couldn't complete that: ${e?.message || e}`, citations: [] });
    } finally {
      // Always clear, on every path. Previously this lived only inside onDone and
      // onError, so any throw left the composer disabled permanently.
      stopWaking();
      if (abortRef.current === ctrl) abortRef.current = null;
      setStreaming(false);
    }
  }

  const mode = health?.retrieval_mode as string | undefined;
  const degraded = health && health.status !== "ok";

  return (
    <div className="flex h-screen overflow-hidden">
      {/* Scrim for the mobile slide-over */}
      {railOpen ? (
        <button
          aria-label="Close menu"
          onClick={() => setRailOpen(false)}
          className="fixed inset-0 z-30 bg-black/20 md:hidden"
        />
      ) : null}

      {/* Rail — identity, threads, and what the system currently knows.
          A slide-over below md so ingest, history and retrieval state stay
          reachable on a phone rather than being hidden with the sidebar. */}
      <aside
        className={`fixed inset-y-0 left-0 z-40 flex w-[236px] shrink-0 flex-col border-r bg-white transition-transform md:static md:translate-x-0 ${
          railOpen ? "translate-x-0" : "-translate-x-full"
        }`}
        style={{ borderColor: "var(--rule)" }}
      >
        <div className="border-b px-4 py-4" style={{ borderColor: "var(--rule)" }}>
          <div className="text-[15px] font-semibold tracking-tight">Aura</div>
          <div className="mt-0.5 text-[11px]" style={{ color: "var(--ink-soft)" }}>
            Clinical reference
          </div>
        </div>

        <div className="px-3 pt-3">
          <button
            onClick={newThread}
            className="w-full rounded-sm border px-3 py-2 text-[12px] font-medium transition-colors hover:bg-[var(--paper)]"
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
            onClick={() => openThread(sessionThread)}
            aria-current={threadId === sessionThread}
            className="block w-full truncate rounded-sm px-2 py-1.5 text-left text-[12px]"
            style={{
              background: threadId === sessionThread ? "var(--paper)" : "transparent",
              color: threadId === sessionThread ? "var(--ink)" : "var(--ink-soft)",
            }}
          >
            This session
          </button>
          {threads.map((t) => (
            <button
              key={t.id}
              onClick={() => openThread(t.id)}
              aria-current={threadId === t.id}
              className="block w-full truncate rounded-sm px-2 py-1.5 text-left text-[12px]"
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
          <AdminUpload compact />
        </div>
      </aside>

      {/* Consultation */}
      <main className="flex min-w-0 flex-1 flex-col">
        <div
          className="flex items-center gap-3 border-b bg-white px-4 py-2.5 md:hidden"
          style={{ borderColor: "var(--rule)" }}
        >
          <button
            onClick={() => setRailOpen(true)}
            aria-label="Open menu"
            aria-expanded={railOpen}
            className="rounded-sm border px-2.5 py-1.5 text-[11px]"
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
        <div ref={scroller} className="flex-1 overflow-y-auto">
          <div className="mx-auto max-w-[68ch] px-6 py-10">
            {messages.length === 0 && (
              <div>
                <h1 className="prose-clinical text-[38px] font-medium leading-[1.15] tracking-[-0.015em]">
                  Ask a clinical question.
                  <br />
                  <span style={{ color: "var(--source)" }}>Read the source it came from.</span>
                </h1>
                <p className="prose-clinical mt-5 text-[16px]" style={{ color: "var(--ink-soft)" }}>
                  Every answer is assembled from indexed guidelines and protocols. Each
                  reference opens the exact passage and page it was drawn from, quoted
                  without alteration. If nothing in the corpus covers your question, Aura
                  says so rather than guessing.
                </p>
                <div className="mt-7">
                  <div className="label">Try</div>
                  <div className="mt-2.5 flex flex-col items-start gap-1.5">
                    {SUGGESTIONS.map((ex) => (
                      <button
                        key={ex}
                        onClick={() => setInput(ex)}
                        className="text-left text-[12px] underline decoration-dotted underline-offset-4 hover:decoration-solid"
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
                    <p className="prose-clinical mt-1.5 text-[19px] font-medium leading-snug">
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
                <span className="font-medium">Waking the server.</span> This deployment runs
                on a free tier that sleeps when idle, so the first request can take around
                30 seconds. Later questions respond immediately.
              </div>
            )}
          </div>
        </div>

        {/* Composer */}
        <div className="border-t bg-white" style={{ borderColor: "var(--rule)" }}>
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
                onClick={send}
                disabled={streaming || !input.trim()}
                className="shrink-0 rounded-sm px-4 py-2 text-[12px] font-medium text-white transition-opacity disabled:opacity-30"
                style={{ background: "var(--ink)" }}
              >
                Send
              </button>
            </div>
            <p className="mt-2 text-[11px]" style={{ color: "var(--ink-soft)" }}>
              Reference tool for clinicians. Not a diagnosis, and not a substitute for
              clinical judgement.
            </p>
          </div>
        </div>
      </main>
    </div>
  );
}
