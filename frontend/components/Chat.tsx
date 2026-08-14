"use client";
import { useState, useRef, useEffect } from "react";
import { streamChat, API_URL } from "@/lib/api";
import { Citation, Message } from "@/lib/types";
import CitationPanel from "./CitationPanel";
import AdminUpload from "./AdminUpload";

function InlineCitations({ text, citations, onCite }: { text: string; citations: Citation[]; onCite: (c: Citation) => void }) {
  const parts = text.split(/(\[\d+\])/g);
  return (
    <>
      {parts.map((p, i) => {
        const m = p.match(/\[(\d+)\]/);
        if (m) {
          const idx = parseInt(m[1], 10);
          const cite = citations.find((c) => c.idx === idx);
          if (cite) {
            return (
              <button
                key={i}
                onClick={() => onCite(cite)}
                className="mx-0.5 inline-flex items-center rounded-md border border-teal-200 bg-teal-50 px-1.5 py-0.5 text-xs font-medium text-teal-800 hover:bg-teal-100 hover:border-teal-300"
                aria-label={`Open citation ${idx}`}
              >
                [{idx}]
              </button>
            );
          }
        }
        return <span key={i}>{p}</span>;
      })}
    </>
  );
}

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

export default function Chat() {
  const [threadId, setThreadId] = useState("default");
  const [threads, setThreads] = useState<any[]>([]);
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [streaming, setStreaming] = useState(false);
  const [activeCite, setActiveCite] = useState<Citation | null>(null);
  const [currentCitations, setCurrentCitations] = useState<Citation[]>([]);
  const scroller = useRef<HTMLDivElement>(null);
  const abortRef = useRef<AbortController | null>(null);
  // This browser's own session thread, used by the "Default session" entry.
  const [sessionThread, setSessionThread] = useState("default");

  useEffect(() => { scroller.current?.scrollTo(0, scroller.current.scrollHeight); }, [messages, streaming]);
  useEffect(() => {
    fetch(`${API_URL}/api/v1/threads`)
      .then((r) => r.json())
      // The list is rendered with .map, so a non-array error body would crash the
      // sidebar rather than just leaving it empty.
      .then((d) => setThreads(Array.isArray(d) ? d : []))
      .catch(() => {});
  }, []);

  // Assign this browser its own thread after mount, so server and client render
  // the same markup on the first pass.
  useEffect(() => {
    const id = initialThreadId();
    setSessionThread(id);
    setThreadId(id);
  }, []);

  // Abandon any in-flight stream when the component goes away, otherwise it keeps
  // reading and calling setState on an unmounted tree.
  useEffect(() => () => abortRef.current?.abort(), []);

  async function newThread() {
    const r = await fetch(`${API_URL}/api/v1/threads`, {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ title: `Consult — ${new Date().toLocaleDateString()} ${new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}` }),
    });
    const t = await r.json();
    setThreads((prev) => [t, ...prev]);
    setThreadId(t.id); setMessages([]); setCurrentCitations([]); setActiveCite(null);
  }

  async function openThread(id: string) {
    // Cancel any stream still writing into the thread we are leaving; without this
    // its tokens land in the newly opened conversation.
    abortRef.current?.abort();
    setStreaming(false);
    setThreadId(id); setActiveCite(null); setCurrentCitations([]);
    try {
      const r = await fetch(`${API_URL}/api/v1/threads/${encodeURIComponent(id)}/messages`);
      const data = await r.json();
      const msgs = Array.isArray(data) ? data : [];
      setMessages(msgs);
      // restore citations from last assistant message if present
      const last = [...msgs].reverse().find((m: any) => m.citations?.length);
      if (last) setCurrentCitations(last.citations);
    } catch {}
  }

  async function send() {
    if (!input.trim() || streaming) return;
    const q = input.trim();
    setInput("");
    setMessages((m) => [...m, { role: "user", content: q }]);
    setStreaming(true);
    let acc = "";
    let metaCites: Citation[] = [];
    // placeholder for assistant
    setMessages((m) => [...m, { role: "assistant", content: "" }]);

    const ctrl = new AbortController();
    abortRef.current = ctrl;
    try {
      await streamChat(q, threadId, {
        onMeta: (cits) => { metaCites = cits as Citation[]; setCurrentCitations(cits as Citation[]); },
        onToken: (tok) => {
          acc += tok;
          setMessages((prev) => { const copy = [...prev]; copy[copy.length - 1] = { role: "assistant", content: acc, citations: metaCites }; return copy; });
        },
        onDone: (full) => {
          setMessages((prev) => { const copy = [...prev]; copy[copy.length - 1] = { role: "assistant", content: full || acc, citations: metaCites }; return copy; });
        },
        onError: (e) => { setMessages((prev) => [...prev.slice(0, -1), { role: "assistant", content: `Error: ${e}` }]); },
      }, ctrl.signal);
    } catch (e: any) {
      setMessages((prev) => [...prev.slice(0, -1), { role: "assistant", content: `Error: ${e?.message || e}` }]);
    } finally {
      // Always clear, on every path. Previously this lived only inside onDone and
      // onError, so any throw left the composer disabled permanently.
      if (abortRef.current === ctrl) abortRef.current = null;
      setStreaming(false);
    }
  }

  return (
    <div className="flex h-screen overflow-hidden">
      {/* Left — threads */}
      <div className="w-[260px] shrink-0 border-r border-slate-200 bg-white flex flex-col">
        <div className="h-12 px-3 border-b border-slate-200 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <div className="h-7 w-7 rounded-md bg-slate-900 flex items-center justify-center text-white text-xs font-semibold">A</div>
            <div>
              <div className="text-sm font-semibold leading-none tracking-tight">Aura</div>
              <div className="text-xs font-mono text-slate-500 leading-none">Clinical</div>
            </div>
          </div>
          <span className="text-xs font-mono px-1.5 py-1 rounded border border-slate-200 bg-slate-50">v0.1</span>
        </div>

        <div className="p-3">
          <button onClick={newThread} className="w-full rounded-md bg-slate-900 px-3 py-2 text-sm font-medium text-white hover:bg-slate-800">New consultation</button>
          <div className="mt-3 flex items-center justify-between">
            <span className="text-xs font-medium text-slate-700">Threads</span>
            <span className="text-xs font-mono text-slate-500">{threads.length + 1}</span>
          </div>
        </div>

        <div className="flex-1 overflow-y-auto px-2 pb-3 space-y-1">
          <button onClick={() => openThread(sessionThread)} className={`w-full text-left rounded-md px-2.5 py-2 border ${threadId === sessionThread ? "bg-slate-900 text-white border-slate-900" : "bg-white border-slate-200 hover:bg-slate-50"}`}>
            <div className="text-sm font-medium leading-none truncate">Default session</div>
            <div className="text-xs font-mono leading-none mt-1 opacity-70">resume · streaming</div>
          </button>
          {threads.map((t) => (
            <button key={t.id} onClick={() => openThread(t.id)} className={`w-full text-left rounded-md px-2.5 py-2 border ${threadId === t.id ? "bg-slate-900 text-white border-slate-900" : "bg-white border-slate-200 hover:bg-slate-50"}`}>
              <div className="text-sm font-medium leading-none truncate">{t.title}</div>
              <div className="text-xs font-mono leading-none mt-1 opacity-60 truncate">{t.id}</div>
            </button>
          ))}
        </div>

        <div className="p-3 border-t border-slate-200 space-y-3">
          <AdminUpload compact />
          <div className="text-xs leading-4 text-slate-600">TTFT &lt;400ms · grounded citations · SSE streaming</div>
        </div>
      </div>

      {/* Center — chat */}
      <div className="flex-1 min-w-0 flex flex-col bg-slate-50">
        <div className="h-12 shrink-0 border-b border-slate-200 bg-white px-4 flex items-center justify-between">
          <div className="min-w-0">
            <div className="text-sm font-semibold tracking-tight truncate">Consultation · {threadId}</div>
            <div className="text-xs font-mono text-slate-500">Streaming RAG · every claim cited · refuses if not verified</div>
          </div>
          <div className="flex items-center gap-2">
            <span className="hidden sm:inline-flex items-center gap-1.5 text-xs font-medium px-2.5 py-1 rounded-full border border-teal-200 bg-teal-50 text-teal-800">
              <span className="h-1.5 w-1.5 rounded-full bg-teal-600 animate-pulse" /> Live
            </span>
            <span className="text-xs font-mono px-2 py-1 rounded-md border border-slate-200 bg-white">{streaming ? "streaming…" : "ready"}</span>
          </div>
        </div>

        <div ref={scroller} className="flex-1 overflow-y-auto">
          <div className="max-w-[780px] mx-auto px-6 py-6 space-y-5">
            {messages.length === 0 && (
              <div className="rounded-lg border border-slate-200 bg-white p-5">
                <div className="text-sm font-semibold">Start a clinical query</div>
                <div className="mt-1 text-sm leading-6 text-slate-600">Ask like a practitioner — e.g. “First-line therapy for hypertension with CKD?” The assistant streams word-by-word with inline citations. Click a number to see the exact source chunk.</div>
                <div className="mt-3 flex flex-wrap gap-2">
                  {[
                    "First-line therapy for hypertension with CKD?",
                    "Contraindications for lisinopril?",
                    "Enoxaparin dosing for VTE prophylaxis?",
                  ].map((ex) => (
                    <button key={ex} onClick={() => setInput(ex)} className="text-xs font-medium px-2.5 py-1.5 rounded-md border border-slate-200 bg-slate-50 hover:bg-white">
                      {ex}
                    </button>
                  ))}
                </div>
              </div>
            )}

            {messages.map((m, i) => (
              <div key={i} className="space-y-1.5">
                <div className="text-xs font-medium tracking-wide text-slate-500">{m.role === "user" ? "You" : "Aura · verified"}</div>
                <div className={`${m.role === "user" ? "bg-slate-900 text-white border-slate-900" : "bg-white text-slate-900 border-slate-200"} rounded-lg border px-4 py-3 text-sm leading-6`}>
                  {m.role === "assistant" && m.citations?.length ? (
                    <InlineCitations text={m.content} citations={m.citations} onCite={setActiveCite} />
                  ) : (
                    m.content || (streaming && i === messages.length - 1 ? <span className="font-mono text-slate-400">▎</span> : null)
                  )}
                </div>
                {m.role === "assistant" && m.citations?.length ? (
                  <div className="flex flex-wrap gap-1.5">
                    {m.citations.map((c) => (
                      <button key={c.id} onClick={() => setActiveCite(c)} className="inline-flex items-center gap-1.5 rounded-md border border-slate-200 bg-white px-2 py-1 text-xs hover:bg-slate-50">
                        <span className="inline-flex h-4 min-w-4 items-center justify-center rounded bg-slate-900 text-white text-xs px-1">{c.idx}</span>
                        <span className="font-medium truncate max-w-[160px]">{c.doc_title}</span>
                        <span className="font-mono text-slate-500">p.{c.page}</span>
                      </button>
                    ))}
                  </div>
                ) : null}
              </div>
            ))}

            {streaming && <div className="text-xs font-mono text-slate-500">Receiving tokens…</div>}
          </div>
        </div>

        <div className="border-t border-slate-200 bg-white p-3">
          <div className="max-w-[780px] mx-auto flex gap-2">
            <input
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && !e.shiftKey && (e.preventDefault(), send())}
              placeholder="Ask a clinical question… (Enter to send)"
              className="flex-1 rounded-md border border-slate-300 bg-white px-3.5 py-2.5 text-sm outline-none focus:border-slate-400 focus:ring-2 focus:ring-slate-900/10"
            />
            <button onClick={send} disabled={streaming || !input.trim()} className="rounded-md bg-teal-600 px-4 py-2.5 text-sm font-medium text-white hover:bg-teal-700 disabled:opacity-40 disabled:cursor-not-allowed">
              Send
            </button>
          </div>
          <div className="max-w-[780px] mx-auto mt-2 flex justify-between text-xs font-mono text-slate-500">
            <span>Grounded in verified guidelines · citations required</span>
            <span>Enter ↵ · citations clickable</span>
          </div>
        </div>
      </div>

      {/* Right — evidence drawer */}
      {activeCite ? <CitationPanel citation={activeCite} onClose={() => setActiveCite(null)} /> : null}
    </div>
  );
}
