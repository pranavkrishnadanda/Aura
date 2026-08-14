export const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export type StreamCallbacks = {
  onMeta: (citations: any[], isRefusal: boolean) => void;
  onToken: (token: string) => void;
  onDone: (fullText: string) => void;
  onError: (err: string) => void;
};

export async function streamChat(
  message: string,
  threadId: string,
  cbs: StreamCallbacks,
  signal?: AbortSignal
) {
  // Every exit path must report exactly once, or the caller's `streaming` flag is
  // never cleared and the composer stays disabled for the rest of the session.
  let settled = false;
  const finish = (fn: () => void) => {
    if (!settled) {
      settled = true;
      fn();
    }
  };

  try {
    const res = await fetch(`${API_URL}/api/v1/chat/stream`, {
      method: "POST",
      headers: { "Content-Type": "application/json", Accept: "text/event-stream" },
      body: JSON.stringify({ message, thread_id: threadId }),
      signal,
    });
    if (!res.ok || !res.body) {
      // Surface the server's own message (rate limit, validation) rather than a
      // bare status code.
      let detail = `HTTP ${res.status}`;
      try {
        const j = await res.json();
        if (j?.detail) detail = j.detail;
      } catch {}
      finish(() => cbs.onError(detail));
      return;
    }

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    let full = "";

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      // Decode incrementally so a multi-byte character split across two network
      // chunks is not corrupted, and only normalise the newly arrived text --
      // re-scanning the whole buffer each pass was quadratic on long answers.
      buffer += decoder.decode(value, { stream: true }).replace(/\r\n/g, "\n");
      const events = buffer.split("\n\n");
      buffer = events.pop() || "";
      for (const evt of events) {
        let event = "message";
        const dataLines: string[] = [];
        for (const l of evt.split("\n")) {
          if (l.startsWith("event:")) event = l.slice(6).trim();
          // SSE allows repeated data: lines for one event; they concatenate.
          else if (l.startsWith("data:")) dataLines.push(l.slice(5).trimStart());
        }
        const data = dataLines.join("\n");
        if (!data) continue;
        let json: any;
        try {
          json = JSON.parse(data);
        } catch {
          continue;
        }
        if (event === "meta") cbs.onMeta(json.citations || [], json.is_refusal);
        else if (event === "token") {
          full += json.token ?? "";
          cbs.onToken(json.token ?? "");
        } else if (event === "done") finish(() => cbs.onDone(json.full_text ?? full));
        // The backend emits this when generation throws mid-stream. It was
        // previously ignored entirely, so the UI just stopped receiving tokens
        // and sat on "streaming…" with no explanation.
        else if (event === "error") finish(() => cbs.onError(json.detail || "stream error"));
      }
    }
    // Stream ended without a done event (proxy cut, backend restart, Render sleep).
    finish(() => (full ? cbs.onDone(full) : cbs.onError("Connection closed before a response completed")));
  } catch (err: any) {
    if (err?.name === "AbortError") {
      finish(() => cbs.onDone(""));
      return;
    }
    finish(() => cbs.onError(err?.message || "Network error"));
  }
}
