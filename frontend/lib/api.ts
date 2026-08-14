export const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export type StreamCallbacks = {
  onMeta: (citations: any[], isRefusal: boolean) => void;
  onToken: (token: string) => void;
  onDone: (fullText: string) => void;
  onError: (err: string) => void;
};

export async function streamChat(message: string, threadId: string, cbs: StreamCallbacks) {
  const res = await fetch(`${API_URL}/api/v1/chat/stream`, {
    method: "POST",
    headers: { "Content-Type": "application/json", Accept: "text/event-stream" },
    body: JSON.stringify({ message, thread_id: threadId }),
  });
  if (!res.ok || !res.body) {
    cbs.onError(`HTTP ${res.status}`);
    return;
  }
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    // handle both \n\n and \r\n\r\n, and keep incomplete chunk in buffer
    buffer = buffer.replace(/\r\n/g, "\n");
    const events = buffer.split("\n\n");
    buffer = events.pop() || "";
    for (const evt of events) {
      const lines = evt.split("\n");
      let event = "message";
      let data = "";
      for (const l of lines) {
        const t = l.trim();
        if (t.startsWith("event:")) event = t.slice(6).trim();
        if (t.startsWith("data:")) data = t.slice(5).trim();
      }
      if (!data) continue;
      try {
        const json = JSON.parse(data);
        if (event === "meta") cbs.onMeta(json.citations || [], json.is_refusal);
        if (event === "token") cbs.onToken(json.token);
        if (event === "done") cbs.onDone(json.full_text);
      } catch {}
    }
  }
  // flush any trailing buffered event (for proxies that don't send final \n\n)
  if (buffer.trim()) {
    try {
      const json = JSON.parse(buffer.split("data:")[1] || "{}");
      if (json.token) cbs.onToken(json.token);
    } catch {}
  }
}
