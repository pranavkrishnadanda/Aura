"use client";
import { useState } from "react";
import { API_URL } from "@/lib/api";

export default function AdminUpload({ compact }: { compact?: boolean }) {
  const [status, setStatus] = useState("");
  const [busy, setBusy] = useState(false);
  async function onFile(e: React.ChangeEvent<HTMLInputElement>) {
    const f = e.target.files?.[0];
    if (!f) return;
    setBusy(true); setStatus("Ingesting — chunking → embedding → pgvector…");
    const fd = new FormData(); fd.append("file", f);
    try {
      const r = await fetch(`${API_URL}/api/v1/documents/upload`, { method: "POST", body: fd });
      const j = await r.json();
      if (r.ok) setStatus(`${j.doc_title} · ${j.pages} pages · ${j.chunks} chunks indexed`);
      else setStatus(`Error: ${j.detail || JSON.stringify(j)}`);
    } catch (err: any) { setStatus(`Error: ${err.message}`); }
    setBusy(false);
  }
  if (compact) {
    return (
      <div className="rounded-lg border border-slate-200 bg-white p-3">
        <div className="text-xs font-semibold">Ingest</div>
        <div className="text-xs text-slate-600 leading-4 mt-1">Upload 100-page PDFs. Auto-chunks to ~600 tokens.</div>
        <label className="mt-2 flex items-center justify-center rounded-md border border-dashed border-slate-300 bg-slate-50 px-3 py-2 text-xs font-medium hover:bg-slate-100 cursor-pointer">
          <input type="file" accept="application/pdf" onChange={onFile} className="hidden" />
          {busy ? "Working…" : "Choose PDF"}
        </label>
        <div className="mt-2 text-xs font-mono text-slate-500 leading-4 break-words">{status}</div>
      </div>
    );
  }
  return (
    <div className="rounded-lg border border-slate-200 bg-white p-4">
      <div className="text-sm font-semibold">Admin — Ingest trial PDFs</div>
      <input type="file" accept="application/pdf" onChange={onFile} className="mt-2 text-sm" />
      <div className="mt-2 text-xs font-mono text-slate-500">{status || "Batch ingest: encrypted at rest, <5min for 100 pages."}</div>
    </div>
  );
}
