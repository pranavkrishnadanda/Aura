"use client";
import { useState } from "react";
import { API_URL, authHeaders } from "@/lib/api";

/** Fallback only. The real limit comes from the server via /health, so the two
 *  cannot disagree -- this constant had already drifted to 10x the backend's,
 *  which means an oversized upload is only rejected after the whole body is sent. */
const DEFAULT_MAX_PDF_MB = 50;
const POLL_MS = 1000;
const POLL_TIMEOUT_MS = 5 * 60 * 1000;

export default function AdminUpload({
  compact,
  maxPdfMb,
}: {
  compact?: boolean;
  maxPdfMb?: number;
}) {
  const MAX_PDF_MB = maxPdfMb ?? DEFAULT_MAX_PDF_MB;
  const [status, setStatus] = useState("");
  const [busy, setBusy] = useState(false);

  async function onFile(e: React.ChangeEvent<HTMLInputElement>) {
    const f = e.target.files?.[0];
    if (!f) return;
    // Reject oversize files before spending the upload; the backend would reject
    // it anyway, but only after the whole body has been sent.
    if (f.size > MAX_PDF_MB * 1024 * 1024) {
      setStatus(`Error: ${(f.size / 1024 / 1024).toFixed(1)}MB exceeds the ${MAX_PDF_MB}MB limit`);
      e.target.value = "";
      return;
    }

    setBusy(true);
    setStatus("Uploading…");
    const fd = new FormData();
    fd.append("file", f);
    try {
      const r = await fetch(`${API_URL}/api/v1/documents/upload`, {
        method: "POST",
        body: fd,
        headers: authHeaders(),
      });
      // 403 means this deployment refuses uploads from the caller's tier. With
      // auth off every caller is anonymous, so the server's "requires an API key"
      // detail would point the visitor at something they cannot do.
      if (r.status === 403) {
        setStatus(
          "Uploads are disabled on this deployment. Adding documents requires an authorised API key."
        );
        return;
      }
      if (!r.ok) {
        // Error bodies are not guaranteed to be JSON (a proxy 413 or 502 returns
        // HTML), so fall back to the status code rather than a parse error.
        let detail = `HTTP ${r.status}`;
        try {
          const j = await r.json();
          if (j?.detail)
            detail = typeof j.detail === "string" ? j.detail : JSON.stringify(j.detail);
        } catch {
          // keep the status-code fallback
        }
        setStatus(`Error: ${detail}`);
        return;
      }
      // Upload only enqueues work: the response carries {job_id, status, filename,
      // bytes}. Document stats and background-task failures come from polling the job.
      const j = await r.json();
      await pollJob(j.job_id, j.filename);
    } catch (err: any) {
      setStatus(`Error: ${err.message}`);
    } finally {
      setBusy(false);
      e.target.value = ""; // allow re-selecting the same file
    }
  }

  async function pollJob(jobId: string, filename: string) {
    const deadline = Date.now() + POLL_TIMEOUT_MS;
    setStatus(`Ingesting ${filename} — chunking → embedding…`);
    while (Date.now() < deadline) {
      await new Promise((res) => setTimeout(res, POLL_MS));
      let job: any;
      try {
        const r = await fetch(`${API_URL}/api/v1/documents/jobs/${jobId}`, {
          headers: authHeaders(),
        });
        if (!r.ok) {
          setStatus(`Error: job ${jobId} not found (HTTP ${r.status})`);
          return;
        }
        job = await r.json();
      } catch (err: any) {
        setStatus(`Error: lost contact while ingesting — ${err.message}`);
        return;
      }
      if (job.status === "completed") {
        const embedded =
          job.embedded > 0 ? `${job.embedded} embedded` : "TF-IDF only (no embeddings)";
        setStatus(`${job.doc_title} · ${job.pages} pages · ${job.chunks} chunks · ${embedded}`);
        return;
      }
      if (job.status === "partial") {
        setStatus(`Partially indexed: ${job.error}. Some of this document is not searchable.`);
        return;
      }
      if (job.status === "failed") {
        setStatus(`Ingest failed: ${job.error || "unknown error"}`);
        return;
      }
    }
    setStatus("Still ingesting — taking longer than expected. Check back shortly.");
  }
  if (compact) {
    return (
      <div>
        <div className="label">Corpus</div>
        <label
          className="mt-2 flex cursor-pointer items-center justify-center rounded-sm border border-dashed px-3 py-2 text-[11px] transition-colors hover:bg-[var(--paper)]"
          style={{ borderColor: "var(--rule)" }}
        >
          <input type="file" accept="application/pdf" onChange={onFile} className="hidden" />
          {busy ? "Working…" : `Add a PDF (max ${MAX_PDF_MB}MB)`}
        </label>
        {status ? (
          <div
            className="mt-2 break-words text-[11px] leading-4"
            style={{ color: "var(--ink-soft)" }}
          >
            {status}
          </div>
        ) : null}
      </div>
    );
  }
  return (
    <div className="rounded-sm border bg-white p-4" style={{ borderColor: "var(--rule)" }}>
      <div className="label">Add to corpus</div>
      <input type="file" accept="application/pdf" onChange={onFile} className="mt-2 text-[12px]" />
      <div className="mt-2 text-[11px]" style={{ color: "var(--ink-soft)" }}>
        {status || `PDF up to ${MAX_PDF_MB}MB. Chunked and indexed in the background.`}
      </div>
    </div>
  );
}
