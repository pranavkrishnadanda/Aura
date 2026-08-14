"use client";
import { Citation } from "@/lib/types";

export default function CitationPanel({ citation, onClose }: { citation: Citation | null; onClose: () => void }) {
  if (!citation) return null;
  return (
    <div className="w-[380px] shrink-0 bg-white border-l border-slate-200 flex flex-col">
      <div className="h-12 px-4 border-b border-slate-200 flex items-center justify-between">
        <div className="flex items-center gap-2">
          <span className="inline-flex h-6 min-w-6 items-center justify-center rounded-md bg-teal-600 px-1.5 text-xs font-medium text-white">[{citation.idx}]</span>
          <span className="text-sm font-semibold tracking-tight">Evidence</span>
        </div>
        <button onClick={onClose} className="text-xs font-medium px-2.5 py-1.5 rounded-md border border-slate-200 hover:bg-slate-50">Close</button>
      </div>
      <div className="p-5 space-y-4 overflow-y-auto">
        <div>
          <div className="text-sm font-medium leading-tight">{citation.doc_title}</div>
          <div className="mt-1 flex items-center gap-2 text-xs">
            <span className="font-mono text-slate-500">p.{citation.page}</span>
            <span className="h-1 w-1 rounded-full bg-slate-300" />
            <span className="font-mono text-slate-500">score {citation.score}</span>
            <span className="h-1 w-1 rounded-full bg-slate-300" />
            <span className="font-mono text-slate-500 truncate">{citation.doc_id}</span>
          </div>
        </div>
        <div className="rounded-lg border border-slate-200 bg-slate-50 p-3.5 text-[13px] leading-6 text-slate-800">
          {citation.chunk_text}
        </div>
        <div className="rounded-md bg-amber-50 border border-amber-200 px-3 py-2 text-xs leading-5 text-amber-900">
          Verified chunk — rendered verbatim. No LLM rewrite.
        </div>
        <div className="pt-2 border-t border-slate-100">
          <div className="text-xs font-medium text-slate-700">Provenance</div>
          <div className="mt-2 grid grid-cols-2 gap-2 text-xs">
            <div className="rounded-md border border-slate-200 p-2"><div className="font-mono text-slate-500">Chunk ID</div><div className="font-medium truncate">{citation.id}</div></div>
            <div className="rounded-md border border-slate-200 p-2"><div className="font-mono text-slate-500">Document</div><div className="font-medium truncate">{citation.doc_id}</div></div>
          </div>
        </div>
      </div>
    </div>
  );
}
