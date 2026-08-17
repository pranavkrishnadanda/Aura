"use client";
import type { Citation } from "@/lib/types";

/**
 * One retrieved source, shown verbatim.
 *
 * This used to be a drawer you had to click a citation to open, which buried the
 * one thing that distinguishes this product from any other chat interface. The
 * evidence is now permanently visible beneath the answer it supports, and the
 * marker in the prose and this card highlight together.
 */
export default function CitationPanel({
  citation,
  linked,
  onHover,
  onClose,
}: {
  citation: Citation | null;
  linked?: boolean;
  onHover?: (idx: number | null) => void;
  onClose?: () => void;
}) {
  if (!citation) return null;
  // Hovering a source to highlight its marker is a pointer-only enhancement on a
  // non-interactive card. The keyboard path to the same linkage already exists on
  // the marker button, which fires the identical callback on focus. Giving this
  // card an interactive role would announce a control that does not exist.
  return (
    // biome-ignore lint/a11y/noStaticElementInteractions: pointer-only enhancement; keyboard path is on the marker button
    <div
      data-testid="citation-panel"
      data-linked={linked ? "true" : "false"}
      onMouseEnter={() => onHover?.(citation.idx)}
      onMouseLeave={() => onHover?.(null)}
      className="evidence rounded-sm border bg-white px-4 py-3 transition-colors"
      style={{ borderColor: "var(--rule)" }}
    >
      <div className="flex items-baseline gap-2.5">
        <span
          className="shrink-0 text-[11px] font-medium tabular-nums"
          style={{ color: "var(--source)" }}
        >
          [{citation.idx}]
        </span>
        <span className="min-w-0 flex-1 truncate text-[13px] font-medium">
          {citation.doc_title}
        </span>
        <span className="shrink-0 text-[11px]" style={{ color: "var(--ink-soft)" }}>
          p.{citation.page}
        </span>
        {onClose ? (
          <button
            type="button"
            onClick={onClose}
            className="shrink-0 text-[11px] underline underline-offset-2"
            style={{ color: "var(--ink-soft)" }}
          >
            Close
          </button>
        ) : null}
      </div>

      {/* Verbatim, in the serif, because this is the source text itself -- the one
          thing on the page that is quoted rather than generated. */}
      <blockquote
        className="prose-clinical mt-2.5 border-l-2 pl-3 text-[15px]"
        style={{ borderColor: "var(--rule)", color: "var(--ink)" }}
      >
        {citation.chunk_text}
      </blockquote>

      <div
        className="mt-2.5 flex items-center gap-3 text-[11px] tabular-nums"
        style={{ color: "var(--ink-soft)" }}
      >
        <span>match {citation.score}</span>
        <span aria-hidden>·</span>
        <span className="truncate">{citation.doc_id}</span>
        <span aria-hidden>·</span>
        <span className="truncate">{citation.id}</span>
      </div>
    </div>
  );
}
