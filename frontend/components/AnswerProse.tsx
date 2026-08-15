"use client";
import { Citation, CitationCheck } from "@/lib/types";

/** Splits the model's prose on citation markers and renders each as a control
 *  linked to its evidence card. A marker with no matching source is shown as
 *  flagged text rather than a link -- it points at nothing. */
export function Prose({
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
export function Provenance({ citations, check }: { citations: Citation[]; check?: CitationCheck | null }) {
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
