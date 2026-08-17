export type Citation = {
  id: string;
  doc_id: string;
  doc_title: string;
  page: number;
  chunk_text: string;
  score: number;
  idx: number;
};

/** Result of checking the model's citation markers against the sources it was given.
 *  `invalid_markers` are references to sources that do not exist -- a fabricated
 *  citation, which reads as verifiable but is not. */
export type CitationCheck = { ok: boolean; invalid_markers: number[]; cited: boolean };

export type Message = {
  role: "user" | "assistant";
  content: string;
  citations?: Citation[];
  check?: CitationCheck | null;
};
