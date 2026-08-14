export type Citation = { id: string; doc_id: string; doc_title: string; page: number; chunk_text: string; score: number; idx: number };
export type Message = { role: "user" | "assistant"; content: string; citations?: Citation[] };
