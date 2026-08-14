import { describe, it, expect, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import CitationPanel from "@/components/CitationPanel";
import type { Citation } from "@/lib/types";

const citation: Citation = {
  id: "chunk_42",
  doc_id: "doc_7",
  doc_title: "ACC/AHA Hypertension Guideline",
  page: 12,
  chunk_text: "First-line therapy for hypertension with CKD is an ACE inhibitor or ARB.",
  score: 0.873,
  idx: 3,
};

describe("CitationPanel", () => {
  it("renders doc_title, page, score, doc_id, chunk_text and the citation index", () => {
    render(<CitationPanel citation={citation} onClose={vi.fn()} />);

    expect(screen.getByText("ACC/AHA Hypertension Guideline")).toBeInTheDocument();
    expect(screen.getByText("p.12")).toBeInTheDocument();
    expect(screen.getByText("score 0.873")).toBeInTheDocument();
    expect(screen.getAllByText("doc_7").length).toBeGreaterThan(0);
    expect(
      screen.getByText("First-line therapy for hypertension with CKD is an ACE inhibitor or ARB.")
    ).toBeInTheDocument();
    expect(screen.getByText("[3]")).toBeInTheDocument();
  });

  it("returns null when citation is null", () => {
    const { container } = render(<CitationPanel citation={null} onClose={vi.fn()} />);
    expect(container).toBeEmptyDOMElement();
  });

  it("fires onClose when the Close button is clicked", async () => {
    const user = userEvent.setup();
    const onClose = vi.fn();
    render(<CitationPanel citation={citation} onClose={onClose} />);

    await user.click(screen.getByRole("button", { name: "Close" }));

    expect(onClose).toHaveBeenCalledTimes(1);
  });

  it("renders chunk_text containing HTML/script markup as plain text, not DOM nodes", () => {
    // Regression guard: chunk_text is untrusted document content rendered via
    // {citation.chunk_text} JSX interpolation. If it were ever rendered with
    // dangerouslySetInnerHTML or similar, this payload would execute/inject.
    const malicious: Citation = {
      ...citation,
      chunk_text: 'Ref <img src=x onerror=alert(1)> and <script>alert(1)</script> below',
    };
    const { container } = render(<CitationPanel citation={malicious} onClose={vi.fn()} />);

    expect(container.querySelector("script")).toBeNull();
    expect(container.querySelector("img")).toBeNull();
    expect(
      screen.getByText('Ref <img src=x onerror=alert(1)> and <script>alert(1)</script> below')
    ).toBeInTheDocument();
  });
});
