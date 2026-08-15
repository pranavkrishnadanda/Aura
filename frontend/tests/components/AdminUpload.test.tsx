import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { render, screen, fireEvent, act } from "@testing-library/react";
import AdminUpload from "@/components/AdminUpload";

/** Locate the hidden file input rendered by the compact variant. */
function fileInput(): HTMLInputElement {
  return document.querySelector('input[type="file"]') as HTMLInputElement;
}

function pdfFile(name = "protocol.pdf", sizeBytes = 1024): File {
  const file = new File([new Uint8Array(Math.min(sizeBytes, 1024))], name, { type: "application/pdf" });
  Object.defineProperty(file, "size", { value: sizeBytes });
  return file;
}

beforeEach(() => {
  vi.useFakeTimers();
});

afterEach(() => {
  // Always restore real timers, even if a test fails mid-way, so a stray fake
  // timer never bleeds into (and hangs) a later test.
  vi.useRealTimers();
});

describe("AdminUpload", () => {
  it("rejects a file over 50MB client-side without calling fetch", async () => {
    const fetchMock = vi.fn();
    vi.stubGlobal("fetch", fetchMock);
    render(<AdminUpload compact />);

    const big = pdfFile("huge.pdf", 51 * 1024 * 1024);
    await act(async () => {
      fireEvent.change(fileInput(), { target: { files: [big] } });
    });

    expect(fetchMock).not.toHaveBeenCalled();
    expect(screen.getByText(/exceeds the 50MB limit/)).toBeInTheDocument();
  });

  it("polls the job endpoint and renders the completed doc_title/pages/chunks", async () => {
    // Regression guard: the upload response only carries {job_id, status,
    // filename, bytes}. Reading doc_title/pages/chunks off *that* object (instead
    // of polling GET /documents/jobs/{id}) rendered
    // "undefined - undefined pages - undefined chunks indexed" on every success.
    let jobCalls = 0;
    const fetchMock = vi.fn(async (url: string) => {
      if (url.includes("/documents/upload")) {
        return {
          ok: true,
          json: async () => ({ job_id: "job_1", status: "queued", filename: "protocol.pdf", bytes: 1024 }),
        };
      }
      if (url.includes("/documents/jobs/job_1")) {
        jobCalls += 1;
        if (jobCalls === 1) {
          return { ok: true, json: async () => ({ status: "processing" }) };
        }
        return {
          ok: true,
          json: async () => ({
            status: "completed",
            doc_title: "Phase III Trial Protocol",
            pages: 42,
            chunks: 118,
            embedded: 118,
          }),
        };
      }
      throw new Error(`unexpected fetch url: ${url}`);
    });
    vi.stubGlobal("fetch", fetchMock);
    render(<AdminUpload compact />);

    await act(async () => {
      fireEvent.change(fileInput(), { target: { files: [pdfFile()] } });
    });
    // First poll tick: "processing", not yet terminal.
    await act(async () => {
      await vi.advanceTimersByTimeAsync(1000);
    });
    // Second poll tick: "completed".
    await act(async () => {
      await vi.advanceTimersByTimeAsync(1000);
    });

    expect(fetchMock.mock.calls.some((c) => String(c[0]).includes("/documents/jobs/job_1"))).toBe(true);
    expect(screen.getByText(/Phase III Trial Protocol/)).toBeInTheDocument();
    expect(screen.getByText(/42 pages/)).toBeInTheDocument();
    expect(screen.getByText(/118 chunks/)).toBeInTheDocument();
    expect(screen.queryByText(/undefined/)).not.toBeInTheDocument();
  });

  it('renders the error when the job status is "failed"', async () => {
    const fetchMock = vi.fn(async (url: string) => {
      if (url.includes("/documents/upload")) {
        return { ok: true, json: async () => ({ job_id: "job_2", status: "queued", filename: "x.pdf", bytes: 1 }) };
      }
      if (url.includes("/documents/jobs/job_2")) {
        return { ok: true, json: async () => ({ status: "failed", error: "corrupt PDF" }) };
      }
      throw new Error(`unexpected fetch url: ${url}`);
    });
    vi.stubGlobal("fetch", fetchMock);
    render(<AdminUpload compact />);

    await act(async () => {
      fireEvent.change(fileInput(), { target: { files: [pdfFile()] } });
    });
    await act(async () => {
      await vi.advanceTimersByTimeAsync(1000);
    });

    expect(screen.getByText(/Ingest failed: corrupt PDF/)).toBeInTheDocument();
  });

  it('renders the partial-indexing warning when the job status is "partial"', async () => {
    const fetchMock = vi.fn(async (url: string) => {
      if (url.includes("/documents/upload")) {
        return { ok: true, json: async () => ({ job_id: "job_3", status: "queued", filename: "x.pdf", bytes: 1 }) };
      }
      if (url.includes("/documents/jobs/job_3")) {
        return { ok: true, json: async () => ({ status: "partial", error: "3 pages failed OCR" }) };
      }
      throw new Error(`unexpected fetch url: ${url}`);
    });
    vi.stubGlobal("fetch", fetchMock);
    render(<AdminUpload compact />);

    await act(async () => {
      fireEvent.change(fileInput(), { target: { files: [pdfFile()] } });
    });
    await act(async () => {
      await vi.advanceTimersByTimeAsync(1000);
    });

    expect(
      screen.getByText(/Partially indexed: 3 pages failed OCR\. Some of this document is not searchable\./)
    ).toBeInTheDocument();
  });

  it("renders the server detail when the upload HTTP request fails", async () => {
    const fetchMock = vi.fn(async (url: string) => {
      if (url.includes("/documents/upload")) {
        return { ok: false, json: async () => ({ detail: "Unsupported file type" }) };
      }
      throw new Error(`unexpected fetch url: ${url}`);
    });
    vi.stubGlobal("fetch", fetchMock);
    render(<AdminUpload compact />);

    await act(async () => {
      fireEvent.change(fileInput(), { target: { files: [pdfFile()] } });
    });

    expect(screen.getByText(/Error: Unsupported file type/)).toBeInTheDocument();
    // Upload failed before a job existed, so nothing should be polled.
    expect(fetchMock.mock.calls.some((c) => String(c[0]).includes("/documents/jobs/"))).toBe(false);
  });

  it("shows a busy indicator while working and clears it once the job settles", async () => {
    let resolveUpload!: (v: any) => void;
    const uploadPromise = new Promise((res) => {
      resolveUpload = res;
    });
    const fetchMock = vi.fn(async (url: string) => {
      if (url.includes("/documents/upload")) return uploadPromise;
      if (url.includes("/documents/jobs/job_4")) {
        return { ok: true, json: async () => ({ status: "completed", doc_title: "Doc", pages: 1, chunks: 1, embedded: 1 }) };
      }
      throw new Error(`unexpected fetch url: ${url}`);
    });
    vi.stubGlobal("fetch", fetchMock);
    render(<AdminUpload compact />);

    expect(screen.getByText(/Add a PDF/)).toBeInTheDocument();

    // setBusy(true) runs synchronously before the upload request is awaited, so
    // the busy label is present immediately -- no need to wait for it (and
    // findByText's internal polling doesn't advance under fake timers anyway).
    act(() => {
      fireEvent.change(fileInput(), { target: { files: [pdfFile()] } });
    });
    expect(screen.getByText("Working…")).toBeInTheDocument();

    await act(async () => {
      resolveUpload({ ok: true, json: async () => ({ job_id: "job_4", status: "queued", filename: "x.pdf", bytes: 1 }) });
    });
    await act(async () => {
      await vi.advanceTimersByTimeAsync(1000);
    });

    expect(screen.getByText(/Add a PDF/)).toBeInTheDocument();
    expect(screen.queryByText("Working…")).not.toBeInTheDocument();
  });
});
