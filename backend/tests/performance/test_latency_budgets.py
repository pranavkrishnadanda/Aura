"""Latency and resource budgets.

Thresholds here are deliberately loose: they exist to catch order-of-magnitude
regressions (an accidental O(n^2), an unbounded cache, a per-chunk DB commit),
not to police normal machine-to-machine variance.
"""
import time

import pytest
from fastapi.testclient import TestClient

pytestmark = [pytest.mark.performance]


@pytest.fixture
def client():
    from sse_starlette.sse import AppStatus
    AppStatus.should_exit_event = None
    from app.main import app
    return TestClient(app)


def test_time_to_first_event_is_fast(client):
    """The meta frame must be flushed before generation begins.

    TTFT is the product's headline number. meta carries the citations and is
    yielded before the first token, so the client can render the evidence panel
    while the answer streams.
    """
    from sse_starlette.sse import AppStatus

    AppStatus.should_exit_event = None
    started = time.perf_counter()
    first_event_at = None
    first_event = ""
    with client.stream("POST", "/api/v1/chat/stream",
                       json={"message": "What is first-line therapy for hypertension with CKD?",
                             "thread_id": "perf_ttft"}) as r:
        # Single pass: record when the first frame lands, then drain the rest so
        # the generator completes cleanly. The stream cannot be iterated twice.
        for line in r.iter_lines():
            if first_event_at is None and line.startswith("event:"):
                first_event_at = time.perf_counter()
                first_event = line

    assert first_event_at is not None, "no event was ever emitted"
    ttft_ms = (first_event_at - started) * 1000
    assert "meta" in first_event, f"first frame should be meta, got {first_event!r}"
    assert ttft_ms < 2000, f"time to first event {ttft_ms:.0f}ms exceeds the 2000ms budget"


def test_health_is_cheap(client):
    """/health is polled by uptime monitors; it must not do heavy work."""
    started = time.perf_counter()
    for _ in range(20):
        assert client.get("/health").status_code == 200
    per_call_ms = (time.perf_counter() - started) * 1000 / 20
    assert per_call_ms < 250, f"/health averages {per_call_ms:.0f}ms per call"


def test_chunking_throughput_is_linear():
    """chunk_text must stay roughly linear in input size.

    A quadratic chunker would make large PDF ingest wildly slow, and the stride
    arithmetic here previously had a clamp bug that could not terminate at all.
    """
    from app.ingest import chunk_text

    small = " ".join(f"word{i}" for i in range(5_000))
    large = " ".join(f"word{i}" for i in range(50_000))

    t0 = time.perf_counter(); chunk_text(small); t_small = time.perf_counter() - t0
    t0 = time.perf_counter(); chunk_text(large); t_large = time.perf_counter() - t0

    # 10x the input should cost well under 50x the time if behaviour is linear-ish.
    assert t_large < max(t_small * 50, 0.5), (
        f"10x input took {t_large/max(t_small,1e-6):.1f}x the time - superlinear chunking"
    )


def test_embed_cache_stays_bounded():
    """The query embedding cache must not grow without limit.

    It is keyed by raw query text, so an unbounded cache both leaks memory and
    retains clinical queries for the lifetime of the process.
    """
    from app import rag

    rag._embed_cache.clear()
    for i in range(rag._EMBED_CACHE_MAX * 3):
        rag._embed_cache[f"query number {i}"] = ([0.0], time.time())
        rag._prune_embed_cache()

    assert len(rag._embed_cache) <= rag._EMBED_CACHE_MAX, (
        f"cache grew to {len(rag._embed_cache)}, above the {rag._EMBED_CACHE_MAX} cap"
    )
    rag._embed_cache.clear()


def test_job_table_stays_bounded():
    """_jobs is a module-level dict; a long-lived instance must not leak it."""
    from app import ingest

    ingest._jobs.clear()
    for _ in range(ingest._MAX_JOBS * 2):
        ingest.create_job()

    assert len(ingest._jobs) <= ingest._MAX_JOBS, (
        f"job table grew to {len(ingest._jobs)}, above the {ingest._MAX_JOBS} cap"
    )
    ingest._jobs.clear()
