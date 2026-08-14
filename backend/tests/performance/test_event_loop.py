"""Event-loop responsiveness budgets.

The product's headline claim is concurrent SSE streaming. That only holds if no
request handler occupies the single uvicorn event loop while it works. Retrieval
does a DB round trip, an outbound embedding call, and -- on the TF-IDF path --
refits a TfidfVectorizer over the entire corpus on every query, and the provider
SDKs are synchronous. All of it originally ran directly on the loop, so concurrent
streams serialised behind each other instead of streaming together.

These tests measure that with a wall-clock proxy: a background ticker coroutine
increments a counter every few milliseconds. If the loop is blocked, the ticker
cannot run and the count stays at zero.
"""
import asyncio
import random
import time

import pytest

pytestmark = [pytest.mark.performance]

# Large enough that a TF-IDF refit takes real time; small enough to stay quick.
CORPUS_SIZE = 3000
CLINICAL_VOCAB = (
    "hypertension ckd lisinopril arb creatinine potassium enoxaparin vte prophylaxis "
    "dosing contraindication angioedema stenosis pregnancy aliskiren diabetes trial "
    "endpoint efficacy anticoagulation warfarin heparin renal hepatic titration"
).split()


@pytest.fixture
def big_corpus(monkeypatch):
    """Populate the in-memory chunk store with a realistically sized corpus."""
    from app import db

    rng = random.Random(1337)
    corpus = {}
    for i in range(CORPUS_SIZE):
        text = " ".join(rng.choice(CLINICAL_VOCAB) for _ in range(80))
        corpus[f"perf_{i}"] = {
            "id": f"perf_{i}", "doc_id": "perf_doc", "doc_title": "Perf Corpus",
            "page": 1 + i // 50, "text": text,
        }
    monkeypatch.setattr(db, "_chunks", corpus)
    return corpus


async def _ticker(stop: asyncio.Event, ticks: list, interval: float = 0.005):
    while not stop.is_set():
        ticks.append(time.perf_counter())
        await asyncio.sleep(interval)


async def _count_ticks_during(coro_factory) -> tuple[int, float]:
    """Run an awaitable and report how many times the loop got to run meanwhile."""
    stop = asyncio.Event()
    ticks: list = []
    task = asyncio.create_task(_ticker(stop, ticks))
    await asyncio.sleep(0.03)  # let the ticker settle
    baseline = len(ticks)
    started = time.perf_counter()
    await coro_factory()
    elapsed = time.perf_counter() - started
    during = len(ticks) - baseline
    stop.set()
    await task
    return during, elapsed


async def test_retrieve_async_does_not_block_the_event_loop(big_corpus):
    """retrieve_async must hand blocking work to a thread.

    Guards the regression where retrieve() ran inline on the loop: measured at
    this corpus size it froze the loop for ~45ms with zero ticks, so every other
    in-flight SSE stream stalled for the duration of each query.
    """
    from app.rag import retrieve_async

    ticks, elapsed = await _count_ticks_during(
        lambda: retrieve_async("hypertension ckd therapy", 5)
    )
    assert elapsed > 0.005, "corpus too small to be a meaningful measurement"
    assert ticks > 0, (
        f"event loop was blocked for the whole {elapsed*1000:.0f}ms retrieval "
        "(0 ticks) - blocking work is running on the loop instead of a thread"
    )


async def test_blocking_retrieve_is_measurably_worse(big_corpus):
    """Sanity-check the measurement itself.

    If the synchronous retrieve() did NOT starve the ticker, the test above would
    prove nothing. This asserts the instrument can actually detect blocking.
    """
    from app.rag import retrieve

    ticks, elapsed = await _count_ticks_during(
        lambda: asyncio.sleep(0) if False else _call_sync(retrieve)
    )
    assert elapsed > 0.005
    assert ticks == 0, (
        "the synchronous path did not starve the ticker, so this measurement "
        "cannot distinguish blocking from non-blocking work"
    )


async def _call_sync(fn):
    # Deliberately called ON the loop, which is the behaviour under test.
    fn("hypertension ckd therapy", 5)


async def test_loop_stays_responsive_under_concurrent_retrieval_load(big_corpus):
    """The loop must keep scheduling while several retrievals are in flight.

    Note what this does NOT claim: offloading to a thread does not make CPU-bound
    TF-IDF work finish faster, because the GIL serialises it anyway. The guarantee
    that matters for SSE is that the loop can still service other streams -- send
    tokens, notice disconnects, accept requests -- instead of freezing for the
    duration. That is what is measured here.
    """
    from app.rag import retrieve_async

    n = 4
    queries = [f"hypertension ckd therapy {i}" for i in range(n)]
    ticks, elapsed = await _count_ticks_during(
        lambda: asyncio.gather(*(retrieve_async(q, 5) for q in queries))
    )

    assert elapsed > 0.01, "workload too small to measure"
    # Roughly one tick per 5ms is available; require the loop to have run at least
    # a few times rather than being starved outright.
    assert ticks >= 2, (
        f"loop ran only {ticks} times during {elapsed*1000:.0f}ms of concurrent "
        "retrieval - it is being starved"
    )
