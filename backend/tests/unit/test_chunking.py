"""Unit tests for app.ingest.chunk_text.

chunk_text splits raw page text into overlapping word-windows. The step between
windows is `size - overlap`; if overlap >= size that step is <= 0 and the index
never advances, so the loop spins forever. The real fix clamps overlap to at most
size - 1 (see app/ingest.py comment above the clamp). These tests pin that
behavior down and also verify the >50-char drop rule and overlap math.
"""
import pytest

from app.ingest import chunk_text


@pytest.mark.unit
def test_overlap_greater_than_size_does_not_hang():
    """Regression: overlap > size used to make step <= 0 -> infinite loop.

    With size=5, overlap=10 (overlap > size), the clamp forces overlap to
    size-1=4, giving step=1. The call must return promptly instead of hanging.
    """
    words = [f"word{i}" for i in range(20)]
    text = " ".join(words)
    # Each chunk of 5 words is well under 50 chars, so nothing is retained --
    # what matters here is that the call terminates at all.
    result = chunk_text(text, size=5, overlap=10)
    assert isinstance(result, list)


@pytest.mark.unit
def test_overlap_equal_to_size_does_not_hang():
    """Regression: overlap == size also makes step <= 0 before the clamp."""
    words = [f"word{i}" for i in range(20)]
    text = " ".join(words)
    result = chunk_text(text, size=5, overlap=5)
    assert isinstance(result, list)


@pytest.mark.unit
def test_overlap_equal_to_size_clamped_step_is_one():
    """overlap==size clamps to overlap=size-1, so step=1 and consecutive chunks
    share size-1 words."""
    # Use long words so each 5-word window exceeds 50 chars and is kept.
    words = [f"medication{i:02d}" for i in range(10)]
    text = " ".join(words)
    result = chunk_text(text, size=5, overlap=5)
    assert len(result) >= 2
    first_words = result[0].split()
    second_words = result[1].split()
    # step=1 means second chunk starts one word after the first.
    assert first_words[1:] == second_words[:-1]


@pytest.mark.unit
def test_overlap_zero_no_repeated_words_between_chunks():
    """An explicit overlap=0 must mean no overlap.

    Guards a falsy-or bug: `overlap = overlap or settings.CHUNK_OVERLAP` treated a
    caller-supplied 0 as "not supplied" and substituted the configured default
    (100), so requesting non-overlapping chunks silently produced overlapping ones.
    The same pattern affected size=0.
    """
    words = [f"medication{i:02d}" for i in range(15)]
    text = " ".join(words)
    result = chunk_text(text, size=5, overlap=0)
    assert len(result) == 3
    all_words = " ".join(result).split()
    assert all_words == words


@pytest.mark.unit
def test_size_one_produces_single_word_chunks_but_all_dropped_as_too_short():
    """size=1 with short words never crosses the 50-char keep threshold."""
    text = "a b c d e f g h i j"
    result = chunk_text(text, size=1, overlap=0)
    assert result == []


@pytest.mark.unit
def test_size_one_with_long_words_kept_individually():
    words = ["x" * 60, "y" * 60, "z" * 60]
    text = " ".join(words)
    result = chunk_text(text, size=1, overlap=0)
    assert result == words


@pytest.mark.unit
def test_empty_text_returns_empty_list():
    assert chunk_text("", size=100, overlap=20) == []


@pytest.mark.unit
def test_whitespace_only_text_returns_empty_list():
    assert chunk_text("   \n\t  ", size=100, overlap=20) == []


@pytest.mark.unit
def test_text_shorter_than_50_chars_is_dropped():
    text = "short clinical note"  # well under 50 chars
    assert len(text) < 50
    result = chunk_text(text, size=100, overlap=0)
    assert result == []


@pytest.mark.unit
def test_text_at_51_chars_is_kept():
    # Build text just over the 50-char threshold.
    text = "a" * 51
    result = chunk_text(text, size=100, overlap=0)
    assert result == [text]


@pytest.mark.unit
def test_chunks_overlap_by_configured_word_count():
    words = [f"term{i:03d}" for i in range(30)]
    text = " ".join(words)
    size, overlap = 10, 3
    result = chunk_text(text, size=size, overlap=overlap)
    step = size - overlap
    # Recompute expected windows the same way chunk_text does.
    expected = []
    i = 0
    while i < len(words):
        c = " ".join(words[i:i + size])
        if len(c.strip()) > 50:
            expected.append(c)
        i += step
    assert result == expected
    # Explicitly confirm the overlap: last `overlap` words of chunk N are the
    # first `overlap` words of chunk N+1.
    first = result[0].split()
    second = result[1].split()
    assert first[-overlap:] == second[:overlap]


@pytest.mark.unit
def test_negative_overlap_env_value_is_clamped_to_zero():
    """Regression: a negative CHUNK_OVERLAP from env must not produce a
    negative step (which would walk backwards / never terminate cleanly)."""
    words = [f"medication{i:02d}" for i in range(15)]
    text = " ".join(words)
    result = chunk_text(text, size=5, overlap=-10)
    # Clamped overlap=0 behaves identically to the explicit overlap=0 case.
    assert len(result) == 3
    all_words = " ".join(result).split()
    assert all_words == words


@pytest.mark.unit
def test_negative_size_env_value_is_clamped_to_one():
    text = "a" * 60
    result = chunk_text(text, size=-5, overlap=0)
    # size clamped to 1 -> single "word" (the whole 60-char blob, no spaces)
    assert result == [text]


@pytest.mark.unit
def test_defaults_come_from_settings_when_not_passed(monkeypatch):
    from app import ingest

    monkeypatch.setattr(ingest.settings, "CHUNK_SIZE", 4)
    monkeypatch.setattr(ingest.settings, "CHUNK_OVERLAP", 0)
    words = [f"medication{i:02d}" for i in range(8)]
    text = " ".join(words)
    result = chunk_text(text)
    assert len(result) == 2
