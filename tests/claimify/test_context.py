import pytest
from aclarai_claimify.utils.context import create_context_window


@pytest.fixture
def sample_sentences():
    """Provides a sample list of sentences for testing."""
    return [
        "Sentence 0: The first.",
        "Sentence 1: The second.",
        "Sentence 2: The third.",
        "Sentence 3: The fourth.",
        "Sentence 4: The fifth.",
        "Sentence 5: The sixth.",
        "Sentence 6: The seventh.",
    ]


def test_create_context_window_middle(sample_sentences):
    """Tests creating a context window from the middle of a document."""
    # Target is "Sentence 3" (index 3), k=2
    # Expects sentences 1, 2 (before) and 4, 5 (after)
    context = create_context_window(sample_sentences, 3, 2)
    expected = (
        "[1] Sentence 1: The second.\n"
        "[2] Sentence 2: The third.\n"
        "[4] Sentence 4: The fifth.\n"
        "[5] Sentence 5: The sixth."
    )
    assert context == expected


def test_create_context_window_start_edge(sample_sentences):
    """Tests creating a context window for the first sentence."""
    # Target is "Sentence 0" (index 0), k=2
    # Expects sentences 1, 2 (after)
    context = create_context_window(sample_sentences, 0, 2)
    expected = "[1] Sentence 1: The second.\n[2] Sentence 2: The third."
    assert context == expected


def test_create_context_window_end_edge(sample_sentences):
    """Tests creating a context window for the last sentence."""
    # Target is "Sentence 6" (index 6), k=2
    # Expects sentences 4, 5 (before)
    context = create_context_window(sample_sentences, 6, 2)
    expected = "[4] Sentence 4: The fifth.\n[5] Sentence 5: The sixth."
    assert context == expected


def test_create_context_window_k_is_zero(sample_sentences):
    """Tests that k=0 results in an empty context string."""
    context = create_context_window(sample_sentences, 3, 0)
    assert context == ""


def test_create_context_window_large_k(sample_sentences):
    """Tests when k is larger than available sentences."""
    # Target is "Sentence 2" (index 2), k=10
    # Expects all other sentences (0, 1, 3, 4, 5, 6)
    context = create_context_window(sample_sentences, 2, 10)
    expected = (
        "[0] Sentence 0: The first.\n"
        "[1] Sentence 1: The second.\n"
        "[3] Sentence 3: The fourth.\n"
        "[4] Sentence 4: The fifth.\n"
        "[5] Sentence 5: The sixth.\n"
        "[6] Sentence 6: The seventh."
    )
    assert context == expected


def test_create_context_window_negative_k_raises_error(sample_sentences):
    """Tests that a negative k raises a ValueError."""
    with pytest.raises(
        ValueError, match="Context window size k must be a non-negative integer."
    ):
        create_context_window(sample_sentences, 3, -1)


def test_create_context_window_empty_sentence_list():
    """Tests that an empty sentence list results in an empty context."""
    context = create_context_window([], 0, 5)
    assert context == ""


def test_create_context_window_single_sentence_list():
    """Tests that a list with one sentence results in an empty context."""
    context = create_context_window(["This is the only sentence."], 0, 5)
    assert context == ""
