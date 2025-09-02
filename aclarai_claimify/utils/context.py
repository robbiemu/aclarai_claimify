"""Context generation utilities for Claimify."""

from typing import List


def create_context_window(
    all_sentences: List[str], target_index: int, k: int
) -> str:
    """Creates a context window string around a target sentence.

    Args:
        all_sentences: A list of all sentences in the document.
        target_index: The index of the target sentence.
        k: The number of sentences to include before and after the target.

    Returns:
        A formatted string representing the local context, with each
        sentence indexed relative to the original document.
        Returns an empty string if k is 0 or there are no context sentences.

    Raises:
        ValueError: If k is negative.
    """
    if k < 0:
        raise ValueError("Context window size k must be a non-negative integer.")

    if not all_sentences or k == 0:
        return ""

    start_index = max(0, target_index - k)
    end_index = min(len(all_sentences), target_index + k + 1)

    context_parts = []
    for i in range(start_index, end_index):
        if i == target_index:
            continue
        context_parts.append(f"[{i}] {all_sentences[i]}")

    return "\n".join(context_parts)
