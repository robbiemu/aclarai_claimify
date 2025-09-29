"""Shared failure-mode definitions for Claimify dataset generation and validation."""

DISAMBIGUATION_FAILURE_MODES = [
    "unresolved_referent",
    "hallucinated_detail",
    "omitted_constraint",
    "formatting_drift",
    "unsupported_resolution",
    "confidence_mismatch",
]

DISAMBIGUATION_FAILURE_DESCRIPTIONS = {
    "unresolved_referent": "Leave at least one pronoun or demonstrative unresolved so the sentence is not self-contained.",
    "hallucinated_detail": "Add a specific detail (name, place, time, motivation) that does not appear in the provided context.",
    "omitted_constraint": "Remove or alter hedges, negations, or temporal qualifiers from the original sentence, changing its scope.",
    "formatting_drift": "Return multiple sentences or otherwise break the one-sentence requirement (e.g., add list markers).",
    "unsupported_resolution": "Replace a pronoun with a vague placeholder even though the context provides a precise referent.",
    "confidence_mismatch": "Keep the rewrite low quality while still reporting confidence >= 0.85.",
}

DECOMPOSITION_FAILURE_MODES = [
    "non_atomic_claim",
    "missing_key_claim",
    "incorrect_metadata",
    "off_topic_hallucination",
    "structure_violation",
    "low_information_reasoning",
]

DECOMPOSITION_FAILURE_DESCRIPTIONS = {
    "non_atomic_claim": "Combine multiple ideas into a single claim candidate instead of splitting them.",
    "missing_key_claim": "Omit at least one major idea from the sentence so no candidate captures it.",
    "incorrect_metadata": "Set the quality flags (is_atomic/is_self_contained/is_verifiable/passes_criteria) inconsistently with the claim text.",
    "off_topic_hallucination": "Introduce a claim candidate that includes entities or facts absent from the source sentence.",
    "structure_violation": "Return an empty list or otherwise structure claim_candidates in a way that makes downstream parsing unhelpful while still valid JSON.",
    "low_information_reasoning": "Keep reasoning fields blank or generic so they provide no guidance.",
}
