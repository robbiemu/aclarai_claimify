"""DSPy signatures for the Claimify pipeline components.

This module defines DSPy Signature classes for each stage of the Claimify pipeline:
Selection, Disambiguation, and Decomposition. These signatures make the inputs and
expected outputs explicit for use with DSPy modules.
"""

from typing import ClassVar

import dspy


class SelectionSignature(dspy.Signature):
    """Signature for the Selection stage of the Claimify pipeline.
    
    Determines if a sentence contains verifiable factual information suitable for claim extraction.
    """

    instructions: ClassVar[str] = (
        "You are the Selection stage in the Claimify pipeline. Given a context passage and a target sentence, "
        "output only a JSON object with keys selected (boolean), confidence (float between 0 and 1), and reasoning "
        "(short string explaining the decision). Example: {\n"
        "  \"selected\": true,\n"
        "  \"confidence\": 0.92,\n"
        "  \"reasoning\": \"States a specific, verifiable fact including the device model and launch date.\"\n"
        "}."
    )

    context_text = dspy.InputField(
        desc="Context (surrounding sentences) for the target sentence"
    )
    target_sentence = dspy.InputField(
        desc="The sentence to evaluate for selection"
    )
    selection_response_json = dspy.OutputField(
        desc="JSON string response with selection decision, confidence, and reasoning"
    )


class DisambiguationSignature(dspy.Signature):
    """Signature for the Disambiguation stage of the Claimify pipeline.

    Rewrites sentences to remove ambiguities and add context.
    """

    instructions: ClassVar[str] = (
        "You are the Disambiguation stage in the Claimify pipeline. Given a context passage and a target sentence, "
        "return a JSON object with disambiguated_text (string), changes_made (list of strings), and confidence (float in [0, 1]). "
        "Respond with JSON only. Avoid the common failure modes: unresolved_referent, hallucinated_detail, omitted_constraint, "
        "formatting_drift (multiple sentences), unsupported_resolution, and confidence_mismatch. Example: {\n"
        "  \"disambiguated_text\": \"The system failed with error code 500.\",\n"
        "  \"changes_made\": [\"Replaced 'It' with 'The system'\"],\n"
        "  \"confidence\": 0.92\n"
        "}."
    )

    context_text = dspy.InputField(
        desc="Context (surrounding sentences) for the target sentence"
    )
    target_sentence = dspy.InputField(
        desc="The sentence to disambiguate"
    )
    disambiguation_response_json = dspy.OutputField(
        desc="JSON string response with disambiguated text, changes made, and confidence"
    )


class DecompositionSignature(dspy.Signature):
    """Signature for the Decomposition stage of the Claimify pipeline.
    
    Breaks disambiguated sentences into atomic, self-contained claims.
    """

    instructions: ClassVar[str] = (
        "You are the Decomposition stage in the Claimify pipeline. Given a disambiguated sentence, output only a JSON "
        "object with key claim_candidates whose value is a list of objects containing text, is_atomic, is_self_contained, "
        "is_verifiable, passes_criteria, confidence, reasoning, and node_type. Avoid failure modes: non_atomic_claim, "
        "missing_key_claim, incorrect_metadata, off_topic_hallucination, structure_violation, and low_information_reasoning. "
        "Example: {\n"
        "  \"claim_candidates\": [\n"
        "    {\n"
        "      \"text\": \"The mission launched on July 16, 1969.\",\n"
        "      \"is_atomic\": true,\n"
        "      \"is_self_contained\": true,\n"
        "      \"is_verifiable\": true,\n"
        "      \"passes_criteria\": true,\n"
        "      \"confidence\": 0.95,\n"
        "      \"reasoning\": \"Single factual statement with explicit date.\",\n"
        "      \"node_type\": \"Claim\"\n"
        "    }\n"
        "  ]\n"
        "}."
    )

    disambiguated_text = dspy.InputField(
        desc="The disambiguated sentence text to decompose into claims"
    )
    decomposition_response_json = dspy.OutputField(
        desc="JSON string response with claim candidates and their evaluations"
    )
