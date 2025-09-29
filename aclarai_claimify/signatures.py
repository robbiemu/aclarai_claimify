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
        '  "selected": true,\n'
        '  "confidence": 0.92,\n'
        '  "reasoning": "States a specific, verifiable fact including the device model and launch date."\n'
        "}."
    )

    context_text = dspy.InputField(
        desc="Context (surrounding sentences) for the target sentence"
    )
    target_sentence = dspy.InputField(desc="The sentence to evaluate for selection")
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
        '  "disambiguated_text": "The system failed with error code 500.",\n'
        "  \"changes_made\": [\"Replaced 'It' with 'The system'\"],\n"
        '  "confidence": 0.92\n'
        "}."
    )

    context_text = dspy.InputField(
        desc="Context (surrounding sentences) for the target sentence"
    )
    target_sentence = dspy.InputField(desc="The sentence to disambiguate")
    disambiguation_response_json = dspy.OutputField(
        desc="JSON string response with disambiguated text, changes made, and confidence"
    )


class DecompositionSignature(dspy.Signature):
    """Signature for the Decomposition stage of the Claimify pipeline.

    Breaks disambiguated sentences into atomic, self-contained claims.
    """

    instructions: ClassVar[str] = (
        "You are an expert at extracting atomic claims from text. Your task is to break down sentences into individual, "
        "verifiable claims that meet strict quality criteria. Each claim must be atomic (a single fact), self-contained "
        "(no ambiguous references), and verifiable (factually checkable).\n\n"
        "Analyze the input sentence and evaluate each potential claim against the three criteria. Your output must be "
        "a single, valid JSON object and nothing else.\n\n"
        'The JSON object must have a single key, "claim_candidates", mapped to a list of claim objects. Each object in '
        "the list MUST contain exactly these 8 keys:\n"
        "* `text` (string): The extracted claim text.\n"
        "* `is_atomic` (boolean): True if the claim contains exactly one verifiable fact (no compound statements).\n"
        "* `is_self_contained` (boolean): True if the claim has no ambiguous pronouns or references.\n"
        "* `is_verifiable` (boolean): True if the claim contains specific, factual information that can be fact-checked.\n"
        "* `passes_criteria` (boolean): True only if `is_atomic`, `is_self_contained`, AND `is_verifiable` are all true.\n"
        "* `confidence` (float): Your confidence in the evaluation, from 0.0 to 1.0.\n"
        "* `reasoning` (string): A brief explanation for the evaluation, especially for claims that fail.\n"
        '* `node_type` (string): Set to "Claim" if `passes_criteria` is true, otherwise set to "Sentence".\n\n'
        "Example of a VALID claim object:\n"
        "{\n"
        '  "text": "The error rate increased to 25% after deployment.",\n'
        '  "is_atomic": true,\n'
        '  "is_self_contained": true,\n'
        '  "is_verifiable": true,\n'
        '  "passes_criteria": true,\n'
        '  "confidence": 0.98,\n'
        '  "reasoning": "This is a single, specific, and verifiable factual statement.",\n'
        '  "node_type": "Claim"\n'
        "}\n\n"
        "Example of an INVALID claim object:\n"
        "{\n"
        '  "text": "The system worked but was slow.",\n'
        '  "is_atomic": false,\n'
        '  "is_self_contained": true,\n'
        '  "is_verifiable": true,\n'
        '  "passes_criteria": false,\n'
        '  "confidence": 0.9,\n'
        '  "reasoning": "This is a compound statement, bundling two separate claims about functionality and performance.",\n'
        '  "node_type": "Sentence"\n'
        "}\n\n"
        "For clairty, here is an example of a full JSON response:\n"
        "{\n"
        '  "claim_candidates": [\n'
        "    {\n"
        '      "text": "The error rate increased to 25% after deployment.",\n'
        '      "is_atomic": true,\n'
        '      "is_self_contained": true,\n'
        '      "is_verifiable": true,\n'
        '      "passes_criteria": true,\n'
        '      "confidence": 0.98,\n'
        '      "reasoning": "This is a single, specific, and verifiable factual statement.",\n'
        "    },\n"
        "    { ... claim object 2 ... }\n"
        "  ]\n"
        "}\n\n"
    )

    disambiguated_text = dspy.InputField(
        desc="The disambiguated sentence text to decompose into claims"
    )
    decomposition_response_json = dspy.OutputField(
        desc="JSON string response with claim candidates and their evaluations"
    )
