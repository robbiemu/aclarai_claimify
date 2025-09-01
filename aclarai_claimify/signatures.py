"""DSPy signatures for the Claimify pipeline components.

This module defines DSPy Signature classes for each stage of the Claimify pipeline:
Selection, Disambiguation, and Decomposition. These signatures make the inputs and
expected outputs explicit for use with DSPy modules.
"""

import dspy


class SelectionSignature(dspy.Signature):
    """Signature for the Selection stage of the Claimify pipeline.
    
    Determines if a sentence contains verifiable factual information suitable for claim extraction.
    """
    
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
    
    disambiguated_text = dspy.InputField(
        desc="The disambiguated sentence text to decompose into claims"
    )
    decomposition_response_json = dspy.OutputField(
        desc="JSON string response with claim candidates and their evaluations"
    )