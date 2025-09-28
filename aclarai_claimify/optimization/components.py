"""Component resolution and program factory for DSPy optimization.

This module provides utilities to resolve DSPy signatures, build programs,
and create evaluation metrics for each Claimify pipeline component.
"""

import functools
import inspect
import json
import re
from difflib import SequenceMatcher
from typing import Callable, Type

import dspy

from ..signatures import SelectionSignature, DisambiguationSignature, DecompositionSignature


class ComponentError(Exception):
    """Raised when component resolution fails."""
    pass


_PRONOUN_PATTERN = re.compile(r"\b(it|they|them|he|she|this|that|these|those)\b", re.IGNORECASE)
_VAGUE_REFERENT_PATTERN = re.compile(
    r"\b(the (person|group|individual|entity)|someone|somebody|something)\b",
    re.IGNORECASE,
)
_NEGATION_WORDS = {"not", "never", "no", "none", "without", "neither", "nor"}


def _normalize_for_similarity(text: str) -> str:
    """Lowercase and collapse whitespace/punctuation for similarity comparison."""

    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return text.strip()


def _tokenize_words(text: str) -> list[str]:
    """Tokenize text into alphanumeric word tokens for heuristic checks."""

    return re.findall(r"[A-Za-z0-9']+", text.lower())


def get_signature(component: str) -> Type[dspy.Signature]:
    """Get the DSPy signature for a component.
    
    Args:
        component: Component name (selection, disambiguation, decomposition)
        
    Returns:
        DSPy signature class
        
    Raises:
        ComponentError: If component is not recognized
    """
    signatures = {
        "selection": SelectionSignature,
        "disambiguation": DisambiguationSignature,
        "decomposition": DecompositionSignature,
    }
    
    if component not in signatures:
        raise ComponentError(
            f"Unknown component '{component}'. Must be one of: {list(signatures.keys())}"
        )
    
    return signatures[component]


def build_program(signature: Type[dspy.Signature], style: str = "cot") -> dspy.Module:
    """Build a DSPy program/module for a signature.
    
    Args:
        signature: DSPy signature class
        style: Program style - "cot" for ChainOfThought, "predict" for basic Predict
        
    Returns:
        DSPy module ready for optimization
        
    Raises:
        ComponentError: If style is not recognized
    """
    if style == "cot":
        try:
            return dspy.ChainOfThought(signature)
        except Exception as e:
            # Fallback to Predict if ChainOfThought fails
            return dspy.Predict(signature)
    elif style == "predict":
        return dspy.Predict(signature)
    else:
        raise ComponentError(f"Unknown program style '{style}'. Must be 'cot' or 'predict'")


def selection_metric(example: dspy.Example, prediction: dspy.Prediction, trace=None) -> float:
    """Evaluation metric for the selection component.
    
    Compares the predicted selection decision with the expected decision.
    
    Args:
        example: DSPy example with expected output
        prediction: DSPy prediction to evaluate
        trace: Optional trace (not used)
        
    Returns:
        Score between 0.0 and 1.0
    """
    try:
        # Parse expected and predicted JSON responses
        expected = json.loads(example.selection_response_json)
        predicted = json.loads(prediction.selection_response_json)
        
        # Extract selection decisions
        expected_selected = expected.get("selected", False)
        predicted_selected = predicted.get("selected", False)
        
        # Simple binary accuracy
        return 1.0 if expected_selected == predicted_selected else 0.0
        
    except (json.JSONDecodeError, KeyError, AttributeError) as e:
        # If JSON parsing fails, score as 0
        return 0.0


def disambiguation_metric(example: dspy.Example, prediction: dspy.Prediction, trace=None) -> float:
    """Evaluation metric for the disambiguation component.

    Compares the predicted disambiguated text with the expected text.

    Args:
        example: DSPy example with expected output
        prediction: DSPy prediction to evaluate
        trace: Optional trace (not used)

    Returns:
        Score between 0.0 and 1.0
    """

    score, _ = _score_disambiguation_output(example, prediction)
    return score


def decomposition_metric(example: dspy.Example, prediction: dspy.Prediction, trace=None) -> float:
    """Evaluation metric for the decomposition component."""

    score, _ = _score_decomposition_output(example, prediction)
    return score


def _detect_disambiguation_failure(
    predicted_text: str,
    changes_made: list,
    confidence: float,
    target_sentence: str,
    context_text: str,
) -> tuple[bool, str]:
    """Return whether the predicted rewrite exhibits a failure mode."""

    if not predicted_text:
        return True, "disambiguated_text must be a non-empty string."

    sentence_terminators = re.findall(r"[.!?]", predicted_text)
    if "\n" in predicted_text or len(sentence_terminators) > 1:
        return True, "Output must be exactly one sentence without newlines."

    lower_pred = predicted_text.lower()
    lower_target = target_sentence.lower()

    unresolved = set(_PRONOUN_PATTERN.findall(lower_target)) & set(
        _PRONOUN_PATTERN.findall(lower_pred)
    )
    if unresolved:
        pronoun_list = ", ".join(sorted(unresolved))
        return True, f"Pronoun(s) left unresolved: {pronoun_list}."

    for neg in _NEGATION_WORDS:
        if neg in lower_target and neg not in lower_pred:
            return True, f"Missing required qualifier '{neg}'."

    if _VAGUE_REFERENT_PATTERN.search(lower_pred):
        return True, "Replaced referent with a vague placeholder."

    allowed_numbers = set(re.findall(r"\d+", target_sentence)) | set(
        re.findall(r"\d+", context_text)
    )
    extra_numbers = set(re.findall(r"\d+", predicted_text)) - allowed_numbers
    if extra_numbers:
        return True, "Introduced numeric detail not present in context."

    if not isinstance(changes_made, list) or any(
        not isinstance(item, str) for item in changes_made
    ):
        return True, "changes_made must be a list of strings."

    return False, ""


def _detect_decomposition_failure(
    claim_candidates: list,
    source_text: str,
) -> tuple[bool, str]:
    """Return whether the predicted decomposition exhibits a failure mode."""

    if not isinstance(claim_candidates, list):
        return True, "claim_candidates must be an array of objects."

    if not claim_candidates:
        return True, "No claim candidates generated."

    source_tokens = set(_tokenize_words(source_text))

    for candidate in claim_candidates:
        if not isinstance(candidate, dict):
            return True, "Each claim candidate must be a JSON object."

        text = str(candidate.get("text", "")).strip()
        if not text:
            return True, "Claim candidate has empty text."

        lower_text = text.lower()
        if candidate.get("is_atomic") and re.search(r"\b(and|;|,\s*and)\b", lower_text):
            return True, "Claim candidate still bundles multiple assertions."

        if candidate.get("passes_criteria") and not all(
            candidate.get(flag) for flag in ("is_atomic", "is_self_contained", "is_verifiable")
        ):
            return True, "passes_criteria is inconsistent with quality flags."

        reasoning = str(candidate.get("reasoning", "")).strip()
        if candidate.get("passes_criteria") and len(reasoning) < 8:
            return True, "Reasoning is too sparse to justify a high-quality claim."

        extra_tokens = set(_tokenize_words(text)) - source_tokens
        if extra_tokens and len(extra_tokens) >= 2:
            return True, "Claim introduces content not present in the source sentence."

    return False, ""


def _score_decomposition_output(
    example: dspy.Example, prediction: dspy.Prediction, trace=None
) -> tuple[float, str]:
    """Compute score and feedback for decomposition predictions."""

    try:
        expected = json.loads(example.decomposition_response_json)
    except (AttributeError, json.JSONDecodeError):
        return 0.0, "Internal error: expected reference JSON is invalid."

    try:
        predicted = json.loads(prediction.decomposition_response_json)
    except (AttributeError, json.JSONDecodeError):
        return 0.0, "Output must be valid JSON with key claim_candidates."

    if not isinstance(predicted, dict):
        return 0.0, "Output JSON must contain claim_candidates list."

    predicted_candidates = predicted.get("claim_candidates", [])
    source_text = getattr(example, "disambiguated_text", "") or ""
    failure, failure_msg = _detect_decomposition_failure(predicted_candidates, source_text)
    if failure:
        return 0.0, failure_msg

    expected_claims = set()
    for candidate in expected.get("claim_candidates", []):
        if isinstance(candidate, dict) and "text" in candidate:
            expected_claims.add(_normalize_for_similarity(candidate["text"]))

    predicted_claims = set()
    for candidate in predicted_candidates:
        if isinstance(candidate, dict) and "text" in candidate:
            predicted_claims.add(_normalize_for_similarity(candidate["text"]))

    if not expected_claims and not predicted_claims:
        return 1.0, "Both reference and prediction contain no claims."

    if not predicted_claims:
        return 0.0, "No claim candidates generated."

    intersection = expected_claims & predicted_claims
    precision = len(intersection) / len(predicted_claims) if predicted_claims else 0.0
    recall = len(intersection) / len(expected_claims) if expected_claims else 0.0

    if precision + recall == 0:
        return 0.0, "Predicted claims did not overlap with reference claims."

    f1 = 2 * (precision * recall) / (precision + recall)
    message = (
        f"Claim overlap F1: {f1:.2f} (precision {precision:.2f}, recall {recall:.2f})."
    )
    return f1, message


def _score_disambiguation_output(
    example: dspy.Example, prediction: dspy.Prediction, trace=None
) -> tuple[float, str]:
    """Compute score and textual feedback for disambiguation predictions."""

    required_fields = {"disambiguated_text", "changes_made", "confidence"}

    try:
        expected_raw = example.disambiguation_response_json
    except AttributeError:
        return 0.0, "Internal error: expected example is missing disambiguation_response_json."

    try:
        predicted_raw = prediction.disambiguation_response_json
    except AttributeError:
        return 0.0, "Prediction must include disambiguation_response_json field."

    try:
        expected = json.loads(expected_raw)
    except json.JSONDecodeError:
        return 0.0, "Internal error: expected reference JSON is invalid."

    try:
        predicted = json.loads(predicted_raw)
    except json.JSONDecodeError:
        return 0.0, "Output must be valid JSON with keys disambiguated_text, changes_made, confidence."

    if not isinstance(predicted, dict):
        return 0.0, "Output JSON must be an object with disambiguated_text, changes_made, confidence."

    missing = sorted(required_fields - set(predicted.keys()))
    if missing:
        return 0.0, f"Output JSON is missing required field(s): {', '.join(missing)}."

    try:
        predicted_confidence = float(predicted.get("confidence", 0.0) or 0.0)
    except (TypeError, ValueError):
        predicted_confidence = 0.0

    disambiguated_text = str(predicted.get("disambiguated_text", "")).strip()
    changes_made = predicted.get("changes_made", [])

    target_sentence = getattr(example, "target_sentence", "") or ""
    context_text = getattr(example, "context_text", "") or ""

    failure, failure_msg = _detect_disambiguation_failure(
        disambiguated_text,
        changes_made,
        predicted_confidence,
        target_sentence,
        context_text,
    )
    if failure:
        if predicted_confidence >= 0.85:
            failure_msg = f"{failure_msg} (reported confidence {predicted_confidence:.2f})"
        return 0.0, failure_msg

    expected_text = str(expected.get("disambiguated_text", "")).strip()
    expected_changes = expected.get("changes_made", []) or []
    predicted_changes = changes_made or []

    expected_norm = _normalize_for_similarity(expected_text)
    predicted_norm = _normalize_for_similarity(disambiguated_text)
    similarity = SequenceMatcher(None, expected_norm, predicted_norm).ratio()

    feedback_parts = [f"Text similarity {similarity:.2f}"]
    score = similarity

    if expected_changes and not predicted_changes:
        feedback_parts.append("Missing changes_made explanations.")
        score *= 0.6
    elif not expected_changes and predicted_changes:
        feedback_parts.append("Unnecessary changes_made entries present.")
        score *= 0.9

    if predicted_confidence >= 0.85 and score < 0.5:
        feedback_parts.append(
            f"Confidence {predicted_confidence:.2f} is inconsistent with rewrite quality."
        )
        score *= 0.8

    score = max(0.0, min(1.0, score))
    feedback = "; ".join(feedback_parts)
    return score, feedback


disambiguation_metric.gepa_feedback = _score_disambiguation_output
decomposition_metric.gepa_feedback = _score_decomposition_output


def _adapt_metric_for_optimizer(metric: Callable) -> Callable:
    """Ensure metrics support optional trace and predictor arguments."""

    signature = inspect.signature(metric)
    param_count = len(signature.parameters)

    if param_count >= 5:
        return metric

    def _preserve_metadata(wrapper):
        wrapper.__name__ = getattr(metric, "__name__", wrapper.__name__)
        wrapper.__doc__ = getattr(metric, "__doc__", wrapper.__doc__)
        wrapper.__module__ = getattr(metric, "__module__", wrapper.__module__)
        if hasattr(metric, "gepa_feedback"):
            wrapper.gepa_feedback = metric.gepa_feedback
        return wrapper

    if param_count <= 2:

        def _wrapped(gold, pred, trace=None, pred_name=None, pred_trace=None):
            return metric(gold, pred)

        return _preserve_metadata(_wrapped)

    if param_count == 3:

        def _wrapped(gold, pred, trace=None, pred_name=None, pred_trace=None):
            return metric(gold, pred, trace)

        return _preserve_metadata(_wrapped)

    if param_count == 4:

        def _wrapped(gold, pred, trace=None, pred_name=None, pred_trace=None):
            return metric(gold, pred, trace, pred_name)

        return _preserve_metadata(_wrapped)

    return metric


def get_metric(component: str) -> Callable:
    """Get the evaluation metric function for a component.
    
    Args:
        component: Component name (selection, disambiguation, decomposition)
        
    Returns:
        Metric function compatible with DSPy evaluation
        
    Raises:
        ComponentError: If component is not recognized
    """
    metrics = {
        "selection": selection_metric,
        "disambiguation": disambiguation_metric,
        "decomposition": decomposition_metric,
    }
    
    if component not in metrics:
        raise ComponentError(
            f"Unknown component '{component}'. Must be one of: {list(metrics.keys())}"
        )
    
    return _adapt_metric_for_optimizer(metrics[component])


def get_component_info(component: str) -> dict:
    """Get comprehensive information about a component.
    
    Args:
        component: Component name
        
    Returns:
        Dictionary with signature, metric, and other component info
        
    Raises:
        ComponentError: If component is not recognized
    """
    try:
        signature = get_signature(component)
        metric = get_metric(component)
        
        return {
            "component": component,
            "signature": signature,
            "signature_name": signature.__name__,
            "input_fields": list(signature.input_fields.keys()),
            "output_fields": list(signature.output_fields.keys()),
            "metric": metric,
            "metric_name": metric.__name__,
        }
    except ComponentError:
        raise
    except Exception as e:
        raise ComponentError(f"Failed to get component info for '{component}': {e}") from e


def validate_component_examples(component: str, examples: list) -> None:
    """Validate that examples are compatible with a component's signature.
    
    Args:
        component: Component name
        examples: List of DSPy examples
        
    Raises:
        ComponentError: If examples don't match component signature
    """
    if not examples:
        return
    
    info = get_component_info(component)
    input_fields = set(info["input_fields"])
    output_fields = set(info["output_fields"])
    
    for i, example in enumerate(examples[:5], 1):  # Check first 5 examples
        try:
            # Get all fields in the example by checking what fields we can access
            example_fields = set()
            for field in input_fields | output_fields:
                if hasattr(example, field):
                    example_fields.add(field)
            
            # Check that example has all required input fields
            missing_inputs = input_fields - example_fields
            if missing_inputs:
                raise ComponentError(
                    f"Example {i} missing required input fields for {component}: {missing_inputs}"
                )
            
            # Check that example has all required output fields
            missing_outputs = output_fields - example_fields
            if missing_outputs:
                raise ComponentError(
                    f"Example {i} missing required output fields for {component}: {missing_outputs}"
                )
                
        except AttributeError as e:
            raise ComponentError(
                f"Example {i} has invalid structure for {component}: {e}"
            ) from e
