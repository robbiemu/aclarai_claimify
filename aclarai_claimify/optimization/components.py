"""Component resolution and program factory for DSPy optimization.

This module provides utilities to resolve DSPy signatures, build programs,
and create evaluation metrics for each Claimify pipeline component.
"""

import functools
import inspect
import json
import math
import re
from difflib import SequenceMatcher
from typing import Callable, Type, Iterable

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

_COORDINATION_EXCEPTION_PATTERNS = (
    re.compile(r"\bboth\b[\s\S]{0,80}?\band\b", re.IGNORECASE),
    re.compile(r"\beither\b[\s\S]{0,80}?\bor\b", re.IGNORECASE),
    re.compile(r"\bneither\b[\s\S]{0,80}?\bnor\b", re.IGNORECASE),
    re.compile(r"\bbetween\b[\s\S]{0,80}?\band\b", re.IGNORECASE),
)

_COORDINATION_PATTERN = re.compile(r"\b(and|;|,\s*(and|or)|or)\b", re.IGNORECASE)

_CLAIM_TEXT_ALIASES = ("text", "claim_text", "claim", "value")


def _is_coordination_exception(text: str) -> bool:
    """Return True when coordination marks appear in an allowed construction."""

    return any(pattern.search(text) for pattern in _COORDINATION_EXCEPTION_PATTERNS)


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
) -> tuple[bool, float, list[str], list[str], list[str]]:
    """Inspect predicted claims and return fatal status, penalties, messages, normalized and raw texts."""

    if not isinstance(claim_candidates, list):
        return True, math.inf, ["claim_candidates must be an array of objects."], [], []

    if not claim_candidates:
        return True, math.inf, ["No claim candidates generated."], [], []

    source_tokens = set(_tokenize_words(source_text))
    penalty_weight = 0.0
    messages: list[str] = []
    normalized_texts: list[str] = []
    raw_texts: list[str] = []

    for candidate in claim_candidates:
        if not isinstance(candidate, dict):
            return True, math.inf, ["Each claim candidate must be a JSON object."], [], []

        text_value = None
        alias_used = None
        for key in _CLAIM_TEXT_ALIASES:
            value = candidate.get(key)
            if isinstance(value, str) and value.strip():
                text_value = value.strip()
                alias_used = key
                break

        if text_value is None:
            penalty_weight += 4.0
            messages.append("Missing claim text; candidate ignored.")
            continue

        if alias_used != "text":
            penalty_weight += 4.0
            messages.append(f"Used nonstandard key '{alias_used}' for claim text.")

        lower_text = text_value.lower()
        if candidate.get("is_atomic") and _COORDINATION_PATTERN.search(lower_text):
            if not _is_coordination_exception(lower_text):
                penalty_weight += 2.0
                messages.append("Potentially bundles multiple assertions.")

        passes_criteria = candidate.get("passes_criteria")
        if passes_criteria is None:
            penalty_weight += 4.0
            messages.append("Missing passes_criteria flag.")
        elif not isinstance(passes_criteria, bool):
            penalty_weight += 4.0
            messages.append("passes_criteria must be boolean.")
        elif passes_criteria and not all(
            candidate.get(flag) is True for flag in ("is_atomic", "is_self_contained", "is_verifiable")
        ):
            penalty_weight += 4.0
            messages.append("passes_criteria inconsistent with quality flags.")

        for field in ("is_atomic", "is_self_contained", "is_verifiable"):
            value = candidate.get(field)
            if value is None:
                penalty_weight += 2.0
                messages.append(f"Missing {field} flag.")
            elif not isinstance(value, bool):
                penalty_weight += 2.0
                messages.append(f"{field} must be boolean.")

        reasoning = str(candidate.get("reasoning", "")).strip()
        if passes_criteria and len(reasoning) < 15:
            penalty_weight += 0.5
            messages.append("Reasoning too short to justify high-quality claim.")

        extra_tokens = set(_tokenize_words(text_value)) - source_tokens
        if extra_tokens and len(extra_tokens) >= 2:
            penalty_weight += 2.0
            messages.append("Introduces tokens absent from source sentence.")

        normalized_texts.append(_normalize_for_similarity(text_value))
        raw_texts.append(text_value)

    return False, penalty_weight, messages, normalized_texts, raw_texts


def _fuzzy_overlap_score(expected_texts: Iterable[str], predicted_texts: Iterable[str]) -> float:
    """Compute a soft overlap score using sequence similarity."""

    expected_list = [text.lower() for text in expected_texts if text]
    predicted_list = [text.lower() for text in predicted_texts if text]

    if not expected_list or not predicted_list:
        return 0.0

    def best_scores(source: list[str], target: list[str]) -> list[float]:
        scores: list[float] = []
        for s in source:
            best = 0.0
            for t in target:
                ratio = SequenceMatcher(None, s, t).ratio()
                if ratio > best:
                    best = ratio
                    if best >= 0.99:
                        break
            scores.append(best)
        return scores

    expected_best = best_scores(expected_list, predicted_list)
    predicted_best = best_scores(predicted_list, expected_list)

    precision = sum(predicted_best) / len(predicted_best)
    recall = sum(expected_best) / len(expected_best)

    if precision + recall == 0:
        return 0.0

    return 2 * (precision * recall) / (precision + recall)


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

    penalty_weight = 0.0
    penalty_messages: list[str] = []

    predicted_candidates = predicted.get("claim_candidates")
    if not isinstance(predicted_candidates, list):
        for alias in ("claims", "candidates", "items"):
            candidate_list = predicted.get(alias)
            if isinstance(candidate_list, list):
                penalty_weight += 2.0
                penalty_messages.append(
                    f"Used nonstandard key '{alias}' for claim list."
                )
                predicted_candidates = candidate_list
                break

    if not isinstance(predicted_candidates, list):
        return 0.0, "Output must include an array of claim candidates."

    source_text = getattr(example, "disambiguated_text", "") or ""
    (
        fatal,
        candidate_penalty,
        candidate_messages,
        normalized_texts,
        raw_texts,
    ) = _detect_decomposition_failure(predicted_candidates, source_text)
    if fatal:
        failure_message = candidate_messages[0] if candidate_messages else "Prediction failed quality checks."
        return 0.0, failure_message

    penalty_weight += candidate_penalty
    penalty_messages.extend(candidate_messages)

    expected_claims = set()
    for candidate in expected.get("claim_candidates", []):
        if isinstance(candidate, dict) and "text" in candidate:
            expected_claims.add(_normalize_for_similarity(candidate["text"]))

    predicted_claims = set(normalized_texts)

    if not expected_claims and not predicted_claims:
        return 1.0, "Both reference and prediction contain no claims."

    if not predicted_claims:
        return 0.0, "No claim candidates generated."

    intersection = expected_claims & predicted_claims
    precision = len(intersection) / len(predicted_claims) if predicted_claims else 0.0
    recall = len(intersection) / len(expected_claims) if expected_claims else 0.0

    if precision + recall == 0:
        fuzzy_f1 = _fuzzy_overlap_score(
            [candidate.get("text", "") for candidate in expected.get("claim_candidates", [])],
            raw_texts,
        )
        if fuzzy_f1 == 0.0:
            if normalized_texts:
                base_score = 0.05
                penalty_messages.append(
                    "Returned claims but none overlapped reference; applying minimal score."
                )
            else:
                base_score = 0.0
        else:
            base_score = fuzzy_f1 * 0.5
            penalty_messages.append(
                "Applied fuzzy matching because exact overlap was zero."
            )
            precision = recall = fuzzy_f1
    else:
        base_score = 2 * (precision * recall) / (precision + recall)

    feedback_parts = [
        f"Claim overlap F1: {base_score:.2f} (precision {precision:.2f}, recall {recall:.2f})."
    ]

    if penalty_weight > 0:
        penalty_factor = 1.0 / (1.0 + penalty_weight)
        adjusted_score = base_score * penalty_factor
        penalty_detail = (
            f"Penalty factor {penalty_factor:.3f} derived from weighted issues ({penalty_weight:.2f})."
        )
        if penalty_messages:
            penalty_detail += " Issues: " + "; ".join(penalty_messages)
        feedback_parts.insert(0, penalty_detail)
        return adjusted_score, " ".join(feedback_parts)

    return base_score, " ".join(feedback_parts)


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
