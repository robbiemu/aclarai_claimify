"""Component resolution and program factory for DSPy optimization.

This module provides utilities to resolve DSPy signatures, build programs,
and create evaluation metrics for each Claimify pipeline component.
"""

import json
from typing import Callable, Type
import dspy
from ..signatures import SelectionSignature, DisambiguationSignature, DecompositionSignature


class ComponentError(Exception):
    """Raised when component resolution fails."""
    pass


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
    try:
        # Parse expected and predicted JSON responses
        expected = json.loads(example.disambiguation_response_json)
        predicted = json.loads(prediction.disambiguation_response_json)
        
        # Extract disambiguated texts
        expected_text = expected.get("disambiguated_text", "").strip()
        predicted_text = predicted.get("disambiguated_text", "").strip()
        
        # Exact match (could be improved with semantic similarity)
        return 1.0 if expected_text == predicted_text else 0.0
        
    except (json.JSONDecodeError, KeyError, AttributeError) as e:
        # If JSON parsing fails, score as 0
        return 0.0


def decomposition_metric(example: dspy.Example, prediction: dspy.Prediction, trace=None) -> float:
    """Evaluation metric for the decomposition component.
    
    Compares the predicted claim candidates with expected candidates using F1 score.
    
    Args:
        example: DSPy example with expected output
        prediction: DSPy prediction to evaluate
        trace: Optional trace (not used)
        
    Returns:
        Score between 0.0 and 1.0 (F1 score)
    """
    try:
        # Parse expected and predicted JSON responses
        expected = json.loads(example.decomposition_response_json)
        predicted = json.loads(prediction.decomposition_response_json)
        
        # Extract claim texts
        expected_claims = set()
        for candidate in expected.get("claim_candidates", []):
            if isinstance(candidate, dict) and "text" in candidate:
                expected_claims.add(candidate["text"].strip())
        
        predicted_claims = set()
        for candidate in predicted.get("claim_candidates", []):
            if isinstance(candidate, dict) and "text" in candidate:
                predicted_claims.add(candidate["text"].strip())
        
        # Calculate F1 score
        if not expected_claims and not predicted_claims:
            return 1.0  # Perfect match for empty sets
        
        intersection = expected_claims & predicted_claims
        if not intersection:
            return 0.0
        
        precision = len(intersection) / len(predicted_claims) if predicted_claims else 0.0
        recall = len(intersection) / len(expected_claims) if expected_claims else 0.0
        
        if precision + recall == 0:
            return 0.0
        
        f1 = 2 * (precision * recall) / (precision + recall)
        return f1
        
    except (json.JSONDecodeError, KeyError, AttributeError) as e:
        # If JSON parsing fails, score as 0
        return 0.0


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
    
    return metrics[component]


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
