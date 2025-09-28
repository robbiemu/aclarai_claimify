"""Dataset loader and schema validation for DSPy optimization.

This module provides utilities to load JSONL datasets and convert them to
DSPy Examples for optimization, with proper validation of the expected
schema for each component.
"""

import json
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple
import dspy

from .failure_modes import (
    DISAMBIGUATION_FAILURE_MODES,
    DECOMPOSITION_FAILURE_MODES,
)


class DataValidationError(Exception):
    """Raised when dataset validation fails."""
    pass


def load_jsonl_dataset(path: Path) -> List[Dict[str, Any]]:
    """Load a JSONL dataset from file.
    
    Args:
        path: Path to the JSONL file
        
    Returns:
        List of dictionaries, one per line
        
    Raises:
        FileNotFoundError: If file doesn't exist
        DataValidationError: If file is empty or has invalid JSON
    """
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")
    
    records = []
    with open(path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
                
            try:
                record = json.loads(line)
                records.append(record)
            except json.JSONDecodeError as e:
                raise DataValidationError(
                    f"Invalid JSON on line {line_num}: {e}"
                ) from e
    
    if not records:
        raise DataValidationError(f"Dataset file is empty: {path}")
    
    return records


def validate_selection_record(record: Dict[str, Any], line_num: int) -> None:
    """Validate a single record for the selection component.
    
    Expected schema:
    {
        "context_text": str,
        "target_sentence": str,
        "selection_response_json": str  # JSON string with selection decision
    }
    """
    required_fields = ["context_text", "target_sentence", "selection_response_json"]
    for field in required_fields:
        if field not in record:
            raise DataValidationError(
                f"Missing required field '{field}' on line {line_num}"
            )
        if not isinstance(record[field], str):
            raise DataValidationError(
                f"Field '{field}' must be string on line {line_num}"
            )
    
    # Validate that selection_response_json is valid JSON
    try:
        selection_data = json.loads(record["selection_response_json"])
        if not isinstance(selection_data, dict):
            raise DataValidationError(
                f"selection_response_json must be JSON object on line {line_num}"
            )
    except json.JSONDecodeError as e:
        raise DataValidationError(
            f"Invalid JSON in selection_response_json on line {line_num}: {e}"
        ) from e


def validate_disambiguation_record(record: Dict[str, Any], line_num: int) -> None:
    """Validate a single record for the disambiguation component.
    
    Expected schema:
    {
        "context_text": str,
        "target_sentence": str,
        "disambiguation_response_json": str  # JSON string with disambiguation result
    }
    """
    required_fields = ["context_text", "target_sentence", "disambiguation_response_json"]
    for field in required_fields:
        if field not in record:
            raise DataValidationError(
                f"Missing required field '{field}' on line {line_num}"
            )
        if not isinstance(record[field], str):
            raise DataValidationError(
                f"Field '{field}' must be string on line {line_num}"
            )

    # Validate that disambiguation_response_json is valid JSON
    try:
        disambiguation_data = json.loads(record["disambiguation_response_json"])
        if not isinstance(disambiguation_data, dict):
            raise DataValidationError(
                f"disambiguation_response_json must be JSON object on line {line_num}"
            )
    except json.JSONDecodeError as e:
        raise DataValidationError(
            f"Invalid JSON in disambiguation_response_json on line {line_num}: {e}"
        ) from e

    sample_type = record.get("sample_type", "positive")
    if sample_type not in {"positive", "negative"}:
        raise DataValidationError(
            f"Invalid sample_type '{sample_type}' on line {line_num}; expected 'positive' or 'negative'"
        )

    if sample_type == "negative":
        failure_mode = record.get("failure_mode")
        if failure_mode not in DISAMBIGUATION_FAILURE_MODES:
            raise DataValidationError(
                f"Negative disambiguation sample missing or invalid failure_mode on line {line_num}"
            )


def validate_decomposition_record(record: Dict[str, Any], line_num: int) -> None:
    """Validate a single record for the decomposition component.
    
    Expected schema:
    {
        "disambiguated_text": str,
        "decomposition_response_json": str  # JSON string with claim candidates
    }
    """
    required_fields = ["disambiguated_text", "decomposition_response_json"]
    for field in required_fields:
        if field not in record:
            raise DataValidationError(
                f"Missing required field '{field}' on line {line_num}"
            )
        if not isinstance(record[field], str):
            raise DataValidationError(
                f"Field '{field}' must be string on line {line_num}"
            )

    # Validate that decomposition_response_json is valid JSON
    try:
        decomposition_data = json.loads(record["decomposition_response_json"])
        if not isinstance(decomposition_data, dict):
            raise DataValidationError(
                f"decomposition_response_json must be JSON object on line {line_num}"
            )
    except json.JSONDecodeError as e:
        raise DataValidationError(
            f"Invalid JSON in decomposition_response_json on line {line_num}: {e}"
        ) from e

    sample_type = record.get("sample_type", "positive")
    if sample_type not in {"positive", "negative"}:
        raise DataValidationError(
            f"Invalid sample_type '{sample_type}' on line {line_num}; expected 'positive' or 'negative'"
        )

    if sample_type == "negative":
        failure_mode = record.get("failure_mode")
        if failure_mode not in DECOMPOSITION_FAILURE_MODES:
            raise DataValidationError(
                f"Negative decomposition sample missing or invalid failure_mode on line {line_num}"
            )


def validate_records_for_component(
    records: List[Dict[str, Any]], 
    component: str
) -> None:
    """Validate all records for a specific component.
    
    Args:
        records: List of dataset records
        component: Component name (selection, disambiguation, decomposition)
        
    Raises:
        DataValidationError: If any record doesn't match expected schema
        ValueError: If component is not recognized
    """
    validators = {
        "selection": validate_selection_record,
        "disambiguation": validate_disambiguation_record,
        "decomposition": validate_decomposition_record,
    }
    
    if component not in validators:
        raise ValueError(
            f"Unknown component '{component}'. Must be one of: {list(validators.keys())}"
        )
    
    validator = validators[component]
    for i, record in enumerate(records, 1):
        validator(record, i)


def map_to_examples(
    records: List[Dict[str, Any]], 
    signature: dspy.Signature
) -> List[dspy.Example]:
    """Map dataset records to DSPy Examples.
    
    Args:
        records: List of validated dataset records
        signature: DSPy signature to map fields for
        
    Returns:
        List of DSPy Examples ready for optimization
        
    Raises:
        DataValidationError: If records don't match signature fields
    """
    signature_inputs = list(signature.input_fields.keys())
    signature_outputs = list(signature.output_fields.keys())
    
    examples = []
    for i, record in enumerate(records, 1):
        if record.get("sample_type", "positive") != "positive":
            continue
        # Check that all signature input fields are present
        missing_inputs = set(signature_inputs) - set(record.keys())
        if missing_inputs:
            raise DataValidationError(
                f"Record {i} missing required input fields: {missing_inputs}"
            )
        
        # Check that all signature output fields are present
        missing_outputs = set(signature_outputs) - set(record.keys())
        if missing_outputs:
            raise DataValidationError(
                f"Record {i} missing required output fields: {missing_outputs}"
            )
        
        # Create DSPy Example with all fields, then mark inputs
        example_dict = {field: record[field] for field in signature_inputs + signature_outputs}
        example = dspy.Example(**example_dict).with_inputs(*signature_inputs)
        examples.append(example)
    
    return examples


def split_examples(
    examples: List[dspy.Example], 
    val_ratio: float = 0.2, 
    seed: int = 42
) -> Tuple[List[dspy.Example], List[dspy.Example]]:
    """Split examples into training and validation sets.
    
    Args:
        examples: List of DSPy examples
        val_ratio: Fraction of examples to use for validation
        seed: Random seed for deterministic splits
        
    Returns:
        Tuple of (train_examples, val_examples)
        
    Raises:
        ValueError: If val_ratio is invalid or dataset too small
    """
    if not 0 < val_ratio < 1:
        raise ValueError("val_ratio must be between 0 and 1")
    
    if len(examples) < 2:
        raise ValueError("Dataset must have at least 2 examples for train/val split")
    
    # Shuffle with fixed seed for reproducibility
    examples_copy = examples.copy()
    random.seed(seed)
    random.shuffle(examples_copy)
    
    # Calculate split point
    n_val = max(1, int(len(examples) * val_ratio))
    n_train = len(examples) - n_val
    
    if n_train < 1:
        raise ValueError("Training set would be empty with current val_ratio")
    
    train_set = examples_copy[:n_train]
    val_set = examples_copy[n_train:]
    
    return train_set, val_set


def get_schema_documentation(component: str) -> str:
    """Get documentation for the expected JSONL schema for a component.
    
    Args:
        component: Component name
        
    Returns:
        Human-readable schema documentation
    """
    schemas = {
        "selection": """
Selection component JSONL schema:
Each line should contain a JSON object with:
- context_text: String containing context (surrounding sentences)
- target_sentence: String with the sentence to evaluate for selection
- selection_response_json: JSON string with selection decision, e.g.:
  '{"selected": true, "confidence": 0.9, "reasoning": "Contains verifiable information"}'

Example line:
{"context_text": "[0] The system was stable.", "target_sentence": "It failed with error 500.", "selection_response_json": "{\\"selected\\": true, \\"confidence\\": 0.85, \\"reasoning\\": \\"Technical failure information\\"}"}
""",
        "disambiguation": """
Disambiguation component JSONL schema:
Each line should contain a JSON object with:
- context_text: String containing context (surrounding sentences)
- target_sentence: String with the sentence to disambiguate
- disambiguation_response_json: JSON string with disambiguation result, e.g.:
  '{"disambiguated_text": "The system failed with error 500.", "changes_made": ["Replaced it with the system"], "confidence": 0.9}'
- sample_type: Optional string ('positive' by default, 'negative' for curated failure cases)
- failure_mode: Required when sample_type == 'negative' (e.g., 'unresolved_referent')

Example line:
{"context_text": "[0] The system was stable.", "target_sentence": "It failed with error 500.", "disambiguation_response_json": "{\\"disambiguated_text\\": \\"The system failed with error 500.\\", \\"changes_made\\": [\\"Replaced 'it' with 'the system'\\"], \\"confidence\\": 0.9}"}
""",
        "decomposition": """
Decomposition component JSONL schema:
Each line should contain a JSON object with:
- disambiguated_text: String with the disambiguated sentence to decompose
- decomposition_response_json: JSON string with claim candidates, e.g.:
  '{"claim_candidates": [{"text": "The system failed with error 500.", "is_atomic": true, "is_self_contained": true, "is_verifiable": true, "passes_criteria": true, "confidence": 0.95, "reasoning": "Single verifiable fact"}]}'
- sample_type: Optional string ('positive' or 'negative')
- failure_mode: Required when sample_type == 'negative' (e.g., 'non_atomic_claim')

Example line:
{"disambiguated_text": "The system failed with error 500.", "decomposition_response_json": "{\\"claim_candidates\\": [{\\"text\\": \\"The system failed with error 500.\\", \\"is_atomic\\": true, \\"is_self_contained\\": true, \\"is_verifiable\\": true, \\"passes_criteria\\": true, \\"confidence\\": 0.95, \\"reasoning\\": \\"Single verifiable fact\\", \\"node_type\\": \\"Claim\\"}]}"}
"""
    }
    
    return schemas.get(component, f"Unknown component: {component}")


def print_schema_help(component: str) -> None:
    """Print schema documentation for a component to stdout."""
    print(get_schema_documentation(component))
