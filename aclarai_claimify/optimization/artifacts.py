"""Artifact schema and serialization utilities for compiled DSPy programs.

This module defines the JSON schema and utilities for saving and loading
compiled DSPy program artifacts, ensuring forward compatibility and
proper serialization of optimization results.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field


class FewShotExample(BaseModel):
    """A few-shot example used in the compiled program."""
    
    inputs: Dict[str, Any] = Field(
        ..., 
        description="Input fields for this example"
    )
    output: Any = Field(
        ..., 
        description="Expected output for this example"
    )
    rationale: Optional[str] = Field(
        None, 
        description="Chain-of-thought rationale (if available)"
    )


class ValidationMetrics(BaseModel):
    """Validation metrics for the compiled program."""
    
    metric_name: str = Field(
        ..., 
        description="Name of the evaluation metric"
    )
    score: float = Field(
        ..., 
        description="Validation score achieved"
    )
    n_val: int = Field(
        ..., 
        description="Number of validation examples"
    )
    per_example_diagnostics: Optional[List[Dict[str, Any]]] = Field(
        None, 
        description="Per-example diagnostics (optional, may be truncated)"
    )


class OptimizerParams(BaseModel):
    """Parameters used for the DSPy optimizer."""
    
    optimizer_name: str = Field(
        ..., 
        description="Name of the DSPy optimizer used"
    )
    max_bootstrapped_demos: Optional[int] = Field(
        None, 
        description="Maximum number of bootstrapped demonstrations"
    )
    max_trials: Optional[int] = Field(
        None, 
        description="Maximum number of optimization trials"
    )
    seed: Optional[int] = Field(
        None, 
        description="Random seed used for optimization"
    )
    other_params: Dict[str, Any] = Field(
        default_factory=dict, 
        description="Other optimizer-specific parameters"
    )


class CompiledArtifact(BaseModel):
    """Schema for a compiled DSPy program artifact."""
    
    # Metadata
    artifact_version: str = Field(
        default="1.0", 
        description="Version of the artifact schema"
    )
    component: str = Field(
        ..., 
        description="Name of the component (selection, disambiguation, decomposition)"
    )
    signature_name: str = Field(
        ..., 
        description="Name of the DSPy signature used"
    )
    created_at: str = Field(
        default_factory=lambda: datetime.now().isoformat(), 
        description="ISO 8601 timestamp when artifact was created"
    )
    
    # Model configuration
    student_model: str = Field(
        ..., 
        description="Model used for final program execution"
    )
    teacher_model: str = Field(
        ..., 
        description="Model used for teaching/optimization guidance"
    )
    k_window_size: Optional[int] = Field(
        default=None,
        description="Context window size (k) used for the training data",
    )
    
    # Optimization parameters
    optimizer_params: OptimizerParams = Field(
        ..., 
        description="Parameters used for optimization"
    )
    
    # Compiled program data
    few_shots: List[FewShotExample] = Field(
        default_factory=list, 
        description="Few-shot examples used by the compiled program"
    )
    system_prompt: Optional[str] = Field(
        None, 
        description="System/instruction prompt (if available)"
    )
    program_style: Optional[str] = Field(
        default=None,
        description="DSPy program style used when compiling (e.g., cot or predict)",
    )
    
    # Validation results
    validation_metrics: Optional[ValidationMetrics] = Field(
        None, 
        description="Validation metrics achieved by the compiled program"
    )
    
    # Raw DSPy serialization (if available)
    dspy_serialized: Optional[Dict[str, Any]] = Field(
        None, 
        description="Raw serialized DSPy module (for round-trip compatibility)"
    )


def save_artifact(artifact: CompiledArtifact, output_path: Path) -> None:
    """Save a compiled artifact to a JSON file.
    
    Args:
        artifact: The compiled artifact to save
        output_path: Path where to save the artifact
        
    Raises:
        OSError: If the file cannot be written
        ValueError: If the artifact is invalid
    """
    # Ensure parent directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Validate the artifact
    validated_artifact = artifact.model_validate(artifact.model_dump())
    
    # Save to JSON
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(
            validated_artifact.model_dump(exclude_none=False), 
            f, 
            indent=2, 
            ensure_ascii=False
        )


def load_artifact(path: Path) -> CompiledArtifact:
    """Load a compiled artifact from a JSON file.
    
    Args:
        path: Path to the artifact JSON file
        
    Returns:
        The loaded and validated artifact
        
    Raises:
        FileNotFoundError: If the file doesn't exist
        ValueError: If the JSON is invalid or doesn't match schema
        OSError: If the file cannot be read
    """
    if not path.exists():
        raise FileNotFoundError(f"Artifact file not found: {path}")
    
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        return CompiledArtifact.model_validate(data)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in artifact file: {e}") from e
    except Exception as e:
        raise ValueError(f"Invalid artifact schema: {e}") from e


def create_artifact_dict(
    component: str,
    signature_name: str,
    student_model: str,
    teacher_model: str,
    optimizer_params: OptimizerParams,
    few_shots: Optional[List[FewShotExample]] = None,
    system_prompt: Optional[str] = None,
    program_style: Optional[str] = None,
    validation_metrics: Optional[ValidationMetrics] = None,
    dspy_serialized: Optional[Dict[str, Any]] = None,
    k_window_size: Optional[int] = None,
) -> CompiledArtifact:
    """Create a compiled artifact with the given parameters.
    
    Args:
        component: Component name
        signature_name: DSPy signature name
        student_model: Student model name
        teacher_model: Teacher model name
        optimizer_params: Optimizer parameters used
        few_shots: Few-shot examples (optional)
        system_prompt: System prompt (optional)
        validation_metrics: Validation metrics (optional)
        dspy_serialized: Raw DSPy serialization (optional)
        k_window_size: Context window size (k) used for training data (optional)
        
    Returns:
        Validated compiled artifact
    """
    return CompiledArtifact(
        component=component,
        signature_name=signature_name,
        student_model=student_model,
        teacher_model=teacher_model,
        optimizer_params=optimizer_params,
        few_shots=few_shots or [],
        system_prompt=system_prompt,
        program_style=program_style,
        validation_metrics=validation_metrics,
        dspy_serialized=dspy_serialized,
        k_window_size=k_window_size,
    )
