"""Tests for artifact serialization functionality."""

import pytest
from pathlib import Path
from datetime import datetime

from ..artifacts import (
    CompiledArtifact,
    OptimizerParams,
    ValidationMetrics,
    FewShotExample,
    save_artifact,
    load_artifact,
)


def test_artifact_save_load_roundtrip(tmp_path: Path) -> None:
    """Test that artifacts can be saved and loaded correctly."""
    # Create a sample artifact with all fields populated
    optimizer_params = OptimizerParams(
        optimizer_name="bootstrap-fewshot",
        seed=42,
        other_params={"max_bootstrapped_demos": 8, "max_labeled_demos": 40}
    )
    
    validation_metrics = ValidationMetrics(
        metric_name="accuracy",
        score=0.85,
        n_val=100,
        per_example_diagnostics=[
            {"example_id": 0, "score": 1.0, "inputs": {"text": "test"}, "expected_output": {"result": "pass"}}
        ]
    )
    
    few_shots = [
        FewShotExample(
            inputs={"question": "What is 2+2?"},
            output={"answer": "4"},
            rationale="Simple addition"
        )
    ]
    
    original_artifact = CompiledArtifact(
        component="selection",
        signature_name="SelectionSignature",
        student_model="gpt-3.5-turbo",
        teacher_model="gpt-4o",
        optimizer_params=optimizer_params,
        few_shots=few_shots,
        system_prompt="You are a helpful assistant.",
        validation_metrics=validation_metrics,
        dspy_serialized={"module": "serialized_data"}
    )
    
    # Save to a temporary file
    artifact_path = tmp_path / "test_artifact.json"
    save_artifact(original_artifact, artifact_path)
    
    # Load the artifact back
    loaded_artifact = load_artifact(artifact_path)
    
    # Assert that all fields match
    assert loaded_artifact.component == original_artifact.component
    assert loaded_artifact.signature_name == original_artifact.signature_name
    assert loaded_artifact.student_model == original_artifact.student_model
    assert loaded_artifact.teacher_model == original_artifact.teacher_model
    
    # Check optimizer params
    assert loaded_artifact.optimizer_params.optimizer_name == optimizer_params.optimizer_name
    assert loaded_artifact.optimizer_params.seed == optimizer_params.seed
    assert loaded_artifact.optimizer_params.other_params == optimizer_params.other_params
    
    # Check validation metrics
    assert loaded_artifact.validation_metrics.metric_name == validation_metrics.metric_name
    assert loaded_artifact.validation_metrics.score == validation_metrics.score
    assert loaded_artifact.validation_metrics.n_val == validation_metrics.n_val
    assert (loaded_artifact.validation_metrics.per_example_diagnostics == 
            validation_metrics.per_example_diagnostics)
    
    # Check few-shots
    assert len(loaded_artifact.few_shots) == len(few_shots)
    assert loaded_artifact.few_shots[0].inputs == few_shots[0].inputs
    assert loaded_artifact.few_shots[0].output == few_shots[0].output
    assert loaded_artifact.few_shots[0].rationale == few_shots[0].rationale
    
    # Check other fields
    assert loaded_artifact.system_prompt == original_artifact.system_prompt
    assert loaded_artifact.dspy_serialized == original_artifact.dspy_serialized
    
    # Check that timestamps are reasonable (created during test)
    loaded_timestamp = datetime.fromisoformat(loaded_artifact.created_at)
    assert loaded_timestamp <= datetime.now()