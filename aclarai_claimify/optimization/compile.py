"""Core compilation pipeline for DSPy optimization.

This module provides the main compilation pipeline that orchestrates
the entire process of optimizing Claimify components using DSPy.
"""

import os
from pathlib import Path
from typing import Optional, List, Dict, Any

try:
    import dspy
    import yaml
except ImportError as e:
    missing = "yaml" if "yaml" in str(e) else "dspy"
    raise ImportError(
        "DSPy optimization features require additional dependencies. Install with:\
"
        "pip install 'aclarai-claimify[optimization]'"
    ) from e

from ..data_models import ClaimifyConfig, OptimizationConfig
from .artifacts import (
    OptimizerParams,
    ValidationMetrics,
    FewShotExample,
    save_artifact,
    create_artifact_dict,
)
from .components import get_component_info, build_program, validate_component_examples
from .data import (
    load_jsonl_dataset,
    validate_records_for_component,
    map_to_examples,
    split_examples,
    DataValidationError,
)


class ModelConfigError(Exception):
    """Raised when model configuration fails."""

    pass


class DSPyVersionError(Exception):
    """Raised when DSPy version compatibility issues occur."""

    pass


class OptimizationError(Exception):
    """Raised when DSPy optimization fails."""

    pass





def _initialize_models(
    student_model: str,
    teacher_model: str,
    verbose: bool = True,
    model_params: Optional[Dict[str, Any]] = None,
) -> tuple:
    """Initialize student and teacher language models.

    Args:
        student_model: Name of the student model (for final program)
        teacher_model: Name of the teacher model (for optimization)
        verbose: Whether to print progress messages
        model_params: Additional model parameters to pass to LiteLLM

    Returns:
        Tuple of (student_lm, teacher_lm) dspy.LM instances

    Raises:
        ModelConfigError: If model configuration fails
        DSPyVersionError: If DSPy version is incompatible
    """
    

    try:
        # Use the new DSPy API (dspy-ai>=2.4.0)
        student_lm = dspy.LM(student_model, **(model_params or {}))
        teacher_lm = dspy.LM(teacher_model, **(model_params or {}))
        if verbose:
            print("   ✅ Using dspy.LM API")

        # Configure DSPy with the student model as default
        dspy.settings.configure(lm=student_lm)

        return student_lm, teacher_lm

    except Exception as e:
        if "api_key" in str(e).lower():
            raise ModelConfigError(f"API key error: {e}") from e
        elif "model" in str(e).lower():
            raise ModelConfigError(f"Model configuration error: {e}") from e
        else:
            raise DSPyVersionError(f"DSPy compatibility error: {e}") from e


def _run_optimizer(
    program: dspy.Module,
    trainset: List[dspy.Example],
    valset: List[dspy.Example],
    metric: callable,
    teacher_lm,
    optimizer_config: dict,
    verbose: bool = True,
) -> dspy.Module:
    """Run DSPy optimization on the program.

    Args:
        program: DSPy program to optimize
        trainset: Training examples
        valset: Validation examples
        metric: Evaluation metric function
        teacher_lm: Teacher language model
        optimizer_config: Optimizer configuration from YAML file
        verbose: Whether to print progress

    Returns:
        Compiled/optimized program

    Raises:
        OptimizationError: If optimization fails
    """
    optimizer_name = optimizer_config["optimizer_name"]
    params = optimizer_config["params"]

    if verbose:
        print(
            f"🚀 Running optimization with {len(trainset)} train, {len(valset)} val examples"
        )
        print(f"   Optimizer: {optimizer_name}")
        print(f"   Parameters: {params}")

    try:
        # Map of supported optimizers
        optimizer_classes = {
            "bootstrap-fewshot": dspy.teleprompt.BootstrapFewShot,
            # Add more optimizers here as needed
        }

        if optimizer_name not in optimizer_classes:
            raise OptimizationError(f"Unsupported optimizer: {optimizer_name}")

        optimizer_class = optimizer_classes[optimizer_name]

        # Add metric to params
        optimizer_params = params.copy()
        optimizer_params["metric"] = metric

        # Try to initialize optimizer with teacher model
        try:
            optimizer = optimizer_class(teacher=teacher_lm, **optimizer_params)
            if verbose:
                print(f"   ✅ Using {optimizer_name} with teacher model")
        except TypeError:
            # Teacher not supported, try without it
            optimizer = optimizer_class(**optimizer_params)
            if verbose:
                print(f"   ✅ Using {optimizer_name} without teacher model")

        if verbose:
            print("   ⏳ Compiling program (this may take a while)...")

        compiled_program = optimizer.compile(program, trainset=trainset)

        if verbose:
            print("   ✅ Optimization completed successfully")

        return compiled_program

    except Exception as e:
        if "api" in str(e).lower() or "openai" in str(e).lower():
            raise OptimizationError(f"API error during optimization: {e}") from e
        else:
            raise OptimizationError(f"Optimization failed: {e}") from e


def _evaluate_program(
    program: dspy.Module,
    valset: List[dspy.Example],
    metric: callable,
    verbose: bool = True,
) -> ValidationMetrics:
    """Evaluate the compiled program on validation set.

    Args:
        program: Compiled program to evaluate
        valset: Validation examples
        metric: Evaluation metric function
        verbose: Whether to print progress

    Returns:
        Validation metrics
    """
    if verbose:
        print(f"📊 Evaluating program on {len(valset)} validation examples")

    scores = []
    diagnostics = []

    for i, example in enumerate(valset):
        score = 0.0  # Default score for failed examples
        inputs = {}  # Default inputs for diagnostics
        try:
            inputs = example.inputs()
            prediction = program(**inputs)
            score = metric(example, prediction)
        except Exception as e:
            if verbose:
                print(f"   ⚠️  Example {i + 1} failed: {e}")

        # Always append the score (0.0 for failed examples)
        scores.append(score)

        # Collect diagnostic info (limit to first 5 examples)
        # Always collect diagnostics, even for failed examples
        if i < 5:
            try:
                inputs_dict = dict(example.inputs().items())
                labels_dict = dict(example.labels().items())

                diagnostics.append(
                    {
                        "example_id": i,
                        "score": score,
                        "inputs": inputs_dict,
                        "expected_output": labels_dict,
                    }
                )
            except Exception as diag_e:
                # If we can't collect diagnostics, at least include basic info
                diagnostics.append(
                    {
                        "example_id": i,
                        "score": score,
                        "inputs": {},
                        "expected_output": {},
                        "error": str(diag_e),
                    }
                )

    avg_score = sum(scores) / len(scores) if scores else 0.0

    if verbose:
        print(f"   📈 Average score: {avg_score:.3f}")

    return ValidationMetrics(
        metric_name=metric.__name__,
        score=avg_score,
        n_val=len(valset),
        per_example_diagnostics=diagnostics[:5],  # Limit diagnostics
    )


def _extract_few_shots(program: dspy.Module, component: str) -> List[FewShotExample]:
    """Extract few-shot examples from compiled program.

    Args:
        program: Compiled DSPy program
        component: Component name for context

    Returns:
        List of few-shot examples
    """
    few_shots = []

    try:
        # Try to introspect the program for demonstrations
        # This is DSPy version-dependent and may not always work
        if hasattr(program, "demos") or hasattr(program, "predictors"):
            # Try to extract from various possible attributes
            demos = getattr(program, "demos", [])
            if not demos and hasattr(program, "predictors"):
                for predictor in program.predictors:
                    if hasattr(predictor, "demos"):
                        demos.extend(predictor.demos)

            for demo in demos[:10]:  # Limit to first 10 demos
                try:
                    if hasattr(demo, "inputs") and hasattr(demo, "__dict__"):
                        inputs = demo.inputs()
                        outputs = {
                            k: v for k, v in demo.__dict__.items() if k not in inputs
                        }

                        few_shot = FewShotExample(
                            inputs=inputs,
                            output=outputs,
                            rationale=getattr(demo, "rationale", None),
                        )
                        few_shots.append(few_shot)
                except Exception:
                    continue  # Skip problematic demos

    except Exception:
        # If introspection fails, return empty list
        pass

    return few_shots


def _extract_system_prompt(program: dspy.Module) -> Optional[str]:
    """Extract system prompt from compiled program.

    Args:
        program: Compiled DSPy program

    Returns:
        System prompt string if available
    """
    try:
        # Try various ways to extract prompt/instructions
        if hasattr(program, "instructions"):
            return program.instructions
        elif hasattr(program, "system_prompt"):
            return program.system_prompt
        elif hasattr(program, "predictors"):
            for predictor in program.predictors:
                if hasattr(predictor, "instructions"):
                    return predictor.instructions
                elif hasattr(predictor, "system_prompt"):
                    return predictor.system_prompt
    except Exception:
        pass

    return None


def compile_component(
    component: str,
    train_path: Path,
    student_model: str,
    teacher_model: str,
    output_path: Path,
    claimify_config: ClaimifyConfig,
    optimizer_config: OptimizationConfig,
    seed: Optional[int] = 42,
    verbose: bool = True,
    model_params: Optional[Dict[str, Any]] = None,
    k_window_size: Optional[int] = None,
    config_path: Optional[Path] = None,  # Keep for logging
) -> None:
    """Compile a Claimify component using DSPy optimization.

    This is the main entry point for the optimization pipeline.

    Args:
        component: Component name (selection, disambiguation, decomposition)
        train_path: Path to training JSONL dataset
        student_model: Model name for final program execution
        teacher_model: Model name for optimization guidance
        output_path: Where to save the compiled artifact
        claimify_config: Main Claimify configuration object
        optimizer_config: Optimizer configuration object
        seed: Random seed for reproducibility
        verbose: Whether to print detailed output
        model_params: Additional model parameters to pass to LiteLLM
        k_window_size: Context window size used for the trainset
        config_path: Path to optimizer config for logging purposes
    """

    if verbose:
        print(f"🎯 Compiling {component} component")
        print(f"   Dataset: {train_path}")
        if config_path:
            print(f"   Config: {config_path}")
        print(f"   Output: {output_path}")

    try:
        # 1. Load and validate component info
        if verbose:
            print("📋 Loading component configuration...")
        component_info = get_component_info(component)
        signature = component_info["signature"]
        metric = component_info["metric"]

        # 2. Load and validate dataset
        if verbose:
            print("📂 Loading and validating dataset...")
        records = load_jsonl_dataset(train_path)
        validate_records_for_component(records, component)

        # Convert to DSPy examples
        examples = map_to_examples(records, signature)
        validate_component_examples(component, examples)

        if verbose:
            print(f"   ✅ Loaded {len(examples)} examples")

        # 3. Split into train/val
        if verbose:
            print("🔀 Splitting dataset...")
        trainset, valset = split_examples(examples, val_ratio=0.2, seed=seed)

        if verbose:
            print(f"   📊 Train: {len(trainset)}, Val: {len(valset)}")

        # 4. Use provided optimizer configuration
        if verbose:
            print("⚙️  Using provided optimizer configuration...")
            print(f"   ✅ Using optimizer: {optimizer_config.optimizer_name}")

        # 5. Initialize models
        student_lm, teacher_lm = _initialize_models(
            student_model, teacher_model, verbose, model_params
        )

        # 6. Build program
        if verbose:
            print("🏗️  Building program...")
        program = build_program(signature, style="cot")

        # 7. Run optimization
        compiled_program = _run_optimizer(
            program,
            trainset,
            valset,
            metric,
            teacher_lm,
            optimizer_config.dict(),
            verbose,
        )

        # 8. Evaluate on validation set
        validation_metrics = _evaluate_program(
            compiled_program, valset, metric, verbose
        )

        # 9. Extract artifacts
        if verbose:
            print("🔍 Extracting program artifacts...")
        few_shots = _extract_few_shots(compiled_program, component)
        system_prompt = _extract_system_prompt(compiled_program)

        if verbose:
            print(f"   📝 Extracted {len(few_shots)} few-shot examples")

        # 10. Create and save artifact
        if verbose:
            print("💾 Creating artifact...")

        # Create optimizer params for artifact
        optimizer_params = OptimizerParams(
            optimizer_name=optimizer_config.optimizer_name,
            seed=seed,
            other_params=optimizer_config.params,
        )

        artifact = create_artifact_dict(
            component=component,
            signature_name=component_info["signature_name"],
            student_model=student_model,
            teacher_model=teacher_model,
            optimizer_params=optimizer_params,
            few_shots=few_shots,
            system_prompt=system_prompt,
            validation_metrics=validation_metrics,
            k_window_size=k_window_size,
        )

        save_artifact(artifact, output_path)

        if verbose:
            print(f"✅ Compilation complete! Artifact saved to: {output_path}")
            print(f"   🎯 Validation score: {validation_metrics.score:.3f}")
            print(f"   📊 Few-shots: {len(few_shots)}")

    except (
        DataValidationError,
        ModelConfigError,
        DSPyVersionError,
        OptimizationError,
    ) as e:
        # These are expected error types with user-friendly messages
        if verbose:
            print(f"❌ {type(e).__name__}: {e}")
        raise
    except Exception as e:
        # Unexpected errors - provide more context
        if verbose:
            print(f"💥 Unexpected error during compilation: {e}")
        raise OptimizationError(f"Compilation failed: {e}") from e
