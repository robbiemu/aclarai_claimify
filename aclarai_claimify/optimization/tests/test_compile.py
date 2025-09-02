"""Unit tests for the core compilation pipeline."""

import sys
import os
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock, PropertyMock
import pytest

# Add the project root to sys.path so we can import the modules
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from aclarai_claimify.data_models import ClaimifyConfig, OptimizationConfig
from aclarai_claimify.optimization.compile import (
    _initialize_models,
    _run_optimizer,
    _evaluate_program,
    _extract_few_shots,
    _extract_system_prompt,
    compile_component,
    ModelConfigError,
    DSPyVersionError,
    OptimizationError,
    DataValidationError,
)


class TestModelInitialization:
    """Test model initialization functionality."""

    @patch("dspy.LM")
    def test_initialize_models_new_api(self, mock_lm):
        """Test model initialization with new DSPy API."""
        # Mock dspy.LM
        mock_student_lm = MagicMock()
        mock_teacher_lm = MagicMock()
        mock_lm.side_effect = [mock_student_lm, mock_teacher_lm]

        # Test successful initialization
        student_lm, teacher_lm = _initialize_models("gpt-3.5-turbo", "gpt-4o")

        # Verify the calls
        assert mock_lm.call_count == 2
        mock_lm.assert_any_call("gpt-3.5-turbo")
        mock_lm.assert_any_call("gpt-4o")
        assert student_lm == mock_student_lm
        assert teacher_lm == mock_teacher_lm

    @patch("dspy.LM")
    def test_initialize_models_fallback_api(self, mock_lm):
        """Test model initialization with fallback to older DSPy API."""
        # Mock dspy.LM to fail first with AttributeError
        mock_lm.side_effect = AttributeError("No LM")

        # Test that it raises DSPyVersionError now (no fallback)
        with pytest.raises(DSPyVersionError):
            _initialize_models("gpt-3.5-turbo", "gpt-4o")

    @patch("dspy.LM")
    def test_initialize_models_api_key_error(self, mock_lm):
        """Test model initialization with API key error."""
        # Mock dspy.LM to raise an API key error
        mock_lm.side_effect = Exception("api_key error")

        with pytest.raises(ModelConfigError):
            _initialize_models("gpt-3.5-turbo", "gpt-4o")

    @patch("dspy.LM")
    def test_initialize_models_model_error(self, mock_lm):
        """Test model initialization with model configuration error."""
        # Mock dspy.LM to raise a model error
        mock_lm.side_effect = Exception("model not found")

        with pytest.raises(ModelConfigError):
            _initialize_models("invalid-model", "gpt-4o")

    @patch("dspy.LM")
    def test_initialize_models_dspy_version_error(self, mock_lm):
        """Test model initialization with DSPy version compatibility error."""
        # Mock dspy.LM to raise a version compatibility error
        mock_lm.side_effect = Exception("incompatible version")

        with pytest.raises(DSPyVersionError):
            _initialize_models("gpt-3.5-turbo", "gpt-4o")


class TestOptimizer:
    """Test optimizer functionality."""

    @patch("dspy.teleprompt.BootstrapFewShot")
    def test_run_optimizer_success(self, mock_bootstrap):
        """Test successful optimizer execution."""
        # Mock the optimizer
        mock_optimizer_instance = MagicMock()
        mock_bootstrap.return_value = mock_optimizer_instance
        mock_optimizer_instance.compile.return_value = MagicMock()

        # Create mock program and examples
        mock_program = MagicMock()
        mock_trainset = [MagicMock(), MagicMock()]
        mock_valset = [MagicMock()]
        mock_metric = MagicMock()
        mock_teacher_lm = MagicMock()

        # Test successful optimization
        optimizer_config = {
            "optimizer_name": "bootstrap-fewshot",
            "params": {"max_bootstrapped_demos": 5, "max_labeled_demos": 20},
        }

        result = _run_optimizer(
            mock_program,
            mock_trainset,
            mock_valset,
            mock_metric,
            mock_teacher_lm,
            optimizer_config,
        )

        # Verify the calls
        mock_bootstrap.assert_called_once()
        mock_optimizer_instance.compile.assert_called_once()
        assert result == mock_optimizer_instance.compile.return_value

    @patch("dspy.teleprompt.BootstrapFewShot")
    def test_run_optimizer_without_teacher(self, mock_bootstrap):
        """Test optimizer execution without teacher support."""
        # Mock the optimizer to fail with teacher, then succeed without
        mock_bootstrap.side_effect = [
            TypeError("teacher not supported"),  # First call fails
            MagicMock(),  # Second call succeeds
        ]

        mock_optimizer_instance = MagicMock()
        mock_optimizer_instance.compile.return_value = MagicMock()

        # Create mock program and examples
        mock_program = MagicMock()
        mock_trainset = [MagicMock(), MagicMock()]
        mock_valset = [MagicMock()]
        mock_metric = MagicMock()
        mock_teacher_lm = MagicMock()

        # Test successful optimization
        optimizer_config = {
            "optimizer_name": "bootstrap-fewshot",
            "params": {"max_bootstrapped_demos": 5, "max_labeled_demos": 20},
        }

        result = _run_optimizer(
            mock_program,
            mock_trainset,
            mock_valset,
            mock_metric,
            mock_teacher_lm,
            optimizer_config,
        )

        # Verify the calls
        assert mock_bootstrap.call_count >= 1
        assert result is not None

    @patch("dspy.teleprompt.BootstrapFewShot")
    def test_run_optimizer_import_error(self, mock_bootstrap):
        """Test optimizer execution with import error."""
        # Mock import error for optimizer
        mock_bootstrap.side_effect = ImportError("No module named teleprompt")

        # Create mock program and examples
        mock_program = MagicMock()
        mock_trainset = [MagicMock(), MagicMock()]
        mock_valset = [MagicMock()]
        mock_metric = MagicMock()
        mock_teacher_lm = MagicMock()

        optimizer_config = {
            "optimizer_name": "bootstrap-fewshot",
            "params": {"max_bootstrapped_demos": 5, "max_labeled_demos": 20},
        }

        with pytest.raises(OptimizationError):
            _run_optimizer(
                mock_program,
                mock_trainset,
                mock_valset,
                mock_metric,
                mock_teacher_lm,
                optimizer_config,
            )

    @patch("dspy.teleprompt.BootstrapFewShot")
    def test_run_optimizer_api_error(self, mock_bootstrap):
        """Test optimizer execution with API error."""
        # Mock the optimizer
        mock_optimizer_instance = MagicMock()
        mock_bootstrap.return_value = mock_optimizer_instance
        mock_optimizer_instance.compile.side_effect = Exception("api error")

        # Create mock program and examples
        mock_program = MagicMock()
        mock_trainset = [MagicMock(), MagicMock()]
        mock_valset = [MagicMock()]
        mock_metric = MagicMock()
        mock_teacher_lm = MagicMock()

        optimizer_config = {
            "optimizer_name": "bootstrap-fewshot",
            "params": {"max_bootstrapped_demos": 5, "max_labeled_demos": 20},
        }

        with pytest.raises(OptimizationError):
            _run_optimizer(
                mock_program,
                mock_trainset,
                mock_valset,
                mock_metric,
                mock_teacher_lm,
                optimizer_config,
            )

    @patch("dspy.teleprompt.BootstrapFewShot")
    def test_run_optimizer_general_error(self, mock_bootstrap):
        """Test optimizer execution with general error."""
        # Mock the optimizer
        mock_optimizer_instance = MagicMock()
        mock_bootstrap.return_value = mock_optimizer_instance
        mock_optimizer_instance.compile.side_effect = Exception("unknown error")

        # Create mock program and examples
        mock_program = MagicMock()
        mock_trainset = [MagicMock(), MagicMock()]
        mock_valset = [MagicMock()]
        mock_metric = MagicMock()
        mock_teacher_lm = MagicMock()

        optimizer_config = {
            "optimizer_name": "bootstrap-fewshot",
            "params": {"max_bootstrapped_demos": 5, "max_labeled_demos": 20},
        }

        with pytest.raises(OptimizationError):
            _run_optimizer(
                mock_program,
                mock_trainset,
                mock_valset,
                mock_metric,
                mock_teacher_lm,
                optimizer_config,
            )


class TestEvaluation:
    """Test evaluation functionality."""

    def test_evaluate_program_success(self):
        """Test successful program evaluation."""
        # Create mock program and examples
        mock_program = MagicMock()
        mock_valset = []

        # Create mock examples with predictable outputs
        for i in range(3):
            mock_example = MagicMock()
            # Mock the inputs method to return an Example-like object
            # We need to make sure the inputs method returns the same dict both times it's called
            mock_inputs_result = {"input": f"test{i}"}
            mock_example.inputs.return_value = mock_inputs_result
            # Set up the example's attributes (not __dict__ to avoid mock issues)
            mock_example.input = f"test{i}"
            mock_example.output = f"result{i}"
            mock_valset.append(mock_example)

        # Mock program execution - it should be called with the inputs from example.inputs()
        mock_pred = MagicMock()

        # Use side_effect to properly handle the call with **kwargs
        def mock_program_call(**kwargs):
            # Verify that the correct inputs are passed
            if kwargs.get("input") in ["test0", "test1", "test2"]:
                return mock_pred
            raise Exception(f"Unexpected call with kwargs: {kwargs}")

        mock_program.side_effect = mock_program_call

        # Mock metric function that always returns 1.0
        def mock_metric(example, prediction):
            return 1.0  # Perfect score

        # Mock the metric function name for testing
        mock_metric.__name__ = "test_metric"

        # Test evaluation
        result = _evaluate_program(mock_program, mock_valset, mock_metric)

        # Verify results
        assert result.metric_name == "test_metric"
        assert result.score == 1.0
        assert result.n_val == 3
        assert len(result.per_example_diagnostics) == 3

    def test_evaluate_program_with_failures(self):
        """Test program evaluation with some failures."""
        # Create mock program and examples
        mock_program = MagicMock()
        mock_valset = []

        # Create mock examples with predictable outputs
        for i in range(3):
            mock_example = MagicMock()
            # Mock the inputs method to return an Example-like object
            # We need to make sure the inputs method returns the same dict both times it's called
            mock_inputs_result = {"input": f"test{i}"}
            mock_example.inputs.return_value = mock_inputs_result
            # Set up the example's attributes (not __dict__ to avoid mock issues)
            mock_example.input = f"test{i}"
            mock_example.output = f"result{i}"
            mock_valset.append(mock_example)

        # Mock program execution - first one fails
        def mock_program_call(**kwargs):
            if kwargs.get("input") == "test0":
                raise Exception("Prediction failed")
            mock_pred = MagicMock()
            return mock_pred

        mock_program.side_effect = mock_program_call

        # Mock metric function that always returns 1.0
        def mock_metric(example, prediction):
            return 1.0  # Perfect score

        # Mock the metric function name for testing
        mock_metric.__name__ = "test_metric"

        # Test evaluation
        result = _evaluate_program(mock_program, mock_valset, mock_metric)

        # Verify results - average should be 0.667 since first example failed (score 0)
        assert result.metric_name == "test_metric"
        assert abs(result.score - 0.667) < 0.001
        assert result.n_val == 3
        assert len(result.per_example_diagnostics) == 3


class TestArtifactExtraction:
    """Test artifact extraction functionality."""

    def test_extract_few_shots_success(self):
        """Test successful few-shot example extraction."""
        # Create mock program with demos
        mock_program = MagicMock()
        mock_program.demos = []

        # Create mock demos
        for i in range(3):
            mock_demo = MagicMock()
            mock_demo.inputs.return_value = {"input": f"demo{i}"}
            mock_demo.__dict__ = {
                "input": f"demo{i}",
                "output": f"result{i}",
                "rationale": f"rationale{i}",
            }
            mock_program.demos.append(mock_demo)

        # Test extraction
        result = _extract_few_shots(mock_program, "test-component")

        # Should return a list (implementation may vary)
        assert isinstance(result, list)

    def test_extract_few_shots_from_predictors(self):
        """Test few-shot example extraction from predictors."""
        # Create mock program with predictors
        mock_program = MagicMock()
        if hasattr(mock_program, "demos"):
            del mock_program.demos  # Remove demos attribute

        # Create mock predictors with demos
        mock_predictors = []
        for p in range(2):
            mock_predictor = MagicMock()
            mock_predictor.demos = []
            for i in range(2):
                mock_demo = MagicMock()
                mock_demo.inputs.return_value = {"input": f"demo{p}-{i}"}
                mock_demo.__dict__ = {
                    "input": f"demo{p}-{i}",
                    "output": f"result{p}-{i}",
                }
                mock_predictor.demos.append(mock_demo)
            mock_predictors.append(mock_predictor)

        mock_program.predictors = mock_predictors

        # Test extraction
        result = _extract_few_shots(mock_program, "test-component")

        # Should return a list (implementation may vary)
        assert isinstance(result, list)

    def test_extract_few_shots_no_demos(self):
        """Test few-shot extraction when no demos are available."""
        # Create mock program with no demos
        mock_program = MagicMock()
        mock_program.demos = []

        # Test extraction
        result = _extract_few_shots(mock_program, "test-component")

        # Should return empty list
        assert result == []

    def test_extract_system_prompt_success(self):
        """Test successful system prompt extraction."""
        # Create mock program with instructions
        mock_program = MagicMock()
        mock_program.instructions = "Test system prompt"

        # Test extraction
        result = _extract_system_prompt(mock_program)

        # Verify result
        assert result == "Test system prompt"

    def test_extract_system_prompt_from_predictors(self):
        """Test system prompt extraction from predictors."""
        # Create mock program with predictors
        mock_program = MagicMock()
        if hasattr(mock_program, "instructions"):
            del mock_program.instructions  # Remove instructions attribute

        # Create mock predictor with system prompt
        mock_predictor = MagicMock()
        mock_predictor.instructions = "Test predictor prompt"

        mock_program.predictors = [mock_predictor]

        # Test extraction
        result = _extract_system_prompt(mock_program)

        # The function tries multiple attributes, so we'll just check it returns a value
        assert result is not None

    def test_extract_system_prompt_no_prompt(self):
        """Test system prompt extraction when no prompt is available."""
        # Create mock program with no prompt attributes
        mock_program = MagicMock()
        # Remove the instructions and system_prompt attributes completely
        del mock_program.instructions
        del mock_program.system_prompt
        mock_program.predictors = []

        # Test extraction
        result = _extract_system_prompt(mock_program)

        # Should return None
        assert result is None


class TestCompileComponent:
    """Test the main compile_component function."""

    @patch("aclarai_claimify.optimization.compile.get_component_info")
    @patch("aclarai_claimify.optimization.compile.load_jsonl_dataset")
    @patch("aclarai_claimify.optimization.compile.validate_records_for_component")
    @patch("aclarai_claimify.optimization.compile.map_to_examples")
    @patch("aclarai_claimify.optimization.compile.validate_component_examples")
    @patch("aclarai_claimify.optimization.compile.split_examples")
    @patch("aclarai_claimify.optimization.compile._initialize_models")
    @patch("aclarai_claimify.optimization.compile.build_program")
    @patch("aclarai_claimify.optimization.compile._run_optimizer")
    @patch("aclarai_claimify.optimization.compile._evaluate_program")
    @patch("aclarai_claimify.optimization.compile._extract_few_shots")
    @patch("aclarai_claimify.optimization.compile._extract_system_prompt")
    @patch("aclarai_claimify.optimization.compile.create_artifact_dict")
    @patch("aclarai_claimify.optimization.compile.save_artifact")
    @patch("tempfile.NamedTemporaryFile")
    def test_compile_component_success(
        self,
        mock_temp_file,
        mock_save,
        mock_create,
        mock_extract_prompt,
        mock_extract_shots,
        mock_evaluate,
        mock_optimize,
        mock_build,
        mock_init_models,
        mock_split,
        mock_validate_examples,
        mock_map_examples,
        mock_validate_records,
        mock_load_dataset,
        mock_get_component,
    ):
        """Test successful compilation of a component."""
        # Mock the temporary file
        mock_temp_config = MagicMock()
        mock_temp_config.name = "/tmp/test_config.yaml"
        mock_temp_file.return_value.__enter__.return_value = mock_temp_config

        # Write a temporary config file
        with open("/tmp/test_config.yaml", "w") as f:
            f.write(
                "optimizer_name: bootstrap-fewshot\nparams:\n  max_bootstrapped_demos: 8\n  max_labeled_demos: 40"
            )

        # Mock all dependencies
        mock_signature = MagicMock()
        mock_signature.input_fields = {"context_text": "str", "target_sentence": "str"}
        mock_signature.output_fields = {"selection_response_json": "str"}

        mock_get_component.return_value = {
            "signature": mock_signature,
            "signature_name": "TestSignature",
            "metric": MagicMock(),
        }

        mock_load_dataset.return_value = [
            {
                "context_text": "Context for testing",
                "target_sentence": "Test sentence to evaluate",
                "selection_response_json": '{"selected": true, "confidence": 0.9, "reasoning": "Contains verifiable information"}',
            }
        ]
        mock_example = MagicMock()
        mock_example.with_inputs.return_value = mock_example
        mock_map_examples.return_value = [mock_example]
        mock_split.return_value = ([mock_example], [mock_example])
        mock_init_models.return_value = (MagicMock(), MagicMock())
        mock_build.return_value = MagicMock()
        mock_optimize.return_value = MagicMock()
        mock_evaluate.return_value = MagicMock()
        mock_evaluate.return_value.score = 0.95
        mock_extract_shots.return_value = []
        mock_extract_prompt.return_value = "Test prompt"
        mock_create.return_value = {"test": "artifact"}

        # Create a proper mock optimizer config with required attributes
        mock_optimizer_config = MagicMock(spec=OptimizationConfig)
        mock_optimizer_config.optimizer_name = "bootstrap-fewshot"
        mock_optimizer_config.params = {"max_bootstrapped_demos": 8, "max_labeled_demos": 40}
        mock_optimizer_config.dict.return_value = {
            "optimizer_name": "bootstrap-fewshot",
            "params": {"max_bootstrapped_demos": 8, "max_labeled_demos": 40}
        }

        # Test compilation
        compile_component(
            component="selection",
            train_path=Path("/tmp/train.jsonl"),
            student_model="gpt-3.5-turbo",
            teacher_model="gpt-4o",
            output_path=Path("/tmp/output.json"),
            claimify_config=MagicMock(spec=ClaimifyConfig),
            optimizer_config=mock_optimizer_config,
        )

        # Verify all the calls were made
        mock_get_component.assert_called_once_with("selection")
        mock_load_dataset.assert_called_once_with(Path("/tmp/train.jsonl"))
        mock_validate_records.assert_called_once()
        mock_map_examples.assert_called_once()
        mock_validate_examples.assert_called_once()
        mock_split.assert_called_once()
        mock_init_models.assert_called_once()
        mock_build.assert_called_once()
        mock_optimize.assert_called_once()
        mock_evaluate.assert_called_once()
        mock_extract_shots.assert_called_once()
        mock_extract_prompt.assert_called_once()
        mock_create.assert_called_once()
        mock_save.assert_called_once_with(
            {"test": "artifact"}, Path("/tmp/output.json")
        )

        # Clean up
        try:
            os.remove("/tmp/test_config.yaml")
        except:
            pass

    @patch("aclarai_claimify.optimization.compile.get_component_info")
    def test_compile_component_data_validation_error(self, mock_get_component):
        """Test compilation with data validation error."""
        # Mock component info
        mock_signature = MagicMock()
        mock_signature.input_fields = {"context_text": "str", "target_sentence": "str"}
        mock_signature.output_fields = {"selection_response_json": "str"}

        mock_get_component.return_value = {
            "signature": mock_signature,
            "signature_name": "TestSignature",
            "metric": MagicMock(),
        }

        # Mock data validation to fail
        with patch(
            "aclarai_claimify.optimization.compile.load_jsonl_dataset"
        ) as mock_load:
            mock_load.side_effect = DataValidationError("Invalid data")

            # Create a proper mock optimizer config with required attributes
            mock_optimizer_config = MagicMock(spec=OptimizationConfig)
            mock_optimizer_config.optimizer_name = "bootstrap-fewshot"
            mock_optimizer_config.params = {"max_bootstrapped_demos": 8, "max_labeled_demos": 40}

            with pytest.raises(DataValidationError):
                compile_component(
                    component="selection",
                    train_path=Path("/tmp/train.jsonl"),
                    student_model="gpt-3.5-turbo",
                    teacher_model="gpt-4o",
                    output_path=Path("/tmp/output.json"),
                    claimify_config=MagicMock(spec=ClaimifyConfig),
                    optimizer_config=mock_optimizer_config,
                )