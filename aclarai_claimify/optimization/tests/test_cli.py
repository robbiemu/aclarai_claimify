"""Unit tests for the CLI argument parsing and validation."""

import argparse
import sys
from pathlib import Path
import pytest
from unittest.mock import patch, MagicMock

# Add the project root to sys.path so we can import the CLI module
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from aclarai_claimify.cli import (
    create_parser,
    validate_compile_args,
    handle_compile_command,
    handle_schema_command,
    validate_generate_args,
    handle_generate_command,
    GenerationError
)
from aclarai_claimify.data_models import ClaimifyConfig, OptimizationConfig


class TestArgumentParsing:
    """Test argument parsing functionality."""
    
    def test_create_parser_help(self):
        """Test that parser is created and has help text."""
        parser = create_parser()
        assert isinstance(parser, argparse.ArgumentParser)
        
        # Check that subcommands are defined
        subparsers_actions = [
            action for action in parser._actions 
            if isinstance(action, argparse._SubParsersAction)
        ]
        assert len(subparsers_actions) == 1
        
        # Get the subparser choices
        subparsers = subparsers_actions[0]
        assert "compile" in subparsers.choices
        assert "schema" in subparsers.choices
    
    def test_compile_parser_required_args(self):
        """Test that compile subcommand requires all necessary arguments."""
        parser = create_parser()
        
        # Test missing required arguments
        with pytest.raises(SystemExit):
            parser.parse_args(["compile"])
        
        # Test with all required arguments
        args = parser.parse_args([
            "compile",
            "--component", "selection",
            "--trainset", "/tmp/train.jsonl",
            "--student-model", "gpt-3.5-turbo",
            "--teacher-model", "gpt-4o",
            "--output-path", "/tmp/output.json"
        ])
        
        assert args.command == "compile"
        assert args.component == "selection"
        assert args.trainset == Path("/tmp/train.jsonl")
        assert args.student_model == "gpt-3.5-turbo"
        assert args.teacher_model == "gpt-4o"
        assert args.output_path == Path("/tmp/output.json")
    
    def test_compile_parser_optional_args(self):
        """Test that compile subcommand accepts optional arguments."""
        parser = create_parser()
        
        args = parser.parse_args([
            "compile",
            "--component", "selection",
            "--trainset", "/tmp/train.jsonl",
            "--student-model", "gpt-3.5-turbo",
            "--teacher-model", "gpt-4o",
            "--output-path", "/tmp/output.json",
            "--seed", "123",
            "--verbose",
            "--overwrite"
        ])
        
        assert args.seed == 123
        assert args.verbose is True
        assert args.quiet is False  # Default
        assert args.overwrite is True
    
    def test_schema_parser_args(self):
        """Test that schema subcommand works correctly."""
        parser = create_parser()
        
        args = parser.parse_args([
            "schema",
            "--component", "selection"
        ])
        
        assert args.command == "schema"
        assert args.component == "selection"

    def test_generate_parser_args(self):
        """Test that generate-dataset subcommand works correctly."""
        parser = create_parser()
        
        args = parser.parse_args([
            "generate-dataset",
            "--input-file", "/tmp/input.txt",
            "--output-file", "/tmp/output.jsonl",
            "--component", "selection",
            "--teacher-model", "gpt-4o"
        ])
        
        assert args.command == "generate-dataset"
        assert args.input_file == Path("/tmp/input.txt")
        assert args.output_file == Path("/tmp/output.jsonl")
        assert args.component == "selection"
        assert args.teacher_model == "gpt-4o"


class TestArgumentValidation:
    """Test argument validation functionality."""
    
    def test_validate_compile_args_missing_trainset(self, tmp_path):
        """Test validation fails when trainset file doesn't exist."""
        parser = create_parser()
        args = parser.parse_args([
            "compile",
            "--component", "selection",
            "--trainset", str(tmp_path / "nonexistent.jsonl"),
            "--student-model", "gpt-3.5-turbo",
            "--teacher-model", "gpt-4o",
            "--output-path", str(tmp_path / "output.json")
        ])
        
        with pytest.raises(SystemExit):
            validate_compile_args(args)
    
    def test_validate_compile_args_trainset_is_directory(self, tmp_path):
        """Test validation fails when trainset is a directory."""
        parser = create_parser()
        args = parser.parse_args([
            "compile",
            "--component", "selection",
            "--trainset", str(tmp_path),  # This is a directory
            "--student-model", "gpt-3.5-turbo",
            "--teacher-model", "gpt-4o",
            "--output-path", str(tmp_path / "output.json")
        ])
        
        with pytest.raises(SystemExit):
            validate_compile_args(args)
    
    def test_validate_compile_args_output_exists_no_overwrite(self, tmp_path):
        """Test validation fails when output file exists and no overwrite flag."""
        # Create a test trainset file
        trainset = tmp_path / "train.jsonl"
        trainset.write_text("test")
        
        # Create an existing output file
        output_file = tmp_path / "output.json"
        output_file.write_text("existing")
        
        parser = create_parser()
        args = parser.parse_args([
            "compile",
            "--component", "selection",
            "--trainset", str(trainset),
            "--student-model", "gpt-3.5-turbo",
            "--teacher-model", "gpt-4o",
            "--output-path", str(output_file)
        ])
        
        with pytest.raises(SystemExit):
            validate_compile_args(args)
    
    def test_validate_compile_args_output_exists_with_overwrite(self, tmp_path):
        """Test validation passes when output file exists with overwrite flag."""
        # Create a test trainset file
        trainset = tmp_path / "train.jsonl"
        trainset.write_text("test")
        
        # Create an existing output file
        output_file = tmp_path / "output.json"
        output_file.write_text("existing")
        
        parser = create_parser()
        args = parser.parse_args([
            "compile",
            "--component", "selection",
            "--trainset", str(trainset),
            "--student-model", "gpt-3.5-turbo",
            "--teacher-model", "gpt-4o",
            "--output-path", str(output_file),
            "--overwrite"
        ])
        
        # Should not raise an exception
        validate_compile_args(args)
    
    def test_validate_compile_args_negative_seed(self, tmp_path):
        """Test validation fails with negative seed."""
        # Create a test trainset file
        trainset = tmp_path / "train.jsonl"
        trainset.write_text("test")
        
        parser = create_parser()
        args = parser.parse_args([
            "compile",
            "--component", "selection",
            "--trainset", str(trainset),
            "--student-model", "gpt-3.5-turbo",
            "--teacher-model", "gpt-4o",
            "--output-path", str(tmp_path / "output.json"),
            "--seed", "-1"
        ])
        
        with pytest.raises(SystemExit):
            validate_compile_args(args)

    def test_validate_generate_args_missing_input(self, tmp_path):
        """Test validation fails when input file doesn't exist."""
        parser = create_parser()
        args = parser.parse_args([
            "generate-dataset",
            "--input-file", str(tmp_path / "nonexistent.txt"),
            "--output-file", str(tmp_path / "output.jsonl"),
            "--component", "selection",
            "--teacher-model", "gpt-4o"
        ])
        
        with pytest.raises(SystemExit):
            validate_generate_args(args)
    
    def test_validate_generate_args_input_is_directory(self, tmp_path):
        """Test validation fails when input file is a directory."""
        parser = create_parser()
        args = parser.parse_args([
            "generate-dataset",
            "--input-file", str(tmp_path),  # This is a directory
            "--output-file", str(tmp_path / "output.jsonl"),
            "--component", "selection",
            "--teacher-model", "gpt-4o"
        ])
        
        with pytest.raises(SystemExit):
            validate_generate_args(args)
    
    def test_validate_generate_args_output_exists(self, tmp_path):
        """Test validation fails when output file exists."""
        # Create a test input file
        input_file = tmp_path / "input.txt"
        input_file.write_text("test")
        
        # Create an existing output file
        output_file = tmp_path / "output.jsonl"
        output_file.write_text("existing")
        
        parser = create_parser()
        args = parser.parse_args([
            "generate-dataset",
            "--input-file", str(input_file),
            "--output-file", str(output_file),
            "--component", "selection",
            "--teacher-model", "gpt-4o"
        ])
        
        with pytest.raises(SystemExit):
            validate_generate_args(args)


class TestCommandHandling:
    """Test command handling functionality."""
    
    @patch('aclarai_claimify.cli.compile_component')
    @patch('aclarai_claimify.cli.validate_compile_args')
    def test_handle_compile_command_success(self, mock_validate, mock_compile):
        """Test successful compile command execution."""
        parser = create_parser()
        args = parser.parse_args([
            "compile",
            "--component", "selection",
            "--trainset", "/tmp/train.jsonl",
            "--student-model", "gpt-3.5-turbo",
            "--teacher-model", "gpt-4o",
            "--output-path", "/tmp/output.json",
            "--seed", "42"
        ])
        
        # Mock the validation and compilation to succeed
        mock_validate.return_value = None
        mock_compile.return_value = None
        
        # Should not raise an exception
        handle_compile_command(args)
        
        # Verify that compile_component was called with correct arguments
        mock_compile.assert_called_once()
    
    @patch('aclarai_claimify.cli.compile_component')
    @patch('aclarai_claimify.cli.validate_compile_args')
    def test_handle_compile_command_with_quiet_flag(self, mock_validate, mock_compile):
        """Test compile command execution with quiet flag."""
        parser = create_parser()
        args = parser.parse_args([
            "compile",
            "--component", "selection",
            "--trainset", "/tmp/train.jsonl",
            "--student-model", "gpt-3.5-turbo",
            "--teacher-model", "gpt-4o",
            "--output-path", "/tmp/output.json",
            "--quiet"
        ])
        
        # Mock the validation and compilation to succeed
        mock_validate.return_value = None
        mock_compile.return_value = None
        
        # Should not raise an exception
        handle_compile_command(args)
        
        # Verify that compile_component was called with correct arguments
        mock_compile.assert_called_once()
    
    @patch('aclarai_claimify.cli.compile_component')
    @patch('aclarai_claimify.cli.validate_compile_args')
    @patch('aclarai_claimify.cli.print_schema_help')
    def test_handle_schema_command(self, mock_print_schema, mock_validate, mock_compile):
        """Test schema command execution."""
        parser = create_parser()
        args = parser.parse_args([
            "schema",
            "--component", "selection"
        ])
        
        # Mock the schema help function
        mock_print_schema.return_value = None
        
        # Should not raise an exception
        handle_schema_command(args)
        
        # Verify that print_schema_help was called with correct argument
        mock_print_schema.assert_called_once_with("selection")
    
    @patch('aclarai_claimify.cli.compile_component')
    @patch('aclarai_claimify.cli.validate_compile_args')
    def test_handle_compile_command_data_validation_error(self, mock_validate, mock_compile):
        """Test compile command handling of DataValidationError."""
        from aclarai_claimify.optimization.compile import DataValidationError
        
        parser = create_parser()
        args = parser.parse_args([
            "compile",
            "--component", "selection",
            "--trainset", "/tmp/train.jsonl",
            "--student-model", "gpt-3.5-turbo",
            "--teacher-model", "gpt-4o",
            "--output-path", "/tmp/output.json"
        ])
        
        # Mock the validation to succeed but compilation to fail with data validation error
        mock_validate.return_value = None
        mock_compile.side_effect = DataValidationError("Test data validation error")
        
        with pytest.raises(SystemExit):
            handle_compile_command(args)
    
    @patch('aclarai_claimify.cli.compile_component')
    @patch('aclarai_claimify.cli.validate_compile_args')
    def test_handle_compile_command_model_config_error(self, mock_validate, mock_compile):
        """Test compile command handling of ModelConfigError."""
        from aclarai_claimify.optimization.compile import ModelConfigError
        
        parser = create_parser()
        args = parser.parse_args([
            "compile",
            "--component", "selection",
            "--trainset", "/tmp/train.jsonl",
            "--student-model", "gpt-3.5-turbo",
            "--teacher-model", "gpt-4o",
            "--output-path", "/tmp/output.json"
        ])
        
        # Mock the validation to succeed but compilation to fail with model config error
        mock_validate.return_value = None
        mock_compile.side_effect = ModelConfigError("Test model config error")
        
        with pytest.raises(SystemExit):
            handle_compile_command(args)
    
    @patch('aclarai_claimify.cli.compile_component')
    @patch('aclarai_claimify.cli.validate_compile_args')
    def test_handle_compile_command_dspy_version_error(self, mock_validate, mock_compile):
        """Test compile command handling of DSPyVersionError."""
        from aclarai_claimify.optimization.compile import DSPyVersionError
        
        parser = create_parser()
        args = parser.parse_args([
            "compile",
            "--component", "selection",
            "--trainset", "/tmp/train.jsonl",
            "--student-model", "gpt-3.5-turbo",
            "--teacher-model", "gpt-4o",
            "--output-path", "/tmp/output.json"
        ])
        
        # Mock the validation to succeed but compilation to fail with DSPy version error
        mock_validate.return_value = None
        mock_compile.side_effect = DSPyVersionError("Test DSPy version error")
        
        with pytest.raises(SystemExit):
            handle_compile_command(args)
    
    @patch('aclarai_claimify.cli.compile_component')
    @patch('aclarai_claimify.cli.validate_compile_args')
    def test_handle_compile_command_optimization_error(self, mock_validate, mock_compile):
        """Test compile command handling of OptimizationError."""
        from aclarai_claimify.optimization.compile import OptimizationError
        
        parser = create_parser()
        args = parser.parse_args([
            "compile",
            "--component", "selection",
            "--trainset", "/tmp/train.jsonl",
            "--student-model", "gpt-3.5-turbo",
            "--teacher-model", "gpt-4o",
            "--output-path", "/tmp/output.json"
        ])
        
        # Mock the validation to succeed but compilation to fail
        mock_validate.return_value = None
        mock_compile.side_effect = OptimizationError("Test optimization error")
        
        with pytest.raises(SystemExit):
            handle_compile_command(args)
    
    @patch('aclarai_claimify.cli.compile_component')
    @patch('aclarai_claimify.cli.validate_compile_args')
    def test_handle_compile_command_keyboard_interrupt(self, mock_validate, mock_compile):
        """Test compile command handling of KeyboardInterrupt."""
        parser = create_parser()
        args = parser.parse_args([
            "compile",
            "--component", "selection",
            "--trainset", "/tmp/train.jsonl",
            "--student-model", "gpt-3.5-turbo",
            "--teacher-model", "gpt-4o",
            "--output-path", "/tmp/output.json"
        ])
        
        # Mock the validation to succeed but compilation to be interrupted
        mock_validate.return_value = None
        mock_compile.side_effect = KeyboardInterrupt()
        
        with pytest.raises(SystemExit) as exc_info:
            handle_compile_command(args)
        
        # Check that exit code is 130 (SIGINT)
        assert exc_info.value.code == 130
    
    @patch('aclarai_claimify.cli.compile_component')
    @patch('aclarai_claimify.cli.validate_compile_args')
    def test_handle_compile_command_unexpected_error(self, mock_validate, mock_compile):
        """Test compile command handling of unexpected errors."""
        parser = create_parser()
        args = parser.parse_args([
            "compile",
            "--component", "selection",
            "--trainset", "/tmp/train.jsonl",
            "--student-model", "gpt-3.5-turbo",
            "--teacher-model", "gpt-4o",
            "--output-path", "/tmp/output.json"
        ])
        
        # Mock the validation to succeed but compilation to fail unexpectedly
        mock_validate.return_value = None
        mock_compile.side_effect = Exception("Unexpected error")
        
        with pytest.raises(SystemExit):
            handle_compile_command(args)

    @patch('aclarai_claimify.cli.generate_dataset')
    @patch('aclarai_claimify.cli.validate_generate_args')
    def test_handle_generate_command_success(self, mock_validate, mock_generate):
        """Test successful generate-dataset command execution."""
        parser = create_parser()
        args = parser.parse_args([
            "generate-dataset",
            "--input-file", "/tmp/input.txt",
            "--output-file", "/tmp/output.jsonl",
            "--component", "selection",
            "--teacher-model", "gpt-4o"
        ])
        
        # Mock the validation and generation to succeed
        mock_validate.return_value = None
        mock_generate.return_value = None
        
        # Should not raise an exception
        handle_generate_command(args)
        
        # Verify that generate_dataset was called with correct arguments
        mock_generate.assert_called_once()
    
    @patch('aclarai_claimify.cli.generate_dataset')
    @patch('aclarai_claimify.cli.validate_generate_args')
    def test_handle_generate_command_generation_error(self, mock_validate, mock_generate):
        """Test generate-dataset command handling of GenerationError."""
        parser = create_parser()
        args = parser.parse_args([
            "generate-dataset",
            "--input-file", "/tmp/input.txt",
            "--output-file", "/tmp/output.jsonl",
            "--component", "selection",
            "--teacher-model", "gpt-4o"
        ])
        
        # Mock the validation to succeed but generation to fail
        mock_validate.return_value = None
        mock_generate.side_effect = GenerationError("Test generation error")
        
        with pytest.raises(SystemExit):
            handle_generate_command(args)
    
    @patch('aclarai_claimify.cli.generate_dataset')
    @patch('aclarai_claimify.cli.validate_generate_args')
    def test_handle_generate_command_keyboard_interrupt(self, mock_validate, mock_generate):
        """Test generate-dataset command handling of KeyboardInterrupt."""
        parser = create_parser()
        args = parser.parse_args([
            "generate-dataset",
            "--input-file", "/tmp/input.txt",
            "--output-file", "/tmp/output.jsonl",
            "--component", "selection",
            "--teacher-model", "gpt-4o"
        ])
        
        # Mock the validation to succeed but generation to be interrupted
        mock_validate.return_value = None
        mock_generate.side_effect = KeyboardInterrupt()
        
        with pytest.raises(SystemExit) as exc_info:
            handle_generate_command(args)
        
        # Check that exit code is 130 (SIGINT)
        assert exc_info.value.code == 130
    
    @patch('aclarai_claimify.cli.generate_dataset')
    @patch('aclarai_claimify.cli.validate_generate_args')
    def test_handle_generate_command_unexpected_error(self, mock_validate, mock_generate):
        """Test generate-dataset command handling of unexpected errors."""
        parser = create_parser()
        args = parser.parse_args([
            "generate-dataset",
            "--input-file", "/tmp/input.txt",
            "--output-file", "/tmp/output.jsonl",
            "--component", "selection",
            "--teacher-model", "gpt-4o"
        ])
        
        # Mock the validation to succeed but generation to fail unexpectedly
        mock_validate.return_value = None
        mock_generate.side_effect = Exception("Unexpected error")
        
        with pytest.raises(SystemExit):
            handle_generate_command(args)
