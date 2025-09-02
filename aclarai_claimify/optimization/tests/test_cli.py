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
    handle_schema_command
)


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
            "--config", "/tmp/config.yaml",
            "--output-path", "/tmp/output.json"
        ])
        
        assert args.command == "compile"
        assert args.component == "selection"
        assert args.trainset == Path("/tmp/train.jsonl")
        assert args.student_model == "gpt-3.5-turbo"
        assert args.teacher_model == "gpt-4o"
        assert args.config == Path("/tmp/config.yaml")
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
            "--config", "/tmp/config.yaml",
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
            "--config", str(tmp_path / "config.yaml"),
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
            "--config", str(tmp_path / "config.yaml"),
            "--output-path", str(tmp_path / "output.json")
        ])
        
        with pytest.raises(SystemExit):
            validate_compile_args(args)
    
    def test_validate_compile_args_output_exists_no_overwrite(self, tmp_path):
        """Test validation fails when output file exists and no overwrite flag."""
        # Create a test trainset file
        trainset = tmp_path / "train.jsonl"
        trainset.write_text("test")
        
        # Create a config file
        config = tmp_path / "config.yaml"
        config.write_text("optimizer_name: bootstrap-fewshot\nparams:\n  max_bootstrapped_demos: 8")
        
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
            "--config", str(config),
            "--output-path", str(output_file)
        ])
        
        with pytest.raises(SystemExit):
            validate_compile_args(args)
    
    def test_validate_compile_args_output_exists_with_overwrite(self, tmp_path):
        """Test validation passes when output file exists with overwrite flag."""
        # Create a test trainset file
        trainset = tmp_path / "train.jsonl"
        trainset.write_text("test")
        
        # Create a config file
        config = tmp_path / "config.yaml"
        config.write_text("optimizer_name: bootstrap-fewshot\nparams:\n  max_bootstrapped_demos: 8")
        
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
            "--config", str(config),
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
        
        # Create a config file
        config = tmp_path / "config.yaml"
        config.write_text("optimizer_name: bootstrap-fewshot\nparams:\n  max_bootstrapped_demos: 8")
        
        parser = create_parser()
        args = parser.parse_args([
            "compile",
            "--component", "selection",
            "--trainset", str(trainset),
            "--student-model", "gpt-3.5-turbo",
            "--teacher-model", "gpt-4o",
            "--config", str(config),
            "--output-path", str(tmp_path / "output.json"),
            "--seed", "-1"
        ])
        
        with pytest.raises(SystemExit):
            validate_compile_args(args)


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
            "--config", "/tmp/config.yaml",
            "--output-path", "/tmp/output.json",
            "--seed", "42"
        ])
        
        # Mock the validation and compilation to succeed
        mock_validate.return_value = None
        mock_compile.return_value = None
        
        # Should not raise an exception
        handle_compile_command(args)
        
        # Verify that compile_component was called with correct arguments
        mock_compile.assert_called_once_with(
            component="selection",
            train_path=Path("/tmp/train.jsonl"),
            student_model="gpt-3.5-turbo",
            teacher_model="gpt-4o",
            config_path=Path("/tmp/config.yaml"),
            output_path=Path("/tmp/output.json"),
            seed=42,
            verbose=True  # Default verbose
        )
    
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
            "--config", "/tmp/config.yaml",
            "--output-path", "/tmp/output.json",
            "--quiet"
        ])
        
        # Mock the validation and compilation to succeed
        mock_validate.return_value = None
        mock_compile.return_value = None
        
        # Should not raise an exception
        handle_compile_command(args)
        
        # Verify that compile_component was called with correct arguments
        mock_compile.assert_called_once_with(
            component="selection",
            train_path=Path("/tmp/train.jsonl"),
            student_model="gpt-3.5-turbo",
            teacher_model="gpt-4o",
            config_path=Path("/tmp/config.yaml"),
            output_path=Path("/tmp/output.json"),
            seed=42,  # Default value
            verbose=False  # Quiet flag disables verbose
        )
    
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
            "--config", "/tmp/config.yaml",
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
            "--config", "/tmp/config.yaml",
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
            "--config", "/tmp/config.yaml",
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
            "--config", "/tmp/config.yaml",
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
            "--config", "/tmp/config.yaml",
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
            "--config", "/tmp/config.yaml",
            "--output-path", "/tmp/output.json"
        ])
        
        # Mock the validation to succeed but compilation to fail unexpectedly
        mock_validate.return_value = None
        mock_compile.side_effect = Exception("Unexpected error")
        
        with pytest.raises(SystemExit):
            handle_compile_command(args)