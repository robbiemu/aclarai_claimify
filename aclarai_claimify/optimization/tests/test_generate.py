"""Integration tests for the dataset generation utility."""

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from aclarai_claimify.optimization.generate import (
    generate_dataset,
    generate_selection_example,
    generate_disambiguation_example,
    generate_decomposition_example,
    GenerationError,
    parse_json_response,
)


class TestParseJSONResponse:
    """Test JSON response parsing functionality."""

    def test_parse_valid_json(self):
        """Test parsing valid JSON response."""
        response = '{"selected": true, "confidence": 0.9, "reasoning": "test"}'
        result = parse_json_response(response)
        assert result == {"selected": True, "confidence": 0.9, "reasoning": "test"}

    def test_parse_json_with_code_block(self):
        """Test parsing JSON response with markdown code block."""
        response = '```json\
{"selected": true, "confidence": 0.9, "reasoning": "test"}\
```'
        result = parse_json_response(response)
        assert result == {"selected": True, "confidence": 0.9, "reasoning": "test"}

    def test_parse_json_with_language_identifier(self):
        """Test parsing JSON response with language identifier."""
        response = '```json\
{"selected": true, "confidence": 0.9, "reasoning": "test"}\
```'
        result = parse_json_response(response)
        assert result == {"selected": True, "confidence": 0.9, "reasoning": "test"}

    def test_parse_invalid_json(self):
        """Test parsing invalid JSON response."""
        response = "invalid json"
        with pytest.raises(GenerationError):
            parse_json_response(response)


class TestGenerateExamples:
    """Test individual example generation functions."""

    @patch('aclarai_claimify.optimization.generate.call_teacher_model')
    def test_generate_selection_example_success(self, mock_call):
        """Test successful selection example generation."""
        # Mock the teacher model response
        mock_call.return_value = '{"selected": true, "confidence": 0.9, "reasoning": "test"}'
        
        result = generate_selection_example(
            context="[0] The system was stable.",
            target="It failed with error code 500.",
            teacher_model="gpt-4o"
        )
        
        assert result is not None
        assert result["context_text"] == "[0] The system was stable."
        assert result["target_sentence"] == "It failed with error code 500."
        assert "selection_response_json" in result
        
        # Verify the JSON is valid
        parsed = json.loads(result["selection_response_json"])
        assert parsed["selected"] is True
        assert parsed["confidence"] == 0.9
        assert parsed["reasoning"] == "test"

    @patch('aclarai_claimify.optimization.generate.call_teacher_model')
    def test_generate_selection_example_missing_fields(self, mock_call):
        """Test selection example generation with missing fields."""
        # Mock the teacher model response with missing fields
        mock_call.return_value = '{"selected": true}'
        
        result = generate_selection_example(
            context="[0] The system was stable.",
            target="It failed with error code 500.",
            teacher_model="gpt-4o"
        )
        
        # Should return None due to validation failure
        assert result is None

    @patch('aclarai_claimify.optimization.generate.call_teacher_model')
    def test_generate_disambiguation_example_success(self, mock_call):
        """Test successful disambiguation example generation."""
        # Mock the teacher model response
        mock_call.return_value = '{"disambiguated_text": "The system failed with error code 500.", "changes_made": ["Replaced It with The system"], "confidence": 0.9}'
        
        result = generate_disambiguation_example(
            context="[0] The system was stable.",
            target="It failed with error code 500.",
            teacher_model="gpt-4o"
        )
        
        assert result is not None
        assert result["context_text"] == "[0] The system was stable."
        assert result["target_sentence"] == "It failed with error code 500."
        assert "disambiguation_response_json" in result
        
        # Verify the JSON is valid
        parsed = json.loads(result["disambiguation_response_json"])
        assert parsed["disambiguated_text"] == "The system failed with error code 500."
        assert parsed["changes_made"] == ["Replaced It with The system"]
        assert parsed["confidence"] == 0.9

    @patch('aclarai_claimify.optimization.generate.call_teacher_model')
    def test_generate_decomposition_example_success(self, mock_call):
        """Test successful decomposition example generation."""
        # Mock the teacher model response
        mock_response = {
            "claim_candidates": [
                {
                    "text": "The system failed with error code 500.",
                    "is_atomic": True,
                    "is_self_contained": True,
                    "is_verifiable": True,
                    "passes_criteria": True,
                    "confidence": 0.95,
                    "reasoning": "Single verifiable technical fact",
                    "node_type": "Claim"
                }
            ]
        }
        mock_call.return_value = json.dumps(mock_response)
        
        result = generate_decomposition_example(
            disambiguated_text="The system failed with error code 500.",
            teacher_model="gpt-4o"
        )
        
        assert result is not None
        assert result["disambiguated_text"] == "The system failed with error code 500."
        assert "decomposition_response_json" in result
        
        # Verify the JSON is valid
        parsed = json.loads(result["decomposition_response_json"])
        assert "claim_candidates" in parsed
        assert len(parsed["claim_candidates"]) == 1
        candidate = parsed["claim_candidates"][0]
        assert candidate["text"] == "The system failed with error code 500."
        assert candidate["is_atomic"] is True
        assert candidate["confidence"] == 0.95


class TestGenerateDataset:
    """Test the main dataset generation function."""

    def test_generate_dataset_nonexistent_input(self):
        """Test dataset generation with nonexistent input file."""
        with pytest.raises(FileNotFoundError):
            generate_dataset(
                input_file=Path("/nonexistent/file.txt"),
                output_file=Path("/tmp/output.jsonl"),
                component="selection",
                teacher_model="gpt-4o",
            )

    def test_generate_dataset_empty_input(self):
        """Test dataset generation with empty input file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("")  # Empty file
            input_file = Path(f.name)

        with pytest.raises(GenerationError, match="Input file is empty"):
            generate_dataset(
                input_file=input_file,
                output_file=Path("/tmp/output.jsonl"),
                component="selection",
                teacher_model="gpt-4o",
            )

        # Clean up
        input_file.unlink()

    def test_generate_dataset_invalid_component(self):
        """Test dataset generation with invalid component."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write(
                "Test sentence.\
"
            )
            input_file = Path(f.name)

        with pytest.raises(GenerationError, match="Unknown component"):
            generate_dataset(
                input_file=input_file,
                output_file=Path("/tmp/output.jsonl"),
                component="invalid-component",
                teacher_model="gpt-4o",
            )

        # Clean up
        input_file.unlink()

    @patch("aclarai_claimify.optimization.generate.call_teacher_model")
    def test_generate_dataset_success(self, mock_call):
        """Test successful dataset generation."""
        # Mock the teacher model response
        mock_call.return_value = '{"selected": true, "confidence": 0.9, "reasoning": "test"}'

        # Create a temporary input file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write(
                "It failed with error code 500.\
"
            )
            input_file = Path(f.name)

        # Create a temporary output file path
        output_file = Path(tempfile.mktemp(suffix=".jsonl"))

        try:
            # Run the generation
            generate_dataset(
                input_file=input_file,
                output_file=output_file,
                component="selection",
                teacher_model="gpt-4o",
            )

            # Verify the output file was created
            assert output_file.exists()

            # Verify the content
            with open(output_file, "r") as f:
                lines = f.readlines()
                assert len(lines) == 1

                # Parse the JSON line
                data = json.loads(lines[0])
                assert data["context_text"] == "[0] It failed with error code 500."
                assert data["target_sentence"] == "It failed with error code 500."
                assert "selection_response_json" in data

        finally:
            # Clean up
            input_file.unlink()
            if output_file.exists():
                output_file.unlink()

    @patch("aclarai_claimify.optimization.generate.call_teacher_model")
    def test_generate_dataset_decomposition_success(self, mock_call):
        """Test successful dataset generation for decomposition component."""
        # Mock the teacher model response
        mock_response = {
            "claim_candidates": [
                {
                    "text": "The system failed with error code 500.",
                    "is_atomic": True,
                    "is_self_contained": True,
                    "is_verifiable": True,
                    "passes_criteria": True,
                    "confidence": 0.95,
                    "reasoning": "Single verifiable technical fact",
                    "node_type": "Claim",
                }
            ]
        }
        mock_call.return_value = json.dumps(mock_response)

        # Create a temporary input file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write(
                "The system failed with error code 500.\
"
            )
            input_file = Path(f.name)

        # Create a temporary output file path
        output_file = Path(tempfile.mktemp(suffix=".jsonl"))

        try:
            # Run the generation
            generate_dataset(
                input_file=input_file,
                output_file=output_file,
                component="decomposition",
                teacher_model="gpt-4o",
            )

            # Verify the output file was created
            assert output_file.exists()

            # Verify the content
            with open(output_file, "r") as f:
                lines = f.readlines()
                assert len(lines) == 1

                # Parse the JSON line
                data = json.loads(lines[0])
                assert (
                    data["disambiguated_text"]
                    == "The system failed with error code 500."
                )
                assert "decomposition_response_json" in data

        finally:
            # Clean up
            input_file.unlink()
            if output_file.exists():
                output_file.unlink()
