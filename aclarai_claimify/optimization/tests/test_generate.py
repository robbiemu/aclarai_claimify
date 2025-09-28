"""Integration tests for the dataset generation utility."""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Check if required dependencies are available
pytest.importorskip("faiss")

from aclarai_claimify.data_models import ClaimifyConfig, GenerateDatasetConfig, GenerateDatasetSemanticConfig, GenerateDatasetSemanticEmbedderConfig, GenerateDatasetSemanticContextConfig, GenerateDatasetStaticConfig
from aclarai_claimify.optimization.generate import (
    GenerationError,
    generate_dataset,
    generate_decomposition_example,
    generate_disambiguation_example,
    generate_selection_example,
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
        response = '```json\n{"selected": true, "confidence": 0.9, "reasoning": "test"}\n```'
        result = parse_json_response(response)
        assert result == {"selected": True, "confidence": 0.9, "reasoning": "test"}

    def test_parse_json_with_language_identifier(self):
        """Test parsing JSON response with language identifier."""
        response = '```json\n{"selected": true, "confidence": 0.9, "reasoning": "test"}\n```'
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


@pytest.mark.skipif(not pytest.importorskip("faiss"), reason="faiss not installed")
class TestGenerateDataset:
    """Test the main dataset generation function."""

    def setup_method(self):
        """Set up a mock ClaimifyConfig for the tests."""
        self.mock_config = MagicMock(spec=ClaimifyConfig)
        self.mock_config.generate_dataset = GenerateDatasetConfig(
            method="semantic",
            semantic=GenerateDatasetSemanticConfig(
                embedder=GenerateDatasetSemanticEmbedderConfig(
                    type="sentence_transformer",
                    model="all-MiniLM-L6-v2"
                ),
                context_params=GenerateDatasetSemanticContextConfig(
                    min_k=3,
                    max_k=20,
                    similarity_threshold=0.75
                )
            ),
            static=GenerateDatasetStaticConfig(k_window_size=3)
        )

    def test_generate_dataset_nonexistent_input(self):
        """Test dataset generation with nonexistent input file."""
        with pytest.raises(FileNotFoundError):
            generate_dataset(
                input_path=Path("/nonexistent/file.txt"),
                output_file=Path("/tmp/output.jsonl"),
                component="selection",
                teacher_model="gpt-4o",
                claimify_config=self.mock_config,
            )

    @patch("aclarai_claimify.optimization.generate.call_teacher_model")
    def test_generate_dataset_empty_input(self, mock_call: MagicMock):
        """Test dataset generation with empty input file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("")  # Empty file
            input_file = Path(f.name)

        with pytest.raises(
            GenerationError, match="No generation jobs were created from the provided input"
        ):
            generate_dataset(
                input_path=input_file,
                output_file=Path("/tmp/output.jsonl"),
                component="selection",
                teacher_model="gpt-4o",
                claimify_config=self.mock_config,
            )

    @patch("aclarai_claimify.optimization.generate.generate_selection_example")
    def test_generate_dataset_curated_selection(self, mock_generate):
        """Curated selection data should produce selection_response_json entries."""

        def fake_generator(context, target, teacher_model, model_params):
            return {
                "context_text": context,
                "target_sentence": target,
                "selection_response_json": json.dumps(
                    {
                        "selected": True,
                        "confidence": 0.9,
                        "reasoning": "demo",
                    }
                ),
            }

        mock_generate.side_effect = fake_generator

        with tempfile.TemporaryDirectory() as tmpdir:
            curated_dir = Path(tmpdir) / "curated"
            curated_dir.mkdir()
            sample = {
                "positive_example": {
                    "target_sentence": "It failed with error 500.",
                    "context_text": "The system was stable. It failed with error 500."
                },
                "negative_example": {
                    "target_sentence": "This is excellent.",
                    "context_text": "This is excellent."
                }
            }
            with open(curated_dir / "sample.json", "w", encoding="utf-8") as f:
                json.dump(sample, f)

            output_file = Path(tmpdir) / "selection.jsonl"

            generate_dataset(
                input_path=curated_dir,
                output_file=output_file,
                component="selection",
                teacher_model="teacher",
                claimify_config=self.mock_config,
                curated_flag=True,
                concurrency=1,
            )

            lines = output_file.read_text().strip().splitlines()
            assert len(lines) == 1
            data = json.loads(lines[0])
            assert data.get("sample_type") == "positive"
            assert "selection_response_json" in data
            parsed = json.loads(data["selection_response_json"])
            assert parsed["selected"] is True

    @patch("aclarai_claimify.optimization.generate.generate_disambiguation_example")
    def test_generate_dataset_curated_disambiguation(self, mock_generate):
        """Curated disambiguation data should produce disambiguation_response_json entries."""

        def fake_generator(context, target, teacher_model, model_params):
            return {
                "context_text": context,
                "target_sentence": target,
                "disambiguation_response_json": json.dumps(
                    {
                        "disambiguated_text": target.replace("It", "The system"),
                        "changes_made": ["Replaced pronoun"],
                        "confidence": 0.8,
                    }
                ),
            }

        mock_generate.side_effect = fake_generator

        with tempfile.TemporaryDirectory() as tmpdir:
            curated_dir = Path(tmpdir) / "curated"
            curated_dir.mkdir()
            sample = {
                "positive_example": {
                    "target_sentence": "It failed with error 500.",
                    "context_text": "The system was stable. It failed with error 500."
                }
            }
            with open(curated_dir / "sample.json", "w", encoding="utf-8") as f:
                json.dump(sample, f)

            output_file = Path(tmpdir) / "disambiguation.jsonl"

            generate_dataset(
                input_path=curated_dir,
                output_file=output_file,
                component="disambiguation",
                teacher_model="teacher",
                claimify_config=self.mock_config,
                curated_flag=True,
                concurrency=1,
            )

            lines = output_file.read_text().strip().splitlines()
            assert len(lines) == 1
            data = json.loads(lines[0])
            assert "disambiguation_response_json" in data

    @patch("aclarai_claimify.optimization.generate.generate_selection_example")
    def test_generate_dataset_curated_invalid_response(self, mock_generate):
        """Missing expected fields should raise GenerationError and remove output file."""

        mock_generate.return_value = {
            "context_text": "ctx",
            "target_sentence": "sentence",
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            curated_dir = Path(tmpdir) / "curated"
            curated_dir.mkdir()
            sample = {
                "positive_example": {
                    "target_sentence": "Sentence",
                    "context_text": "Context"
                }
            }
            curated_path = curated_dir / "sample.json"
            curated_path.write_text(json.dumps(sample))

            output_file = Path(tmpdir) / "selection.jsonl"

            with pytest.raises(GenerationError):
                generate_dataset(
                    input_path=curated_dir,
                    output_file=output_file,
                    component="selection",
                    teacher_model="teacher",
                    claimify_config=self.mock_config,
                    curated_flag=True,
                    concurrency=1,
                )

            assert not output_file.exists()

        mock_generate.assert_called()

    def test_generate_dataset_invalid_component(self):
        """Test dataset generation with invalid component."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write(
                "Test sentence.\n"
            )
            input_file = Path(f.name)

        with pytest.raises(GenerationError, match="Unknown component"):
            generate_dataset(
                input_path=input_file,
                output_file=Path("/tmp/output.jsonl"),
                component="invalid-component",
                teacher_model="gpt-4o",
                claimify_config=self.mock_config,
            )

        # Clean up
        input_file.unlink()

    @patch("aclarai_claimify.optimization.generate.generate_disambiguation_negative_example")
    @patch("aclarai_claimify.optimization.generate.generate_disambiguation_example")
    def test_generate_dataset_curated_disambiguation_with_negatives(
        self,
        mock_positive,
        mock_negative,
    ):
        """Curated disambiguation generation can emit labelled negative samples."""

        mock_positive.return_value = {
            "context_text": "[0] It failed with error 500.",
            "target_sentence": "It failed with error 500.",
            "disambiguation_response_json": json.dumps(
                {
                    "disambiguated_text": "The system failed with error 500.",
                    "changes_made": ["Replaced pronoun"],
                    "confidence": 0.9,
                }
            ),
        }

        mock_negative.return_value = {
            "context_text": "[0] It failed with error 500.",
            "target_sentence": "It failed with error 500.",
            "disambiguation_response_json": json.dumps(
                {
                    "disambiguated_text": "It failed with error 500.",
                    "changes_made": ["Left pronoun unresolved"],
                    "confidence": 0.95,
                }
            ),
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            curated_dir = Path(tmpdir) / "curated"
            curated_dir.mkdir()
            sample = {
                "positive_example": {
                    "target_sentence": "It failed with error 500.",
                    "context_text": "The system was stable. It failed with error 500.",
                    "rationale": "High-impact pronoun."
                },
                "negative_examples": [
                    {
                        "failure_mode": "unresolved_referent",
                        "target_sentence": "It failed with error 500.",
                        "context_text": "The system was stable. It failed with error 500.",
                        "rationale": "Likely to keep the pronoun."
                    }
                ],
            }
            (curated_dir / "sample.json").write_text(json.dumps(sample), encoding="utf-8")

            output_file = Path(tmpdir) / "disambiguation.jsonl"

            generate_dataset(
                input_path=curated_dir,
                output_file=output_file,
                component="disambiguation",
                teacher_model="teacher",
                claimify_config=self.mock_config,
                curated_flag=True,
                concurrency=1,
                include_negatives=True,
                negative_quota=1,
            )

            lines = output_file.read_text().strip().splitlines()
            assert len(lines) == 2

            positive = json.loads(lines[0])
            negative = json.loads(lines[1])

            assert positive["sample_type"] == "positive"
            assert negative["sample_type"] == "negative"
            assert negative["failure_mode"] == "unresolved_referent"
            parsed_negative = json.loads(negative["disambiguation_response_json"])
            assert parsed_negative["disambiguated_text"].startswith("It failed")
            assert negative.get("prospect_label") == "negative_unresolved_referent"
            mock_negative.assert_called_once()

    @patch("aclarai_claimify.optimization.generate.call_teacher_model")
    def test_generate_dataset_success(self, mock_call):
        """Test successful dataset generation."""
        # Mock the teacher model response
        mock_call.return_value = '{"selected": true, "confidence": 0.9, "reasoning": "test"}'

        # Create a temporary input file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write(
                "It failed with error code 500.\n"
            )
            input_file = Path(f.name)

        # Create a temporary output file path
        output_file = Path(tempfile.mktemp(suffix=".jsonl"))

        try:
            # Run the generation
            generate_dataset(
                input_path=input_file,
                output_file=output_file,
                component="selection",
                teacher_model="gpt-4o",
                claimify_config=self.mock_config,
            )

            # Verify the output file was created
            assert output_file.exists()

            # Verify the content
            with open(output_file, "r") as f:
                lines = f.readlines()
                assert len(lines) == 1

                # Parse the JSON line
                data = json.loads(lines[0])
                assert data["context_text"] == ""
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
                "The system failed with error code 500.\n"
            )
            input_file = Path(f.name)

        # Create a temporary output file path
        output_file = Path(tempfile.mktemp(suffix=".jsonl"))

        try:
            # Run the generation
            generate_dataset(
                input_path=input_file,
                output_file=output_file,
                component="decomposition",
                teacher_model="gpt-4o",
                claimify_config=self.mock_config,
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

    @patch("aclarai_claimify.optimization.generate.call_teacher_model")
    def test_generate_dataset_curated_handles_sparse_files(
        self, mock_call, tmp_path
    ):
        """Ensure curated directories skip invalid JSON structures."""

        mock_call.return_value = '{"selected": true, "confidence": 0.9, "reasoning": "ok"}'

        curated_dir = tmp_path / "curated"
        curated_dir.mkdir()

        valid_payload = {
            "positive_example": {
                "target_sentence": "System uptime reached 99.9% in Q1.",
                "context_text": "[0] System uptime reached 99.9% in Q1. [1] Maintenance windows remained unchanged.",
            },
            "negative_example": {
                "target_sentence": "I think the rollout felt smooth overall.",
                "context_text": "[0] Stakeholders attended the rollout meeting. [1] I think the rollout felt smooth overall. [2] Follow-up actions are pending.",
            },
        }
        (curated_dir / "valid.json").write_text(json.dumps(valid_payload), encoding="utf-8")
        (curated_dir / "empty.json").write_text(json.dumps(""), encoding="utf-8")

        output_file = tmp_path / "out.jsonl"

        generate_dataset(
            input_path=curated_dir,
            output_file=output_file,
            component="selection",
            teacher_model="gpt-4o",
            claimify_config=self.mock_config,
            curated_flag=True,
        )

        assert output_file.exists()
        with output_file.open() as handle:
            lines = handle.readlines()
        assert len(lines) == 1
        assert mock_call.call_count == 1
