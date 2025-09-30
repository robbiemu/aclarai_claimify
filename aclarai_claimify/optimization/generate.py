"""Dataset generation utility for creating gold standard training datasets.

This module provides functionality to generate high-quality training datasets
by using powerful teacher models to produce structured outputs from raw text inputs.
"""

import asyncio
import faiss
import importlib
import json
import numpy as np
import re
import sys
import time
import traceback
import litellm
from litellm import RateLimitError
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Optional
from tqdm import tqdm

from ..data_models import ClaimifyConfig
from ..utils.context import create_context_window
from .failure_modes import (
    DISAMBIGUATION_FAILURE_DESCRIPTIONS,
    DISAMBIGUATION_FAILURE_MODES,
    DECOMPOSITION_FAILURE_DESCRIPTIONS,
    DECOMPOSITION_FAILURE_MODES,
)


class GenerationError(Exception):
    """Raised when dataset generation fails."""

    pass


class APICallError(Exception):
    """Raised when API call fails."""

    pass


# Teacher prompts for each component
TEACHER_PROMPTS = {
    "selection": """You are an expert at identifying sentences that contain verifiable factual claims from text.

Given a context and a target sentence, determine if the target sentence should be selected for further processing. A sentence should be selected if it contains factual, verifiable information rather than subjective opinions or generic statements.

Input:
- context_text: The surrounding context (may include sentence indices)
- target_sentence: The sentence to evaluate for selection

Output a JSON object with:
- selected: Boolean indicating if the sentence should be selected
- confidence: Float between 0.0-1.0 indicating your confidence
- reasoning: Brief explanation of your decision

Example output:
{{"selected": true, "confidence": 0.9, "reasoning": "Contains specific technical failure information that is verifiable"}}

Context: {context_text}
Target Sentence: {target_sentence}

Respond ONLY with the JSON object, nothing else:""",
    "disambiguation": """You are an expert at resolving ambiguities in text to make sentences self-contained and clear.

Given a context and a target sentence, rewrite the target sentence to be self-contained by resolving pronouns and references using the context, while preserving the original meaning.

Avoid these failure modes at all costs:
- unresolved_referent (pronoun left ambiguous)
- hallucinated_detail (added facts not present in context)
- omitted_constraint (dropped hedges/negations/temporal qualifiers)
- formatting_drift (output is not exactly one sentence of plain text)
- unsupported_resolution (replacing a pronoun with a vague placeholder when a precise referent exists)
- confidence_mismatch (reporting high confidence for an obviously flawed rewrite)

Input:
- context_text: The surrounding context (may include sentence indices)
- target_sentence: The sentence to disambiguate

Output a JSON object with:
- disambiguated_text: The rewritten sentence with ambiguities resolved
- changes_made: Array of strings describing what changes you made
- confidence: Float between 0.0-1.0 indicating your confidence

Example output:
{{"disambiguated_text": "The system failed with error code 500.", "changes_made": ["Replaced 'It' with 'The system'"], "confidence": 0.9}}

Context: {context_text}
Target Sentence: {target_sentence}

Respond ONLY with the JSON object, nothing else:""",
    "decomposition": """You are an expert at decomposing complex sentences into atomic, verifiable claims.

Given a disambiguated sentence, break it down into one or more atomic claims that are:
1. Self-contained (no external references needed)
2. Verifiable (can be confirmed true/false with evidence)
3. Atomic (express one clear idea)

Avoid these failure modes:
- non_atomic_claim (bundling several ideas together)
- missing_key_claim (failing to capture important content from the sentence)
- incorrect_metadata (quality flags disagree with the claim text)
- off_topic_hallucination (inventing claims absent from the sentence)
- structure_violation (malformed JSON or empty results without justification)
- low_information_reasoning (reasoning field is blank or unhelpful)

Input:
- disambiguated_text: The sentence to decompose

Output a JSON object with an array of claim candidates, each having:
- text: The claim text
- is_atomic: Boolean (should be true)
- is_self_contained: Boolean (should be true)
- is_verifiable: Boolean (should be true)
- passes_criteria: Boolean (should be true if all above are true)
- confidence: Float between 0.0-1.0 indicating your confidence
- reasoning: Brief explanation of why this is a good claim
- node_type: String (should be "Claim")

Example output:
{{"claim_candidates": [{{"text": "The system failed with error code 500.", "is_atomic": true, "is_self_contained": true, "is_verifiable": true, "passes_criteria": true, "confidence": 0.95, "reasoning": "Single verifiable technical fact", "node_type": "Claim"}}]}}

Disambiguated Text: {disambiguated_text}

Respond ONLY with the JSON object, nothing else:""",
}


def exponential_backoff(
    attempt: int, base_delay: float = 1.0, max_delay: float = 60.0
) -> float:
    """Calculate delay with exponential backoff."""
    import random

    delay = min(base_delay * (2**attempt) + random.uniform(0, 1), max_delay)
    return delay


def call_teacher_model(
    prompt: str, model: str, model_params: Optional[Dict[str, Any]] = None
) -> str:
    """Call the teacher model with the given prompt.

    Args:
        prompt: The prompt to send to the model
        model: The model name to use
        model_params: Additional model parameters to pass to LiteLLM

    Returns:
        The model's response text

    Raises:
        APICallError: If the API call fails
    """
    max_retries = 5
    for attempt in range(max_retries):
        try:
            # Prepare the completion call with model_params
            completion_kwargs = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
            }

            # Add any additional model parameters
            if model_params:
                completion_kwargs.update(model_params)

            response = litellm.completion(**completion_kwargs)
            return response.choices[0].message.content.strip()
        except RateLimitError as e:
            if attempt < max_retries - 1:
                delay = exponential_backoff(attempt)
                print(
                    f"Rate limit hit, waiting {delay:.1f}s before retry {attempt + 1}/{max_retries}",
                    file=sys.stderr,
                )
                time.sleep(delay)
            else:
                raise APICallError(
                    f"Rate limit exceeded after {max_retries} attempts: {e}"
                )
        except litellm.exceptions.AuthenticationError as e:
            raise APICallError(f"Authentication failed - check your API keys: {e}")
        except litellm.exceptions.InvalidRequestError as e:
            raise APICallError(
                f"Invalid request - check model name and parameters: {e}"
            )
        except litellm.exceptions.ServiceUnavailableError as e:
            if attempt < max_retries - 1:
                delay = exponential_backoff(attempt)
                print(
                    f"Service unavailable, waiting {delay:.1f}s before retry {attempt + 1}/{max_retries}",
                    file=sys.stderr,
                )
                time.sleep(delay)
            else:
                raise APICallError(
                    f"Service unavailable after {max_retries} attempts: {e}"
                )
        except Exception as e:
            raise APICallError(f"API call failed: {e}")


def parse_json_response(response: str) -> Dict[str, Any]:
    """Parse JSON response from the teacher model.

    Args:
        response: The raw response text

    Returns:
        Parsed JSON object

    Raises:
        GenerationError: If JSON parsing fails
    """
    try:
        # Handle potential markdown code blocks
        if response.startswith("```"):
            # Find the end of the code block
            end_idx = response.find("```", 3)
            if end_idx != -1:
                response = response[3:end_idx].strip()
                # Remove potential language identifier
                if response.startswith("json"):
                    response = response[4:].strip()

        return json.loads(response)
    except json.JSONDecodeError as e:
        raise GenerationError(
            f"Failed to parse JSON response: {e}\nResponse: {response}"
        )


def generate_selection_example(
    context: str,
    target: str,
    teacher_model: str,
    model_params: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    """Generate a single example for the selection component.

    Args:
        context: Context text
        target: Target sentence to evaluate
        teacher_model: Teacher model name
        model_params: Additional model parameters to pass to LiteLLM

    Returns:
        Generated example dictionary or None if failed
    """
    prompt = TEACHER_PROMPTS["selection"].format(
        context_text=context, target_sentence=target
    )

    try:
        response = call_teacher_model(prompt, teacher_model, model_params)
        parsed = parse_json_response(response)

        # Validate required fields
        required_fields = ["selected", "confidence", "reasoning"]
        for field in required_fields:
            if field not in parsed:
                raise GenerationError(f"Missing required field '{field}' in response")

        return {
            "context_text": context,
            "target_sentence": target,
            "selection_response_json": json.dumps(parsed),
        }
    except (APICallError, GenerationError) as e:
        print(
            f"Warning: Failed to generate selection example for '{target}': {e}",
            file=sys.stderr,
        )
        return None


def generate_disambiguation_example(
    context: str,
    target: str,
    teacher_model: str,
    model_params: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    """Generate a single example for the disambiguation component.

    Args:
        context: Context text
        target: Target sentence to disambiguate
        teacher_model: Teacher model name
        model_params: Additional model parameters to pass to LiteLLM

    Returns:
        Generated example dictionary or None if failed
    """
    prompt = TEACHER_PROMPTS["disambiguation"].format(
        context_text=context, target_sentence=target
    )

    try:
        response = call_teacher_model(prompt, teacher_model, model_params)
        parsed = parse_json_response(response)

        # Validate required fields
        required_fields = ["disambiguated_text", "changes_made", "confidence"]
        for field in required_fields:
            if field not in parsed:
                raise GenerationError(f"Missing required field '{field}' in response")

        return {
            "context_text": context,
            "target_sentence": target,
            "disambiguation_response_json": json.dumps(parsed),
        }
    except (APICallError, GenerationError) as e:
        print(
            f"Warning: Failed to generate disambiguation example for '{target}': {e}",
            file=sys.stderr,
        )
        return None


def generate_decomposition_example(
    disambiguated_text: str,
    teacher_model: str,
    model_params: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    """Generate a single example for the decomposition component.

    Args:
        disambiguated_text: Text to decompose
        teacher_model: Teacher model name
        model_params: Additional model parameters to pass to LiteLLM

    Returns:
        Generated example dictionary or None if failed
    """
    prompt = TEACHER_PROMPTS["decomposition"].format(
        disambiguated_text=disambiguated_text
    )

    try:
        response = call_teacher_model(prompt, teacher_model, model_params)
        parsed = parse_json_response(response)

        # Validate required fields
        if "claim_candidates" not in parsed:
            raise GenerationError(
                "Missing required field 'claim_candidates' in response"
            )

        if not isinstance(parsed["claim_candidates"], list):
            raise GenerationError("'claim_candidates' must be an array")

        # Validate each claim candidate
        for candidate in parsed["claim_candidates"]:
            if not isinstance(candidate, dict):
                raise GenerationError("Each claim candidate must be an object")

            required_fields = [
                "text",
                "is_atomic",
                "is_self_contained",
                "is_verifiable",
                "passes_criteria",
                "confidence",
                "reasoning",
                "node_type",
            ]
            for field in required_fields:
                if field not in candidate:
                    raise GenerationError(
                        f"Missing required field '{field}' in claim candidate"
                    )

        return {
            "disambiguated_text": disambiguated_text,
            "decomposition_response_json": json.dumps(parsed),
        }
    except (APICallError, GenerationError) as e:
        print(
            f"Warning: Failed to generate decomposition example for '{disambiguated_text}': {e}",
            file=sys.stderr,
        )
        return None


def generate_disambiguation_negative_example(
    context: str,
    target: str,
    failure_mode: str,
    teacher_model: str,
    model_params: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    """Generate a flawed disambiguation example for a specific failure mode."""

    if failure_mode not in DISAMBIGUATION_FAILURE_MODES:
        raise GenerationError(f"Unknown disambiguation failure mode '{failure_mode}'")

    description = DISAMBIGUATION_FAILURE_DESCRIPTIONS[failure_mode]
    prompt = f"""You are intentionally crafting a flawed disambiguation output.

Failure mode: {failure_mode}
Definition: {description}

Input:
- context_text: {context}
- target_sentence: {target}

Produce a JSON object with keys disambiguated_text (string), changes_made (array of strings), and confidence (float in [0,1]).
- The JSON must remain syntactically valid.
- The contents MUST clearly exhibit the specified failure mode.
- Keep changes_made honest about what you altered, even if the change is incorrect.
- Confidence may be high or low depending on the failure mode definition.

Return ONLY the JSON object.
"""

    try:
        response = call_teacher_model(prompt, teacher_model, model_params)
        parsed = parse_json_response(response)

        required_fields = ["disambiguated_text", "changes_made", "confidence"]
        for field in required_fields:
            if field not in parsed:
                raise GenerationError(
                    f"Missing required field '{field}' in negative disambiguation response for mode '{failure_mode}'"
                )

        return {
            "context_text": context,
            "target_sentence": target,
            "disambiguation_response_json": json.dumps(parsed),
        }
    except (APICallError, GenerationError) as e:
        print(
            f"Warning: Failed to generate disambiguation negative ({failure_mode}) for '{target}': {e}",
            file=sys.stderr,
        )
        return None


def generate_decomposition_negative_example(
    disambiguated_text: str,
    failure_mode: str,
    teacher_model: str,
    model_params: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    """Generate a flawed decomposition example for a specific failure mode."""

    if failure_mode not in DECOMPOSITION_FAILURE_MODES:
        raise GenerationError(f"Unknown decomposition failure mode '{failure_mode}'")

    description = DECOMPOSITION_FAILURE_DESCRIPTIONS[failure_mode]
    prompt = f"""You are intentionally crafting a flawed decomposition output.

Failure mode: {failure_mode}
Definition: {description}

Input sentence: {disambiguated_text}

Return ONLY a JSON object with key claim_candidates.
- claim_candidates must be an array (even if empty).
- Each element should remain a JSON object with the standard fields (text, is_atomic, is_self_contained, is_verifiable, passes_criteria, confidence, reasoning, node_type).
- Adjust the contents so the overall output exemplifies the specified failure mode.
- Reasoning should acknowledge the (flawed) rationale when applicable.
"""

    try:
        response = call_teacher_model(prompt, teacher_model, model_params)
        parsed = parse_json_response(response)

        if "claim_candidates" not in parsed:
            raise GenerationError(
                f"Missing required field 'claim_candidates' in negative decomposition response for mode '{failure_mode}'"
            )

        if not isinstance(parsed["claim_candidates"], list):
            raise GenerationError(
                f"'claim_candidates' must be an array for negative decomposition mode '{failure_mode}'"
            )

        return {
            "disambiguated_text": disambiguated_text,
            "decomposition_response_json": json.dumps(parsed),
        }
    except (APICallError, GenerationError) as e:
        print(
            f"Warning: Failed to generate decomposition negative ({failure_mode}) for '{disambiguated_text}': {e}",
            file=sys.stderr,
        )
        return None


def _load_embedder(embedder_config):
    """Dynamically load the embedder plugin."""
    embedder_type = embedder_config.type
    try:
        module_path = f"aclarai_claimify.embeddings.{embedder_type}"
        module = importlib.import_module(module_path)

        # Convention: class name is CamelCase version of module name (e.g., sentence_transformer -> SentenceTransformerEmbedder)
        class_name = (
            "".join(word.capitalize() for word in embedder_type.split("_")) + "Embedder"
        )
        embedder_class = getattr(module, class_name)

        return embedder_class(model_name=embedder_config.model)
    except (ImportError, AttributeError) as e:
        raise GenerationError(f"Failed to load embedder '{embedder_type}': {e}")


def clean_markdown(content: str) -> str:
    """Removes common markdown syntax from a string."""
    # Remove headings
    content = re.sub(r"^#+\s+", "", content, flags=re.MULTILINE)
    # Remove bold and italics
    content = re.sub(r"(\*\*|\*|__|_)(.*?)\1", r"\2", content)
    # Remove links
    content = re.sub(r"\[(.*?)\]\((.*?))\)", r"\1", content)
    # Remove lists
    content = re.sub(r"^\s*[-*+]\s+", "", content, flags=re.MULTILINE)
    # Remove newlines and extra spaces
    content = content.replace("\n", " ").strip()
    return content


def _log_job_exception(metadata: Dict[str, Any], error: Exception) -> None:
    """Log a job failure with context-aware messaging."""

    file_path = metadata.get("file_path")
    if metadata.get("curated"):
        print(
            f"Error processing prospect from file {file_path}: {error}",
            file=sys.stderr,
        )
    else:
        line_index = metadata.get("line_index")
        if line_index is not None:
            print(
                f"Error processing line {line_index + 1} in file {file_path}: {error}",
                file=sys.stderr,
            )
        else:
            print(f"Error processing file {file_path}: {error}", file=sys.stderr)

    if hasattr(error, "__traceback__"):
        traceback.print_exception(
            type(error), error, error.__traceback__, file=sys.stderr
        )


def _iter_curated_examples(data: Any, file_path: Path) -> list[dict[str, Any]]:
    """Normalize curated data structures into example dictionaries."""

    candidates: list[dict[str, Any]] = []

    if isinstance(data, dict):
        for key, value in data.items():
            if isinstance(value, dict):
                candidate = value.copy()
                candidate.setdefault("prospect_label", key)
                candidates.append(candidate)
    elif isinstance(data, list):
        for idx, item in enumerate(data):
            if isinstance(item, dict):
                candidate = item.copy()
                candidate.setdefault("prospect_label", f"item_{idx}")
                candidates.append(candidate)
    else:
        print(
            f"Skipping curated file {file_path}: unsupported JSON structure ({type(data).__name__})",
            file=sys.stderr,
        )
        return []

    if not candidates:
        print(
            f"Skipping curated file {file_path}: no usable examples found",
            file=sys.stderr,
        )

    return candidates


def generate_dataset(
    input_path: Path,
    output_file: Path,
    component: str,
    teacher_model: str,
    claimify_config: ClaimifyConfig,
    model_params: Optional[Dict[str, Any]] = None,
    clean_markdown_flag: bool = False,
    curated_flag: bool = False,
    concurrency: int = 1,
    include_negatives: bool = False,
    negative_quota: int = 0,
    max_examples: Optional[int] = None,
) -> None:
    """Generate a gold standard dataset from raw text inputs.

    Args:
        input_path: Path to input text file (one sentence per line) or a directory of text files.
        output_file: Path to output JSONL file
        component: Component name (selection, disambiguation, decomposition)
        teacher_model: Teacher model name
        claimify_config: The main Claimify configuration object.
        model_params: Additional model parameters to pass to LiteLLM
        clean_markdown_flag: Whether to clean markdown from input files.
        curated_flag: Whether the input path is a directory of curated JSON files.
        concurrency: Maximum number of concurrent generation tasks.
        include_negatives: Whether to synthesize negative samples (disambiguation/decomposition only).
        negative_quota: Minimum count per failure mode for negatives (0 disables enforcement).

    Raises:
        GenerationError: If generation fails
        FileNotFoundError: If input path doesn't exist
    """
    if not input_path.exists():
        raise FileNotFoundError(f"Input path not found: {input_path}")

    if curated_flag:
        files_to_process = sorted(input_path.glob("*.json"))
    else:
        files_to_process = []
        if input_path.is_dir():
            files_to_process.extend(sorted(input_path.glob("*.md")))
            files_to_process.extend(sorted(input_path.glob("*.txt")))
        else:
            files_to_process.append(input_path)

    if not files_to_process:
        raise GenerationError(f"No files found at: {input_path}")

    if concurrency <= 0:
        raise GenerationError("--concurrency must be a positive integer")

    if include_negatives and component == "selection":
        print(
            "ℹ️  Ignoring request for negatives: selection component already encodes negatives via labels.",
            file=sys.stderr,
        )
        include_negatives = False

    if negative_quota and not include_negatives:
        negative_quota = 0

    settings = claimify_config.generate_dataset

    generators = {
        "selection": generate_selection_example,
        "disambiguation": generate_disambiguation_example,
        "decomposition": generate_decomposition_example,
    }
    if component not in generators:
        raise GenerationError(
            f"Unknown component '{component}'. Must be one of: {list(generators.keys())}"
        )
    generator = generators[component]

    final_teacher_model = teacher_model or claimify_config.get_model_for_stage(
        component
    )
    print(
        f"Generating examples for {component} component using {final_teacher_model}..."
    )
    print("⚠️  WARNING: This may incur costs depending on your API provider and usage!")

    jobs: list[dict[str, Any]] = []
    negative_counts: Counter[str] = Counter()
    requested_failure_modes: Counter[str] = Counter()

    expected_field_key = {
        "selection": "selection_response_json",
        "disambiguation": "disambiguation_response_json",
        "decomposition": "decomposition_response_json",
    }

    for file_path in files_to_process:
        if curated_flag:
            with open(file_path, "r", encoding="utf-8") as f_in:
                prospects = json.load(f_in)

            def _enqueue_positive(payload: Dict[str, Any], label: str) -> None:
                target_sentence = payload.get("target_sentence")
                if not target_sentence:
                    print(
                        f"Skipping prospect in {file_path}: missing target_sentence",
                        file=sys.stderr,
                    )
                    return

                metadata = {
                    "file_path": file_path,
                    "curated": True,
                    "prospect_label": payload.get("prospect_label", label),
                    "sample_type": "positive",
                    "rationale": payload.get("rationale"),
                }

                if component == "decomposition":
                    args = (target_sentence, final_teacher_model, model_params)
                else:
                    context_text = payload.get("context_text")
                    if context_text is None:
                        print(
                            f"Skipping prospect in {file_path}: missing context_text",
                            file=sys.stderr,
                        )
                        return
                    metadata["context_text"] = context_text
                    args = (
                        context_text,
                        target_sentence,
                        final_teacher_model,
                        model_params,
                    )

                jobs.append(
                    {
                        "func": generator,
                        "args": args,
                        "metadata": metadata,
                    }
                )

            def _enqueue_negative(payload: Dict[str, Any]) -> None:
                failure_mode = payload.get("failure_mode")
                if not failure_mode:
                    print(
                        f"Skipping negative prospect in {file_path}: missing failure_mode",
                        file=sys.stderr,
                    )
                    return

                if component == "disambiguation":
                    if failure_mode not in DISAMBIGUATION_FAILURE_MODES:
                        print(
                            f"Skipping unknown disambiguation failure mode '{failure_mode}' in {file_path}",
                            file=sys.stderr,
                        )
                        return
                    context_text = payload.get("context_text")
                    if context_text is None:
                        print(
                            f"Skipping negative prospect in {file_path}: missing context_text",
                            file=sys.stderr,
                        )
                        return
                    target_sentence = payload.get("target_sentence")
                    if not target_sentence:
                        print(
                            f"Skipping negative prospect in {file_path}: missing target_sentence",
                            file=sys.stderr,
                        )
                        return
                    args = (
                        context_text,
                        target_sentence,
                        failure_mode,
                        final_teacher_model,
                        model_params,
                    )
                    func = generate_disambiguation_negative_example
                elif component == "decomposition":
                    if failure_mode not in DECOMPOSITION_FAILURE_MODES:
                        print(
                            f"Skipping unknown decomposition failure mode '{failure_mode}' in {file_path}",
                            file=sys.stderr,
                        )
                        return
                    target_sentence = payload.get("target_sentence")
                    if not target_sentence:
                        print(
                            f"Skipping negative prospect in {file_path}: missing target_sentence",
                            file=sys.stderr,
                        )
                        return
                    args = (
                        target_sentence,
                        failure_mode,
                        final_teacher_model,
                        model_params,
                    )
                    func = generate_decomposition_negative_example
                else:
                    return

                requested_failure_modes[failure_mode] += 1

                jobs.append(
                    {
                        "func": func,
                        "args": args,
                        "metadata": {
                            "file_path": file_path,
                            "curated": True,
                            "prospect_label": payload.get(
                                "prospect_label", f"negative_{failure_mode}"
                            ),
                            "sample_type": "negative",
                            "failure_mode": failure_mode,
                            "rationale": payload.get("rationale"),
                            "context_text": payload.get("context_text"),
                            "target_sentence": payload.get("target_sentence"),
                        },
                    }
                )

            if isinstance(prospects, dict) and (
                "positive_example" in prospects or "negative_examples" in prospects
            ):
                positive_payload = prospects.get("positive_example")
                if isinstance(positive_payload, dict):
                    _enqueue_positive(positive_payload, "positive_example")
                else:
                    print(
                        f"Skipping {file_path}: positive_example missing or malformed",
                        file=sys.stderr,
                    )

                if include_negatives and component in {
                    "disambiguation",
                    "decomposition",
                }:
                    negative_payloads = prospects.get("negative_examples") or []
                    if not isinstance(negative_payloads, list):
                        print(
                            f"Negative examples in {file_path} must be a list; skipping.",
                            file=sys.stderr,
                        )
                    else:
                        valid_negatives = [
                            payload
                            for payload in negative_payloads
                            if isinstance(payload, dict)
                        ]
                        if valid_negatives:
                            # Choose the failure mode that has been requested least so far
                            def _mode_count(payload: dict[str, Any]) -> int:
                                return requested_failure_modes[
                                    payload.get("failure_mode", "")
                                ]

                            selected_negative = min(valid_negatives, key=_mode_count)
                            _enqueue_negative(selected_negative)
            else:
                for prospect in _iter_curated_examples(prospects, file_path):
                    _enqueue_positive(
                        prospect, prospect.get("prospect_label", "prospect")
                    )
        else:
            with open(file_path, "r", encoding="utf-8") as f_in:
                content = f_in.read()

            if clean_markdown_flag:
                content = clean_markdown(content)

            lines = re.split(r"(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=[\.?!])\s", content)
            lines = [line.strip() for line in lines if line.strip()]

            if not lines:
                continue

            contexts: list[str] = []
            if component != "decomposition":
                if settings.method == "static":
                    k = settings.static.k_window_size
                    contexts = [
                        create_context_window(lines, i, k) for i in range(len(lines))
                    ]
                elif settings.method == "semantic":
                    embedder_config = settings.semantic.embedder
                    context_params = settings.semantic.context_params

                    embedder = _load_embedder(embedder_config)
                    embeddings = embedder.embed(lines)

                    if np.any(np.linalg.norm(embeddings, axis=1) - 1.0 > 1e-6):
                        faiss.normalize_L2(embeddings)

                    index = faiss.IndexFlatIP(embeddings.shape[1])
                    index.add(embeddings)

                    for i in range(len(lines)):
                        query_vector = embeddings[i : i + 1]
                        search_k = min(len(lines), context_params.max_k + 1)
                        distances, indices = index.search(query_vector, search_k)

                        context_sentences = []
                        for neighbor_idx, similarity in zip(indices[0], distances[0]):
                            if neighbor_idx == i:
                                continue

                            if similarity >= context_params.similarity_threshold:
                                context_sentences.append(lines[neighbor_idx])

                        if len(context_sentences) < context_params.min_k:
                            all_neighbors = [idx for idx in indices[0] if idx != i]
                            context_sentences = [
                                lines[idx]
                                for idx in all_neighbors[: context_params.min_k]
                            ]

                        final_context = "\n".join(
                            context_sentences[: context_params.max_k]
                        )
                        contexts.append(final_context)
                else:
                    raise ValueError(
                        f"Unknown generation method in config: '{settings.method}'"
                    )

            for i, line in enumerate(lines):
                if component == "decomposition":
                    args = (line, final_teacher_model, model_params)
                else:
                    context = contexts[i]
                    args = (context, line, final_teacher_model, model_params)

                jobs.append(
                    {
                        "func": generator,
                        "args": args,
                        "metadata": {
                            "file_path": file_path,
                            "line_index": i,
                            "curated": False,
                        },
                    }
                )

    if not jobs:
        raise GenerationError("No generation jobs were created from the provided input")

    if max_examples is not None:
        jobs = jobs[:max_examples]

    with open(output_file, "w", encoding="utf-8") as f_out:

        async def _execute_jobs() -> tuple[int, int, int, int]:
            nonlocal negative_counts
            successful = 0
            failed = 0
            positive_generated = 0
            negative_generated = 0
            progress_bar = tqdm(total=len(jobs), desc="Generating examples")

            async def _run_job(job: dict[str, Any]):
                nonlocal successful, failed, positive_generated, negative_generated
                try:
                    example = await asyncio.to_thread(job["func"], *job["args"])
                except Exception as exc:  # noqa: BLE001
                    failed += 1
                    _log_job_exception(job["metadata"], exc)
                    progress_bar.update(1)
                    return

                if example is None:
                    failed += 1
                    progress_bar.update(1)
                    return

                expected_key = expected_field_key[component]
                if expected_key not in example:
                    raise GenerationError(
                        f"Generated example missing expected field '{expected_key}'"
                    )

                metadata = job.get("metadata", {})
                sample_type = metadata.get("sample_type", "positive")
                example.setdefault("sample_type", sample_type)
                prospect_label = metadata.get("prospect_label")
                if prospect_label is not None:
                    example.setdefault("prospect_label", prospect_label)

                rationale = metadata.get("rationale")
                if rationale:
                    example.setdefault("prospect_rationale", rationale)

                if sample_type == "negative":
                    failure_mode = metadata.get("failure_mode")
                    if failure_mode:
                        example.setdefault("failure_mode", failure_mode)
                        negative_counts[failure_mode] += 1
                    negative_generated += 1
                else:
                    # ensure failure_mode is removed for positives if set
                    example.pop("failure_mode", None)
                    positive_generated += 1

                if metadata.get("context_text") and component == "disambiguation":
                    example.setdefault("context_text", metadata["context_text"])

                if metadata.get("target_sentence") and component != "decomposition":
                    example.setdefault("target_sentence", metadata["target_sentence"])

                f_out.write(json.dumps(example) + "\n")
                successful += 1
                progress_bar.update(1)

            semaphore = asyncio.Semaphore(concurrency)

            async def _bounded(job: dict[str, Any]):
                async with semaphore:
                    return await _run_job(job)

            try:
                await asyncio.gather(*[_bounded(job) for job in jobs])
            finally:
                progress_bar.close()

            return successful, failed, positive_generated, negative_generated

        try:
            (
                successful_examples,
                failed_examples,
                positive_examples,
                negative_examples,
            ) = asyncio.run(_execute_jobs())
        except GenerationError as _exc:
            f_out.close()
            raise

        if include_negatives and component in {"disambiguation", "decomposition"}:
            mode_list = (
                DISAMBIGUATION_FAILURE_MODES
                if component == "disambiguation"
                else DECOMPOSITION_FAILURE_MODES
            )

            print("\n📊 Negative failure-mode coverage:")
            for mode in mode_list:
                count = negative_counts.get(mode, 0)
                marker = "✅" if count else "⚠️"
                print(f"   {marker} {mode}: {count}")

            if negative_quota:
                modes_to_enforce = requested_failure_modes or set(mode_list)
                missing = [
                    mode
                    for mode in modes_to_enforce
                    if negative_counts.get(mode, 0) < negative_quota
                ]
                if missing:
                    raise GenerationError(
                        "Negative quota not met for modes: " + ", ".join(missing)
                    )

    print("\n✅ Generation complete!")
    print(f"   Successfully generated: {successful_examples} examples")
    print(f"   Failed to generate: {failed_examples} examples")
    print(f"   Output saved to: {output_file}")
    print(f"   Positives: {positive_examples}")
    if include_negatives and component in {"disambiguation", "decomposition"}:
        print(f"   Negatives: {negative_examples}")

    if successful_examples == 0:
        raise GenerationError("Failed to generate any examples successfully")


def main():
    """Main entry point for testing the module directly."""
    print("This module is intended to be used through the CLI.")
    print("Use: aclarai-claimify generate-dataset --help")


if __name__ == "__main__":
    main()
