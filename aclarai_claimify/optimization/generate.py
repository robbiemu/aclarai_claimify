"""Dataset generation utility for creating gold standard training datasets.

This module provides functionality to generate high-quality training datasets
by using powerful teacher models to produce structured outputs from raw text inputs.
"""

import asyncio
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, Optional
import time
import traceback
import tqdm
import litellm
from litellm import RateLimitError

from ..data_models import ClaimifyConfig
from ..utils.context import create_context_window


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


import importlib
import numpy as np
import faiss


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


async def _run_generation_jobs(
    jobs: list[dict[str, Any]],
    concurrency: int,
    output_handle,
) -> tuple[int, int]:
    """Execute generation jobs concurrently while respecting the concurrency limit."""

    semaphore = asyncio.Semaphore(concurrency)
    progress_bar = tqdm.tqdm(total=len(jobs), desc="Generating examples")

    successful = 0
    failed = 0

    async def run_job(job: dict[str, Any]):
        async with semaphore:
            try:
                result = await asyncio.to_thread(job["func"], *job["args"])
                return job, result, None
            except Exception as exc:  # noqa: BLE001 - log and continue
                return job, None, exc

    tasks = [asyncio.create_task(run_job(job)) for job in jobs]

    try:
        for task in asyncio.as_completed(tasks):
            job, result, error = await task
            progress_bar.update(1)

            if error is not None:
                failed += 1
                _log_job_exception(job["metadata"], error)
                continue

            if result is None:
                failed += 1
                continue

            output_handle.write(json.dumps(result) + "\n")
            successful += 1
    finally:
        progress_bar.close()

    return successful, failed


def _iter_curated_examples(data: Any, file_path: Path) -> list[dict[str, Any]]:
    """Normalize curated data structures into example dictionaries."""

    if isinstance(data, dict):
        candidates = [value for value in data.values() if isinstance(value, dict)]
    elif isinstance(data, list):
        candidates = [item for item in data if isinstance(item, dict)]
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

    for file_path in files_to_process:
        if curated_flag:
            with open(file_path, "r", encoding="utf-8") as f_in:
                prospects = json.load(f_in)

            for prospect in _iter_curated_examples(prospects, file_path):
                target_sentence = prospect.get("target_sentence")
                if not target_sentence:
                    print(
                        f"Skipping prospect in {file_path}: missing target_sentence",
                        file=sys.stderr,
                    )
                    continue

                if component == "decomposition":
                    args = (target_sentence, final_teacher_model, model_params)
                else:
                    context_text = prospect.get("context_text", "")
                    if context_text is None:
                        print(
                            f"Skipping prospect in {file_path}: missing context_text",
                            file=sys.stderr,
                        )
                        continue
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
                        "metadata": {
                            "file_path": file_path,
                            "curated": True,
                        },
                    }
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

    with open(output_file, "w", encoding="utf-8") as f_out:
        successful_examples, failed_examples = asyncio.run(
            _run_generation_jobs(jobs, concurrency, f_out)
        )

    print("\n✅ Generation complete!")
    print(f"   Successfully generated: {successful_examples} examples")
    print(f"   Failed to generate: {failed_examples} examples")
    print(f"   Output saved to: {output_file}")

    if successful_examples == 0:
        raise GenerationError("Failed to generate any examples successfully")


def main():
    """Main entry point for testing the module directly."""
    print("This module is intended to be used through the CLI.")
    print("Use: aclarai-claimify generate-dataset --help")


if __name__ == "__main__":
    main()
