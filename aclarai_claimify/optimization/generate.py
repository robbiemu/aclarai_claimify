"""Dataset generation utility for creating gold standard training datasets.

This module provides functionality to generate high-quality training datasets
by using powerful teacher models to produce structured outputs from raw text inputs.
"""

import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional
import time
import traceback
import tqdm
import litellm
from litellm import RateLimitError


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


def call_teacher_model(prompt: str, model: str) -> str:
    """Call the teacher model with the given prompt.

    Args:
        prompt: The prompt to send to the model
        model: The model name to use

    Returns:
        The model's response text

    Raises:
        APICallError: If the API call fails
    """
    max_retries = 5
    for attempt in range(max_retries):
        try:
            response = litellm.completion(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=500,
            )
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
    context: str, target: str, teacher_model: str
) -> Optional[Dict[str, Any]]:
    """Generate a single example for the selection component.

    Args:
        context: Context text
        target: Target sentence to evaluate
        teacher_model: Teacher model name

    Returns:
        Generated example dictionary or None if failed
    """
    prompt = TEACHER_PROMPTS["selection"].format(
        context_text=context, target_sentence=target
    )

    try:
        response = call_teacher_model(prompt, teacher_model)
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
    context: str, target: str, teacher_model: str
) -> Optional[Dict[str, Any]]:
    """Generate a single example for the disambiguation component.

    Args:
        context: Context text
        target: Target sentence to disambiguate
        teacher_model: Teacher model name

    Returns:
        Generated example dictionary or None if failed
    """
    prompt = TEACHER_PROMPTS["disambiguation"].format(
        context_text=context, target_sentence=target
    )

    try:
        response = call_teacher_model(prompt, teacher_model)
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
    disambiguated_text: str, teacher_model: str
) -> Optional[Dict[str, Any]]:
    """Generate a single example for the decomposition component.

    Args:
        disambiguated_text: Text to decompose
        teacher_model: Teacher model name

    Returns:
        Generated example dictionary or None if failed
    """
    prompt = TEACHER_PROMPTS["decomposition"].format(
        disambiguated_text=disambiguated_text
    )

    try:
        response = call_teacher_model(prompt, teacher_model)
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


def generate_dataset(
    input_file: Path,
    output_file: Path,
    component: str,
    teacher_model: str,
) -> None:
    """Generate a gold standard dataset from raw text inputs.

    Args:
        input_file: Path to input text file (one sentence per line)
        output_file: Path to output JSONL file
        component: Component name (selection, disambiguation, decomposition)
        teacher_model: Teacher model name

    Raises:
        GenerationError: If generation fails
        FileNotFoundError: If input file doesn't exist
    """
    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")

    # Select the appropriate generator function
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

    # Read input lines
    with open(input_file, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    if not lines:
        raise GenerationError("Input file is empty")

    print(
        f"Generating {len(lines)} examples for {component} component using {teacher_model}..."
    )
    print("⚠️  WARNING: This may incur costs depending on your API provider and usage!")

    # Process each line
    successful_examples = 0
    failed_examples = 0

    with open(output_file, "w", encoding="utf-8") as f_out:
        for i, line in enumerate(tqdm.tqdm(lines, desc="Generating examples")):
            try:
                example = None
                if component == "decomposition":
                    # For decomposition, the line is the disambiguated text
                    example = generator(line, teacher_model)
                else:
                    # For selection and disambiguation, we need context and target
                    # We'll create a simple context with just the sentence itself
                    # In practice, users should provide better context
                    context = f"[0] {line}"
                    example = generator(context, line, teacher_model)

                if example is not None:
                    f_out.write(json.dumps(example) + "\n")
                    successful_examples += 1
                else:
                    failed_examples += 1

            except Exception as e:
                failed_examples += 1
                print(f"Error processing line {i + 1}: {e}", file=sys.stderr)
                if hasattr(e, "__traceback__"):
                    traceback.print_exception(
                        type(e), e, e.__traceback__, file=sys.stderr
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
