"""This script is a "Prospector" agent that analyzes a directory of markdown files and, for each file, selects the best positive and negative examples for a given Claimify component. It then saves these curated examples to a new directory of JSON files."""

import argparse
import asyncio
import glob
import json
import os
import re
import sys
from pathlib import Path
import litellm
from tqdm.asyncio import tqdm as async_tqdm
from json_repair import loads as json_repair_loads


def get_prospector_prompt(content: str, component: str) -> str:
    """Constructs a detailed prompt for the prospector agent based on the component.

    Args:
        content: The content of the document to analyze.
        component: The Claimify component to prospect for.

    Returns:
        A detailed prompt string.
    """
    if component == "selection":
        return f"""You are an expert in data quality for training AI models.
Your task is to analyze the following document and select the single best positive example and the single best negative example of a verifiable factual claim.

**Definitions:**
- A **positive example** is a sentence that contains a specific, verifiable factual claim (e.g., contains numbers, dates, specific events, or technical specifications).
- A **negative example** is a sentence that is clearly subjective, an opinion, a recommendation, or a rhetorical question.

**Document Content:**
---
{content}
---

**Your Task:**
1.  Read the entire document.
2.  Identify the single best positive example of a verifiable factual claim.
3.  Identify the single best negative example (a subjective or opinionated sentence).
4.  For each of the two selected sentences, provide the sentence itself and the two sentences immediately before and after it as context.
5.  Return a single JSON object with the keys "positive_example" and "negative_example". Each key should contain the "target_sentence" and "context_text".

**JSON Output Format:**
{{
  "positive_example": {{
    "target_sentence": "...",
    "context_text": "..."
  }},
  "negative_example": {{
    "target_sentence": "...",
    "context_text": "..."
  }}
}}
"""
    elif component == "disambiguation":
        return f"""You are an expert in data quality for training AI models.
Your task is to analyze the following document and surface:
- one **positive example** that clearly requires disambiguation, and
- as many **negative examples** as you can find that illustrate different failure modes for disambiguation outputs.

**Positive example:**
- Target a sentence whose pronouns, deictic phrases, or elliptical references demand carefully grounded rewrites.
- Choose a case that, when rewritten correctly, would avoid all common disambiguation failure modes (no hallucinations, no lost qualifiers, single well-formed sentence, etc.).

**Negative failure modes:**
Label each negative example with one of the following identifiers (use snake_case exactly as written):
- unresolved_referent
- hallucinated_detail
- omitted_constraint
- formatting_drift
- unsupported_resolution
- confidence_mismatch

Negative examples should be sentences where an inexperienced model would likely fall into that failure mode when rewriting. Provide context that makes the pitfall obvious.

**Document Content:**
---
{content}
---

**Your Task:**
1.  Read the document.
2.  Select the strongest positive example per guidance above.
3.  Collect a diverse set of negative examples, ideally one per failure mode. Skip a mode if the document truly lacks a representative sentence.
4.  For every example, include the sentence itself and the two surrounding sentences as context.
5.  Return a JSON object matching this schema:
{{
  "positive_example": {{
    "target_sentence": "...",
    "context_text": "...",
    "rationale": "Explain why this is an excellent disambiguation case."
  }},
  "negative_examples": [
    {{
      "failure_mode": "unresolved_referent",
      "target_sentence": "...",
      "context_text": "...",
      "rationale": "Explain why this context tempts that failure mode."
    }},
    ...
  ]
}}
Ensure all rationales are concise (<= 2 sentences each).
"""
    elif component == "decomposition":
        return f"""You are an expert in data quality for training AI models.
Your task is to analyze the following document and surface:
- one **positive example** that truly requires decomposition into multiple atomic claims, and
- as many **negative examples** as you can find that demonstrate different decomposition failure modes.

**Positive example:**
- Pick a sentence whose faithful decomposition should yield multiple high-quality claims.
- The sentence should cover several verifiable ideas with clear distinctions so a careful model can avoid common pitfalls.

**Negative failure modes:**
Label each negative example with one of the following identifiers (use snake_case exactly as written):
- non_atomic_claim
- missing_key_claim
- incorrect_metadata
- off_topic_hallucination
- structure_violation
- low_information_reasoning

Negative examples should be sentences where a naïve decomposition step would likely trigger that failure (e.g., fusing claims, losing key content, returning empty or malformed metadata).

**Document Content:**
---
{content}
---

**Your Task:**
1.  Read the document.
2.  Select the strongest positive example per guidance above.
3.  Collect a diverse set of negative examples, ideally one per failure mode. Skip a mode if the document lacks a representative sentence.
4.  For every example, include the sentence itself and the two surrounding sentences as context.
5.  Return a JSON object matching this schema:
{{
  "positive_example": {{
    "target_sentence": "...",
    "context_text": "...",
    "rationale": "Explain why this supports a rich decomposition."
  }},
  "negative_examples": [
    {{
      "failure_mode": "non_atomic_claim",
      "target_sentence": "...",
      "context_text": "...",
      "rationale": "Explain why this context encourages that failure."
    }},
    ...
  ]
}}
Keep rationales concise (<= 2 sentences each).
"""
    else:
        raise ValueError(f"Invalid component: {component}")


def clean_markdown(content: str) -> str:
    """Removes common markdown syntax from a string to prepare it for analysis.

    Args:
        content: The raw markdown content.

    Returns:
        The cleaned text content.
    """
    content = re.sub(r"^#+\s+", "", content, flags=re.MULTILINE)
    content = re.sub(r"(\*\*|\*|__|_)(.*?)\1", r"\2", content)
    content = re.sub(r"\[(.*?)\]\((.*?)\)", r"\1", content)
    content = re.sub(r"^\s*[-*+]\s+", "", content, flags=re.MULTILINE)
    content = re.sub(r"`{{1,3}}[\s\S]*?`{{1,3}}", "", content)
    content = re.sub(r"^---\s*$", "", content, flags=re.MULTILINE)
    content = re.sub(r"^>\s?", "", content, flags=re.MULTILINE)
    return content


async def read_text_async(path: Path) -> str:
    """Reads a text file asynchronously to avoid blocking the event loop.

    Args:
        path: The path to the file to read.

    Returns:
        The content of the file.
    """
    return await asyncio.to_thread(path.read_text, encoding="utf-8")


async def write_json_async(path: Path, payload: dict) -> None:
    """Writes a JSON payload to a file asynchronously.

    Args:
        path: The path to the file to write.
        payload: The JSON payload to write.
    """

    def _write() -> None:
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    await asyncio.to_thread(_write)


async def get_prospect(file_path, output_dir, component, model, k):
    """Processes a single file to get the best and worst examples, making an async API call.

    Args:
        file_path: The path to the input file.
        output_dir: The directory to save the output JSON file.
        component: The Claimify component to prospect for.
        model: The name of the model to use for the API call.
        k: The context window size.
    """
    try:
        source_path = Path(file_path)
        content = await read_text_async(source_path)

        cleaned_content = clean_markdown(content)
        prompt = get_prospector_prompt(cleaned_content[:4096], component)

        response = await litellm.acompletion(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=1.0,
        )
        print(f"---MODEL RESPONSE---\n{response}\n---")
        prospects = json_repair_loads(response.choices[0].message.content.strip())
        print(f"Prospects for {file_path}: {prospects}")

        output_path = Path(output_dir) / (source_path.stem + ".json")
        await write_json_async(output_path, prospects)

    except Exception as e:
        print(f"Error processing file {file_path}: {e}", file=sys.stderr)


async def main():
    """Main entry point for the prospector script.
    Parses command-line arguments and orchestrates the concurrent processing of files.
    """
    parser = argparse.ArgumentParser(
        description="Select best and worst sentences from markdown files."
    )
    parser.add_argument(
        "component",
        choices=["selection", "disambiguation", "decomposition"],
        help="The component to prospect for",
    )
    parser.add_argument(
        "-i",
        "--input",
        required=True,
        dest="input_dir",
        help="Input directory of markdown files",
    )
    parser.add_argument(
        "-o",
        "--output",
        required=True,
        dest="output_dir",
        help="Output directory for curated JSON files",
    )
    parser.add_argument(
        "--model", default="gpt-3.5-turbo", help="The model to use for evaluation."
    )
    parser.add_argument(
        "--k",
        type=int,
        default=2,
        help="The number of sentences to include before and after the target sentence for context.",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="The number of concurrent requests to make.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Process at most this many files; useful for debugging.",
    )
    args = parser.parse_args()

    if not os.path.isdir(args.input_dir):
        print(f"Error: Input directory not found: {args.input_dir}", file=sys.stderr)
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)

    files = glob.glob(os.path.join(args.input_dir, "*.md"))

    if args.limit is not None:
        if args.limit <= 0:
            print("Error: --limit must be a positive integer.", file=sys.stderr)
            sys.exit(1)
        files = files[: args.limit]

    if not files:
        print("No markdown files found to process.")
        return

    semaphore = asyncio.Semaphore(args.concurrency)

    progress_bar = async_tqdm(total=len(files), desc="Processing files")

    async def run_with_semaphore(file_path: str) -> None:
        try:
            async with semaphore:
                await get_prospect(
                    file_path,
                    args.output_dir,
                    args.component,
                    args.model,
                    args.k,
                )
        finally:
            progress_bar.update(1)

    tasks = [asyncio.create_task(run_with_semaphore(file)) for file in files]
    try:
        await asyncio.gather(*tasks)
    finally:
        progress_bar.close()

    print(
        f"Successfully processed {len(files)} files and created curated JSON files in {args.output_dir}"
    )


if __name__ == "__main__":
    asyncio.run(main())
