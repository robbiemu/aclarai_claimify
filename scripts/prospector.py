"""This script is a "Prospector" agent that analyzes a directory of markdown files and, for each file, selects the best positive and negative examples for a given Claimify component. It then saves these curated examples to a new directory of JSON files.
"""

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
Your task is to analyze the following document and select the single best positive example and the single best negative example of a sentence that requires disambiguation.

**Definitions:**
- A **positive example** is a sentence that contains pronouns or ambiguous references that need to be resolved using the context.
- A **negative example** is a sentence that is already self-contained and does not require any disambiguation.

**Document Content:**
---
{content}
---

**Your Task:**
1.  Read the entire document.
2.  Identify the single best positive example of a sentence that requires disambiguation.
3.  Identify the single best negative example (a self-contained sentence).
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
    elif component == "decomposition":
        return f"""You are an expert in data quality for training AI models.
Your task is to analyze the following document and select the single best positive example and the single best negative example of a sentence that requires decomposition.

**Definitions:**
- A **positive example** is a long, complex sentence with multiple clauses that can be broken down into smaller, atomic claims.
- A **negative example** is a short, simple, atomic sentence that cannot be broken down further.

**Document Content:**
---
{content}
---

**Your Task:**
1.  Read the entire document.
2.  Identify the single best positive example of a sentence that requires decomposition.
3.  Identify the single best negative example (an atomic sentence).
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
    content = re.sub(r"^---\s*$", "", content, flags=re.MULTILLINE)
    content = re.sub(r"^>\s?", "", content, flags=re.MULTILINE)
    return content


def create_context_window(sentences: list[str], index: int, k: int) -> str:
    """Creates a context window of k sentences before and after the target sentence.

    Args:
        sentences: A list of sentences.
        index: The index of the target sentence.
        k: The number of sentences to include before and after the target.

    Returns:
        A string containing the context window.
    """
    start = max(0, index - k)
    end = min(len(sentences), index + k + 1)
    return "\n".join(sentences[start:end])


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
