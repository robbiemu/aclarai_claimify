"""
This script demonstrates how to run the original, agent-based Claimify pipeline.
It shows how to orchestrate the three core agents (Selection, Disambiguation, Decomposition)
in a sequential manner and integrate them with a modern LLM provider like DSPy.

This example processes text from a file or STDIN, cleans it, and extracts claims sentence by sentence.

Usage:
    # From a file
    python examples/agent_pipeline_runtime.py --input-file <path_to_input_file> --output-file <path_to_output_file>

    # From STDIN
    cat <path_to_input_file> | python examples/agent_pipeline_runtime.py
"""
import argparse
import logging
import os
import re
import sys
from typing import Any, List

import dspy

from aclarai_claimify.agents import (
    DecompositionAgent,
    DisambiguationAgent,
    LLMInterface,
    SelectionAgent,
)
from aclarai_claimify.config import load_claimify_config
from aclarai_claimify.data_models import ClaimifyContext, SentenceChunk
from aclarai_claimify.utils.context import create_context_window

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class DSPyLLM(LLMInterface):
    """
    A wrapper class to make a DSPy language model compatible with the
    Claimify pipeline's LLMInterface protocol.
    """

    def __init__(self, llm: dspy.dsp.LM):
        self.llm = llm

    def complete(self, prompt: str, **kwargs: Any) -> str:
        response = self.llm(prompt, **kwargs)
        return response[0] if isinstance(response, list) and response else ""


def _clean_markdown_text(text_content: str) -> str:
    """
    Strip markdown syntax to plain text.
    A simplified version for this example.
    """
    if not text_content:
        return ""
    text = re.sub(r"#+\s+", "", text_content)  # Headings
    text = re.sub(r"(\*\*|\*|__|_)(.*?)\1", r"\2", text)  # Bold/Italics
    text = re.sub(r"\[(.*?)\]\(.*?\)", r"\1", text)  # Links
    text = re.sub(r"^\s*[-*+]\s+", "", text, flags=re.MULTILINE)  # Lists
    text = text.replace("\n", " ").strip()
    text = re.sub(r"\s{2,}", " ", text)
    return text


def split_into_sentences(text: str) -> List[str]:
    """Splits a text into sentences using a simple regex."""
    sentences = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in sentences if s.strip()]


def main():
    """
    Main function to set up and run the agent-based Claimify pipeline.
    """
    parser = argparse.ArgumentParser(
        description="Agent-based runtime for the Claimify pipeline."
    )
    parser.add_argument(
        "--input-file",
        help="Path to the input file. If not provided, reads from STDIN.",
    )
    parser.add_argument(
        "--output-file",
        help="Path to the output file. If not provided, prints to STDOUT.",
    )
    parser.add_argument(
        "--model", default="gemma", help="Ollama model to use (e.g., 'gemma', 'llama3')."
    )
    parser.add_argument(
        "--k-window-size",
        type=int,
        default=3,
        help="Number of preceding/following sentences for context.",
    )
    parser.add_argument(
        "--verbosity",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Set the logging verbosity level.",
    )
    args = parser.parse_args()

    # Configure logging level
    logger.setLevel(args.verbosity)

    # 1. Configure DSPy with an LLM provider
    logger.info(f"Configuring DSPy with Ollama model: {args.model}")
    llm = dspy.Ollama(model=args.model, base_url=os.getenv("OLLAMA_BASE_URL"))
    dspy.settings.configure(lm=llm)

    # 2. Create the LLM wrapper for the Claimify agents
    dspy_llm_wrapper = DSPyLLM(llm)

    # 3. Load the Claimify configuration
    config = load_claimify_config()

    # 4. Initialize the three core agents
    selection_agent = SelectionAgent(llm=dspy_llm_wrapper, config=config)
    disambiguation_agent = DisambiguationAgent(llm=dspy_llm_wrapper, config=config)
    decomposition_agent = DecompositionAgent(llm=dspy_llm_wrapper, config=config)

    # 5. Read and clean the input from file or STDIN
    if args.input_file:
        logger.info(f"Reading from file: {args.input_file}")
        with open(args.input_file, "r", encoding="utf-8") as f:
            raw_text = f.read()
        source_id = args.input_file
    else:
        logger.info("Reading from STDIN...")
        raw_text = sys.stdin.read()
        source_id = "<stdin>"

    cleaned_text = _clean_markdown_text(raw_text)
    sentences = split_into_sentences(cleaned_text)
    logger.info(f"Found {len(sentences)} sentences to process from '{source_id}'.")

    all_claims = []
    # 6. Process each sentence through the pipeline
    for i, sentence_text in enumerate(sentences):
        logger.info(f"--- Processing sentence {i+1}/{len(sentences)}: '{sentence_text}' ---")

        # Create context for the current sentence
        context_str = create_context_window(sentences, i, k=args.k_window_size)
        context_sentences = [
            SentenceChunk(text=s) for s in context_str.split("\n") if s
        ]

        # This is a simplification; a real implementation would need to parse the indices from create_context_window
        current_sentence_chunk = SentenceChunk(text=sentence_text, source_id=source_id, chunk_id=str(i))
        context = ClaimifyContext(
            current_sentence=current_sentence_chunk,
            preceding_sentences=[s for s in context_sentences if sentences.index(s.text) < i],
            following_sentences=[s for s in context_sentences if sentences.index(s.text) > i]
        )

        # Stage 1: Selection
        selection_result = selection_agent.process(context)
        logger.info(f"Selection result: {selection_result.is_selected}, Reason: {selection_result.reasoning}")

        if selection_result.is_selected:
            # Stage 2: Disambiguation
            disambiguation_result = disambiguation_agent.process(
                sentence=context.current_sentence, context=context
            )
            logger.info(f"Disambiguated text: '{disambiguation_result.disambiguated_text}'")

            # Stage 3: Decomposition
            decomposition_result = decomposition_agent.process(
                disambiguated_text=disambiguation_result.disambiguated_text,
                _original_sentence=context.current_sentence,
            )
            logger.info(f"Decomposition found {len(decomposition_result.valid_claims)} valid claims.")

            if decomposition_result.valid_claims:
                for claim in decomposition_result.valid_claims:
                    all_claims.append(claim.text)

    # 7. Write claims to output file or STDOUT
    if args.output_file:
        with open(args.output_file, "w", encoding="utf-8") as f:
            for claim in all_claims:
                f.write(f"- {claim}\n")
        logger.info(f"\n✅ Successfully extracted {len(all_claims)} claims to {args.output_file}")
    else:
        print("\n--- Extracted Claims ---")
        if all_claims:
            for claim in all_claims:
                print(f"- {claim}")
        else:
            print("No claims were extracted.")
        print("----------------------")


if __name__ == "__main__":
    main()