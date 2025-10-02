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
import json
import sys
import re
import numpy as np
import faiss
from typing import Any
from json_repair import repair_json

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
from examples.common import clean_markdown_text, split_into_sentences, load_embedder

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class DSPyLLM(LLMInterface):
    """
    A wrapper class to make a DSPy language model compatible with the
    Claimify pipeline's LLMInterface protocol.
    """

    def __init__(self, llm: dspy.LM):
        self.llm = llm

    def complete(self, prompt: str, **kwargs: Any) -> str:
        response = self.llm(prompt, **kwargs)
        if not response:
            return ""
        # The response could be a list of strings
        if isinstance(response, list):
            completion = response[0]
        else:
            completion = str(response)

        return repair_json(completion)


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
        "--model",
        default="openai/gpt-5",
        help="DSPy model to use.",
    )
    parser.add_argument(
        "--model-params",
        type=str,
        default="{}",
        help='JSON string with additional model parameters (e.g., \'{"temperature": 0.7, "max_tokens": 1000}\')',
    )
    parser.add_argument(
        "--context-method",
        default="static",
        choices=["static", "semantic"],
        help="Context generation method to use.",
    )
    parser.add_argument(
        "--k-window-size",
        type=int,
        default=3,
        help="Window size for static context generation.",
    )
    parser.add_argument(
        "--similarity-threshold",
        type=float,
        default=0.7,
        help="Similarity threshold for semantic context generation.",
    )
    parser.add_argument(
        "--min-k",
        type=int,
        default=1,
        help="Minimum number of context sentences for semantic context generation.",
    )
    parser.add_argument(
        "--max-k",
        type=int,
        default=5,
        help="Maximum number of context sentences for semantic context generation.",
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

    # Parse model_params if provided
    model_params = {}
    if args.model_params:
        try:
            model_params = json.loads(args.model_params)
        except json.JSONDecodeError as e:
            logger.error(f"\n❌ Invalid JSON in --model-params: {e}")
            sys.exit(1)

    # Load the Claimify configuration
    config = load_claimify_config()

    # Use config values for model if not provided via CLI
    if not args.model:
        args.model = config.model.claimify.default

    # Set model_params from config, override with CLI if provided
    config_params = {
        'temperature': config.temperature,
        'max_tokens': config.max_tokens,
    }
    config_params.update(model_params)

    # For gpt-5, force required params
    if "gpt-5" in args.model:
        config_params['temperature'] = 1.0
        if 'max_tokens' not in model_params:
            config_params['max_tokens'] = 16000

    model_params = config_params

    # Override config with final model params for agent calls
    config.temperature = model_params.get('temperature', config.temperature)
    config.max_tokens = model_params.get('max_tokens', config.max_tokens)

    # 1. Configure DSPy with an LLM provider
    logger.info(f"Using model: {args.model}")
    lm = dspy.LM(model=args.model, **model_params)
    dspy.configure(lm=lm, adapter=dspy.JSONAdapter())

    # 2. Create the LLM wrapper for the Claimify agents
    dspy_llm_wrapper = DSPyLLM(lm)

    # 3. Initialize the three core agents
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

    cleaned_text = clean_markdown_text(raw_text)
    sentences = split_into_sentences(cleaned_text)

    # Context generation setup if semantic
    embeddings = None
    index = None
    if args.context_method == "semantic":
        logger.info("Generating semantic context...")
        embedder_config = config.generate_dataset.semantic.embedder
        embedder = load_embedder(embedder_config)
        embeddings = embedder.embed(sentences)
        if np.any(np.linalg.norm(embeddings, axis=1) - 1.0 > 1e-6):
            faiss.normalize_L2(embeddings)
        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)

    logger.info(f"Found {len(sentences)} sentences to process from '{source_id}'.")

    all_claims = []
    # 6. Process each sentence through the pipeline
    for i, sentence_text in enumerate(sentences):
        logger.info(
            f"--- Processing sentence {i + 1}/{len(sentences)}: '{sentence_text}' ---"
        )

        preceding_sentences = []
        following_sentences = []
        if args.context_method == "static":
            context_str = create_context_window(sentences, i, k=args.k_window_size)
            for line in context_str.split("\n"):
                if not line:
                    continue
                match = re.match(r"\[(\d+)\] (.*)", line)
                if match:
                    idx = int(match.group(1))
                    text = match.group(2)
                    chunk = SentenceChunk(
                        text=text,
                        source_id=source_id,
                        chunk_id=f"{i}-{idx}",
                        sentence_index=idx,
                    )
                    if idx < i:
                        preceding_sentences.append(chunk)
                    elif idx > i:
                        following_sentences.append(chunk)

        elif args.context_method == "semantic":
            if embeddings is None or index is None:
                preceding_sentences = []
                following_sentences = []
            else:
                query_vector = embeddings[i : i + 1]
                search_k = min(len(sentences), args.max_k + 1)
                distances, indices = index.search(query_vector, search_k)

                context_sentences_list = []
                for neighbor_idx_raw, similarity in zip(indices[0], distances[0]):
                    neighbor_idx = int(neighbor_idx_raw)
                    if neighbor_idx == i:
                        continue
                    if similarity >= args.similarity_threshold:
                        context_sentences_list.append(
                            SentenceChunk(
                                text=sentences[neighbor_idx],
                                source_id=source_id,
                                chunk_id=f"{i}-{neighbor_idx}",
                                sentence_index=neighbor_idx,
                            )
                        )

                if len(context_sentences_list) < args.min_k:
                    all_neighbors = [
                        int(idx_raw) for idx_raw in indices[0] if int(idx_raw) != i
                    ][: args.min_k]
                    context_sentences_list = [
                        SentenceChunk(
                            text=sentences[neighbor_idx],
                            source_id=source_id,
                            chunk_id=f"{i}-{neighbor_idx}",
                            sentence_index=neighbor_idx,
                        )
                        for neighbor_idx in all_neighbors
                    ]

                preceding_sentences = [
                    s for s in context_sentences_list if s.sentence_index < i
                ]
                following_sentences = []  # Semantic uses preceding only

        else:
            preceding_sentences = []
            following_sentences = []

        current_sentence_chunk = SentenceChunk(
            text=sentence_text, source_id=source_id, chunk_id=str(i), sentence_index=i
        )
        context = ClaimifyContext(
            current_sentence=current_sentence_chunk,
            preceding_sentences=preceding_sentences,
            following_sentences=following_sentences,
        )

        # Stage 1: Selection
        selection_result = selection_agent.process(context)
        logger.info(
            f"Selection result: {selection_result.is_selected}, Reason: {selection_result.reasoning}"
        )

        if selection_result.is_selected:
            # Stage 2: Disambiguation
            disambiguation_result = disambiguation_agent.process(
                sentence=context.current_sentence, context=context
            )
            logger.info(
                f"Disambiguated text: '{disambiguation_result.disambiguated_text}'"
            )

            # Stage 3: Decomposition
            decomposition_result = decomposition_agent.process(
                disambiguated_text=disambiguation_result.disambiguated_text,
                _original_sentence=context.current_sentence,
            )
            logger.info(
                f"Decomposition found {len(decomposition_result.valid_claims)} valid claims."
            )

            if decomposition_result.valid_claims:
                for claim in decomposition_result.valid_claims:
                    all_claims.append(claim.text)

    # 7. Write claims to output file or STDOUT
    if args.output_file:
        with open(args.output_file, "w", encoding="utf-8") as f:
            for claim in all_claims:
                f.write(f"- {claim}\n")
        logger.info(
            f"\n✅ Successfully extracted {len(all_claims)} claims to {args.output_file}"
        )
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
