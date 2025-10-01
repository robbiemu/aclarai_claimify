"""
Example of a pure DSPy runtime for the Claimify pipeline.

This script demonstrates how to use the Claimify components with a DSPy LLM
to process a file and extract claims.

Usage:
    python examples/dspy_runtime.py --input-file <path_to_input_file> --output-file <path_to_output_file>
"""

import argparse
import dspy
import faiss
import logging
import numpy as np
import re
import sys
from attachments import attach, load, present

from aclarai_claimify.config import load_claimify_config
from aclarai_claimify.components.state import ClaimifyState
from aclarai_claimify.components.selection import SelectionComponent
from aclarai_claimify.components.disambiguation import DisambiguationComponent
from aclarai_claimify.components.decomposition import DecompositionComponent
from aclarai_claimify.utils.context import create_context_window
from aclarai_claimify.data_models import (
    ClaimifyContext,
    SentenceChunk,
)

from .common import clean_markdown_text, load_embedder, split_into_sentences


def main():
    """Main function to run the DSPy runtime."""
    parser = argparse.ArgumentParser(
        description="Pure DSPy runtime for the Claimify pipeline."
    )
    parser.add_argument(
        "--input-file",
        help="Path to the input file. If not provided, reads from STDIN.",
    )
    parser.add_argument(
        "--output-file",
        help="Path to the output file. If not provided, output will be printed to STDOUT.",
    )
    parser.add_argument("--model", default="openai/gpt-5", help="DSPy model to use.")
    parser.add_argument(
        "--context-method",
        default="semantic",
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
        default="WARNING",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Set the logging verbosity level.",
    )
    parser.add_argument(
        "--model-params",
        type=str,
        default="{}",
        help='JSON string with additional model parameters (e.g., \'{"temperature": 0.7, "max_tokens": 1000}\')',
    )
    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(level=args.verbosity)
    logger = logging.getLogger(__name__)

    logger.info(f"Using model: {args.model}")

    # Parse model_params if provided
    model_params = {}
    if hasattr(args, "model_params") and args.model_params:
        import json

        try:
            model_params = json.loads(args.model_params)
        except json.JSONDecodeError as e:
            logger.error(f"\n❌ Invalid JSON in --model-params: {e}")
            sys.exit(1)

    # Configure the DSPy LLM
    lm = dspy.LM(model=args.model, **model_params)
    dspy.configure(lm=lm, adapter=dspy.JSONAdapter())

    # Verify the configuration was set
    if dspy.settings.lm is None:
        logger.error("Failed to configure DSPy LM")
        sys.exit(1)

    logger.info(f"DSPy LM configured: {dspy.settings.lm}")

    # Load the config
    config = load_claimify_config()

    # Initialize the components WITHOUT passing llm
    # They will use the global dspy.settings.lm
    selection_comp = SelectionComponent(llm=dspy.settings.lm, config=config)
    disambiguation_comp = DisambiguationComponent(llm=dspy.settings.lm, config=config)
    decomposition_comp = DecompositionComponent(llm=dspy.settings.lm, config=config)

    # Read and clean the input - either from file or STDIN
    if args.input_file:
        # Use attachments for file input
        logger.info(f"Reading from file: {args.input_file}")
        attachment = (
            attach(args.input_file) | load.text_to_string | present.markdown_to_text
        )
        cleaned_text = attachment.text
        source_id = args.input_file
    else:
        # Read from STDIN and use the same cleaning logic
        logger.info("Reading from STDIN")
        raw_text = sys.stdin.read()
        cleaned_text = clean_markdown_text(raw_text)
        source_id = "<stdin>"

    logger.debug(f"\n--- Processing text from: {source_id} ---\n{cleaned_text}")

    sentences = split_into_sentences(cleaned_text)

    # Context generation
    if args.context_method == "semantic":
        print("Generating semantic context...")
        embedder_config = config.generate_dataset.semantic.embedder
        embedder = load_embedder(embedder_config)
        embeddings = embedder.embed(sentences)
        if np.any(np.linalg.norm(embeddings, axis=1) - 1.0 > 1e-6):
            faiss.normalize_L2(embeddings)
        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)

    # Process the sentences
    all_claims = []
    for i, sentence_text in enumerate(sentences):
        sentence_text = sentence_text.strip()
        if not sentence_text:
            continue

        logger.info(f"Processing sentence {i + 1}/{len(sentences)}: {sentence_text}")

        sentence = SentenceChunk(
            text=sentence_text,
            source_id=source_id,
            chunk_id=str(i),
            sentence_index=i,
        )

        if args.context_method == "static":
            context_sentences_str = create_context_window(
                sentences, i, args.k_window_size
            )
            context_sentences_list = []
            for line in context_sentences_str.split("\n"):
                if not line:
                    continue
                match = re.match(r"\\[(\d+)\\] (.*)", line)
                if match:
                    idx, text = match.groups()
                    context_sentences_list.append((text, int(idx)))

        elif args.context_method == "semantic":
            query_vector = embeddings[i : i + 1]
            search_k = min(len(sentences), args.max_k + 1)
            distances, indices = index.search(query_vector, search_k)

            context_sentences_list = []
            for neighbor_idx, similarity in zip(indices[0], distances[0]):
                if neighbor_idx == i:
                    continue

                if similarity >= args.similarity_threshold:
                    context_sentences_list.append(
                        (sentences[neighbor_idx], neighbor_idx)
                    )

            if len(context_sentences_list) < args.min_k:
                all_neighbors = [idx for idx in indices[0] if idx != i]
                context_sentences_list = [
                    (sentences[idx], idx) for idx in all_neighbors[: args.min_k]
                ]

        preceding_chunks = [
            SentenceChunk(
                text=s.strip(),
                source_id=source_id,
                chunk_id=f"{i}-{j}",
                sentence_index=idx,
            )
            for j, (s, idx) in enumerate(context_sentences_list)
        ]

        context = ClaimifyContext(
            current_sentence=sentence,
            preceding_sentences=preceding_chunks,
            following_sentences=[],
        )

        state = ClaimifyState(context=context)
        logger.info(f"Context for sentence {i + 1}: {context}")

        # Run the components in sequence
        state = selection_comp(state)
        logger.info(f"Selection result: {state.selection_result}")

        if state.was_selected:
            state = disambiguation_comp(state)
            logger.info(f"Disambiguation result: {state.disambiguation_result}")
            state = decomposition_comp(state)
            logger.info(f"Decomposition result: {state.decomposition_result}")

        if state.final_claims:
            for claim in state.final_claims:
                all_claims.append(claim.text)

    # Write the claims to the output file or STDOUT
    if args.output_file:
        with open(args.output_file, "w") as f:
            for claim in all_claims:
                f.write(f"{claim}\n")
        print(
            f"\nProcessing complete. {len(all_claims)} claims extracted to {args.output_file}"
        )
    else:
        print("\n--- Extracted Claims ---")
        for claim in all_claims:
            print(claim)


if __name__ == "__main__":
    main()
