"""
Example of running the Claimify pipeline as a graph using LangGraph.

This script demonstrates how to orchestrate the stateless Claimify components
(Selection, Disambiguation, Decomposition) using LangGraph. It processes text
from a file or standard input and extracts claims.

Usage:
    - Ensure you have a compatible LLM server running.
    - Run with a file:
      python examples/langgraph_runtime.py --input-file <path_to_input_file>
    - Run with stdin:
      cat <path_to_input_file> | python examples/langgraph_runtime.py
"""

import argparse
import dspy
import logging
import re
import sys
from langgraph.graph import StateGraph, END
from typing import Literal, Dict, Any
import numpy as np
import faiss

from common import (
    clean_markdown_text,
    split_into_sentences,
    load_embedder,
)

from attachments import attach, load, present

from aclarai_claimify.config import load_claimify_config
from aclarai_claimify.components.state import ClaimifyState
from aclarai_claimify.components.selection import SelectionComponent
from aclarai_claimify.components.disambiguation import DisambiguationComponent
from aclarai_claimify.components.decomposition import DecompositionComponent
from aclarai_claimify.data_models import ClaimifyContext, SentenceChunk
from aclarai_claimify.utils.context import create_context_window


# --- Graph Nodes ---
# Each node in the graph must return a dictionary containing only the fields
# of the state that it has modified. LangGraph uses this dictionary to update
# the central state object.

selection_component: SelectionComponent = None
disambiguation_component: DisambiguationComponent = None
decomposition_component: DecompositionComponent = None


def selection_node(state: ClaimifyState) -> Dict[str, Any]:
    """Runs the SelectionComponent and returns the updated part of the state."""
    logging.info("\n--- Running Selection Node ---")
    updated_state = selection_component(state)
    logging.info(f"\n reason: {updated_state.selection_result.reasoning}")
    return {"selection_result": updated_state.selection_result}


def disambiguation_node(state: ClaimifyState) -> Dict[str, Any]:
    """Runs the DisambiguationComponent and returns the updated part of the state."""
    logging.info("--- Running Disambiguation Node ---")
    updated_state = disambiguation_component(state)
    if updated_state.disambiguation_result:
        logging.info(
            f"Disambiguated text: {updated_state.disambiguation_result.disambiguated_text}"
        )
    return {"disambiguation_result": updated_state.disambiguation_result}


def decomposition_node(state: ClaimifyState) -> Dict[str, Any]:
    """Runs the DecompositionComponent and returns the updated part of the state."""
    logging.info("--- Running Decomposition Node ---")
    updated_state = decomposition_component(state)
    if updated_state.final_claims:
        logging.info(f"Extracted {len(updated_state.final_claims)} claims.")
    return {
        "decomposition_result": updated_state.decomposition_result,
        "final_claims": updated_state.final_claims,
    }


# --- Conditional Edge ---


def should_continue(state: ClaimifyState) -> Literal["continue", "end"]:
    """Determines the next step after the selection node based on its output."""
    logging.info(
        f"--- Checking Condition: Was sentence selected? -> {state.was_selected} ---"
    )
    if state.was_selected:
        return "continue"
    else:
        return "end"


# --- Utility Functions ---


# --- Main Execution ---


def main():
    """Assemble the graph and process input from a file or stdin."""
    global selection_component, disambiguation_component, decomposition_component

    parser = argparse.ArgumentParser(
        description="Run the Claimify pipeline using LangGraph."
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
        help="DSPy model to use (e.g., 'openai/gpt-5', 'ollama/gemma:2b').",
    )
    parser.add_argument(
        "--model-params",
        type=str,
        default="{}",
        help='JSON string with additional model parameters (e.g., \'{"temperature": 0.7, "max_tokens": 1000}\')',
    )
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
    dspy.configure(lm=lm)

    # Verify the configuration was set
    if dspy.settings.lm is None:
        logger.error("Failed to configure DSPy LM")
        sys.exit(1)

    logger.info(f"DSPy LM configured: {dspy.settings.lm}")

    # Load the config
    config = load_claimify_config()

    # Initialize components
    selection_component = SelectionComponent(llm=dspy.settings.lm, config=config)
    disambiguation_component = DisambiguationComponent(
        llm=dspy.settings.lm, config=config
    )
    decomposition_component = DecompositionComponent(
        llm=dspy.settings.lm, config=config
    )

    # 2. Define the Graph
    workflow = StateGraph(ClaimifyState)
    workflow.add_node("selection", selection_node)
    workflow.add_node("disambiguation", disambiguation_node)
    workflow.add_node("decomposition", decomposition_node)

    workflow.set_entry_point("selection")
    workflow.add_conditional_edges(
        "selection",
        should_continue,
        {"continue": "disambiguation", "end": END},
    )
    workflow.add_edge("disambiguation", "decomposition")
    workflow.add_edge("decomposition", END)

    app = workflow.compile()

    # 3. Read and process input
    if args.input_file:
        logger.info(f"Reading from file: {args.input_file}")
        attachment = (
            attach(args.input_file) | load.text_to_string | present.markdown_to_text
        )
        text_content = attachment.text
        source_id = args.input_file
    else:
        logger.info("Reading from STDIN...")
        raw_text = sys.stdin.read()
        text_content = clean_markdown_text(raw_text)
        source_id = "<stdin>"

    logger.debug(f"\n--- Processing text from: {source_id} ---\n{text_content}")

    sentences = split_into_sentences(text_content)
    logger.info(f"Found {len(sentences)} sentences to process.")
    all_claims = []

    # Context generation setup
    embedder = None
    index = None
    embeddings = None
    if args.context_method == "semantic":
        logger.info("Generating semantic context...")
        embedder_config = config.generate_dataset.semantic.embedder
        embedder = load_embedder(embedder_config)
        embeddings = embedder.embed(sentences)
        if np.any(np.linalg.norm(embeddings, axis=1) - 1.0 > 1e-6):
            faiss.normalize_L2(embeddings)
        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)

    # 4. Loop through sentences and run the graph
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

        # Generate context
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

        else:
            context_sentences_list = []

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

        initial_state = ClaimifyState(context=context)
        logger.info(f"Context for sentence {i + 1}: {context}")

        # The `invoke` method returns the final state of the graph as a dictionary.
        final_state = app.invoke(initial_state)

        final_claims = final_state.get("final_claims")
        if final_claims:
            for claim in final_claims:
                all_claims.append(claim.text)

    # 5. Output results
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
