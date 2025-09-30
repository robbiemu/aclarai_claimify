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
import json
import os
import re
import sys
from langgraph.graph import StateGraph, END
from typing import Literal

# Add the project root to the path to allow importing from aclarai_claimify
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from aclarai_claimify.components.state import ClaimifyState
from aclarai_claimify.components.selection import SelectionComponent
from aclarai_claimify.components.disambiguation import DisambiguationComponent
from aclarai_claimify.components.decomposition import DecompositionComponent
from aclarai_claimify.data_models import ClaimifyContext, SentenceChunk

# --- Graph Nodes ---

def selection_node(state: ClaimifyState) -> ClaimifyState:
    """Runs the SelectionComponent to decide if the sentence is worth processing."""
    print("\n--- Running Selection Node ---")
    # Initialize component within the node to use the globally configured LLM
    selection_component = SelectionComponent(llm=dspy.settings.lm)
    updated_state = selection_component(state)
    print(f"Selection result: {updated_state.was_selected}")
    return updated_state

def disambiguation_node(state: ClaimifyState) -> ClaimifyState:
    """Runs the DisambiguationComponent to clarify the sentence."""
    print("--- Running Disambiguation Node ---")
    disambiguation_component = DisambiguationComponent(llm=dspy.settings.lm)
    updated_state = disambiguation_component(state)
    if updated_state.disambiguation_result:
        print(f"Disambiguated text: {updated_state.disambiguation_result.disambiguated_text}")
    return updated_state

def decomposition_node(state: ClaimifyState) -> ClaimifyState:
    """Runs the DecompositionComponent to extract atomic claims."""
    print("--- Running Decomposition Node ---")
    decomposition_component = DecompositionComponent(llm=dspy.settings.lm)
    updated_state = decomposition_component(state)
    if updated_state.final_claims:
        print(f"Extracted {len(updated_state.final_claims)} claims.")
    return updated_state

# --- Conditional Edge ---

def should_continue(state: ClaimifyState) -> Literal["continue", "end"]:
    """Determines the next step after the selection node based on its output."""
    print(f"--- Checking Condition: Was sentence selected? -> {state.was_selected} ---")
    if state.was_selected:
        return "continue"
    else:
        return "end"

# --- Utility Functions ---

def split_into_sentences(text: str) -> list[str]:
    """Splits a text into sentences using a simple regex."""
    sentences = re.split(r'(?<=[.!?])\s+|\n+', text)
    return [s.strip() for s in sentences if s.strip()]

# --- Main Execution ---

def main():
    """Assemble the graph and process input from a file or stdin."""
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
        default="gpt-3.5-turbo",
        help="DSPy model to use (e.g., 'gpt-3.5-turbo', 'ollama/gemma:2b')."
    )
    parser.add_argument(
        "--model-params",
        type=str,
        default='{}',
        help='JSON string with additional model parameters (e.g., \'{"api_key": "...", "base_url": "..."}\')',
    )
    args = parser.parse_args()

    # 1. Configure DSPy LM
    try:
        model_params = json.loads(args.model_params)
        lm = dspy.LM(model=args.model, **model_params)
        dspy.configure(lm=lm)
        print(f"DSPy LM configured successfully with model: {args.model}")
    except Exception as e:
        print(f"Failed to configure DSPy LLM. Error: {e}")
        print("Please ensure your model parameters (e.g., API keys, base URLs) are correct.")
        return

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
        print(f"Reading from file: {args.input_file}")
        with open(args.input_file, 'r') as f:
            text_content = f.read()
        source_id = args.input_file
    else:
        print("Reading from STDIN...")
        text_content = sys.stdin.read()
        source_id = "<stdin>"

    sentences = split_into_sentences(text_content)
    print(f"Found {len(sentences)} sentences to process.")
    all_claims = []

    # 4. Loop through sentences and run the graph
    for i, sentence_text in enumerate(sentences):
        print(f"\n--- Processing sentence {i+1}/{len(sentences)} ---")
        print(f"Text: {sentence_text}")

        sentence = SentenceChunk(
            text=sentence_text,
            source_id=source_id,
            chunk_id=str(i),
            sentence_index=i,
        )
        context = ClaimifyContext(current_sentence=sentence)
        initial_state = ClaimifyState(context=context)

        final_state = app.invoke(initial_state)

        if final_state.final_claims:
            for claim in final_state.final_claims:
                all_claims.append(claim.text)

    # 5. Output results
    output_content = "\n".join(all_claims)
    if args.output_file:
        with open(args.output_file, 'w') as f:
            f.write(output_content)
        print(f"\n--- Processing complete. {len(all_claims)} claims extracted to {args.output_file} ---")
    else:
        print("\n--- Extracted Claims ---")
        print(output_content)

if __name__ == "__main__":
    main()