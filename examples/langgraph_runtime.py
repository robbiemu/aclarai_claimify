"""
Example of running the Claimify pipeline as a graph using LangGraph.

This script demonstrates how to orchestrate the stateless Claimify components
(Selection, Disambiguation, Decomposition) using LangGraph. This provides
a powerful, visual, and robust way to manage the pipeline's control flow.

Usage:
    - Ensure you have a compatible LLM server running (e.g., Ollama with gemma).
    - Run the script: python examples/langgraph_runtime.py
"""
import dspy
import os
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


# --- DSPy LLM Wrapper ---
# LangGraph's `invoke` method for LLMs is slightly different from DSPy's.
# This wrapper makes a DSPy LLM compatible with the LLMInterface protocol
# that LangGraph expects, by forwarding the call to dspy.Predict.
class DSPyLLMWrapper:
    def __init__(self, llm):
        self.llm = llm

    def __call__(self, messages, **kwargs):
        # Langchain ecosystem uses 'messages' and expects a string back.
        # We'll take the last message content as the prompt for DSPy.
        prompt = messages[-1].content
        # Use a simple dspy.Predict to get the completion
        predictor = dspy.Predict(dspy.Signature("prompt -> response", "Receives a prompt and returns a string response."))
        result = predictor(prompt=prompt)
        return result.response

# --- Graph State ---
# We use the existing ClaimifyState as the state object for the graph.
# The graph will modify this state as it passes through each node.

# --- Graph Nodes ---
# Each node in the graph is a function that takes the current state
# and returns a modified version of it.

def selection_node(state: ClaimifyState) -> ClaimifyState:
    """
    Runs the SelectionComponent to decide if the sentence is worth processing.
    """
    print("--- Running Selection Node ---")
    selection_component = SelectionComponent(llm=dspy.settings.lm)
    updated_state = selection_component(state)
    print(f"Selection result: {updated_state.was_selected}")
    return updated_state

def disambiguation_node(state: ClaimifyState) -> ClaimifyState:
    """
    Runs the DisambiguationComponent to clarify the sentence.
    """
    print("--- Running Disambiguation Node ---")
    disambiguation_component = DisambiguationComponent(llm=dspy.settings.lm)
    updated_state = disambiguation_component(state)
    if updated_state.disambiguation_result:
        print(f"Disambiguated text: {updated_state.disambiguation_result.disambiguated_text}")
    return updated_state

def decomposition_node(state: ClaimifyState) -> ClaimifyState:
    """
    Runs the DecompositionComponent to extract atomic claims.
    """
    print("--- Running Decomposition Node ---")
    decomposition_component = DecompositionComponent(llm=dspy.settings.lm)
    updated_state = decomposition_component(state)
    if updated_state.final_claims:
        print(f"Extracted {len(updated_state.final_claims)} claims.")
    return updated_state

# --- Conditional Edge ---
# This function determines the next step after the selection node.

def should_continue(state: ClaimifyState) -> Literal["continue", "end"]:
    """
    Checks the 'was_selected' flag in the state.
    - If True, the graph continues to the disambiguation step.
    - If False, the graph ends.
    """
    print(f"--- Checking Condition: Was sentence selected? -> {state.was_selected} ---")
    if state.was_selected:
        return "continue"
    else:
        return "end"

# --- Graph Assembly ---
def main():
    """
    Main function to assemble and run the LangGraph pipeline.
    """
    # 1. Configure DSPy LM
    # This example uses a local LLM server (e.g., Ollama with gemma).
    # Replace with your desired DSPy configuration.
    try:
        llm = dspy.OllamaLocal(model="gemma:2b", model_type="text")
        dspy.configure(lm=llm)
    except Exception as e:
        print(f"Failed to configure DSPy LLM. Make sure your LLM server is running. Error: {e}")
        return

    # 2. Define the Graph
    workflow = StateGraph(ClaimifyState)

    # Add nodes
    workflow.add_node("selection", selection_node)
    workflow.add_node("disambiguation", disambiguation_node)
    workflow.add_node("decomposition", decomposition_node)

    # Define the workflow edges
    workflow.set_entry_point("selection")
    workflow.add_conditional_edges(
        "selection",
        should_continue,
        {
            "continue": "disambiguation",
            "end": END,
        },
    )
    workflow.add_edge("disambiguation", "decomposition")
    workflow.add_edge("decomposition", END)

    # Compile the graph
    app = workflow.compile()

    # 3. Create Initial State and Run the Graph
    # This is the same setup as in the stateless_components.py example.
    sentence = SentenceChunk(
        text="The system, after processing, it returned error code 500.",
        source_id="doc_abc",
        chunk_id="chunk_123",
        sentence_index=0,
    )
    context = ClaimifyContext(
        current_sentence=sentence,
        preceding_sentences=[
            SentenceChunk(text="User authenticated successfully.", source_id="doc_abc", chunk_id="chunk_122", sentence_index=0)
        ],
        following_sentences=[],
    )
    initial_state = ClaimifyState(context=context)

    print("\n--- Invoking LangGraph Claimify Pipeline ---")
    final_state = app.invoke(initial_state)
    print("\n--- Pipeline Finished ---")

    # 4. Display Results
    print("\n--- Final State ---")
    if final_state.final_claims:
        print("Extracted Claims:")
        for claim in final_state.final_claims:
            print(f"- {claim.text}")
    else:
        print("No claims were extracted.")

    if final_state.final_sentences:
        print("\nProcessed Sentences (Nodes):")
        for sent_node in final_state.final_sentences:
            print(f"- {sent_node.text}")

if __name__ == "__main__":
    main()