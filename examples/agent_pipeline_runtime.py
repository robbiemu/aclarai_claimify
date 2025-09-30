"""
This script demonstrates how to run the original, agent-based Claimify pipeline.
It shows how to orchestrate the three core agents (Selection, Disambiguation, Decomposition)
in a sequential manner and integrate them with a modern LLM provider like DSPy.
"""
import logging
import os
from typing import Any, Dict, Optional

import dspy

from aclarai_claimify.agents import (
    DecompositionAgent,
    DisambiguationAgent,
    LLMInterface,
    SelectionAgent,
)
from aclarai_claimify.config import load_claimify_config
from aclarai_claimify.data_models import ClaimifyContext, SentenceChunk

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class DSPyLLM(LLMInterface):
    """
    A wrapper class to make a DSPy language model compatible with the
    Claimify pipeline's LLMInterface protocol.
    """

    def __init__(self, llm: dspy.dsp.LM):
        """
        Initializes the wrapper with a DSPy language model.
        Args:
            llm: An instance of a DSPy language model (e.g., dspy.Ollama).
        """
        self.llm = llm

    def complete(self, prompt: str, **kwargs: Any) -> str:
        """
        Generates a completion for the given prompt using the DSPy LLM.
        Args:
            prompt: The input prompt for the language model.
            **kwargs: Additional parameters for the LLM call (e.g., temperature).
        Returns:
            The raw string response from the language model.
        """
        # Note: DSPy's __call__ method is a straightforward way to get a completion.
        # It may not support all kwargs directly, depending on the underlying LM.
        # For this example, we assume basic completion is sufficient.
        response = self.llm(prompt, **kwargs)
        # dspy.Ollama returns a list with one string
        return response[0] if isinstance(response, list) and response else ""


def main():
    """
    Main function to set up and run the agent-based Claimify pipeline.
    """
    # 1. Configure DSPy with an LLM provider
    # This example uses Ollama with the 'gemma' model.
    # Ensure you have a running Ollama server with the 'gemma' model pulled.
    # You can change the model and base_url as needed.
    llm = dspy.Ollama(model="gemma", base_url=os.getenv("OLLAMA_BASE_URL"))
    dspy.settings.configure(lm=llm)

    # 2. Create the LLM wrapper for the Claimify agents
    dspy_llm_wrapper = DSPyLLM(llm)

    # 3. Load the Claimify configuration
    # This loads the default configuration from the package.
    config = load_claimify_config()

    # 4. Initialize the three core agents
    selection_agent = SelectionAgent(llm=dspy_llm_wrapper, config=config)
    disambiguation_agent = DisambiguationAgent(llm=dspy_llm_wrapper, config=config)
    decomposition_agent = DecompositionAgent(llm=dspy_llm_wrapper, config=config)

    # 5. Define the input text and create the initial context
    # This example text contains a mix of verifiable claims and other sentences.
    input_text = (
        "The system failed to connect to the database. This was due to an "
        "invalid credential configuration. The error code was 500. "
        "I think this is a critical issue. We should fix it immediately."
    )

    # For this example, we process one sentence at a time.
    # Here, we focus on the second sentence, providing the first as context.
    sentence_to_process = "This was due to an invalid credential configuration."
    context = ClaimifyContext(
        current_sentence=SentenceChunk(text=sentence_to_process, start_char=42, end_char=95),
        preceding_sentences=[
            SentenceChunk(text="The system failed to connect to the database.", start_char=0, end_char=41)
        ],
        following_sentences=[
            SentenceChunk(text="The error code was 500.", start_char=96, end_char=119),
            SentenceChunk(text="I think this is a critical issue.", start_char=120, end_char=153),
            SentenceChunk(text="We should fix it immediately.", start_char=154, end_char=182),
        ],
    )

    logging.info(f"Processing sentence: '{context.current_sentence.text}'")

    # 6. Run the sequential agent pipeline
    # Stage 1: Selection
    selection_result = selection_agent.process(context)
    logging.info(f"Selection result: {selection_result.is_selected}, Reason: {selection_result.reasoning}")

    if selection_result.is_selected:
        # Stage 2: Disambiguation
        disambiguation_result = disambiguation_agent.process(
            sentence=context.current_sentence, context=context
        )
        logging.info(f"Disambiguated text: '{disambiguation_result.disambiguated_text}'")
        logging.info(f"Changes made: {disambiguation_result.changes_made}")

        # Stage 3: Decomposition
        decomposition_result = decomposition_agent.process(
            disambiguated_text=disambiguation_result.disambiguated_text,
            _original_sentence=context.current_sentence,
        )
        logging.info(f"Decomposition found {len(decomposition_result.valid_claims)} valid claims.")

        # 7. Print the final claims
        if decomposition_result.valid_claims:
            print("\n--- Extracted Claims ---")
            for i, claim in enumerate(decomposition_result.valid_claims, 1):
                print(f"{i}. {claim.text}")
            print("----------------------")
        else:
            print("\nNo valid claims were extracted.")
    else:
        print("\nSentence was not selected for claim extraction.")


if __name__ == "__main__":
    main()