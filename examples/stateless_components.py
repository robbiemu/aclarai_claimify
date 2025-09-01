"""
Example usage of the stateless components in the Claimify pipeline.

This script demonstrates how to use the new stateless components
as an alternative to the monolithic pipeline approach.
"""

import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from aclarai_claimify.components.state import ClaimifyState
from aclarai_claimify.components.selection import SelectionComponent
from aclarai_claimify.components.disambiguation import DisambiguationComponent
from aclarai_claimify.components.decomposition import DecompositionComponent
from aclarai_claimify.data_models import (
    ClaimifyContext,
    SentenceChunk,
    ClaimifyConfig,
)


class MockLLM:
    """Mock LLM for demonstration purposes."""
    
    def __init__(self, name="MockLLM"):
        self.name = name

    def complete(self, prompt, **kwargs):
        """Mock completion method that returns appropriate JSON responses."""
        if "You are an expert at identifying verifiable factual content" in prompt:
            # Selection response
            return '{"selected": true, "confidence": 0.9, "reasoning": "Contains verifiable technical information"}'
        elif "You are an expert at disambiguating text" in prompt:
            # Disambiguation response
            return '''{
                "disambiguated_text": "The authentication system returned error code 500.",
                "changes_made": ["Replaced 'it' with 'the authentication system'"],
                "confidence": 0.85
            }'''
        elif "You are an expert at extracting atomic claims" in prompt:
            # Decomposition response
            return '''{
                "claim_candidates": [
                    {
                        "text": "The authentication system returned error code 500.",
                        "is_atomic": true,
                        "is_self_contained": true,
                        "is_verifiable": true,
                        "passes_criteria": true,
                        "confidence": 0.95,
                        "reasoning": "Single verifiable fact about system behavior",
                        "node_type": "Claim"
                    }
                ]
            }'''
        else:
            return '{"text": "Default response", "confidence": 0.5}'


def main():
    """Demonstrate usage of stateless components."""
    print("Claimify Stateless Components Example")
    print("=" * 40)
    
    # Create mock LLMs
    selection_llm = MockLLM("SelectionLLM")
    disambiguation_llm = MockLLM("DisambiguationLLM")
    decomposition_llm = MockLLM("DecompositionLLM")
    
    # Create components
    selection_component = SelectionComponent(llm=selection_llm)
    disambiguation_component = DisambiguationComponent(llm=disambiguation_llm)
    decomposition_component = DecompositionComponent(llm=decomposition_llm)
    
    # Create test data
    sentence = SentenceChunk(
        text="It returned error code 500.",
        source_id="doc_123",
        chunk_id="chunk_456",
        sentence_index=0,
    )
    
    context = ClaimifyContext(
        current_sentence=sentence,
        preceding_sentences=[],
        following_sentences=[],
    )
    
    print(f"Original sentence: {sentence.text}")
    print()
    
    # Create initial state
    state = ClaimifyState(context=context)
    
    # Process through components
    print("1. Selection Stage:")
    state = selection_component(state)
    print(f"   Selected: {state.was_selected}")
    if state.selection_result:
        print(f"   Confidence: {state.selection_result.confidence}")
        print(f"   Reasoning: {state.selection_result.reasoning}")
    print()
    
    if state.was_selected:
        print("2. Disambiguation Stage:")
        state = disambiguation_component(state)
        if state.disambiguation_result:
            print(f"   Disambiguated text: {state.disambiguation_result.disambiguated_text}")
            print(f"   Changes made: {state.disambiguation_result.changes_made}")
            print(f"   Confidence: {state.disambiguation_result.confidence}")
        print()
        
        print("3. Decomposition Stage:")
        state = decomposition_component(state)
        if state.decomposition_result:
            print(f"   Found {len(state.decomposition_result.claim_candidates)} claim candidates")
            print(f"   Valid claims: {len(state.final_claims)}")
            print(f"   Sentence nodes: {len(state.final_sentences)}")
            if state.final_claims:
                print("   Claims extracted:")
                for i, claim in enumerate(state.final_claims, 1):
                    print(f"     {i}. {claim.text}")
    
    print()
    print("Processing complete!")


if __name__ == "__main__":
    main()