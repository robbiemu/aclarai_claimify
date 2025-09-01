"""
Simple test to verify that the integration conversion methods work correctly.
"""

import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from aclarai_claimify.data_models import (
    ClaimCandidate,
    ClaimifyContext,
    ClaimifyResult,
    DecompositionResult,
    SelectionResult,
    SentenceChunk,
)
from aclarai_claimify.integration import ClaimifyGraphIntegration
from aclarai_claimify.outputs import ClaimInput


def test_conversion_functionality():
    """Test the conversion functionality of the decoupled claimify integration."""
    print("Testing conversion functionality...")

    # Create test data
    test_chunk = SentenceChunk(
        text="The system failed at startup.",
        source_id="blk_001",
        chunk_id="chunk_001",
        sentence_index=0,
    )

    # Test integration
    integration = ClaimifyGraphIntegration()
    print(f"Created integration: {integration}")

    # Create a valid claim candidate
    valid_claim = ClaimCandidate(
        text="The system failed at startup.",
        is_atomic=True,
        is_self_contained=True,
        is_verifiable=True,
        confidence=0.95,
    )

    # Create decomposition result with valid claim
    decomposition_result = DecompositionResult(
        original_text="The system failed at startup.",
        claim_candidates=[valid_claim],
    )

    # Create a successful claimify result
    result = ClaimifyResult(
        original_chunk=test_chunk,
        context=ClaimifyContext(current_sentence=test_chunk),
        selection_result=SelectionResult(sentence_chunk=test_chunk, is_selected=True),
        decomposition_result=decomposition_result,
    )

    # Convert to inputs (this is the core functionality we want to test)
    claim_inputs, sentence_inputs = integration._convert_result_to_inputs(result)

    # Check claim properties
    assert len(claim_inputs) == 1
    claim = claim_inputs[0]
    assert isinstance(claim, ClaimInput)
    assert claim.text == "The system failed at startup."
    assert claim.block_id == "blk_001"
    assert claim.verifiable
    assert claim.self_contained
    assert claim.context_complete

    print("Conversion functionality test passed!")


if __name__ == "__main__":
    test_conversion_functionality()
