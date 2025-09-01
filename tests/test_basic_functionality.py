"""
Simple test to verify that the decoupled claimify package works correctly.
"""

import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from aclarai_claimify.data_models import ClaimifyConfig, ClaimCandidate
from aclarai_claimify.outputs import ClaimInput, SentenceInput
from aclarai_claimify.config import load_claimify_config
from aclarai_claimify.integration import ClaimifyGraphIntegration

def test_basic_functionality():
    """Test basic functionality of the decoupled claimify package."""
    print("Testing basic functionality...")
    
    # Test config loading
    config = load_claimify_config()
    print(f"Loaded config: {config}")
    
    # Test data models
    candidate = ClaimCandidate(
        text="Test claim",
        is_atomic=True,
        is_self_contained=True,
        is_verifiable=True,
        confidence=0.9
    )
    print(f"Created claim candidate: {candidate}")
    
    # Test outputs
    claim_input = ClaimInput(
        text="Test claim",
        block_id="test_block"
    )
    print(f"Created claim input: {claim_input}")
    
    sentence_input = SentenceInput(
        text="Test sentence",
        block_id="test_block"
    )
    print(f"Created sentence input: {sentence_input}")
    
    # Test integration
    integration = ClaimifyGraphIntegration()
    print(f"Created integration: {integration}")
    
    print("All basic functionality tests passed!")

if __name__ == "__main__":
    test_basic_functionality()