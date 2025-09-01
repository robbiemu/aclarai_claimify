"""Simple verification script to test component functionality."""

import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def test_component_imports():
    """Test that components can be imported."""
    try:
        from aclarai_claimify.components.state import ClaimifyState
        from aclarai_claimify.components.selection import SelectionComponent
        from aclarai_claimify.components.disambiguation import DisambiguationComponent
        from aclarai_claimify.components.decomposition import DecompositionComponent
        from aclarai_claimify.components.example import process_sentence_with_components, process_sentences_with_components
        
        print("✓ All components imported successfully")
        return True
    except Exception as e:
        print(f"✗ Failed to import components: {e}")
        return False

def test_main_package_imports():
    """Test that components are available through main package."""
    try:
        from aclarai_claimify import (
            ClaimifyState,
            SelectionComponent,
            DisambiguationComponent,
            DecompositionComponent,
            process_sentence_with_components,
            process_sentences_with_components,
        )
        
        print("✓ All components accessible through main package")
        return True
    except Exception as e:
        print(f"✗ Failed to import from main package: {e}")
        return False

if __name__ == "__main__":
    print("Testing component imports...")
    success1 = test_component_imports()
    success2 = test_main_package_imports()
    
    if success1 and success2:
        print("\n✓ All import tests passed!")
    else:
        print("\n✗ Some import tests failed!")
        sys.exit(1)