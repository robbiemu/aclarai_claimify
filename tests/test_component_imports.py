"""Test that components can be imported correctly."""

import pytest

def test_component_imports():
    """Test that all components can be imported."""
    try:
        from aclarai_claimify.components.state import ClaimifyState
        from aclarai_claimify.components.selection import SelectionComponent
        from aclarai_claimify.components.disambiguation import DisambiguationComponent
        from aclarai_claimify.components.decomposition import DecompositionComponent
        from aclarai_claimify.components.example import process_sentence_with_components, process_sentences_with_components
    except ImportError as e:
        pytest.fail(f"Failed to import components: {e}")

def test_main_imports():
    """Test that components are exposed through main package."""
    try:
        from aclarai_claimify import (
            ClaimifyState,
            SelectionComponent,
            DisambiguationComponent,
            DecompositionComponent,
            process_sentence_with_components,
            process_sentences_with_components,
        )
    except ImportError as e:
        pytest.fail(f"Failed to import from main package: {e}")