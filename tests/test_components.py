"""Tests for the stateless components in the Claimify pipeline."""

import pytest
from unittest.mock import Mock

from aclarai_claimify.components.state import ClaimifyState
from aclarai_claimify.components.selection import SelectionComponent
from aclarai_claimify.components.disambiguation import DisambiguationComponent
from aclarai_claimify.components.decomposition import DecompositionComponent
from aclarai_claimify.data_models import (
    ClaimifyContext,
    SentenceChunk,
    SelectionResult,
    DisambiguationResult,
    DecompositionResult,
    ClaimCandidate,
)


class MockLLM:
    """Mock LLM for testing components."""

    def __init__(self, response_text=""):
        self.response_text = response_text

    def complete(self, prompt, **kwargs):
        return self.response_text


def test_selection_component():
    """Test the SelectionComponent with a mock LLM."""
    # Create mock LLM with a valid JSON response
    mock_response = '{"selected": true, "confidence": 0.9, "reasoning": "Test reasoning"}'
    mock_llm = MockLLM(mock_response)
    
    # Create component
    component = SelectionComponent(llm=mock_llm)
    
    # Create test state
    sentence = SentenceChunk(
        text="The system returned error code 500.",
        source_id="test_source",
        chunk_id="test_chunk",
        sentence_index=0,
    )
    context = ClaimifyContext(current_sentence=sentence)
    state = ClaimifyState(context=context)
    
    # Process state
    result_state = component(state)
    
    # Verify results
    assert result_state.selection_result is not None
    assert result_state.selection_result.is_selected is True
    assert result_state.selection_result.confidence == 0.9
    assert result_state.selection_result.reasoning == "Test reasoning"


def test_disambiguation_component():
    """Test the DisambiguationComponent with a mock LLM."""
    # Create mock LLM with a valid JSON response
    mock_response = '''{
        "disambiguated_text": "The system failed.",
        "changes_made": ["Added subject"],
        "confidence": 0.8
    }'''
    mock_llm = MockLLM(mock_response)
    
    # Create component
    component = DisambiguationComponent(llm=mock_llm)
    
    # Create test state with a selection result
    sentence = SentenceChunk(
        text="It failed.",
        source_id="test_source",
        chunk_id="test_chunk",
        sentence_index=0,
    )
    context = ClaimifyContext(current_sentence=sentence)
    state = ClaimifyState(context=context)
    
    # Add a selection result to simulate a selected sentence
    state.selection_result = SelectionResult(
        sentence_chunk=sentence,
        is_selected=True,
        confidence=0.9,
        reasoning="Test reasoning",
    )
    
    # Process state
    result_state = component(state)
    
    # Verify results
    assert result_state.disambiguation_result is not None
    assert result_state.disambiguation_result.disambiguated_text == "The system failed."
    assert result_state.disambiguation_result.changes_made == ["Added subject"]
    assert result_state.disambiguation_result.confidence == 0.8


def test_decomposition_component():
    """Test the DecompositionComponent with a mock LLM."""
    # Create mock LLM with a valid JSON response
    mock_response = '''{
        "claim_candidates": [
            {
                "text": "The system failed.",
                "is_atomic": true,
                "is_self_contained": true,
                "is_verifiable": true,
                "passes_criteria": true,
                "confidence": 0.9,
                "reasoning": "Valid claim",
                "node_type": "Claim"
            }
        ]
    }'''
    mock_llm = MockLLM(mock_response)
    
    # Create component
    component = DecompositionComponent(llm=mock_llm)
    
    # Create test state with selection and disambiguation results
    sentence = SentenceChunk(
        text="It failed.",
        source_id="test_source",
        chunk_id="test_chunk",
        sentence_index=0,
    )
    context = ClaimifyContext(current_sentence=sentence)
    state = ClaimifyState(context=context)
    
    # Add a selection result
    state.selection_result = SelectionResult(
        sentence_chunk=sentence,
        is_selected=True,
        confidence=0.9,
        reasoning="Test reasoning",
    )
    
    # Add a disambiguation result
    state.disambiguation_result = DisambiguationResult(
        original_sentence=sentence,
        disambiguated_text="The system failed.",
        changes_made=["Added subject"],
        confidence=0.8,
    )
    
    # Process state
    result_state = component(state)
    
    # Verify results
    assert result_state.decomposition_result is not None
    assert len(result_state.decomposition_result.claim_candidates) == 1
    candidate = result_state.decomposition_result.claim_candidates[0]
    assert candidate.text == "The system failed."
    assert candidate.is_atomic is True
    assert candidate.is_self_contained is True
    assert candidate.is_verifiable is True
    assert candidate.confidence == 0.9
    assert candidate.reasoning == "Valid claim"


def test_component_chaining():
    """Test chaining components together."""
    # Create mock LLMs with valid JSON responses
    selection_response = '{"selected": true, "confidence": 0.9, "reasoning": "Test reasoning"}'
    disambiguation_response = '''{
        "disambiguated_text": "The system failed.",
        "changes_made": ["Added subject"],
        "confidence": 0.8
    }'''
    decomposition_response = '''{
        "claim_candidates": [
            {
                "text": "The system failed.",
                "is_atomic": true,
                "is_self_contained": true,
                "is_verifiable": true,
                "passes_criteria": true,
                "confidence": 0.9,
                "reasoning": "Valid claim",
                "node_type": "Claim"
            }
        ]
    }'''
    
    selection_llm = MockLLM(selection_response)
    disambiguation_llm = MockLLM(disambiguation_response)
    decomposition_llm = MockLLM(decomposition_response)
    
    # Create components
    selection_component = SelectionComponent(llm=selection_llm)
    disambiguation_component = DisambiguationComponent(llm=disambiguation_llm)
    decomposition_component = DecompositionComponent(llm=decomposition_llm)
    
    # Create test state
    sentence = SentenceChunk(
        text="It failed.",
        source_id="test_source",
        chunk_id="test_chunk",
        sentence_index=0,
    )
    context = ClaimifyContext(current_sentence=sentence)
    state = ClaimifyState(context=context)
    
    # Process through components
    state = selection_component(state)
    state = disambiguation_component(state)
    state = decomposition_component(state)
    
    # Verify final state
    assert state.was_selected is True
    assert state.disambiguation_result is not None
    assert state.decomposition_result is not None
    assert len(state.final_claims) == 1
    assert state.final_claims[0].text == "The system failed."