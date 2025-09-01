"""Example of using stateless components in the Claimify pipeline.

This module demonstrates how to use the stateless components as an alternative
to the monolithic pipeline approach.
"""

from typing import List, Optional

from ..components.state import ClaimifyState
from ..components.selection import SelectionComponent
from ..components.disambiguation import DisambiguationComponent
from ..components.decomposition import DecompositionComponent
from ..data_models import (
    ClaimifyContext,
    SentenceChunk,
    ClaimifyConfig,
)


def process_sentence_with_components(
    context: ClaimifyContext,
    selection_component: SelectionComponent,
    disambiguation_component: DisambiguationComponent,
    decomposition_component: DecompositionComponent,
) -> ClaimifyState:
    """Process a single sentence through the component-based pipeline.

    Args:
        context: ClaimifyContext with the sentence and surrounding context
        selection_component: Component for the selection stage
        disambiguation_component: Component for the disambiguation stage
        decomposition_component: Component for the decomposition stage

    Returns:
        Final ClaimifyState with all processing results
    """
    # Initialize state with context
    state = ClaimifyState(context=context)
    
    # Stage 1: Selection
    state = selection_component(state)
    
    # If not selected, stop processing
    if not state.was_selected:
        return state
    
    # Stage 2: Disambiguation
    state = disambiguation_component(state)
    
    # Stage 3: Decomposition
    state = decomposition_component(state)
    
    return state


def process_sentences_with_components(
    sentences: List[SentenceChunk],
    config: Optional[ClaimifyConfig] = None,
    selection_llm=None,
    disambiguation_llm=None,
    decomposition_llm=None,
) -> List[ClaimifyState]:
    """Process a list of sentence chunks through the component-based pipeline.

    Args:
        sentences: List of sentence chunks to process
        config: Pipeline configuration
        selection_llm: LLM instance for Selection stage
        disambiguation_llm: LLM instance for Disambiguation stage
        decomposition_llm: LLM instance for Decomposition stage

    Returns:
        List of ClaimifyState objects with processing results
    """
    if not sentences:
        return []
    
    config = config or ClaimifyConfig()
    
    # Initialize components with their respective LLMs
    selection_component = SelectionComponent(llm=selection_llm, config=config)
    disambiguation_component = DisambiguationComponent(llm=disambiguation_llm, config=config)
    decomposition_component = DecompositionComponent(llm=decomposition_llm, config=config)
    
    results = []
    for i, sentence in enumerate(sentences):
        # Build context window for this sentence
        context = _build_context_window(sentence, sentences, i, config)
        
        # Process the sentence through the components
        result = process_sentence_with_components(
            context,
            selection_component,
            disambiguation_component,
            decomposition_component,
        )
        results.append(result)
    
    return results


def _build_context_window(
    current_sentence: SentenceChunk,
    all_sentences: List[SentenceChunk],
    current_index: int,
    config: ClaimifyConfig,
) -> ClaimifyContext:
    """Build context window with preceding and following sentences.

    Args:
        current_sentence: The sentence to process
        all_sentences: All available sentences
        current_index: Index of current sentence in the list
        config: Configuration with context window settings

    Returns:
        ClaimifyContext with context window
    """
    # Get preceding sentences (p)
    start_p = max(0, current_index - config.context_window_p)
    preceding = all_sentences[start_p:current_index]
    
    # Get following sentences (f)
    end_f = min(len(all_sentences), current_index + 1 + config.context_window_f)
    following = all_sentences[current_index + 1 : end_f]
    
    context = ClaimifyContext(
        current_sentence=current_sentence,
        preceding_sentences=preceding,
        following_sentences=following,
    )
    
    return context