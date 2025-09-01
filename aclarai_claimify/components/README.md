# Claimify Components

This directory contains the refactored, stateless components of the Claimify pipeline. Each component implements a single stage of the claim extraction process and can be composed with others to build custom pipelines.

## Component Architecture

The components follow a stateless design pattern where each component:

1. Accepts a `ClaimifyState` object as input
2. Performs its specific transformation
3. Returns a modified `ClaimifyState` object as output
4. Maintains no internal state between calls

This design makes the components framework-agnostic and easily integrable with systems like LangGraph or LlamaIndex.

## Components

### SelectionComponent
Responsible for identifying sentence chunks that contain verifiable information relevant for claim extraction.

### DisambiguationComponent
Rewrites sentences to remove ambiguities and add context, making them self-contained.

### DecompositionComponent
Breaks sentences into atomic claims that meet the Claimify quality criteria.

## Usage

```python
from aclarai_claimify.components import (
    ClaimifyState,
    SelectionComponent,
    DisambiguationComponent,
    DecompositionComponent
)

# Initialize components with LLM instances
selection = SelectionComponent(llm=your_llm_instance)
disambiguation = DisambiguationComponent(llm=your_llm_instance)
decomposition = DecompositionComponent(llm=your_llm_instance)

# Create initial state
state = ClaimifyState(context=your_context)

# Process through components
state = selection(state)
if state.was_selected:
    state = disambiguation(state)
    state = decomposition(state)

# Access results
claims = state.final_claims
```

## Benefits

1. **Framework Agnostic**: Components can be used with any orchestration framework
2. **Composable**: Easy to rearrange or extend the pipeline
3. **Testable**: Each component can be tested independently
4. **Maintainable**: Clear separation of concerns