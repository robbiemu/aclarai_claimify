# Claimify Stateless Components Architecture

## Overview

This document describes the refactored architecture of the Claimify pipeline, which has been re-architected from a monolithic pipeline into a set of independent, stateless, and composable components.

## Motivation

The original monolithic pipeline architecture had several limitations:
1. Tight coupling between pipeline stages
2. Difficulty integrating with external orchestration frameworks
3. Limited reusability of individual stages
4. Challenges in testing and debugging individual components

The new stateless component architecture addresses these issues by:
1. Decoupling pipeline stages into independent components
2. Making components framework-agnostic for easy integration
3. Enabling composition and reuse of individual stages
4. Simplifying testing and debugging

## Architecture

### Core Concepts

1. **ClaimifyState**: A central data model that carries information between components
2. **Stateless Components**: Each component performs a single transformation on the state
3. **Composability**: Components can be chained together in any order
4. **Framework Agnostic**: Components can be used with any orchestration system

### Component Structure

Each component follows this pattern:
```python
class Component:
    def __init__(self, llm=None, config=None):
        self.llm = llm
        self.config = config or ClaimifyConfig()
    
    def __call__(self, state: ClaimifyState) -> ClaimifyState:
        # Perform transformation
        # Return modified state
        pass
```

### Data Flow

1. Initialize `ClaimifyState` with input context
2. Pass state through `SelectionComponent`
3. If selected, pass state through `DisambiguationComponent`
4. If disambiguated, pass state through `DecompositionComponent`
5. Extract final claims from the resulting state

## Components

### SelectionComponent

Responsible for identifying sentence chunks that contain verifiable information relevant for claim extraction.

**Input**: `ClaimifyState` with context
**Output**: `ClaimifyState` with selection result
**Key Properties**:
- Uses LLM to evaluate sentences against selection criteria
- Applies confidence threshold to filter low-quality selections
- Handles errors gracefully with informative messages

### DisambiguationComponent

Rewrites sentences to remove ambiguities and add context, making them self-contained.

**Input**: `ClaimifyState` with selection result
**Output**: `ClaimifyState` with disambiguation result
**Key Properties**:
- Only processes sentences that were selected
- Resolves pronouns and vague references using context
- Preserves original meaning while adding clarity
- Applies confidence threshold for quality control

### DecompositionComponent

Breaks sentences into atomic claims that meet the Claimify quality criteria.

**Input**: `ClaimifyState` with disambiguation result
**Output**: `ClaimifyState` with decomposition result
**Key Properties**:
- Only processes sentences that were disambiguated
- Extracts atomic, self-contained, verifiable claims
- Evaluates each candidate against quality criteria
- Separates valid claims from sentence nodes

## Usage Examples

### Basic Component Usage

```python
from aclarai_claimify import (
    ClaimifyState,
    SelectionComponent,
    DisambiguationComponent,
    DecompositionComponent
)

# Initialize components
selection = SelectionComponent(llm=your_llm)
disambiguation = DisambiguationComponent(llm=your_llm)
decomposition = DecompositionComponent(llm=your_llm)

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

### Batch Processing

```python
from aclarai_claimify import process_sentences_with_components

results = process_sentences_with_components(
    sentences=your_sentences,
    config=your_config,
    selection_llm=your_selection_llm,
    disambiguation_llm=your_disambiguation_llm,
    decomposition_llm=your_decomposition_llm,
)
```

## Integration with External Frameworks

The stateless component architecture makes it easy to integrate with external orchestration frameworks:

### LangGraph Integration

```python
# Example of how components could be used in LangGraph
from langgraph import StateGraph

graph = StateGraph(ClaimifyState)
graph.add_node("selection", SelectionComponent(llm=llm))
graph.add_node("disambiguation", DisambiguationComponent(llm=llm))
graph.add_node("decomposition", DecompositionComponent(llm=llm))

graph.add_edge("selection", "disambiguation")
graph.add_edge("disambiguation", "decomposition")
```

### LlamaIndex Integration

```python
# Example of how components could be used in LlamaIndex
from llama_index.core.workflow import Workflow

class ClaimifyWorkflow(Workflow):
    def __init__(self):
        super().__init__()
        self.selection = SelectionComponent(llm=llm)
        self.disambiguation = DisambiguationComponent(llm=llm)
        self.decomposition = DecompositionComponent(llm=llm)
    
    async def run(self, state: ClaimifyState):
        state = self.selection(state)
        if state.was_selected:
            state = self.disambiguation(state)
            state = self.decomposition(state)
        return state
```

## Benefits

1. **Framework Agnostic**: Components can be used with any orchestration framework
2. **Composable**: Easy to rearrange or extend the pipeline
3. **Testable**: Each component can be tested independently
4. **Maintainable**: Clear separation of concerns
5. **Scalable**: Components can be distributed across different services
6. **Flexible**: Easy to customize or replace individual components

## Migration from Legacy Pipeline

The legacy `ClaimifyPipeline` class is now deprecated but still available for backward compatibility. New implementations should use the stateless components directly.

To migrate:
1. Replace `ClaimifyPipeline` instantiation with individual component creation
2. Replace `pipeline.process_sentence()` calls with component chaining
3. Update error handling to work with the new state-based approach
4. Adjust any code that depends on the specific structure of `ClaimifyResult`

## Testing

Each component has comprehensive unit tests that verify:
1. Correct processing with valid inputs
2. Error handling with invalid inputs
3. Proper state management
4. Component chaining behavior

The tests can be run with:
```bash
uv run python -m pytest tests/test_components.py -v
```