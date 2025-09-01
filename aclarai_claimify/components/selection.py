"""Selection component for the Claimify pipeline.

This module contains the SelectionComponent class that implements the first stage
of the Claimify pipeline as a stateless component.
"""

import time
from typing import Optional, Protocol

from ..data_models import SelectionResult
from ..llm_schemas import SelectionResponse
from ..components.state import ClaimifyState
from ..data_models import ClaimifyConfig


class LLMInterface(Protocol):
    """Protocol defining the interface for LLM interactions."""

    def complete(self, prompt: str, **kwargs) -> str:
        """Generate a completion for the given prompt."""
        ...


class SelectionComponent:
    """Component responsible for identifying sentence chunks that contain verifiable information.

    This is the first stage of the Claimify pipeline implemented as a stateless component.
    It accepts a ClaimifyState object and returns a modified ClaimifyState object.
    """

    def __init__(
        self,
        llm: Optional[LLMInterface] = None,
        config: Optional[ClaimifyConfig] = None,
    ):
        self.llm = llm
        self.config = config or ClaimifyConfig()

    def __call__(self, state: ClaimifyState) -> ClaimifyState:
        """Process a sentence to determine if it should be selected for further processing.

        Args:
            state: ClaimifyState containing the context to process

        Returns:
            Modified ClaimifyState with selection result
        """
        start_time = time.time()
        sentence = state.context.current_sentence
        
        try:
            # LLM is required for selection processing
            if self.llm is None:
                raise ValueError("LLM is required for Selection component processing")
            
            result = self._llm_selection(state.context)
            processing_time = time.time() - start_time
            
            # Update state with result
            new_state = state.model_copy()
            new_state.selection_result = result
            new_state.selection_result.processing_time = processing_time
            return new_state
            
        except Exception as e:
            processing_time = time.time() - start_time
            error_msg = f"Error in selection processing: {e}"
            new_state = state.model_copy()
            new_state.errors.append(error_msg)
            
            # Create a default selection result for error handling
            default_result = SelectionResult(
                sentence_chunk=sentence,
                is_selected=False,
                reasoning=f"Error during processing: {e}",
                processing_time=processing_time,
            )
            new_state.selection_result = default_result
            return new_state

    def _llm_selection(self, context) -> SelectionResult:
        """LLM-based selection following the Claimify approach.

        This implements Stage 1: Selection from the Claimify pipeline, which:
        1. Identifies if the sentence contains verifiable factual information
        2. Uses JSON output format as specified in claimify_selection.yaml prompt
        """
        assert self.llm is not None, "LLM must be initialized for this method"

        sentence = context.current_sentence
        context_text = self._build_context_text(context)
        
        # Use JSON prompt format matching claimify_selection.yaml
        prompt = f"""You are an expert at identifying verifiable factual content in text. Your task is to determine whether a given sentence contains information that could be extracted as verifiable claims.
Analyze the following sentence within its context to determine if it contains verifiable, factual information.
Context (surrounding sentences):
{context_text}
Target sentence: "{sentence.text}"
Consider these criteria:
1. Does this sentence contain factual, verifiable information?
2. Is it a statement (not a question, command, or exclamation)?
3. Could this information be fact-checked or validated?
4. Does it describe events, relationships, measurements, or properties?
5. Is it specific enough to be meaningful?
Sentences to REJECT:
- Questions ("What should we do?")
- Commands ("Please fix this.")
- Opinions without factual basis ("I think it's bad.")
- Vague statements ("Something happened.")
- Very short fragments ("Yes.", "OK.")
Sentences to SELECT:
- Technical facts ("The system returned error code 500.")
- Event descriptions ("The deployment occurred at 10:30 AM.")
- Measurements ("The response time was 2.3 seconds.")
- Relationships ("User A reported the bug to Team B.")
- Specific observations ("The CPU usage spiked to 95%.")
Respond with valid JSON only:
{{
  "selected": true/false,
  "confidence": 0.0-1.0,
  "reasoning": "Brief explanation of decision"
}}"""
        
        try:
            # Call the LLM with the prompt
            response = self.llm.complete(
                prompt,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens or 500,
            ).strip()
            
            # Parse JSON response using Pydantic model
            try:
                result_data = SelectionResponse.model_validate_json(response)
                is_selected = result_data.selected
                confidence = result_data.confidence
                reasoning = result_data.reasoning
                
                # Apply confidence threshold - if LLM confidence is below threshold, reject selection
                if (
                    is_selected
                    and confidence < self.config.selection_confidence_threshold
                ):
                    is_selected = False
                    reasoning = f"LLM selected but confidence {confidence:.2f} below threshold {self.config.selection_confidence_threshold:.2f}"
                    
                return SelectionResult(
                    sentence_chunk=sentence,
                    is_selected=is_selected,
                    reasoning=reasoning,
                    confidence=confidence,
                    rewritten_text=sentence.text if is_selected else None,
                )
            except Exception as e:
                raise ValueError(f"Invalid JSON response from LLM: {e}") from e
        except Exception as e:
            # If LLM fails, we cannot perform selection without heuristics
            raise ValueError(
                f"LLM selection failed and no fallback available: {e}"
            ) from e

    def _build_context_text(self, context) -> str:
        """Build context text from surrounding sentences."""
        parts = []
        # Add preceding sentences
        for i, sent in enumerate(context.preceding_sentences):
            parts.append(f"[{-len(context.preceding_sentences) + i}] {sent.text}")
        # Add current sentence marker
        parts.append(f"[0] {context.current_sentence.text} ← TARGET")
        # Add following sentences
        for i, sent in enumerate(context.following_sentences):
            parts.append(f"[{i + 1}] {sent.text}")
        return "\n".join(parts)