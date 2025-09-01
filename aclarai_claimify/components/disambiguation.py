"""Disambiguation component for the Claimify pipeline.

This module contains the DisambiguationComponent class that implements the second stage
of the Claimify pipeline as a stateless component.
"""

import time
from typing import Optional, Protocol

from ..data_models import DisambiguationResult, SentenceChunk
from ..llm_schemas import DisambiguationResponse
from ..components.state import ClaimifyState
from ..data_models import ClaimifyConfig


class LLMInterface(Protocol):
    """Protocol defining the interface for LLM interactions."""

    def complete(self, prompt: str, **kwargs) -> str:
        """Generate a completion for the given prompt."""
        ...


class DisambiguationComponent:
    """Component responsible for rewriting sentences to remove ambiguities and add context.

    This is the second stage of the Claimify pipeline implemented as a stateless component.
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
        """Process a selected sentence to remove ambiguities and add inferred subjects.

        Args:
            state: ClaimifyState containing the context to process

        Returns:
            Modified ClaimifyState with disambiguation result
        """
        # Verify that the sentence was selected for processing
        if not state.was_selected:
            # If not selected, don't process and return state unchanged
            return state
            
        start_time = time.time()
        sentence = state.current_sentence
        
        try:
            # LLM is required for disambiguation processing
            if self.llm is None:
                raise ValueError("LLM is required for Disambiguation component processing")
                
            result = self._llm_disambiguation(sentence, state.context)
            processing_time = time.time() - start_time
            result.processing_time = processing_time
            
            # Update state with result
            new_state = state.model_copy()
            new_state.disambiguation_result = result
            return new_state
            
        except Exception as e:
            processing_time = time.time() - start_time
            error_msg = f"Error in disambiguation processing: {e}"
            new_state = state.model_copy()
            new_state.errors.append(error_msg)
            
            # Create a default disambiguation result for error handling
            default_result = DisambiguationResult(
                original_sentence=sentence,
                disambiguated_text=sentence.text,  # Return original on error
                changes_made=[f"Error during processing: {e}"],
                processing_time=processing_time,
            )
            new_state.disambiguation_result = default_result
            return new_state

    def _llm_disambiguation(
        self, sentence: SentenceChunk, context
    ) -> DisambiguationResult:
        """LLM-based disambiguation following the Claimify approach.

        This implements Stage 2: Disambiguation from the Claimify pipeline, which:
        1. Identifies ambiguities (pronouns, time references, structural ambiguities)
        2. Uses context to resolve ambiguities confidently
        3. Uses JSON output format as specified in claimify_disambiguation.yaml prompt
        """
        assert self.llm is not None, "LLM must be initialized for this method"

        context_text = self._build_context_text(context)
        # Use JSON prompt format matching claimify_disambiguation.yaml
        prompt = f"""You are an expert at disambiguating text by resolving pronouns, adding missing context, and making implicit information explicit. Your goal is to rewrite sentences to be clear and self-contained while preserving their original meaning.
Rewrite the following sentence to remove ambiguities and make it self-contained. Use the surrounding context to resolve pronouns and add missing subjects or objects.
Context (surrounding sentences):
{context_text}
Target sentence to disambiguate: "{sentence.text}"
Disambiguation guidelines:
1. Replace ambiguous pronouns (it, this, that, they) with specific entities
2. Add missing subjects for sentences starting with verbs
3. Clarify vague references ("the error", "the issue", "the problem")
4. Make temporal and causal relationships explicit
5. Preserve the original meaning and factual content
6. Keep the sentence concise but complete
Examples:
- "It failed." → "[The system] failed."
- "This caused problems." → "This [configuration change] caused problems."
- "Reported an error." → "[The application] reported an error."
- "The error occurred when..." → "The [authentication] error occurred when..."
Respond with valid JSON only:
{{
  "disambiguated_text": "The rewritten sentence",
  "changes_made": ["List of specific changes"],
  "confidence": 0.0-1.0
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
                result_data = DisambiguationResponse.model_validate_json(response)
                disambiguated_text = result_data.disambiguated_text
                changes_made = result_data.changes_made
                confidence = result_data.confidence
                
                # Apply confidence threshold - if LLM confidence is below threshold, use original text
                if confidence < self.config.disambiguation_confidence_threshold:
                    disambiguated_text = sentence.text
                    changes_made = [
                        f"LLM confidence {confidence:.2f} below threshold {self.config.disambiguation_confidence_threshold:.2f}, using original text"
                    ]
                    
                return DisambiguationResult(
                    original_sentence=sentence,
                    disambiguated_text=disambiguated_text,
                    changes_made=changes_made,
                    confidence=confidence,
                )
            except Exception as e:
                raise ValueError(f"Invalid JSON response from LLM: {e}") from e
        except Exception as e:
            # If LLM fails, we cannot perform disambiguation without heuristics
            raise ValueError(
                f"LLM disambiguation failed and no fallback available: {e}"
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