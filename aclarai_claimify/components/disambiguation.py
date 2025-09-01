"""Disambiguation component for the Claimify pipeline.

This module contains the DisambiguationComponent class that implements the second stage
of the Claimify pipeline as a stateless component.
"""

import time
from typing import Optional, Protocol

import dspy
from pydantic import ValidationError

from ..data_models import DisambiguationResult, SentenceChunk
from ..llm_schemas import DisambiguationResponse
from ..components.state import ClaimifyState
from ..data_models import ClaimifyConfig
from ..config import load_prompt_template
from ..prompt_utils import format_prompt_with_schema
from ..signatures import DisambiguationSignature


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
        # Initialize DSPy module
        self.dspy_module = dspy.Predict(DisambiguationSignature)

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
                raise ValueError(
                    "LLM is required for Disambiguation component processing"
                )

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
        self,
        sentence: SentenceChunk,
        context
    ) -> DisambiguationResult:
        """LLM-based disambiguation following the Claimify approach.

        This implements Stage 2: Disambiguation from the Claimify pipeline, which:
        1. Identifies ambiguities (pronouns, time references, structural ambiguities)
        2. Uses context to resolve ambiguities confidently
        3. Uses JSON output format as specified in claimify_disambiguation.yaml prompt
        """
        assert self.llm is not None, "LLM must be initialized for this method"

        context_text = self._build_context_text(context)

        try:
            # Call the DSPy module
            response = self.dspy_module(
                context_text=context_text,
                target_sentence=sentence.text,
            )
            response = response.disambiguation_response_json.strip()

            # Parse JSON response using Pydantic model with proper error handling
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
            except ValidationError as e:
                # Log detailed validation error for debugging
                error_details = "\n".join([f"- {error}" for error in e.errors()])
                raise ValueError(
                    f"Invalid JSON response from LLM does not match expected schema:\n{error_details}"
                ) from e
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
