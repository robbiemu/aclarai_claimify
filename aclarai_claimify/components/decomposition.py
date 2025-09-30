"""Decomposition component for the Claimify pipeline.

This module contains the DecompositionComponent class that implements the third stage
of the Claimify pipeline as a stateless component.
"""

import time
from typing import Optional, Protocol

import dspy
from pydantic import ValidationError

from ..data_models import DecompositionResult
from ..llm_schemas import DecompositionResponse
from ..components.state import ClaimifyState
from ..data_models import ClaimifyConfig
from ..signatures import DecompositionSignature


class LLMInterface(Protocol):
    """Protocol defining the interface for LLM interactions."""

    def complete(self, prompt: str, **kwargs) -> str:
        """Generate a completion for the given prompt."""
        ... 


class DecompositionComponent:
    """Component responsible for breaking disambiguated sentences into atomic, self-contained claims.

    This is the third and final stage of the Claimify pipeline implemented as a stateless component.
    It accepts a ClaimifyState object and returns a modified ClaimifyState object.
    """

    def __init__(
        self,
        llm: Optional[LLMInterface] = None,
        config: Optional[ClaimifyConfig] = None,
        compiled_prompt_path: Optional[str] = None,
    ):
        self.llm = llm
        self.config = config or ClaimifyConfig()
        # Initialize DSPy module
        self.dspy_module = dspy.Predict(DecompositionSignature)
        if compiled_prompt_path:
            self.dspy_module.load(compiled_prompt_path)

    def __call__(self, state: ClaimifyState) -> ClaimifyState:
        """Process a disambiguated sentence to extract atomic claims.

        Args:
            state: ClaimifyState containing the context to process

        Returns:
            Modified ClaimifyState with decomposition result
        """
        # Verify that the sentence was selected and disambiguated for processing
        if not state.was_selected or state.disambiguation_result is None:
            # If not selected or disambiguated, don't process and return state unchanged
            return state

        start_time = time.time()
        disambiguated_text = state.disambiguation_result.disambiguated_text
        original_sentence = state.current_sentence

        try:
            # LLM is required for decomposition processing
            if self.llm is None:
                raise ValueError(
                    "LLM is required for Decomposition component processing"
                )

            result = self._llm_decomposition(disambiguated_text)
            processing_time = time.time() - start_time
            result.processing_time = processing_time

            # Update state with result
            new_state = state.model_copy()
            new_state.decomposition_result = result
            return new_state

        except Exception as e:
            processing_time = time.time() - start_time
            error_msg = f"Error in decomposition processing: {e}"
            new_state = state.model_copy()
            new_state.errors.append(error_msg)

            # Create a default decomposition result for error handling
            default_result = DecompositionResult(
                original_text=disambiguated_text,
                claim_candidates=[],
                processing_time=processing_time,
            )
            new_state.decomposition_result = default_result
            return new_state

    def _llm_decomposition(self, text: str) -> DecompositionResult:
        """LLM-based decomposition following the Claimify approach.

        This implements Stage 3: Decomposition from the Claimify pipeline, which:
        1. Breaks the sentence into atomic, self-contained claims
        2. Uses JSON output format as specified in claimify_decomposition.yaml prompt
        3. Gets quality criteria evaluation from the LLM rather than hardcoding
        """
        assert self.llm is not None, "LLM must be initialized for this method"

        try:
            # Call the DSPy module with the correct LLM
            with dspy.settings.context(lm=self.llm):
                response = self.dspy_module(
                    disambiguated_text=text,
                )
            response = response.decomposition_response_json.strip()

            # Parse JSON response using Pydantic model with proper error handling
            try:
                result_data = DecompositionResponse.model_validate_json(response)
                claim_candidates = []
                for candidate_data in result_data.claim_candidates:
                    claim_text = candidate_data.text.strip()
                    if not claim_text:
                        continue

                    # Get quality flags from LLM output
                    is_atomic = candidate_data.is_atomic
                    is_self_contained = candidate_data.is_self_contained
                    is_verifiable = candidate_data.is_verifiable
                    passes_criteria = candidate_data.passes_criteria
                    reasoning = candidate_data.reasoning

                    # Get confidence from LLM response or calculate based on quality flags
                    confidence = candidate_data.confidence or 0.5
                    if confidence is None:
                        # Fallback calculation if LLM doesn't provide confidence
                        if (
                            passes_criteria
                            and is_atomic
                            and is_self_contained
                            and is_verifiable
                        ):
                            confidence = 0.9
                        elif sum([is_atomic, is_self_contained, is_verifiable]) >= 2:
                            confidence = 0.6
                        else:
                            confidence = 0.3

                    # Apply confidence threshold - only include candidates above threshold
                    if confidence >= self.config.decomposition_confidence_threshold:
                        # Import here to avoid circular imports
                        from ..data_models import ClaimCandidate

                        candidate = ClaimCandidate(
                            text=claim_text,
                            is_atomic=is_atomic,
                            is_self_contained=is_self_contained,
                            is_verifiable=is_verifiable,
                            confidence=confidence,
                            reasoning=reasoning,
                        )
                        claim_candidates.append(candidate)

                return DecompositionResult(
                    original_text=text,
                    claim_candidates=claim_candidates,
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
            # If LLM fails, we cannot perform decomposition without heuristics
            raise ValueError(
                f"LLM decomposition failed and no fallback available: {e}"
            ) from e
