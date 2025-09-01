"""Decomposition component for the Claimify pipeline.

This module contains the DecompositionComponent class that implements the third stage
of the Claimify pipeline as a stateless component.
"""

import time
from typing import Optional, Protocol

from ..data_models import DecompositionResult
from ..llm_schemas import DecompositionResponse
from ..components.state import ClaimifyState
from ..data_models import ClaimifyConfig


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
    ):
        self.llm = llm
        self.config = config or ClaimifyConfig()

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
                raise ValueError("LLM is required for Decomposition component processing")
                
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

        # Use JSON prompt format matching claimify_decomposition.yaml
        prompt = f"""You are an expert at extracting atomic claims from text. Your task is to break down sentences into individual, verifiable claims that meet strict quality criteria. Each claim must be atomic (single fact), self-contained (no ambiguous references), and verifiable (factually checkable).
Analyze the following disambiguated sentence and extract atomic claims that meet the Claimify quality criteria.
Input sentence: "{text}"
Quality Criteria for Claims:
1. ATOMIC: Contains exactly one verifiable fact (no compound statements)
2. SELF-CONTAINED: No ambiguous pronouns or references (all entities clearly identified)
3. VERIFIABLE: Contains specific, factual information that can be fact-checked
Examples of VALID claims:
- "The user received an error from Pylance."
- "In Python, a slice cannot be assigned to a parameter of type int in __setitem__."
- "The error rate increased to 25% after deployment."
Examples of INVALID claims:
- "The error occurred while calling __setitem__ with a slice." (vague reference "the error")
- "The system worked but was slow." (compound statement - not atomic)
- "Something went wrong." (not specific enough to verify)
Instructions:
1. Split compound sentences (connected by "and", "but", "or", "because", etc.)
2. Evaluate each potential claim against the three criteria
3. Only include claims that pass ALL criteria
4. For claims that fail criteria, explain why they should become :Sentence nodes instead
Respond with valid JSON only:
{{
  "claim_candidates": [
    {{
      "text": "The extracted claim text",
      "is_atomic": true/false,
      "is_self_contained": true/false,
      "is_verifiable": true/false,
      "passes_criteria": true/false,
      "confidence": 0.0-1.0,
      "reasoning": "Explanation of evaluation",
      "node_type": "Claim" or "Sentence"
    }}
  ]
}}"""
        
        try:
            # Call the LLM with the prompt
            response = self.llm.complete(
                prompt,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens or 1000,
            ).strip()
            
            # Parse JSON response using Pydantic model
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
                    confidence = candidate_data.confidence
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
                    original_text=text, claim_candidates=claim_candidates
                )
            except Exception as e:
                raise ValueError(f"Invalid JSON response from LLM: {e}") from e
        except Exception as e:
            # If LLM fails, we cannot perform decomposition without heuristics
            raise ValueError(
                f"LLM decomposition failed and no fallback available: {e}"
            ) from e