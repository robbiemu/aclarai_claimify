"""State model for the Claimify pipeline components.

This module defines the ClaimifyState Pydantic model that serves as the data
contract between stateless components in the refactored pipeline.
"""

from typing import List, Optional
from pydantic import BaseModel, Field

from ..data_models import (
    ClaimifyContext,
    SelectionResult,
    DisambiguationResult,
    DecompositionResult,
    SentenceChunk,
    ClaimCandidate,
)


class ClaimifyState(BaseModel):
    """State model that carries data between components in the Claimify pipeline.

    This model serves as the central data contract for the stateless component architecture,
    replacing the monolithic pipeline approach with a series of independent transformations.
    """

    # Input data
    context: ClaimifyContext = Field(
        ..., description="Context window for Claimify processing"
    )

    # Stage results (populated as pipeline progresses)
    selection_result: Optional[SelectionResult] = Field(
        None, description="Result of the Selection stage"
    )
    disambiguation_result: Optional[DisambiguationResult] = Field(
        None, description="Result of the Disambiguation stage"
    )
    decomposition_result: Optional[DecompositionResult] = Field(
        None, description="Result of the Decomposition stage"
    )

    # Error tracking
    errors: List[str] = Field(
        default_factory=list, description="List of errors encountered during processing"
    )

    @property
    def current_sentence(self) -> SentenceChunk:
        """Get the current sentence being processed."""
        return self.context.current_sentence

    @property
    def was_selected(self) -> bool:
        """Check if the sentence was selected for processing."""
        return self.selection_result is not None and self.selection_result.is_selected

    @property
    def final_claims(self) -> List[ClaimCandidate]:
        """Get the final valid claims from this processing."""
        if self.decomposition_result:
            return self.decomposition_result.valid_claims
        return []

    @property
    def final_sentences(self) -> List[ClaimCandidate]:
        """Get candidates that should become Sentence nodes."""
        if self.decomposition_result:
            return self.decomposition_result.sentence_nodes
        return []