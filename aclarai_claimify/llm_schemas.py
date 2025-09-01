"""
Pydantic models for LLM JSON responses in the Claimify pipeline.
These models define the expected structure of JSON responses from LLMs
for each stage of the pipeline.
"""

from typing import List, Optional
from pydantic import BaseModel, Field


class SelectionResponse(BaseModel):
    """Schema for Selection agent LLM response."""
    selected: bool = Field(
        ..., 
        description="Whether the sentence should be selected for processing"
    )
    confidence: float = Field(
        ..., 
        ge=0.0, 
        le=1.0,
        description="Confidence score for the selection decision"
    )
    reasoning: str = Field(
        ..., 
        description="Explanation of the selection decision"
    )


class DisambiguationResponse(BaseModel):
    """Schema for Disambiguation agent LLM response."""
    disambiguated_text: str = Field(
        ..., 
        description="The disambiguated sentence text"
    )
    changes_made: List[str] = Field(
        default_factory=list,
        description="List of specific changes made during disambiguation"
    )
    confidence: float = Field(
        ..., 
        ge=0.0, 
        le=1.0,
        description="Confidence score for the disambiguation"
    )


class ClaimCandidateResponse(BaseModel):
    """Schema for individual claim candidate in Decomposition response."""
    text: str = Field(
        ..., 
        description="The claim candidate text"
    )
    is_atomic: bool = Field(
        ..., 
        description="Whether the candidate is atomic (single fact)"
    )
    is_self_contained: bool = Field(
        ..., 
        description="Whether the candidate is self-contained (no ambiguous references)"
    )
    is_verifiable: bool = Field(
        ..., 
        description="Whether the candidate is verifiable (factually checkable)"
    )
    passes_criteria: bool = Field(
        ..., 
        description="Whether the candidate passes all Claimify criteria"
    )
    confidence: Optional[float] = Field(
        None, 
        ge=0.0, 
        le=1.0,
        description="Confidence score for the candidate quality"
    )
    reasoning: str = Field(
        ..., 
        description="Explanation of the quality evaluation"
    )
    node_type: str = Field(
        ..., 
        description="Whether this should be a Claim or Sentence node"
    )


class DecompositionResponse(BaseModel):
    """Schema for Decomposition agent LLM response."""
    claim_candidates: List[ClaimCandidateResponse] = Field(
        default_factory=list,
        description="List of claim candidates extracted from the sentence"
    )