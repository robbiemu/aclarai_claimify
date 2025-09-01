"""
Output data models for the Claimify pipeline.
Defines the data structures used for Neo4j persistence, including ClaimInput and SentenceInput.
"""

import uuid
from typing import Optional
from pydantic import BaseModel, Field


class ClaimInput(BaseModel):
    """
    Input data structure for creating Claim nodes in Neo4j.
    Represents a valid claim extracted by the Claimify pipeline that meets all quality criteria.
    """

    text: str = Field(
        ..., 
        description="The text content of the claim"
    )
    block_id: str = Field(
        ..., 
        description="The ID of the source block"
    )
    entailed_score: Optional[float] = Field(
        None, 
        ge=0.0, 
        le=1.0,
        description="Entailment score from evaluation"
    )
    coverage_score: Optional[float] = Field(
        None, 
        ge=0.0, 
        le=1.0,
        description="Coverage score from evaluation"
    )
    decontextualization_score: Optional[float] = Field(
        None, 
        ge=0.0, 
        le=1.0,
        description="Decontextualization score from evaluation"
    )
    verifiable: bool = Field(
        default=True,
        description="Whether the claim is verifiable"
    )
    self_contained: bool = Field(
        default=True,
        description="Whether the claim is self-contained"
    )
    context_complete: bool = Field(
        default=True,
        description="Whether the claim has complete context"
    )
    id: str = Field(
        default_factory=lambda: f"claim_{uuid.uuid4().hex}",
        description="Unique identifier for the claim"
    )


class SentenceInput(BaseModel):
    """
    Input data structure for creating Sentence nodes in Neo4j.
    Represents sentences that were processed but did not meet claim quality criteria,
    or sentences that were not selected for processing.
    """

    text: str = Field(
        ..., 
        description="The text content of the sentence"
    )
    block_id: str = Field(
        ..., 
        description="The ID of the source block"
    )
    ambiguous: bool = Field(
        default=False,
        description="Whether the sentence is ambiguous"
    )
    verifiable: bool = Field(
        default=False,
        description="Whether the sentence is verifiable"
    )
    failed_decomposition: bool = Field(
        default=False,
        description="Whether the sentence failed decomposition"
    )
    rejection_reason: Optional[str] = Field(
        None, 
        description="Reason why the sentence was rejected"
    )
    id: str = Field(
        default_factory=lambda: f"sentence_{uuid.uuid4().hex}",
        description="Unique identifier for the sentence"
    )