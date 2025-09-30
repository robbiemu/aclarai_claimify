from typing import List, Optional
from pydantic import BaseModel, Field, AliasChoices, field_validator


class SelectionResponse(BaseModel):
    """Schema for Selection agent LLM response."""

    selected: bool = Field(
        ...,
        validation_alias=AliasChoices(
            "selection",
            "selected",
            "select",
            "is_selected",
            "is_selectable",
            "selection_decision",
        ),
        description="Whether the sentence should be selected for processing",
    )
    confidence: Optional[float] = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Confidence score for the selection decision",
    )
    reasoning: Optional[str] = Field(
        default=None, description="Explanation of the selection decision"
    )

    @field_validator("selected", mode="before")
    @classmethod
    def coerce_to_bool(cls, v):
        if isinstance(v, str):
            if v.lower() in ["yes", "true", "include", "selected"]:
                return True
            elif v.lower() in ["no", "false", "exclude", "not_selected"]:
                return False
        return v


class DisambiguationResponse(BaseModel):
    """Schema for Disambiguation agent LLM response."""

    disambiguated_text: str = Field(
        ...,
        validation_alias=AliasChoices("disambiguated_text", "rewritten_text"),
        description="The disambiguated sentence text",
    )
    changes_made: List[str] = Field(
        default_factory=list,
        validation_alias=AliasChoices("changes_made", "changes"),
        description="List of specific changes made during disambiguation",
    )
    confidence: Optional[float] = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Confidence score for the disambiguation",
    )

    @field_validator("changes_made", mode="before")
    @classmethod
    def coerce_changes(cls, v):
        if isinstance(v, list):
            changed = []
            for item in v:
                if isinstance(item, dict):
                    changed.append(item.get("rationale", str(item)))
                else:
                    changed.append(str(item))
            return changed
        return [str(v)] if v else []


class ClaimCandidateResponse(BaseModel):
    """Schema for individual claim candidate in Decomposition response."""

    text: str = Field(
        ...,
        validation_alias=AliasChoices("claim_text", "claim", "value", "text"),
        description="The claim candidate text",
    )
    is_atomic: bool = Field(
        default=False,
        validation_alias=AliasChoices("is_atomic", "atomic"),
        description="Whether the candidate is atomic (single fact)",
    )
    is_self_contained: bool = Field(
        default=False,
        validation_alias=AliasChoices("is_self_contained", "self_contained"),
        description="Whether the candidate is self-contained (no ambiguous references)",
    )
    is_verifiable: bool = Field(
        default=False,
        validation_alias=AliasChoices("is_verifiable", "verifiable"),
        description="Whether the candidate is verifiable (factually checkable)",
    )
    passes_criteria: bool = Field(
        default=False,
        validation_alias=AliasChoices("passes_criteria", "passes"),
        description="Whether the candidate passes all Claimify criteria",
    )
    confidence: Optional[float] = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Confidence score for the candidate quality",
    )
    reasoning: str = Field(
        default="", description="Explanation of the quality evaluation"
    )
    node_type: str = Field(
        default="Sentence",
        description="Whether this should be a Claim or Sentence node",
    )


class DecompositionResponse(BaseModel):
    """Schema for Decomposition agent LLM response."""

    claim_candidates: List[ClaimCandidateResponse] = Field(
        default_factory=list,
        validation_alias=AliasChoices(
            "claim_candidates", "claims", "candidates", "items"
        ),
        description="List of claim candidates extracted from the sentence",
    )
