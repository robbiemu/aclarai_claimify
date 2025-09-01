"""
Output data models for the Claimify pipeline.
Defines the data structures used for Neo4j persistence, including ClaimInput and SentenceInput.
"""

import uuid
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ClaimInput:
    """
    Input data structure for creating Claim nodes in Neo4j.
    Represents a valid claim extracted by the Claimify pipeline that meets all quality criteria.
    """

    text: str
    block_id: str
    entailed_score: Optional[float] = None
    coverage_score: Optional[float] = None
    decontextualization_score: Optional[float] = None
    verifiable: bool = True
    self_contained: bool = True
    context_complete: bool = True
    id: str = field(default_factory=lambda: f"claim_{uuid.uuid4().hex}")

    def __post_init__(self):
        """Generate a unique ID if not provided."""
        if not self.id:
            self.id = f"claim_{uuid.uuid4().hex}"


@dataclass
class SentenceInput:
    """
    Input data structure for creating Sentence nodes in Neo4j.
    Represents sentences that were processed but did not meet claim quality criteria,
    or sentences that were not selected for processing.
    """

    text: str
    block_id: str
    ambiguous: bool = False
    verifiable: bool = False
    failed_decomposition: bool = False
    rejection_reason: Optional[str] = None
    id: str = field(default_factory=lambda: f"sentence_{uuid.uuid4().hex}")

    def __post_init__(self):
        """Generate a unique ID if not provided."""
        if not self.id:
            self.id = f"sentence_{uuid.uuid4().hex}"