"""
Data models for the Claimify pipeline.
Defines the core data structures used throughout the Selection → Disambiguation → Decomposition
pipeline, including input/output types and configuration models.
"""

import logging
from enum import Enum
from typing import List, Optional, Dict
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class NodeType(str, Enum):
    """Types of nodes that can be created from processed content."""

    CLAIM = "Claim"
    SENTENCE = "Sentence"


class SentenceChunk(BaseModel):
    """
    A sentence chunk to be processed by the Claimify pipeline.
    Represents individual sentences extracted from Tier 1 content,
    which serve as input to the Selection stage.
    """

    text: str = Field(..., description="The text content of the sentence")
    source_id: str = Field(..., description="Original block/document ID")
    chunk_id: str = Field(..., description="Unique identifier for this chunk")
    sentence_index: int = Field(..., description="Position within the source")


class ClaimifyContext(BaseModel):
    """
    Context window for Claimify processing.
    Contains preceding and following sentences to provide context
    during Selection and Disambiguation stages.
    """

    current_sentence: SentenceChunk
    preceding_sentences: List[SentenceChunk] = Field(
        default_factory=list, description="Preceding sentences (p sentences)"
    )
    following_sentences: List[SentenceChunk] = Field(
        default_factory=list, description="Following sentences (f sentences)"
    )

    @property
    def context_window_size(self) -> tuple[int, int]:
        """Returns (p, f) context window size."""
        return len(self.preceding_sentences), len(self.following_sentences)


class SelectionResult(BaseModel):
    """Result of the Selection stage."""

    sentence_chunk: SentenceChunk
    is_selected: bool = Field(
        ..., description="Whether the sentence was selected for processing"
    )
    confidence: Optional[float] = Field(
        None, ge=0.0, le=1.0, description="Confidence score for the selection decision"
    )
    reasoning: Optional[str] = Field(
        None, description="Explanation of the selection decision"
    )
    processing_time: Optional[float] = Field(
        None, description="Time taken to process the selection"
    )
    rewritten_text: Optional[str] = Field(
        None, description="Cleaned sentence text from LLM"
    )


class DisambiguationResult(BaseModel):
    """Result of the Disambiguation stage."""

    original_sentence: SentenceChunk
    disambiguated_text: str = Field(..., description="The disambiguated sentence text")
    changes_made: List[str] = Field(
        default_factory=list,
        description="Description of changes made during disambiguation",
    )
    confidence: Optional[float] = Field(
        None, ge=0.0, le=1.0, description="Confidence score for the disambiguation"
    )
    processing_time: Optional[float] = Field(
        None, description="Time taken to process the disambiguation"
    )


class ClaimCandidate(BaseModel):
    """A candidate claim from the Decomposition stage."""

    text: str = Field(..., description="The claim candidate text")
    is_atomic: bool = Field(
        ..., description="Whether the candidate is atomic (single fact)"
    )
    is_self_contained: bool = Field(
        ...,
        description="Whether the candidate is self-contained (no ambiguous references)",
    )
    is_verifiable: bool = Field(
        ..., description="Whether the candidate is verifiable (factually checkable)"
    )
    confidence: Optional[float] = Field(
        None, ge=0.0, le=1.0, description="Confidence score for the candidate quality"
    )
    reasoning: Optional[str] = Field(
        None, description="Explanation of the quality evaluation"
    )

    @property
    def passes_criteria(self) -> bool:
        """Check if the claim meets all Claimify criteria."""
        return self.is_atomic and self.is_self_contained and self.is_verifiable

    @property
    def node_type(self) -> NodeType:
        """Determine the appropriate node type for this candidate."""
        return NodeType.CLAIM if self.passes_criteria else NodeType.SENTENCE


class DecompositionResult(BaseModel):
    """Result of the Decomposition stage."""

    original_text: str = Field(..., description="The original text that was decomposed")
    claim_candidates: List[ClaimCandidate] = Field(
        default_factory=list,
        description="List of claim candidates extracted from the sentence",
    )
    processing_time: Optional[float] = Field(
        None, description="Time taken to process the decomposition"
    )

    @property
    def valid_claims(self) -> List[ClaimCandidate]:
        """Get candidates that pass Claimify criteria."""
        return [claim for claim in self.claim_candidates if claim.passes_criteria]

    @property
    def sentence_nodes(self) -> List[ClaimCandidate]:
        """Get candidates that should become Sentence nodes."""
        return [claim for claim in self.claim_candidates if not claim.passes_criteria]


class ClaimifyResult(BaseModel):
    """Complete result of processing a sentence through the Claimify pipeline."""

    original_chunk: SentenceChunk
    context: ClaimifyContext
    selection_result: Optional[SelectionResult] = Field(
        None, description="Result of the Selection stage"
    )
    disambiguation_result: Optional[DisambiguationResult] = Field(
        None, description="Result of the Disambiguation stage"
    )
    decomposition_result: Optional[DecompositionResult] = Field(
        None, description="Result of the Decomposition stage"
    )
    total_processing_time: Optional[float] = Field(
        None, description="Total time taken to process the sentence"
    )
    errors: List[str] = Field(
        default_factory=list, description="List of errors encountered during processing"
    )

    @property
    def was_processed(self) -> bool:
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


class GenerateDatasetSemanticEmbedderConfig(BaseModel):
    """Configuration for the semantic embedder."""

    type: str = Field(
        default="sentence_transformer",
        description="Specifies which plugin to use from aclarai_claimify/embeddings/",
    )
    model: str = Field(
        default="all-MiniLM-L6-v2",
        description="The name of the embedding model to use.",
    )


class GenerateDatasetSemanticContextConfig(BaseModel):
    """Parameters for semantic context generation."""

    min_k: int = Field(
        default=3,
        ge=1,
        description="Minimum number of context sentences to retrieve.",
    )
    max_k: int = Field(
        default=20,
        ge=1,
        description="Maximum number of context sentences to retrieve.",
    )
    similarity_threshold: float = Field(
        default=0.75,
        ge=0.0,
        le=1.0,
        description="Minimum similarity score for a sentence to be included in the context.",
    )


class GenerateDatasetSemanticConfig(BaseModel):
    """Configuration for the 'semantic' method."""

    embedder: GenerateDatasetSemanticEmbedderConfig = Field(
        default_factory=GenerateDatasetSemanticEmbedderConfig,
        description="Embedding model configuration.",
    )
    context_params: GenerateDatasetSemanticContextConfig = Field(
        default_factory=GenerateDatasetSemanticContextConfig,
        description="Parameters for controlling the semantic context generation.",
    )


class GenerateDatasetStaticConfig(BaseModel):
    """Configuration for the 'static' method."""

    k_window_size: int = Field(
        default=3,
        ge=0,
        description="The number of sentences to include before and after the target sentence.",
    )


class GenerateDatasetConfig(BaseModel):
    """Configuration for the generate-dataset tool."""

    method: str = Field(
        default="semantic",
        description='The context generation method. Can be "semantic" or "static".',
    )
    semantic: GenerateDatasetSemanticConfig = Field(
        default_factory=GenerateDatasetSemanticConfig
    )
    static: GenerateDatasetStaticConfig = Field(
        default_factory=GenerateDatasetStaticConfig
    )


class AgentsReactConfig(BaseModel):
    """Configuration for ReAct agents."""

    max_iterations: int = Field(
        default=7,
        ge=1,
        le=50,
        description="Maximum number of iterations for ReAct loops in research agents",
    )


class SupervisorConfig(BaseModel):
    """Configuration for the supervisor agent."""

    max_consecutive_same_agent: int = Field(
        default=1,
        ge=0,
        le=5,
        description="Maximum consecutive calls to the same agent without failure",
    )
    max_total_steps: int = Field(
        default=100,
        ge=1,
        le=1000,
        description="Maximum total steps before terminating to prevent infinite loops",
    )

    # Agent retry configurations
    agent_max_retries: Dict[str, int] = Field(
        default_factory=lambda: {
            "research": 2,
            "fitness": 1,
            "archive": 1,
            "synthetic": 1,
        },
        description="Maximum retries per agent type before switching",
    )

    # Fallback chain configuration
    fallback_chain: Dict[str, str] = Field(
        default_factory=lambda: {
            "fitness": "research",
            "research": "synthetic",
            "synthetic": "archive",
            "archive": "end",
        },
        description="Default fallback agents when an agent fails repeatedly",
    )


class AgentsConfig(BaseModel):
    """Configuration for all agents."""

    react: AgentsReactConfig = Field(
        default_factory=AgentsReactConfig, description="Configuration for ReAct agents"
    )
    supervisor: SupervisorConfig = Field(
        default_factory=SupervisorConfig,
        description="Configuration for the supervisor agent",
    )



class ClaimifyConfig(BaseModel):
    """Configuration for the Claimify pipeline."""

    # Context window parameters
    context_window_p: int = Field(
        default=3, ge=0, description="Previous sentences in context window"
    )
    context_window_f: int = Field(
        default=1, ge=0, description="Following sentences in context window"
    )

    # NEW: Configuration for agents (ReAct, Supervisor, etc.)
    agents: AgentsConfig = Field(
        default_factory=AgentsConfig,
        description="Configuration for all agents in the system",
    )

    # NEW: Configuration for the generate-dataset tool
    generate_dataset: GenerateDatasetConfig = Field(
        default_factory=GenerateDatasetConfig,
        description="Settings for the 'generate-dataset' tool.",
    )

    # Model configuration
    selection_model: Optional[str] = Field(
        None, description="Model to use for selection stage"
    )
    disambiguation_model: Optional[str] = Field(
        None, description="Model to use for disambiguation stage"
    )
    decomposition_model: Optional[str] = Field(
        None, description="Model to use for decomposition stage"
    )
    default_model: str = Field(
        default="gpt-3.5-turbo",
        description="Default model to use when stage model is not specified",
    )

    # Processing parameters
    max_retries: int = Field(
        default=3, ge=0, description="Maximum number of retries for LLM calls"
    )
    timeout_seconds: int = Field(
        default=30, ge=1, description="Timeout for LLM calls in seconds"
    )
    temperature: float = Field(
        default=0.1, ge=0.0, le=1.0, description="Temperature for LLM calls"
    )
    max_tokens: int = Field(
        default=1000, ge=1, description="Maximum tokens for LLM calls"
    )

    # Quality thresholds
    selection_confidence_threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Minimum confidence threshold for selection",
    )
    disambiguation_confidence_threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Minimum confidence threshold for disambiguation",
    )
    decomposition_confidence_threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Minimum confidence threshold for decomposition",
    )

    # Logging configuration
    log_decisions: bool = Field(
        default=True, description="Whether to log agent decisions"
    )
    log_transformations: bool = Field(
        default=True, description="Whether to log transformations"
    )
    log_timing: bool = Field(
        default=True, description="Whether to log processing times"
    )

    def get_model_for_stage(self, stage: str) -> str:
        """Get the configured model for a specific pipeline stage."""
        stage_models = {
            "selection": self.selection_model,
            "disambiguation": self.disambiguation_model,
            "decomposition": self.decomposition_model,
        }
        return stage_models.get(stage) or self.default_model


class OptimizationConfig(BaseModel):
    """Configuration for the DSPy optimizer."""

    optimizer_name: str = Field(
        ...,
        description="Name of the DSPy optimizer to use (e.g., 'bootstrap-fewshot')",
    )
    params: dict = Field(
        default_factory=dict,
        description="Parameters to pass to the optimizer",
    )
