# state.py
"""
Defines the overall state for the Data Scout agent graph.
"""
from typing import TypedDict, List, Dict, Optional, Annotated
from langgraph.graph.message import add_messages

# Import the new model
from .models import FitnessReport

class DataScoutState(TypedDict):
    strategy_block: str
    """
    Represents the state of the Data Scout graph.

    Attributes:
        messages: The conversation history, managed by `add_messages`.
        run_id: The unique identifier for the current agent run (thread_id).
        progress: A dictionary tracking the mission progress.
        current_task: The current task being processed by the agent.
        research_findings: A list of dictionaries containing research results.
        research_session_cache: A list of raw results accumulated during current research session.
        pedigree_path: The file path for the audit trail, loaded from config.
        next_agent: The name of the next agent to be invoked by the supervisor.
        decision_history: A list of recent supervisor decisions to prevent loops.
        tool_execution_failures: A counter for consecutive tool execution failures.
        research_attempts: A counter for research attempts on current question.
        samples_generated: A counter for tracking completed samples.
        total_samples_target: The total number of samples to generate.
        current_mission: The current mission being processed.
        synthetic_samples_generated: A counter for synthetic samples created.
        research_samples_generated: A counter for research-based samples created.
        consecutive_failures: A counter for consecutive agent failures.
        last_action_status: The status of the last agent action (success/failure).
        last_action_agent: The name of the last agent that took action.
        synthetic_budget: The maximum allowed percentage of synthetic samples (0.0-1.0).
        fitness_report: A structured report from the FitnessAgent evaluating a source document.
        task_history: A list of (characteristic, topic, failure_reason) tuples.
    """
    messages: Annotated[list, add_messages]
    run_id: str
    progress: Dict
    current_task: Optional[Dict]
    research_findings: List[Dict]
    research_session_cache: List[Dict]
    pedigree_path: str
    next_agent: str
    decision_history: List[str]
    tool_execution_failures: int
    research_attempts: int
    samples_generated: int
    total_samples_target: int
    current_mission: str
    synthetic_samples_generated: int
    research_samples_generated: int
    consecutive_failures: int
    last_action_status: str
    last_action_agent: str
    synthetic_budget: float
    fitness_report: Optional[FitnessReport]
    task_history: List[tuple[str, str, str]]
