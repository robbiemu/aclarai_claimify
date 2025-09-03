# state.py
"""
Defines the overall state for the Data Scout agent graph.
"""
from typing import TypedDict, List, Dict, Optional, Annotated
from langgraph.graph.message import add_messages

class DataScoutState(TypedDict):
    """
    Represents the state of the Data Scout graph.

    Attributes:
        messages: The conversation history, managed by `add_messages`.
        task_queue: A list of tasks for the agents to perform.
        research_findings: A list of dictionaries containing research results.
        pedigree_path: The file path for the audit trail, loaded from config.
        next_agent: The name of the next agent to be invoked by the supervisor.
    """
    messages: Annotated[list, add_messages]
    task_queue: List[Dict]
    research_findings: List[Dict]
    pedigree_path: str
    next_agent: str
