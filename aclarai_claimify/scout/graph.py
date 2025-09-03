# graph.py
"""
Builds the LangGraph application graph for the Data Scout agent.
"""
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.sqlite import SqliteSaver

from .state import DataScoutState
from .nodes import supervisor_node, research_node, archive_node, fitness_node
from .tools import get_tools_for_role

def build_graph(checkpointer: SqliteSaver):
    """
    Builds and compiles the multi-agent graph with a supervisor.

    Args:
        checkpointer: A LangGraph checkpointer instance for persisting state.

    Returns:
        A compiled LangGraph app.
    """
    workflow = StateGraph(DataScoutState)

    # --- Define Agent Nodes ---
    workflow.add_node("supervisor", supervisor_node)
    workflow.add_node("research", research_node)
    workflow.add_node("archive", archive_node)
    workflow.add_node("fitness", fitness_node)

    # --- Define Role-Specific Tool Nodes ---
    research_tools = get_tools_for_role("research")
    archive_tools = get_tools_for_role("archive")

    workflow.add_node("research_tools", ToolNode(research_tools))
    workflow.add_node("archive_tools", ToolNode(archive_tools))

    # --- Wire the Graph ---
    workflow.set_entry_point("supervisor")

    # The supervisor decides which agent to run next.
    workflow.add_conditional_edges(
        "supervisor",
        lambda state: state["next_agent"],
        {
            "research": "research",
            "archive": "archive",
            "fitness": "fitness",
            "end": END,
        }
    )

    # Agent nodes route to their tool nodes, which then route back to the supervisor.
    workflow.add_edge("research", "research_tools")
    workflow.add_edge("research_tools", "supervisor")

    workflow.add_edge("archive", "archive_tools")
    workflow.add_edge("archive_tools", "supervisor")

    # The fitness agent has no tools, so it routes directly back to the supervisor.
    workflow.add_edge("fitness", "supervisor")

    # Compile the graph with the provided checkpointer
    return workflow.compile(checkpointer=checkpointer)
