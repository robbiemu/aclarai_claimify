# graph.py
"""
Builds the LangGraph application graph for the Data Scout agent.
"""
import sqlite3
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.sqlite import SqliteSaver

from scout_agent.state import DataScoutState
from scout_agent.nodes import supervisor_node, research_node, archive_node, fitness_node
from scout_agent.tools import get_tools_for_role

def build_graph():
    """Builds and compiles the multi-agent graph."""
    workflow = StateGraph(DataScoutState)

    # --- Define Agent Nodes ---
    workflow.add_node("supervisor", supervisor_node)
    workflow.add_node("research", research_node)
    workflow.add_node("archive", archive_node)
    workflow.add_node("fitness", fitness_node)

    # --- Define Role-Specific Tool Nodes ---
    # Each agent that can use tools gets its own ToolNode.
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
            "end": END, # Supervisor can decide to end the process
        }
    )

    # After an agent node runs, it can either call a tool or return to the supervisor.
    # The ToolNode will route back to the supervisor after executing the tool.
    workflow.add_edge("research", "research_tools")
    workflow.add_edge("archive", "archive_tools")

    # The ToolNodes execute the chosen tool and their output is added to the state.
    # Then, control returns to the supervisor to decide the next step.
    workflow.add_edge("research_tools", "supervisor")
    workflow.add_edge("archive_tools", "supervisor")

    # The fitness agent has no tools, so it always returns directly to the supervisor.
    workflow.add_edge("fitness", "supervisor")

    # --- Compile the Graph with a Checkpointer ---
    # The checkpointer allows the graph to be persistent.
    conn = sqlite3.connect("checkpointer.sqlite", check_same_thread=False)
    checkpointer = SqliteSaver(conn=conn)

    return workflow.compile(checkpointer=checkpointer)
