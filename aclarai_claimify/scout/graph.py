from langgraph.graph import StateGraph, END
from .state import DataScoutState
from .nodes import (
    deconstruct_goal_node,
    plan_node,
    web_search_node,
    fitness_check_node,
    archiving_node,
)

def should_continue(state: DataScoutState) -> str:
    """
    Determines whether to continue the loop or end.
    """
    if state['iteration'] >= state['max_iterations']:
        return "end"
    return "continue"

from langgraph.checkpoint.sqlite import SqliteSaver

def create_graph(checkpointer: SqliteSaver = None):
    """
    Creates the Data Scout agent graph.
    """
    workflow = StateGraph(DataScoutState)

    # Add the nodes
    workflow.add_node("deconstruct_goal", deconstruct_goal_node)
    workflow.add_node("plan", plan_node)
    workflow.add_node("web_search", web_search_node)
    workflow.add_node("fitness_check", fitness_check_node)
    workflow.add_node("archive", archiving_node)

    # Set the entry point
    workflow.set_entry_point("deconstruct_goal")

    # Add the edges
    workflow.add_edge("deconstruct_goal", "plan")
    workflow.add_edge("plan", "web_search")
    workflow.add_edge("web_search", "fitness_check")
    workflow.add_edge("fitness_check", "archive")

    # Add the conditional edge
    workflow.add_conditional_edges(
        "archive",
        should_continue,
        {
            "continue": "web_search",
            "end": END,
        },
    )

    return workflow.compile(checkpointer=checkpointer)
