import os
import litellm
from langsmith import Client
from aclarai_claimify.config import load_claimify_config
from .graph import create_graph, SqliteSaver
from .state import DataScoutState

def setup_observability():
    """Initializes LangSmith tracing for all litellm calls if configured."""
    if os.getenv("LANGCHAIN_API_KEY"):
        print("LangSmith API Key found. Setting up litellm callback...")
        client = Client() # Auto-configures from environment variables
        litellm.success_callback = [client.log_run]
        litellm.failure_callback = [client.log_run]
        print("Observability configured.")
    else:
        print("Warning: LANGCHAIN_API_KEY not set. Skipping LangSmith integration.")

def run():
    # Load the unified config, which includes the scout_agent section
    config = load_claimify_config()
    mission_plan = config.scout_agent

    setup_observability()

    print("Data Scout Agent started.")
    if not mission_plan:
        print("Error: scout_agent configuration not found in config.yaml.")
        return

    print(f"Mission Goal: {mission_plan.mission_plan.goal}")

    checkpointer = SqliteSaver.from_conn_string(mission_plan.checkpointer_path)
    graph = create_graph(checkpointer=checkpointer)

    thread = {"configurable": {"thread_id": "DATA_SCOUT_THREAD"}}

    # Check if there is a saved state for the thread
    saved_state = checkpointer.get(thread)
    if saved_state:
        print("Resuming from saved state.")
        initial_state = None
    else:
        print("Starting new mission.")
        initial_state: DataScoutState = {
            "mission_goal": mission_plan.mission_plan.goal,
            "deconstructed_goal": {},
            "search_plan": {},
            "search_queries": [],
            "search_results": [],
            "extraction_results": [],
            "fitness_check_results": [],
            "archived_data": [],
            "iteration": 0,
            "max_iterations": mission_plan.mission_plan.max_iterations,
        }

    for event in graph.stream(initial_state, thread):
        for key, value in event.items():
            print(f"Node: {key}")
            print(f"State: {value}")
            print("-" * 20)

    print("Data Scout Agent finished.")
