# main.py
"""
CLI entry point for the Data Scout Agent.
"""
import os
import uuid
import litellm
import sqlite3

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.sqlite import SqliteSaver

from .graph import build_graph
from .tools import _HAVE_LIBCRAWLER
from ..config import load_claimify_config


def setup_observability():
    """Initializes LangSmith tracing for all litellm calls if configured."""
    if os.getenv("LANGCHAIN_API_KEY"):
        print("LangSmith API Key found. Setting up litellm callback...")
        litellm.success_callback = ["langsmith"]
        litellm.failure_callback = ["langsmith"]
        print("Observability configured.")
    else:
        print("Warning: LANGCHAIN_API_KEY not set. Skipping LangSmith integration.")


def run():
    """Main function to run the CLI interactive loop for the scout agent."""
    config = load_claimify_config()
    scout_config = config.scout_agent
    if not scout_config:
        print("Error: `scout_agent` section not found in configuration.")
        return

    setup_observability()

    if not _HAVE_LIBCRAWLER:
        print("\n\033[93mWarning: `libcrawler` is not installed.\033[0m")
        print("Crawler functionality will be disabled.")
        print("To enable, run: pip install -e .[crawler]\n")

    # Use the checkpointer as a context manager for the entire session
    with SqliteSaver.from_conn_string(scout_config.checkpointer_path) as checkpointer:

        app = build_graph(checkpointer=checkpointer)

        thread_id = str(uuid.uuid4())
        thread_config = {"configurable": {"thread_id": thread_id}}

        print(f"🤖 \033[92mData Scout Agent is ready. Starting new thread: {thread_id}\033[0m")
        print("Type your request or 'exit' to quit.")

        while True:
            try:
                user_input = input(">> ")
                if user_input.lower() in ["exit", "quit"]:
                    print("Exiting...")
                    break

                # Check if state for this thread already exists. If not, create it.
                thread_state = app.get_state(thread_config)
                if not thread_state or not thread_state.values().get("messages"):
                    print("   -> Creating new state for this thread.")
                    initial_state = {
                        "messages": [],
                        "task_queue": [],
                        "research_findings": [],
                        "pedigree_path": scout_config.writer.audit_trail_path,
                    }
                    app.update_state(thread_config, initial_state)

                inputs = {"messages": [HumanMessage(content=user_input)]}

                for event in app.stream(inputs, config=thread_config):
                    for node_name, node_output in event.items():
                        if node_name == "__end__":
                            print("\n🏁 \033[92mAgent has finished the task.\033[0m")
                            continue

                        print(f"\033[1m-> Executing Node: {node_name.upper()}\033[0m")
                        if "messages" in node_output:
                            message = node_output["messages"][-1]
                            if message.tool_calls:
                                tool_calls_str = ", ".join([f"{tc['name']}(...)" for tc in message.tool_calls])
                                print(f"   - \033[94mDecided to call tools\033[0m: {tool_calls_str}")
                            else:
                                print(f"   - \033[92mResponded\033[0m: {message.content}")
                        else:
                            print(f"   - \033[90mOutput: {str(node_output)[:300]}...\033[0m")
                        print("-" * 30)

            except KeyboardInterrupt:
                print("\nExiting...")
                break
            except Exception as e:
                print(f"\n\033[91mAn error occurred: {e}\033[0m")
                print("Please try again.")
    # The 'with' block ensures the connection is closed automatically.
