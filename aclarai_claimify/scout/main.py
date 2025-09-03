# main.py
"""
CLI entry point for the Data Scout Agent.
"""

import os
import sys
from typing import Optional
import uuid
import litellm
import yaml
import typer

# Force unbuffered output for live progress updates
sys.stdout.reconfigure(line_buffering=True)

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.sqlite import SqliteSaver

from .graph import build_graph
from .tools import _HAVE_LIBCRAWLER
from ..config import load_claimify_config


def get_mission_names(mission_plan_path: str) -> list[str]:
    """Loads the mission YAML and returns a list of mission names."""
    try:
        with open(mission_plan_path, "r") as f:
            content = f.read()
            if content.startswith("#"):
                first_newline = content.find("\n")
                if first_newline != -1:
                    content = content[first_newline + 1 :]
            mission_plan = yaml.safe_load(content)
        return [mission.get("name") for mission in mission_plan.get("missions", [])]
    except Exception:
        return []


def load_mission_plan(
    mission_plan_path: str = "settings/scout_mission.yaml",
    mission_name: Optional[str] = None,
) -> dict:
    """
    Load and parse the mission plan YAML file to calculate target sample counts.

    Args:
        mission_plan_path: Path to the mission plan YAML file

    Returns:
        Dictionary with mission plan information including total target samples
    """
    try:
        with open(mission_plan_path, "r") as f:
            # Skip the first line which is a comment
            content = f.read()
            if content.startswith("#"):
                # Find the first newline and skip to after it
                first_newline = content.find("\n")
                if first_newline != -1:
                    content = content[first_newline + 1 :]
            mission_plan = yaml.safe_load(content)

        # Calculate total target samples
        total_samples = 0
        missions = mission_plan.get("missions", [])

        if mission_name:
            for mission in missions:
                if mission.get("name") == mission_name:
                    target_size = mission.get("target_size", 0)
                    goals = mission.get("goals", [])
                    total_samples += target_size * len(goals)
                    break
        else:
            for mission in missions:
                target_size = mission.get("target_size", 0)
                goals = mission.get("goals", [])
                # Each goal is a characteristic with topics
                # Total samples = target_size * number_of_characteristics
                total_samples += target_size * len(goals)

        return {
            "missions": missions,
            "total_samples_target": total_samples,
            "mission_plan_path": mission_plan_path,
        }
    except Exception as e:
        print(f"Warning: Could not load mission plan from {mission_plan_path}: {e}")
        # Return default values
        return {
            "missions": [],
            "total_samples_target": 1200,  # Default from documentation
            "mission_plan_path": mission_plan_path,
        }


def setup_observability():
    """Initializes LangSmith tracing for all litellm calls if configured."""
    if os.getenv("LANGCHAIN_API_KEY"):
        print("LangSmith API Key found. Setting up litellm callback...")
        litellm.success_callback = ["langsmith"]
        litellm.failure_callback = ["langsmith"]
        print("Observability configured.")
    else:
        print("Warning: LANGCHAIN_API_KEY not set. Skipping LangSmith integration.")


def run(
    mission_name: Optional[str] = typer.Option(
        None, "--mission", "-m", help="The name of the mission to run."
    ),
    max_iterations: int = None,
    recursion_limit: int = typer.Option(None, "--recursion-limit"),
):
    """Main function to run the CLI interactive loop for the scout agent."""
    config = load_claimify_config()
    scout_config = config.scout_agent
    if not scout_config:
        print("Error: `scout_agent` section not found in configuration.")
        return

    mission_plan_path = "settings/scout_mission.yaml"
    available_missions = get_mission_names(mission_plan_path)

    if not mission_name:
        print("❌ Error: No mission specified.")
        print("Available missions are:")
        for name in available_missions:
            print(f"  - {name}")
        print("\n💡 Please specify one with the --mission flag.")
        return

    if mission_name not in available_missions:
        print(f"❌ Error: Mission '{mission_name}' not found in {mission_plan_path}.")
        return

    setup_observability()
    sys.stdout.flush()  # Ensure setup messages appear immediately

    if not _HAVE_LIBCRAWLER:
        print("\033[93mWarning: `libcrawler` is not installed.\033[0m")
        print("Crawler functionality will be disabled.")
        print("To enable, run: pip install -e .[crawler]\n")
        sys.stdout.flush()

    # Load mission plan to get target sample counts
    mission_plan_info = load_mission_plan(mission_plan_path, mission_name)
    total_samples_target = mission_plan_info["total_samples_target"]
    print(
        f"📊 Mission Plan: Targeting {total_samples_target} samples for mission '{mission_name}'"
    )
    sys.stdout.flush()

    # --- START: Dynamic Recursion Limit Calculation ---
    if recursion_limit is None:  # Only calculate if not provided via CLI
        STEPS_PER_SAMPLE = 4
        RETRY_MARGIN = 20
        OVERHEAD_FACTOR = 1.2
        calculated_limit = int(
            STEPS_PER_SAMPLE * (total_samples_target + RETRY_MARGIN) * OVERHEAD_FACTOR
        )
        recursion_limit = calculated_limit
        print(f"🔐 Calculated recursion limit: {recursion_limit}")
    else:
        print(f"🔐 Using CLI-provided recursion limit: {recursion_limit}")
    sys.stdout.flush()
    # --- END: Dynamic Recursion Limit Calculation ---

    # Use the checkpointer as a context manager for the entire session
    with SqliteSaver.from_conn_string(scout_config.checkpointer_path) as checkpointer:
        app = build_graph(checkpointer=checkpointer)

        thread_id = str(uuid.uuid4())
        thread_config = {
            "configurable": {"thread_id": thread_id, "recursion_limit": recursion_limit}
        }

        print(
            f"🤖 \033[92mData Scout Agent is ready. Starting new thread: {thread_id}\033[0m"
        )

        iteration_count = 0
        # Convert max_iterations to int if it's a mock or other non-int type
        if max_iterations is not None:
            try:
                max_iterations = int(max_iterations)
            except (ValueError, TypeError):
                max_iterations = None

        while max_iterations is None or iteration_count < max_iterations:
            try:
                # Show progress before prompt
                try:
                    current_state = app.get_state(thread_config).values
                    samples_generated = current_state.get("samples_generated", 0)
                    total_target = current_state.get(
                        "total_samples_target", total_samples_target
                    )
                    if total_target > 0:
                        progress_pct = (samples_generated / total_target) * 100
                        progress_display = f"\r📊 Progress: {samples_generated}/{total_target} samples ({progress_pct:.1f}%)"
                    else:
                        progress_display = "\r📊 Progress: Calculating..."
                except:
                    progress_display = "\r📊 Progress: Initializing..."

                # Clear line and show progress
                print(progress_display, end="", flush=True)
                user_input = input("\n>>")
                if user_input.lower() in ["exit", "quit"]:
                    print("Exiting...")
                    break

                # Check if state for this thread already exists. If not, create it.
                thread_state = app.get_state(thread_config)
                # Check if there are messages in the state
                has_messages = False
                if thread_state and hasattr(thread_state, "values"):
                    state_values = thread_state.values
                    if isinstance(state_values, dict) and "messages" in state_values:
                        has_messages = len(state_values["messages"]) > 0

                if not thread_state or not has_messages:
                    print("   -> Creating new state for this thread.")

                    synthetic_budget = 0.2  # Default to 20%
                    for mission in mission_plan_info.get("missions", []):
                        if mission.get("name") == mission_name:
                            synthetic_budget = mission.get("synthetic_budget", 0.2)
                            break

                    initial_state = {
                        "run_id": thread_id,
                        "messages": [],
                        "progress": {},
                        "current_task": None,
                        "research_findings": [],
                        "pedigree_path": scout_config.writer.audit_trail_path,
                        "decision_history": [],
                        "tool_execution_failures": 0,
                        "research_attempts": 0,
                        "samples_generated": 0,
                        "total_samples_target": total_samples_target,
                        "current_mission": mission_name,
                        "synthetic_samples_generated": 0,
                        "research_samples_generated": 0,
                        "consecutive_failures": 0,
                        "last_action_status": "success",
                        "last_action_agent": "",
                        "synthetic_budget": synthetic_budget,
                    }
                    app.update_state(thread_config, initial_state)

                inputs = {"messages": [HumanMessage(content=user_input)]}

                for event in app.stream(inputs, config=thread_config):
                    for node_name, node_output in event.items():
                        if node_name == "__end__":
                            # Clear progress line before showing final message
                            print("\033[K", end="")
                            print("\n🏁 \033[92mAgent has finished the task.\033[0m")
                            continue

                        # Clear progress line before showing node execution
                        print("\033[K", end="")
                        print(f"\033[1m-> Executing Node: {node_name.upper()}\033[0m")
                        sys.stdout.flush()
                        if "messages" in node_output:
                            messages = node_output["messages"]
                            if messages:  # Check if messages list is not empty
                                message = messages[-1]
                                if (
                                    hasattr(message, "tool_calls")
                                    and message.tool_calls
                                ):
                                    # Debug the structure of tool_calls
                                    try:
                                        tool_calls_str = ", ".join(
                                            [
                                                f"{tc['name']}(...)"
                                                for tc in message.tool_calls
                                            ]
                                        )
                                        print(
                                            f"   - \033[94mDecided to call tools\033[0m: {tool_calls_str}"
                                        )
                                    except (TypeError, KeyError) as _e:
                                        # Handle case where tc is not a dict with 'name' key
                                        tool_calls_info = []
                                        for tc in message.tool_calls:
                                            if isinstance(tc, dict) and "name" in tc:
                                                tool_calls_info.append(
                                                    f"{tc['name']}(...)"
                                                )
                                            else:
                                                tool_calls_info.append(
                                                    f"{type(tc).__name__}(...)"
                                                )
                                        print(
                                            f"   - \033[94mDecided to call tools\033[0m: {', '.join(tool_calls_info)}"
                                        )
                                else:
                                    print(
                                        f"   - \033[92mResponded\033[0m: {message.content}"
                                    )
                            else:
                                print("   - \033[90mNo messages in response\033[0m")
                        else:
                            print(
                                f"   - \033[90mOutput: {str(node_output)[:300]}...\033[0m"
                            )
                        print("-" * 30)

                iteration_count += 1

            except KeyboardInterrupt:
                # Show final progress on interrupt
                try:
                    current_state = app.get_state(thread_config).values
                    samples_generated = current_state.get("samples_generated", 0)
                    total_target = current_state.get(
                        "total_samples_target", total_samples_target
                    )
                    if total_target > 0:
                        progress_pct = (samples_generated / total_target) * 100
                        print(
                            f"\r📊 Final Progress: {samples_generated}/{total_target} samples ({progress_pct:.1f}%)"
                        )
                    else:
                        print(f"\r📊 Final Progress: {samples_generated} samples")
                except:
                    print("\r📊 Final Progress: Unknown")
                print("\nExiting...")
                break
            except Exception as e:
                import traceback

                print(f"\n\033[91mAn error occurred: {e}\033[0m")
                print(f"Error type: {type(e).__name__}")
                print(f"Traceback: {traceback.format_exc()}")
                print("Please try again.")
    # The 'with' block ensures the connection is closed automatically.


# Create the CLI app that will be used by the entry point
cli_app = typer.Typer()
cli_app.command()(run)


# This function is the actual entry point called by the console script
def main():
    """Entry point for the aclarai-claimify-scout console script."""
    cli_app()


if __name__ == "__main__":
    main()
