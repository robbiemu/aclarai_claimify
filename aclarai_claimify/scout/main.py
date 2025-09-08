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
import json

from langgraph.checkpoint.sqlite import SqliteSaver

# Force unbuffered output for live progress updates
# Only reconfigure if stdout supports it (not redirected to StringIO in TUI)
try:
    sys.stdout.reconfigure(line_buffering=True)
except AttributeError:
    # stdout doesn't support reconfigure (e.g., when redirected to StringIO in TUI)
    pass

from langchain_core.messages import HumanMessage

from .graph import build_graph
from .tools import _HAVE_LIBCRAWLER
from ..config import load_claimify_config
from .mission_runner import MissionRunner


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


def run_agent_process(
    mission_name: str,
    max_iterations: Optional[int] = None,
    recursion_limit: Optional[int] = None,
    resume_from: Optional[str] = None,
    non_interactive: bool = False,
):
    """Main function to run the CLI interactive loop for the scout agent."""
    with SqliteSaver.from_conn_string(
        "checkpoints/mission_checkpoints.db"
    ) as checkpointer:
        try:
            app = build_graph(checkpointer=checkpointer)
            mission_runner = MissionRunner(
                checkpointer=checkpointer,
                app=app,
                mission_plan_path="settings/scout_mission.yaml",
            )

            available_missions = mission_runner.get_mission_names()

            if not mission_name:
                print("❌ Error: No mission specified.")
                print("Available missions are:")
                for name in available_missions:
                    print(f"  - {name}")
                print("\n💡 Please specify one with the --mission flag.")
                return

            if mission_name not in available_missions:
                print(f"❌ Error: Mission '{mission_name}' not found in mission plan.")
                return

            setup_observability()
            sys.stdout.flush()  # Ensure setup messages appear immediately

            if not _HAVE_LIBCRAWLER:
                print("\033[93mWarning: `libcrawler` is not installed.\033[0m")
                print("Crawler functionality will be disabled.")
                print("To enable, run: pip install -e .[crawler]\n")
                sys.stdout.flush()

            # Load mission plan to get target sample counts
            total_samples_target = mission_runner.calculate_total_samples(
                mission_name
            )
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
                    STEPS_PER_SAMPLE
                    * (total_samples_target + RETRY_MARGIN)
                    * OVERHEAD_FACTOR
                )
                recursion_limit = calculated_limit
                print(f"🔐 Calculated recursion limit: {recursion_limit}")
            else:
                print(f"🔐 Using CLI-provided recursion limit: {recursion_limit}")
            sys.stdout.flush()
            # --- END: Dynamic Recursion Limit Calculation ---

            # Start or resume mission
            if resume_from:
                thread_id = resume_from
                success = mission_runner.resume_mission(thread_id, recursion_limit)
                if not success:
                    print(
                        f"❌ Error: Could not resume mission with thread ID: {thread_id}"
                    )
                    return
                print(f"🔄 Resuming mission from thread: {thread_id}")
            else:
                thread_id = mission_runner.start_new_mission(
                    mission_name, recursion_limit
                )
                print(
                    f"🤖 \033[92mData Scout Agent is ready. Starting new thread: {thread_id}\033[0m"
                )

            if non_interactive:
                # Non-interactive path for TUI
                initial_prompt = sys.stdin.readline().strip()
                if initial_prompt:
                    mission_runner.run_full_mission(thread_id, initial_prompt, recursion_limit)
                else:
                    print("Error: Non-interactive mode requires an initial prompt via stdin.")
            else:
                # Keep the existing interactive loop for debugging
                # This loop calls mission_runner.run_mission_step() on each iteration
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
                        progress_info = mission_runner.get_progress(thread_id)
                        samples_generated = progress_info["samples_generated"]
                        total_target = progress_info["total_target"]
                        progress_pct = progress_info["progress_pct"]

                        if total_target > 0:
                            progress_display = f"\r📊 Progress: {samples_generated}/{total_target} samples ({progress_pct:.1f}%)"
                        else:
                            progress_display = "\r📊 Progress: Calculating..."

                        # Clear line and show progress
                        print(progress_display, end="", flush=True)
                        user_input = input(">>")
                        if user_input.lower() in ["exit", "quit"]:
                            print("Exiting...")
                            break

                        # Run mission step
                        step_result = mission_runner.run_mission_step(
                            thread_id, user_input, recursion_limit
                        )

                        # Display results
                        for result in step_result["results"]:
                            node_name = result["node_name"]
                            node_output = result["node_output"]

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
                        progress_info = mission_runner.get_progress(thread_id)
                        samples_generated = progress_info["samples_generated"]
                        total_target = progress_info["total_target"]
                        progress_pct = progress_info["progress_pct"]

                        if total_target > 0:
                            print(
                                f"\r📊 Final Progress: {samples_generated}/{total_target} samples ({progress_pct:.1f}%)"
                            )
                        else:
                            print(f"\r📊 Final Progress: {samples_generated} samples")
                        print("\nExiting...")
                        break
                    except Exception as e:
                        import traceback

                        print(f"\n\033[91mAn error occurred: {e}\033[0m")
                        print(f"Error type: {type(e).__name__}")
                        print(f"Traceback: {traceback.format_exc()}")
                        print("Please try again.")
        finally:
            print("\nConnection closed")



# This function is the actual entry point called by the console script
def run(
    mission_name: Optional[str] = typer.Option(
        None, "--mission", "-m", help="The name of the mission to run."
    ),
    max_iterations: int = None,
    recursion_limit: int = typer.Option(None, "--recursion-limit"),
    resume_from: Optional[str] = typer.Option(
        None, "--resume-from", help="The thread ID to resume from."
    ),
    non_interactive: bool = typer.Option(False, "--non-interactive", help="Run in non-interactive mode for automated execution."),
):
    config = load_claimify_config()
    scout_config = config.scout_agent
    if not scout_config:
        print("Error: `scout_agent` section not found in configuration.")
        return

    run_agent_process(
        mission_name=mission_name,
        max_iterations=max_iterations,
        recursion_limit=recursion_limit,
        resume_from=resume_from,
        non_interactive=non_interactive,
    )


# Create the CLI app that will be used by the entry point
cli_app = typer.Typer()
cli_app.command()(run)


def main():
    """Entry point for the aclarai-claimify-scout console script.""" 
    cli_app()


if __name__ == "__main__":
    main()