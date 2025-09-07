# mission_runner.py
"""
Mission Runner for the Data Scout Agent.
This class orchestrates the mission execution and handles checkpointing for resumable missions.
"""

import json
import uuid
from typing import Dict, Any, Optional, List
import yaml
import sqlite3
from contextlib import closing

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.sqlite import SqliteSaver

from .graph import build_graph
from .checkpoint_db import create_connection, create_table


class MissionRunner:
    """A high-level orchestrator for Data Scout missions with checkpointing support."""

    def __init__(self, mission_plan_path: str = "settings/scout_mission.yaml"):
        """Initialize the MissionRunner with the mission plan."""
        self.mission_plan_path = mission_plan_path
        self.mission_plan = self._load_mission_plan()
        self.checkpoint_db_path = "checkpoints/mission_checkpoints.db"

        # --- FIX: Instantiate a persistent checkpointer and the graph ---
        self.checkpointer = SqliteSaver.from_conn_string(self.checkpoint_db_path)
        self.app = build_graph(checkpointer=self.checkpointer)

    def _load_mission_plan(self) -> Dict[str, Any]:
        """Load and parse the mission plan YAML file."""
        try:
            with open(self.mission_plan_path, "r") as f:
                content = f.read()
                if content.startswith("#"):
                    first_newline = content.find("\n")
                    if first_newline != -1:
                        content = content[first_newline + 1:]
                return yaml.safe_load(content)
        except Exception as e:
            print(f"Error loading mission plan: {e}")
            return {}

    def get_mission_names(self) -> List[str]:
        """Get a list of available mission names from the mission plan."""
        missions = self.mission_plan.get("missions", [])
        return [mission.get("name") for mission in missions]

    def get_mission_by_name(self, mission_name: str) -> Optional[Dict[str, Any]]:
        """Get a mission configuration by name."""
        missions = self.mission_plan.get("missions", [])
        for mission in missions:
            if mission.get("name") == mission_name:
                return mission
        return None

    def calculate_total_samples(self, mission_name: str) -> int:
        """Calculate the total target samples for a mission."""
        mission = self.get_mission_by_name(mission_name)
        if not mission:
            return 0

        target_size = mission.get("target_size", 0)
        goals = mission.get("goals", [])
        # Each goal is a characteristic with topics
        # Total samples = target_size * number_of_characteristics
        return target_size * len(goals)

    def start_new_mission(self, mission_name: str, recursion_limit: int = 1000) -> str:
        """
        Start a new mission with a fresh thread ID.
        
        Args:
            mission_name: Name of the mission to run
            recursion_limit: Maximum recursion limit for the graph
            
        Returns:
            Thread ID for the new mission
        """
        thread_id = str(uuid.uuid4())
        print(f"🤖 Starting new mission '{mission_name}' with thread ID: {thread_id}")
        
        # Get mission details
        mission = self.get_mission_by_name(mission_name)
        total_samples_target = self.calculate_total_samples(mission_name)
        
        # Initialize state
        synthetic_budget = mission.get("synthetic_budget", 0.2) if mission else 0.2
        
        initial_state = {
            "run_id": thread_id,
            "messages": [],
            "progress": {},
            "current_task": None,
            "research_findings": [],
            "pedigree_path": "examples/PEDIGREE.md",  # Default path
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
        
        # --- FIX: Save initial state using the app's checkpointer ---
        thread_config = {"configurable": {"thread_id": thread_id}}
        self.app.update_state(thread_config, initial_state)
        
        return thread_id

    def resume_mission(self, thread_id: str, recursion_limit: int = 1000) -> bool:
        """
        Resume a mission from a checkpoint.
        
        Args:
            thread_id: Thread ID of the mission to resume
            
        Returns:
            True if mission was successfully resumed, False otherwise
        """
        print(f"🔄 Resuming mission with thread ID: {thread_id}")
        
        # --- FIX: Check for checkpoint existence using the checkpointer ---
        thread_config = {"configurable": {"thread_id": thread_id}}
        checkpoint = self.checkpointer.get(thread_config)

        if checkpoint is None:
            print(f"❌ No checkpoint found for thread ID: {thread_id}")
            return False

        print("   -> Found existing checkpoint. Mission can be resumed.")
        return True

    def run_mission_step(self, thread_id: str, user_input: str, recursion_limit: int = 1000) -> Dict[str, Any]:
        """
        Run a single step of the mission.
        
        Args:
            thread_id: Thread ID of the mission
            user_input: User input for this step
            recursion_limit: Maximum recursion limit for the graph
            
        Returns:
            Dictionary with results of the step execution
        """
        # --- FIX: Use the persistent app and checkpointer ---
        thread_config = {
            "configurable": {"thread_id": thread_id, "recursion_limit": recursion_limit}
        }
        
        # Prepare inputs
        inputs = {"messages": [HumanMessage(content=user_input)]}
        
        # Run the graph - the checkpointer will handle state loading and saving automatically
        results = []
        for event in self.app.stream(inputs, config=thread_config):
            for node_name, node_output in event.items():
                results.append({
                    "node_name": node_name,
                    "node_output": node_output
                })

                if node_name == "__end__":
                    print("🏁 Agent has finished the task.")

        # Get the latest state from the checkpointer
        current_state = self.app.get_state(thread_config).values

        return {
            "results": results,
            "current_state": current_state
        }

    def get_progress(self, thread_id: str) -> Dict[str, Any]:
        """
        Get the current progress of a mission.
        
        Args:
            thread_id: Thread ID of the mission
            
        Returns:
            Dictionary with progress information
        """
        # --- FIX: Load state directly from the checkpointer ---
        thread_config = {"configurable": {"thread_id": thread_id}}
        state = self.checkpointer.get(thread_config)

        if state:
            try:
                # The state object from the checkpointer has a 'values' attribute
                # which contains the actual state dictionary.
                loaded_state = state.values
                samples_generated = loaded_state.get("samples_generated", 0)
                total_target = loaded_state.get("total_samples_target", 0)

                return {
                    "samples_generated": samples_generated,
                    "total_target": total_target,
                    "progress_pct": (samples_generated / total_target) * 100 if total_target > 0 else 0
                }
            except Exception as e:
                print(f"Error getting progress from checkpoint: {e}")
        
        # Default values if no state found
        return {
            "samples_generated": 0,
            "total_target": 0,
            "progress_pct": 0
        }