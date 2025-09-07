# mission_runner.py
"""
Mission Runner for the Data Scout Agent.
This class orchestrates the mission execution and handles checkpointing for resumable missions.
"""

import json
import uuid
from typing import Dict, Any, Optional, List
import yaml

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.sqlite import SqliteSaver

from .graph import build_graph
from .checkpoint_db import create_connection, create_table, save_mission_state, load_mission_state


class MissionRunner:
    """A high-level orchestrator for Data Scout missions with checkpointing support."""

    def __init__(self, mission_plan_path: str = "settings/scout_mission.yaml"):
        """Initialize the MissionRunner with the mission plan."""
        self.mission_plan_path = mission_plan_path
        self.mission_plan = self._load_mission_plan()
        self.checkpoint_db_path = "checkpoints/mission_checkpoints.db"

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
        
        # Save initial state to checkpoint
        self._save_checkpoint(thread_id, initial_state)
        
        return thread_id

    def resume_mission(self, thread_id: str, recursion_limit: int = 1000) -> bool:
        """
        Resume a mission from a checkpoint.
        
        Args:
            thread_id: Thread ID of the mission to resume
            recursion_limit: Maximum recursion limit for the graph
            
        Returns:
            True if mission was successfully resumed, False otherwise
        """
        print(f"🔄 Resuming mission with thread ID: {thread_id}")
        
        # Connect to checkpoint database
        conn = create_connection(self.checkpoint_db_path)
        if conn is None:
            print(f"❌ Error: Could not connect to checkpoint database at {self.checkpoint_db_path}")
            return False
            
        # Load saved state
        loaded_state_str = load_mission_state(conn, thread_id)
        if not loaded_state_str:
            print(f"❌ No checkpoint found for thread ID: {thread_id}")
            conn.close()
            return False
            
        try:
            # Parse the saved state
            loaded_state = json.loads(loaded_state_str)
            print("   -> Loaded state from checkpoint.")
            
            conn.close()
            return True
        except Exception as e:
            print(f"❌ Error resuming mission: {e}")
            conn.close()
            return False

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
        import sqlite3
        from contextlib import closing
        
        # Create a proper SqliteSaver checkpointer for this step
        with closing(sqlite3.connect(":memory:", check_same_thread=False)) as conn:
            checkpointer = SqliteSaver(conn)
            app = build_graph(checkpointer=checkpointer)
            
            # Create thread config
            thread_config = {
                "configurable": {"thread_id": thread_id, "recursion_limit": recursion_limit}
            }
            
            # Load state from checkpoint if it exists
            conn_db = create_connection(self.checkpoint_db_path)
            if conn_db is not None:
                loaded_state_str = load_mission_state(conn_db, thread_id)
                if loaded_state_str:
                    try:
                        loaded_state = json.loads(loaded_state_str)
                        # Update the graph state with loaded state
                        app.update_state(thread_config, loaded_state)
                        print(f"   -> Loaded state from checkpoint for thread {thread_id}")
                    except Exception as e:
                        print(f"⚠️  Warning: Could not load state from checkpoint: {e}")
                else:
                    print(f"   -> No existing checkpoint found for thread {thread_id} (this is normal for new missions)")
                conn_db.close()
            
            # Prepare inputs
            inputs = {"messages": [HumanMessage(content=user_input)]}
            
            # Run the graph
            results = []
            for event in app.stream(inputs, config=thread_config):
                for node_name, node_output in event.items():
                    results.append({
                        "node_name": node_name,
                        "node_output": node_output
                    })
                    
                    if node_name == "__end__":
                        print("🏁 Agent has finished the task.")
            
            # Save state after each iteration
            try:
                current_state = app.get_state(thread_config).values
                self._save_checkpoint(thread_id, current_state)
            except Exception as e:
                print(f"⚠️  Warning: Could not save state to checkpoint: {e}")
                current_state = {}
            
            return {
                "results": results,
                "current_state": current_state
            }

    def _save_checkpoint(self, thread_id: str, state: Dict[str, Any]) -> None:
        """
        Save the current mission state to the checkpoint database.
        
        Args:
            thread_id: Thread ID of the mission
            state: Current state to save
        """
        conn = create_connection(self.checkpoint_db_path)
        if conn is None:
            print(f"❌ Error: Could not connect to checkpoint database at {self.checkpoint_db_path}")
            return
            
        create_table(conn)
        save_mission_state(conn, thread_id, json.dumps(state))
        conn.close()

    def get_progress(self, thread_id: str) -> Dict[str, Any]:
        """
        Get the current progress of a mission.
        
        Args:
            thread_id: Thread ID of the mission
            
        Returns:
            Dictionary with progress information
        """
        # Load state from checkpoint
        conn = create_connection(self.checkpoint_db_path)
        if conn is not None:
            loaded_state_str = load_mission_state(conn, thread_id)
            if loaded_state_str:
                try:
                    loaded_state = json.loads(loaded_state_str)
                    samples_generated = loaded_state.get("samples_generated", 0)
                    total_target = loaded_state.get("total_samples_target", 0)
                    
                    conn.close()
                    return {
                        "samples_generated": samples_generated,
                        "total_target": total_target,
                        "progress_pct": (samples_generated / total_target) * 100 if total_target > 0 else 0
                    }
                except Exception as e:
                    print(f"Error getting progress: {e}")
            conn.close()
        
        # Default values if no state found
        return {
            "samples_generated": 0,
            "total_target": 0,
            "progress_pct": 0
        }