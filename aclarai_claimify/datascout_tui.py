#!/usr/bin/env python3
"""
Data Scout Agent TUI - A modern terminal interface for sample generation.
"""

import asyncio
import os
import sys
from datetime import datetime
from typing import Optional
import io
import contextlib

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container
from textual.widgets import Header, Footer
import typer

from .scout.tui.components.stats_header import StatsHeader, GenerationStats
from .scout.tui.components.progress_panel import ProgressPanel
from .scout.tui.components.mission_panel import MissionPanel
from .scout.tui.components.conversation_panel import ConversationPanel
from .scout.tui.components.mission_selector import MissionSelector
from .scout.tui.agent_process_manager import AgentProcessManager
from .scout.tui.agent_output_parser import (
    AgentOutputParser,
    ProgressUpdate,
    NewMessage,
    ErrorMessage,
)
from .config import load_claimify_config
from .scout.config import load_scout_config
from .scout.scout_utils import get_mission_details_from_file


class DataScoutTUI(App):
    """Main TUI application."""

    # Get the absolute path to the CSS file
    @property
    def CSS_PATH(self) -> str:
        import os
        return os.path.join(os.path.dirname(__file__), "scout", "tui", "styles.css")

    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("ctrl+c", "quit", "Quit", show=False),
    ]

    def __init__(
        self,
        mission_plan_path: str,
        log_file: Optional[str] = None,
        debug: bool = False,
        scout_config_path: Optional[str] = None,
    ):
        super().__init__()
        self.mission_plan_path = mission_plan_path
        self.log_file = log_file
        self.debug_enabled = debug
        self.scout_config_path = scout_config_path
        if debug:
            print("Writing debug log to /tmp/tui_debug.log")
        self.stats = GenerationStats()
        self.agent_process_manager: Optional[AgentProcessManager] = None
        self.agent_output_parser = AgentOutputParser()
        self.log_handle: Optional[io.TextIOWrapper] = None
    
    def debug_log(self, message: str):
        """Log debug message if debug mode is enabled."""
        if self.debug_enabled:
            try:
                with open("/tmp/tui_debug.log", "a") as f:
                    f.write(f"{message}\n")
            except Exception:
                pass  # Ignore debug logging errors

    def compose(self) -> ComposeResult:
        yield Header()
        yield Footer()

    def on_mount(self):
        """Start the mission selector when the app mounts."""
        mission_details = get_mission_details_from_file(self.mission_plan_path)
        if not mission_details or not mission_details["mission_names"]:
            self.exit(message=f"❌ No missions found in {self.mission_plan_path}")
            return

        self.push_screen(
            MissionSelector(mission_details["mission_names"]), self.on_mission_selected
        )

    def on_mission_selected(self, mission_name: str):
        """Called when a mission is selected from the modal."""
        if not mission_name:
            self.exit(message="❌ No mission selected")
            return

        # Open log file if specified
        if self.log_file:
            try:
                self.log_handle = open(self.log_file, "a")
                self.log_handle.write(
                    f"\n=== Data Scout TUI Session Started at {datetime.now().isoformat()} ===\n"
                )
                self.log_handle.flush()
            except Exception as e:
                # Log error but continue
                pass

        mission_details = get_mission_details_from_file(self.mission_plan_path)
        total_samples_target = mission_details["mission_targets"].get(mission_name, 1200)
        self.stats.target = total_samples_target

        scout_config = load_scout_config(self.scout_config_path)
        recursion_limit = scout_config.get("recursion_per_sample", 27)

        self.agent_process_manager = AgentProcessManager(
            mission_name=mission_name,
            recursion_limit=recursion_limit,
        )

        self.stats_header = StatsHeader()
        self.progress_panel = ProgressPanel()
        self.mission_panel = MissionPanel(
            self.mission_plan_path,
            mission_name,
            total_samples_target,
            scout_config,
        )
        self.conversation = ConversationPanel(debug=self.debug_enabled)

        # Mount the main UI components
        self.debug_log(f"MOUNTING COMPONENTS:")
        self.debug_log(f"  - stats_header: {self.stats_header}")
        self.debug_log(f"  - progress_panel: {self.progress_panel}")
        self.debug_log(f"  - mission_panel: {self.mission_panel}")
        self.debug_log(f"  - conversation: {self.conversation}")
        
        self.mount(
            Container(
                self.stats_header,
                Container(
                    self.progress_panel,
                    self.mission_panel,
                    id="main-container",
                ),
                self.conversation,
            )
        )

        # Add initial messages to the conversation
        if self.log_file:
            self.conversation.add_message("info", f"Logging to: {self.log_file}")
        self.conversation.add_message("info", "Starting Data Scout Agent...")
        self.conversation.add_message("info", f"Mission: {mission_name}")
        self.conversation.add_message("info", f"Target: {total_samples_target} samples")

        self.run_worker(self._run_agent(), name="agent")

    async def _run_agent(self):
        """Run the Data Scout Agent as a subprocess and parse its output."""
        self.stats.started_at = datetime.now()
        try:
            process = await self.agent_process_manager.start()
            while True:
                line_bytes = await process.stdout.readline()
                if not line_bytes:
                    break
                line = line_bytes.decode("utf-8").strip()
                if line:
                    # Log to file if log file is specified
                    if self.log_handle:
                        try:
                            timestamp = datetime.now().strftime("%H:%M:%S")
                            self.log_handle.write(f"[{timestamp}] {line}\n")
                            self.log_handle.flush()
                        except Exception:
                            pass  # Continue if log writing fails
                    
                    # Parse the line and handle events
                    events = list(self.agent_output_parser.parse_line(line))
                    if events:
                        with open("/tmp/tui_debug.log", "a") as f:
                            f.write(f"GENERATED {len(events)} EVENTS for line: {line[:100]}...\n")
                    for event in events:
                        self._handle_agent_event(event)
        except Exception as e:
            self.conversation.add_message("error", f"Agent error: {str(e)}")
        finally:
            self.conversation.add_message("info", "Agent process completed")
            # Start a 5-second timer to auto-close the TUI
            self.set_timer(5.0, self._auto_close_after_completion)

    def _handle_agent_event(self, event):
        """Handle events from the agent output parser."""
        # Debug: Log all events being handled
        self.debug_log(f"HANDLING EVENT: {type(event).__name__} - {event}")
        
        if isinstance(event, ProgressUpdate):
            self.stats.completed = event.completed
            self.stats.target = event.target
            self.stats_header.update_stats(self.stats)
            self.progress_panel.update_progress(self.stats)
            
            # Update mission status when progress is made
            if event.completed > 0:
                self.mission_panel.update_status(f"Generating... ({event.completed}/{event.target})")
            
            self.debug_log(f"PROGRESS UPDATE: {event.completed}/{event.target}")
        elif isinstance(event, NewMessage):
            self.debug_log(f"NEW MESSAGE EVENT: {event.role} -> {event.content[:100]}...")
            self.conversation.add_message(event.role, event.content)
            
            # Update mission status based on message content
            import re
            
            # Check for Graph Router patterns
            if "Graph Router: Routing to" in event.content:
                route_match = re.search(r"Graph Router: Routing to (\w+)", event.content)
                if route_match:
                    route_name = route_match.group(1)
                    if route_name.lower() == "end":
                        self.mission_panel.update_status("Sample Completed")
                    elif route_name.lower() == "archive":
                        self.mission_panel.update_status("Archiving Sample...")
                    elif route_name.lower() == "fitness":
                        self.mission_panel.update_status("Checking Fitness...")
                    elif route_name.lower() == "synthetic":
                        self.mission_panel.update_status("Generating Synthetic...")
                    else:
                        self.mission_panel.update_status(f"Routing to {route_name}...")
            
            # Check for node execution patterns like "🔍 RESEARCH NODE" or "🎨 SYNTHETIC NODE"
            elif ("NODE" in event.content and "🔍" in event.content) or ("SYNTHETIC NODE" in event.content and "🎨" in event.content):
                if "🎨 SYNTHETIC NODE" in event.content:
                    self.mission_panel.update_status("Generating Synthetic Content...")
                else:
                    node_match = re.search(r"🔍\s+(\w+)\s+NODE", event.content)
                    if node_match:
                        node_name = node_match.group(1)
                        self.mission_panel.update_status(f"Working on {node_name}...")
            
            # Check for agent starting work (move from Initializing)
            elif ("▶ Iteration" in event.content or 
                  "🔧 Tool calls:" in event.content or
                  "📊 CONTEXT:" in event.content) and self.mission_panel.current_status == "Initializing...":
                self.mission_panel.update_status("Working...")
            
            # Check for routing to END (fallback pattern)
            elif "Routing to END" in event.content or "Decided on 'end'" in event.content:
                self.mission_panel.update_status("Sample Completed")
            
            # Check for sample archival and add to recent samples
            elif "sample #" in event.content.lower() and "archived" in event.content.lower():
                sample_match = re.search(r"#(\d+)", event.content)
                if sample_match:
                    sample_num = int(sample_match.group(1))
                    
                    # Extract completion percentage if available
                    pct_match = re.search(r"\((\d+\.?\d*)% complete\)", event.content)
                    completion_pct = pct_match.group(1) if pct_match else "?"
                    
                    # Try to extract sample excerpt from the most recent content
                    sample_excerpt = self._extract_recent_sample_excerpt()
                    
                    if sample_excerpt:
                        description = f"{sample_excerpt} ({completion_pct}%)"
                    else:
                        description = f"Archived ({completion_pct}% complete)"
                    
                    self.progress_panel.add_sample(sample_num, description)
        elif isinstance(event, ErrorMessage):
            self.debug_log(f"ERROR MESSAGE EVENT: {event.message[:100]}...")
            self.conversation.add_message("error", event.message)
            self.stats.errors += 1
            self.stats_header.update_stats(self.stats)

    def action_quit(self):
        """Quit the application."""
        if self.agent_process_manager:
            self.agent_process_manager.terminate()
        
        # Close log file if open
        if self.log_handle:
            try:
                self.log_handle.write(
                    f"\n=== Data Scout TUI Session Ended at {datetime.now().isoformat()} ===\n"
                )
                self.log_handle.close()
            except Exception:
                pass  # Ignore errors during cleanup
        
        self.exit()
    
    def _auto_close_after_completion(self):
        """Auto-close the TUI after agent process completion with a countdown."""
        self.conversation.add_message("info", "TUI will close in 5 seconds... (Press 'q' to exit immediately)")
        # Set another timer to actually close
        self.set_timer(5.0, self.action_quit)
        
    def _extract_recent_sample_excerpt(self) -> str:
        """Extract a content excerpt from the most recently archived sample.
        
        Scans the entire conversation history to find the most recent
        '## Retrieved Content (Markdown)' or '## Generated Content (Synthetic)' section
        and extracts content between that marker and the end boundary.
        
        Supports both boundary patterns:
        - '📝 ---  END RAW LLM RESPONSE  ---'
        - '📝 ---  END LLM RESPONSE  ---'
        """
        try:
            # Scan the entire conversation history (not just recent messages)
            messages = self.conversation.messages
            self.debug_log(f"EXCERPT EXTRACTION: Scanning {len(messages)} total messages for content boundaries")
            
            # Find the most recent content marker (either retrieved or synthetic)
            content_start = -1
            content_message_idx = -1
            full_content = ""
            content_type = "unknown"
            
            # Search backwards through all messages to find the latest content
            for i in range(len(messages) - 1, -1, -1):
                message = messages[i]
                content = message.get('content', '')
                
                # Check for retrieved content first
                if "## Retrieved Content (Markdown)" in content:
                    content_start = content.find("## Retrieved Content (Markdown)")
                    content_message_idx = i
                    full_content = content
                    content_type = "retrieved"
                    self.debug_log(f"FOUND Retrieved Content marker in message {i} at position {content_start}")
                    break
                # Also check for synthetic content
                elif "## Generated Content (Synthetic)" in content:
                    content_start = content.find("## Generated Content (Synthetic)")
                    content_message_idx = i
                    full_content = content
                    content_type = "synthetic"
                    self.debug_log(f"FOUND Generated Content marker in message {i} at position {content_start}")
                    break
            
            if content_start == -1:
                self.debug_log("NO Content marker found in conversation history")
                return None
            
            # Extract content from the marker onwards (handle both types)
            if content_type == "retrieved":
                marker_text = "## Retrieved Content (Markdown)"
            else:  # synthetic
                marker_text = "## Generated Content (Synthetic)"
            
            content_after_marker = full_content[content_start + len(marker_text):]
            
            # Look for the end boundary in the same message or subsequent messages
            # Support both "END RAW LLM RESPONSE" and "END LLM RESPONSE" patterns
            end_boundary_raw = "📝 ---  END RAW LLM RESPONSE  ---"
            end_boundary_simple = "📝 ---  END LLM RESPONSE  ---"
            sample_content = content_after_marker
            end_boundary = None
            
            # Check if either end boundary is in the same message
            if end_boundary_raw in content_after_marker:
                end_boundary = end_boundary_raw
                sample_content = content_after_marker.split(end_boundary)[0]
                self.debug_log(f"FOUND RAW end boundary in same message, extracted {len(sample_content)} chars")
            elif end_boundary_simple in content_after_marker:
                end_boundary = end_boundary_simple
                sample_content = content_after_marker.split(end_boundary)[0]
                self.debug_log(f"FOUND simple end boundary in same message, extracted {len(sample_content)} chars")
            elif end_boundary is None:
                # Search subsequent messages for either end boundary
                self.debug_log("End boundary not in same message, searching subsequent messages")
                accumulated_content = [content_after_marker]
                
                for i in range(content_message_idx + 1, len(messages)):
                    next_message = messages[i]
                    next_content = next_message.get('content', '')
                    
                    if end_boundary_raw in next_content:
                        end_boundary = end_boundary_raw
                        content_before_boundary = next_content.split(end_boundary)[0]
                        accumulated_content.append(content_before_boundary)
                        self.debug_log(f"FOUND RAW end boundary in message {i}, total accumulated content")
                        break
                    elif end_boundary_simple in next_content:
                        end_boundary = end_boundary_simple
                        content_before_boundary = next_content.split(end_boundary)[0]
                        accumulated_content.append(content_before_boundary)
                        self.debug_log(f"FOUND simple end boundary in message {i}, total accumulated content")
                        break
                    else:
                        # Add the entire message content to our sample
                        accumulated_content.append(next_content)
                        self.debug_log(f"Adding full message {i} to sample content")
                
                sample_content = '\n'.join(accumulated_content)
            
            # Clean up and extract meaningful excerpt from the sample content
            sample_content = sample_content.strip()
            self.debug_log(f"RAW SAMPLE CONTENT length: {len(sample_content)} chars")
            
            if not sample_content:
                self.debug_log("Empty sample content after extraction")
                return None
            
            # Handle cache reference tokens
            if "[CACHE_REFERENCE:" in sample_content:
                self.debug_log("Sample contains cache reference")
                # Try to extract meaningful content around the cache reference
                lines = sample_content.split('\n')
                meaningful_lines = []
                
                for line in lines:
                    line = line.strip()
                    if (line and 
                        not line.startswith('[CACHE_REFERENCE') and
                        not line.startswith('`') and 
                        not line.startswith('#') and 
                        len(line) > 20):
                        meaningful_lines.append(line)
                        if len(meaningful_lines) >= 2:
                            break
                
                if meaningful_lines:
                    excerpt = ' '.join(meaningful_lines)
                    if len(excerpt) > 120:
                        excerpt = excerpt[:120] + "..."
                    self.debug_log(f"EXTRACTED cache excerpt: {excerpt[:50]}...")
                    return excerpt
                else:
                    return f"{content_type.title()} sample (from cache)"
            
            # Extract meaningful content lines
            lines = sample_content.split('\n')
            clean_lines = []
            
            for line in lines:
                line = line.strip()
                # Skip markdown formatting, empty lines, timestamps, and other noise
                if (line and 
                    not line.startswith('`') and 
                    not line.startswith('#') and 
                    not line.startswith('[') and
                    not line.startswith('---') and
                    not line.startswith('**') and
                    not line.startswith('*') and
                    not line.startswith('|') and  # Skip table formatting
                    len(line) > 25):  # Only meaningful content
                    clean_lines.append(line)
                    if len(clean_lines) >= 3:  # Get first 3 meaningful lines
                        break
            
            if clean_lines:
                excerpt = ' '.join(clean_lines)
                # Truncate to reasonable display length
                if len(excerpt) > 120:
                    excerpt = excerpt[:120] + "..."
                self.debug_log(f"FINAL EXCERPT: {excerpt[:50]}...")
                return excerpt
            else:
                self.debug_log("No clean meaningful lines found in sample content")
                return "Sample content (formatting only)"
                
        except Exception as e:
            self.debug_log(f"Error extracting sample excerpt: {e}")
            import traceback
            self.debug_log(f"Exception traceback: {traceback.format_exc()}")
            return None


cli_app = typer.Typer()


@cli_app.command()
def generate(
    mission: Optional[str] = typer.Argument(None, help="Path to mission plan YAML"),
    log: Optional[str] = typer.Option(
        None, "--log", help="Log file to save terminal output for debugging"
    ),
    debug: bool = typer.Option(
        False, "--debug", help="Enable debug logging to /tmp/tui_debug.log"
    ),
    config: Optional[str] = typer.Option(
        None, "--config", "-c", help="Path to scout configuration file (defaults to scout_config.yaml)"
    ),
):
    """Start the Data Scout Agent TUI for sample generation."""
    mission_plan_path = mission or "settings/mission_config.yaml"
    if not os.path.exists(mission_plan_path):
        typer.echo(f"❌ Mission file not found: {mission_plan_path}", err=True)
        raise typer.Exit(1)

    app = DataScoutTUI(mission_plan_path=mission_plan_path, log_file=log, debug=debug, scout_config_path=config)
    app.run()


if __name__ == "__main__":
    cli_app()