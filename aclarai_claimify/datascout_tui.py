#!/usr/bin/env python3
"""
Data Scout Agent TUI - A modern terminal interface for sample generation.

This replaces the bash script with a real-time TUI showing progress and agent conversation.
"""

import asyncio
import os
import sys
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import Optional
import io
import contextlib

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, Vertical
from textual.widgets import Header, Footer, Static, ProgressBar, Log
import typer

# Import config without triggering patches yet
from aclarai_claimify.config import ClaimifyConfig, load_claimify_config
from aclarai_claimify.datascout_mission_selector import MissionSelector


@dataclass
class GenerationStats:
    """Statistics about the sample generation process."""

    target: int = 1200
    completed: int = 0
    errors: int = 0
    started_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    agent_activities: int = 0  # Count of agent actions/responses

    @property
    def elapsed_seconds(self) -> float:
        if not self.started_at:
            return 0
        return (
            self.updated_at or datetime.now()
        ).timestamp() - self.started_at.timestamp()

    @property
    def samples_per_second(self) -> float:
        if self.elapsed_seconds <= 0:
            return 0
        return self.completed / self.elapsed_seconds

    @property
    def eta_seconds(self) -> float:
        if self.samples_per_second <= 0:
            return 0
        return (self.target - self.completed) / self.samples_per_second

    @property
    def eta_human(self) -> str:
        if self.completed >= self.target:
            return "Done"
        if self.samples_per_second <= 0 or self.elapsed_seconds < 60:
            return "Calculating..."

        eta_delta = timedelta(seconds=int(self.eta_seconds))
        if eta_delta.total_seconds() < 60:
            return f"{int(eta_delta.total_seconds())}s"
        elif eta_delta.total_seconds() < 3600:
            return f"{int(eta_delta.total_seconds() / 60)}m"
        else:
            hours = int(eta_delta.total_seconds() / 3600)
            minutes = int((eta_delta.total_seconds() % 3600) / 60)
            return f"{hours}h {minutes}m"

    @property
    def elapsed_human(self) -> str:
        if self.elapsed_seconds < 60:
            return f"{int(self.elapsed_seconds)}s"
        elif self.elapsed_seconds < 3600:
            return f"{int(self.elapsed_seconds / 60)}m"
        else:
            hours = int(self.elapsed_seconds / 3600)
            minutes = int((self.elapsed_seconds % 3600) / 60)
            return f"{hours}h {minutes}m"


class StatsHeader(Static):
    """Header showing generation statistics."""

    def __init__(self):
        super().__init__(id="stats-header")
        self.stats = GenerationStats()

    def update_stats(self, stats: GenerationStats):
        self.stats = stats
        progress_pct = int(100 * stats.completed / max(1, stats.target))

        status_line = (
            f"📊 Progress: {stats.completed}/{stats.target} ({progress_pct}%) | "
            f"⏱️  Elapsed: {stats.elapsed_human} | "
            f"🎯 ETA: {stats.eta_human} | "
            f"🤖 Activity: {stats.agent_activities} | "
            f"❌ Errors: {stats.errors}"
        )
        self.update(status_line)


class ProgressPanel(Static):
    """Left panel with progress bar and recent samples."""

    def __init__(self):
        super().__init__(id="progress-panel")
        self.progress_bar = None

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Static("📈 Generation Progress", classes="panel-title")
            self.progress_bar = ProgressBar(
                total=100, show_percentage=True, id="main-progress"
            )
            yield self.progress_bar
            yield Static("Recent samples will appear here...", id="recent-samples")

    def update_progress(self, stats: GenerationStats):
        if self.progress_bar:
            progress_pct = int(100 * stats.completed / max(1, stats.target))
            self.progress_bar.update(progress=progress_pct)


class MissionPanel(Static):
    """Right panel with mission status and details."""

    def __init__(
        self,
        mission_path: str,
        mission_name: str,
        total_samples_target: int,
        config: "ClaimifyConfig",
    ):
        super().__init__(id="mission-panel")
        self.mission_path = mission_path
        self.mission_name = mission_name
        self.total_samples_target = total_samples_target
        self.config = config

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Static("🎯 Mission Status", classes="panel-title")
            mission_info = (
                f"Mission: {self.mission_name}\n"
                f"Target: {self.total_samples_target} samples\n"
                f"Model: {self.config.scout_agent.mission_plan.nodes[0].model}\n"
                f"Provider: {self.config.scout_agent.search_provider}\n"
                f"Status: Initializing..."
            )
            yield Static(mission_info, id="mission-info")


class ConversationPanel(Log):
    """Bottom panel showing agent conversation with auto-scroll."""

    def __init__(self):
        super().__init__(auto_scroll=True, id="conversation")
        # Add initial placeholder
        self.write("🤖 Data Scout Agent TUI")
        self.write("Waiting for agent to start...")
        self._initialized = False

    def add_message(self, role: str, content: str):
        """Add a message to the conversation log with role-based coloring."""
        # Clear placeholder on first real message
        if not self._initialized:
            self.clear()
            self._initialized = True

        colors = {
            "system": "bright_black",
            "user": "cyan",
            "assistant": "green",
            "tool": "yellow",
            "debug": "magenta",
            "info": "blue",
        }
        color = colors.get(role, "white")
        timestamp = datetime.now().strftime("%H:%M:%S")

        # Use plain text without markup to avoid rendering issues
        role_str = f"{role:>9}"
        message_line = f"{timestamp} {role_str}: {content}"

        # Use write() with explicit newline to ensure each message is on its own line
        try:
            # Add explicit newline to ensure proper line separation
            self.write(message_line + "\n")
        except Exception as e:
            # Debug: write to a file if widget writing fails
            with open("/tmp/tui_debug.log", "a") as f:
                f.write(f"ERROR writing message: {e}\n")
                f.write(f"Role: {role}, Content: {content[:100]}...\n")


class DataScoutTUI(App):
    """Main TUI application."""

    CSS = """
    Screen {
        layout: grid;
        grid-size: 1 4;
        grid-rows: 3 1fr 1fr 12;
    }
    
    #stats-header {
        dock: top;
        height: 3;
        background: $panel;
        color: $text;
        content-align: center middle;
    }
    
    #main-container {
        layout: horizontal;
        height: 1fr;
    }
    
    #progress-panel {
        width: 70%;
        padding: 1;
        border: solid $primary;
    }
    
    #mission-panel {
        width: 30%;
        padding: 1;
        border: solid $secondary;
    }
    
    #conversation {
        dock: bottom;
        height: 12;
        border: solid $success;
        padding: 1;
    }
    
    .panel-title {
        text-style: bold;
        color: $accent;
        margin-bottom: 1;
    }
    
    #main-progress {
        margin: 1 0;
    }
    """

    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("ctrl+c", "quit", "Quit", show=False),
        Binding("p", "toggle_pause", "Pause/Resume"),
        Binding("r", "restart", "Restart"),
        Binding("e", "show_errors", "Show Errors"),
    ]

    def __init__(
        self,
        initial_prompt: str,
        mission_path: str = "settings/scout_mission.yaml",
        stub: bool = False,
        log_file: Optional[str] = None,
        recursion_limit: int = 1000,
        mission_name: str = "",
        total_samples_target: int = 1200,
        resume_from: Optional[str] = None,
    ):
        super().__init__()
        # STORE the prompt
        self.initial_prompt = initial_prompt
        self.mission_path = mission_path
        self.stub = stub
        self.stats = GenerationStats(target=total_samples_target)
        self.agent_process = None
        self.paused = False
        self.log_file = log_file
        self.log_handle = None
        self.recursion_limit = recursion_limit
        self.mission_name = mission_name
        self.resume_from = resume_from

        # Components
        self.stats_header = None
        self.progress_panel = None
        self.mission_panel = None
        self.conversation = None
        self.error_log = []  # Track errors for inspection

    def _clear_debug_log(self):
        """Clear the debug log file at the start of each session."""
        try:
            with open("/tmp/tui_debug.log", "w") as f:
                f.write(
                    f"=== Data Scout TUI Session Started at {datetime.now().isoformat()} ===\n"
                )
        except Exception:
            pass  # Ignore if we can't write to the debug log

    def compose(self) -> ComposeResult:
        yield Header()

        self.stats_header = StatsHeader()
        yield self.stats_header

        with Container(id="main-container"):
            self.progress_panel = ProgressPanel()
            yield self.progress_panel

            self.mission_panel = MissionPanel(
                self.mission_path,
                self.mission_name,
                self.stats.target,
                load_claimify_config(),
            )
            yield self.mission_panel

        self.conversation = ConversationPanel()
        yield self.conversation

        yield Footer()

    def on_mount(self):
        """Start the agent when the app mounts."""
        # Clear the debug log file at the start of each session
        self._clear_debug_log()

        # Helper to safely update UI from main thread during initialization
        def safe_init_call(func, *args, **kwargs):
            try:
                func(*args, **kwargs)
            except Exception as e:
                with open("/tmp/tui_debug.log", "a") as f:
                    f.write(f"Init UI call error: {e}\n")

        # Open log file if specified
        if self.log_file:
            try:
                self.log_handle = open(self.log_file, "a")
                self.log_handle.write(
                    f"\n=== Data Scout TUI Session Started at {datetime.now().isoformat()} ===\n"
                )
                self.log_handle.flush()
                safe_init_call(
                    self.conversation.add_message,
                    "info",
                    f"Logging to: {self.log_file}",
                )
            except Exception as e:
                safe_init_call(
                    self.conversation.add_message,
                    "debug",
                    f"Failed to open log file {self.log_file}: {e}",
                )

        safe_init_call(
            self.conversation.add_message, "info", "Starting Data Scout Agent..."
        )
        safe_init_call(
            self.conversation.add_message, "info", f"Mission: {self.mission_name}"
        )
        safe_init_call(
            self.conversation.add_message,
            "info",
            f"Target: {self.stats.target} samples",
        )

        # Start the agent process
        self.run_worker(self._run_agent(), name="agent")

        # Start a timer to update elapsed time every second
        self.set_interval(1.0, self._update_elapsed_time)

    def _update_elapsed_time(self):
        """Update elapsed time in stats header every second."""
        if self.stats.started_at and self.stats_header:
            # Update the current time for elapsed calculation
            self.stats.updated_at = datetime.now()
            self.stats_header.update_stats(self.stats)

    def _schedule_graceful_exit(self, reason: str, delay_seconds: float = 3.0):
        """Schedule a graceful exit after a delay, giving user time to see final messages."""
        try:
            self.conversation.add_message("system", f"🏁 {reason}")
            self.conversation.add_message(
                "info",
                f"TUI will exit in {delay_seconds:.1f} seconds... (Press 'q' to exit now)",
            )
        except Exception as e:
            with open("/tmp/tui_debug.log", "a") as f:
                f.write(f"Schedule exit UI call error: {e}\n")

        # Schedule the actual exit using Textual's timer system
        self.set_timer(delay_seconds, self._delayed_exit)

    def _delayed_exit(self):
        """Perform the actual exit after the delay."""
        try:
            self.conversation.add_message("system", "Goodbye! 👋")
        except Exception:
            pass  # Ignore errors during final cleanup

        # Give a moment for the final message to render
        self.set_timer(0.5, self._final_exit)

    def _final_exit(self):
        """Final exit after ensuring agent process is terminated."""
        # Ensure agent process is terminated before exiting
        if self.agent_process and self.agent_process.returncode is None:
            try:
                self.agent_process.terminate()
                # Small delay to allow for graceful termination
                self.set_timer(0.1, self.exit)
                return
            except Exception:
                pass  # Continue with exit if termination fails

        self.exit()

    async def _run_agent(self):
        """Run the Data Scout Agent as a subprocess and parse its output."""
        self.stats.started_at = datetime.now()
        self.stats.updated_at = datetime.now()

        # Helper to safely update UI from worker thread
        def safe_ui_call(func, *args, **kwargs):
            try:
                # Try call_from_thread first, fall back to direct call if in main thread
                try:
                    self.call_from_thread(func, *args, **kwargs)
                except RuntimeError as e:
                    if "must run in a different thread" in str(e):
                        # We're already in the main thread, call directly
                        func(*args, **kwargs)
                    else:
                        raise
            except Exception as e:
                with open("/tmp/tui_debug.log", "a") as f:
                    f.write(f"Agent UI call error: {e}\n")

        if self.stub:
            # Run simulation mode for demo
            safe_ui_call(
                self.conversation.add_message, "system", "Running in stub/demo mode"
            )
            await self._simulate_agent_run()
            return

        try:
            # Check if the CLI tool exists
            safe_ui_call(
                self.conversation.add_message,
                "system",
                "Starting aclarai-claimify-scout...",
            )

            # Start the actual agent process
            command = [
                "aclarai-claimify-scout",
                "--mission",
                self.mission_name,  # Use mission_name directly
                "--recursion-limit",
                str(self.recursion_limit),
            ]
            if self.resume_from:
                command.extend(["--resume-from", self.resume_from])

            process = await asyncio.create_subprocess_exec(
                *command,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                cwd=os.getcwd(),
            )

            self.agent_process = process

            # Send initial user input to start the agent
            if process.stdin:
                initial_request = self.initial_prompt + "\n"

                process.stdin.write(initial_request.encode())
                await process.stdin.drain()
                safe_ui_call(
                    self.conversation.add_message, "user", initial_request.strip()
                )

            # Read output line by line
            while True:
                line = await process.stdout.readline()
                if not line:
                    break

                line_str = line.decode("utf-8").strip()
                if line_str:
                    await self._parse_agent_output(line_str)

            # Wait for process to complete
            return_code = await process.wait()

            if return_code == 0:
                safe_ui_call(
                    self.conversation.add_message,
                    "system",
                    "Agent completed successfully!",
                )
                # Schedule graceful exit after successful completion
                safe_ui_call(
                    self._schedule_graceful_exit, "Agent completed successfully", 3.0
                )
            else:
                safe_ui_call(
                    self.conversation.add_message,
                    "debug",
                    f"Agent exited with code {return_code}",
                )
                self.stats.errors += 1
                # For non-zero exit codes, also schedule exit but with a longer delay
                safe_ui_call(
                    self._schedule_graceful_exit,
                    f"Agent exited with code {return_code}",
                    5.0,
                )

        except FileNotFoundError:
            safe_ui_call(
                self.conversation.add_message,
                "debug",
                "aclarai-claimify-scout not found. Make sure it's installed and in PATH.",
            )
            self.stats.errors += 1
        except Exception as e:
            safe_ui_call(
                self.conversation.add_message, "debug", f"Agent error: {str(e)}"
            )
            self.stats.errors += 1
        finally:
            safe_ui_call(
                self.conversation.add_message, "info", "Agent process completed"
            )

    async def _parse_agent_output(self, line: str):
        """Parse a line of output from the agent and update the UI accordingly."""
        import re

        self.stats.updated_at = datetime.now()

        # Helper to safely update UI from worker thread
        def safe_ui_call(func, *args, **kwargs):
            try:
                # Try call_from_thread first, fall back to direct call if in main thread
                try:
                    self.call_from_thread(func, *args, **kwargs)
                except RuntimeError as e:
                    if "must run in a different thread" in str(e):
                        # We're already in the main thread, call directly
                        func(*args, **kwargs)
                    else:
                        raise
            except Exception as e:
                with open("/tmp/tui_debug.log", "a") as f:
                    f.write(f"UI call error: {e}\n")

        # Skip empty lines
        if not line.strip():
            return

        # Remove ANSI escape sequences for cleaner parsing
        ansi_escape = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
        clean_line = ansi_escape.sub("", line)

        # Debug: Log all agent output to file
        with open("/tmp/tui_debug.log", "a") as f:
            f.write(f"PARSING: {repr(clean_line)}\n")

        # Also log to user-specified log file if available
        if self.log_handle:
            try:
                timestamp = datetime.now().strftime("%H:%M:%S")
                self.log_handle.write(f"[{timestamp}] {clean_line}\n")
                self.log_handle.flush()
            except Exception as e:
                with open("/tmp/tui_debug.log", "a") as f:
                    f.write(f"Log file write error: {e}\n")

        # Parse specific Data Scout Agent patterns (use clean_line for pattern matching, original line for display)

        # 1. Progress patterns: "📊 Progress: 123/1200 samples (10.3%)"
        progress_match = re.search(
            r"📊 Progress: (\d+)/(\d+) samples \((\d+\.\d+)%\)", clean_line
        )
        if progress_match:
            self.stats.completed = int(progress_match.group(1))
            self.stats.target = int(progress_match.group(2))

            # Update progress stats only - don't add to conversation log
            if self.stats_header:
                safe_ui_call(self.stats_header.update_stats, self.stats)
            if self.progress_panel:
                safe_ui_call(self.progress_panel.update_progress, self.stats)
            with open("/tmp/tui_debug.log", "a") as f:
                f.write("MATCHED: Progress pattern\n")
            return

        # 2. Mission plan info: "📊 Mission Plan: Targeting 1200 samples"
        if clean_line.startswith("📊 Mission Plan:"):
            # Extract target from mission plan
            target_match = re.search(r"Targeting (\d+) samples", clean_line)
            if target_match:
                self.stats.target = int(target_match.group(1))
            safe_ui_call(self.conversation.add_message, "info", clean_line)
            return

        # 3. Agent status: "🤖 Data Scout Agent is ready..."
        if clean_line.startswith("🤖 "):
            safe_ui_call(self.conversation.add_message, "system", clean_line)
            self.stats.agent_activities += 1  # Agent activity detected
            return

        # 4. Supervisor messages: "🔍 Supervisor: ..."
        if (
            clean_line.startswith("🔍 Supervisor:")
            and "Decided on 'end'" not in clean_line
        ):
            safe_ui_call(self.conversation.add_message, "assistant", clean_line)
            self.stats.agent_activities += 1  # Agent activity detected
            return

        # 5. Warning/error messages: "⚠️ Supervisor: ..."
        if clean_line.startswith("⚠️ "):
            safe_ui_call(self.conversation.add_message, "debug", clean_line)
            self.stats.errors += 1
            # Track error for inspection
            error_entry = {
                "timestamp": datetime.now(),
                "message": clean_line,
                "type": "warning",
            }
            self.error_log.append(error_entry)
            return

        # 6. Success messages: "✅ Supervisor: ..."
        if (
            clean_line.startswith("✅ Supervisor:")
            and "Decided on 'end'" not in clean_line
        ):
            safe_ui_call(self.conversation.add_message, "assistant", clean_line)
            return

        # 7. Supervisor end decision - detect when agent decides to end
        if "Decided on 'end'" in clean_line or "Routing to END" in clean_line:
            safe_ui_call(
                self.conversation.add_message,
                "system",
                "Agent decided to end conversation",
            )
            # Schedule graceful exit when supervisor decides to end
            safe_ui_call(
                self._schedule_graceful_exit, "Agent decided to end conversation", 3.0
            )
            return

        # 7. Research node messages: "🔍 RESEARCH NODE DEBUG:"
        if "RESEARCH NODE DEBUG:" in clean_line:
            safe_ui_call(self.conversation.add_message, "tool", clean_line)
            return

        # 8. Sample archived: "📊 Sample #123 archived (10.3% complete)"
        if re.search(r"📊 Sample #(\d+) archived", clean_line):
            sample_match = re.search(r"Sample #(\d+)", clean_line)
            if sample_match:
                self.stats.completed = int(sample_match.group(1))
            safe_ui_call(self.conversation.add_message, "assistant", clean_line)
            # Update UI
            if self.stats_header:
                safe_ui_call(self.stats_header.update_stats, self.stats)
            if self.progress_panel:
                safe_ui_call(self.progress_panel.update_progress, self.stats)
            return

        # 9. Node execution: "-> Executing Node: RESEARCH"
        if clean_line.startswith("-> Executing Node:"):
            safe_ui_call(self.conversation.add_message, "system", clean_line)
            return

        # 10. Tool calls: "- Decided to call tools: web_search(...)"
        if "Decided to call tools:" in clean_line:
            safe_ui_call(self.conversation.add_message, "tool", clean_line)
            return

        # 11. Agent responses: "- Responded: ..."
        if clean_line.strip().startswith("- Responded:"):
            safe_ui_call(self.conversation.add_message, "assistant", clean_line)
            return

        # 12. Final completion: "🏁 Agent has finished the task."
        if clean_line.startswith("🏁 "):
            safe_ui_call(self.conversation.add_message, "system", clean_line)
            return

        # 13. User input prompts and responses
        if clean_line.strip() == ">>":  # Input prompt
            return  # Skip the input prompt

        if clean_line.startswith(">>"):  # User input line
            content = clean_line[2:].strip()
            if content:
                safe_ui_call(self.conversation.add_message, "user", content)
            return

        # 14. State creation messages
        if "Creating new state for this thread" in clean_line:
            safe_ui_call(self.conversation.add_message, "system", clean_line)
            return

        # 15. Warning messages (without emoji)
        if clean_line.startswith("Warning:"):
            safe_ui_call(self.conversation.add_message, "debug", clean_line)
            return

        # 16. Error patterns
        clean_lower = clean_line.lower()
        if any(
            keyword in clean_lower
            for keyword in ["error", "failed", "exception", "traceback"]
        ):
            safe_ui_call(self.conversation.add_message, "debug", clean_line)
            self.stats.errors += 1
            # Track error for inspection
            error_entry = {
                "timestamp": datetime.now(),
                "message": clean_line,
                "type": "error",
            }
            self.error_log.append(error_entry)
            return

        # 17. Tool operation messages
        if any(
            keyword in clean_lower
            for keyword in ["search", "found", "fetching", "downloading", "crawling"]
        ):
            safe_ui_call(self.conversation.add_message, "tool", clean_line)
            return

        # 18. Creation/generation messages
        if any(
            keyword in clean_lower
            for keyword in ["generating", "created", "completed", "writing", "saved"]
        ):
            safe_ui_call(self.conversation.add_message, "assistant", clean_line)
            return

        # 19. Thread/connection messages
        if any(keyword in clean_line for keyword in ["thread", "Thread", "config"]):
            safe_ui_call(self.conversation.add_message, "system", clean_line)
            return

        # 20. Supervisor end decision - detect when agent decides to end
        if "Decided on 'end'" in clean_line or "Routing to END" in clean_line:
            safe_ui_call(
                self.conversation.add_message,
                "system",
                "Agent decided to end conversation",
            )
            # Schedule graceful exit when supervisor decides to end
            safe_ui_call(
                self._schedule_graceful_exit, "Agent decided to end conversation", 3.0
            )
            return

        # 21. Default: system message for anything else
        safe_ui_call(self.conversation.add_message, "system", clean_line)

        # Update UI after any message that might affect progress
        if self.stats_header:
            self.stats_header.update_stats(self.stats)
        if self.progress_panel:
            self.progress_panel.update_progress(self.stats)

    async def _simulate_agent_run(self):
        """Simulate the agent run for demo purposes."""

        # Helper to safely update UI from worker thread
        def safe_ui_call(func, *args, **kwargs):
            try:
                # Try call_from_thread first, fall back to direct call if in main thread
                try:
                    self.call_from_thread(func, *args, **kwargs)
                except RuntimeError as e:
                    if "must run in a different thread" in str(e):
                        # We're already in the main thread, call directly
                        func(*args, **kwargs)
                    else:
                        raise
            except Exception as e:
                with open("/tmp/tui_debug.log", "a") as f:
                    f.write(f"Simulate UI call error: {e}\n")

        safe_ui_call(self.conversation.add_message, "system", "🚀 Agent initialized")
        safe_ui_call(
            self.conversation.add_message,
            "user",
            "Generate samples for verifiability from news reports",
        )

        # Simulate progress
        for i in range(1201):
            if self.paused:
                safe_ui_call(self.conversation.add_message, "system", "⏸️  Agent paused")
                while self.paused:
                    await asyncio.sleep(0.1)
                safe_ui_call(
                    self.conversation.add_message, "system", "▶️  Agent resumed"
                )

            # Update stats
            self.stats.completed = i
            self.stats.updated_at = datetime.now()

            # Update UI
            if self.stats_header:
                safe_ui_call(self.stats_header.update_stats, self.stats)
            if self.progress_panel:
                safe_ui_call(self.progress_panel.update_progress, self.stats)

            # Simulate agent messages
            if i % 100 == 0:
                safe_ui_call(
                    self.conversation.add_message,
                    "assistant",
                    f"Generated {i} samples so far...",
                )

            if i % 200 == 0 and i > 0:
                safe_ui_call(
                    self.conversation.add_message,
                    "tool",
                    f"Web search: Found relevant content for sample {i}",
                )

            # Simulate different speeds
            await asyncio.sleep(0.01 if i < 50 else 0.001)

        safe_ui_call(
            self.conversation.add_message,
            "system",
            "✅ All 1200 samples generated successfully!",
        )
        # Schedule graceful exit after successful stub completion
        safe_ui_call(
            self._schedule_graceful_exit, "Stub simulation completed successfully", 3.0
        )

    def action_toggle_pause(self):
        """Toggle pause/resume."""
        self.paused = not self.paused
        status = "⏸️ Paused" if self.paused else "▶️ Resumed"
        # These actions run in the main thread, so no need for call_from_thread
        try:
            self.conversation.add_message("debug", status)
        except Exception as e:
            with open("/tmp/tui_debug.log", "a") as f:
                f.write(f"Action UI call error: {e}\n")

    def action_restart(self):
        """Restart the generation process."""
        try:
            self.conversation.add_message("debug", "🔄 Restart requested")
        except Exception as e:
            with open("/tmp/tui_debug.log", "a") as f:
                f.write(f"Action UI call error: {e}\n")
        # In a real implementation, this would restart the agent process

    def action_show_errors(self):
        """Show a focused view of all errors encountered."""
        try:
            if not self.error_log:
                self.conversation.add_message("info", "🎉 No errors encountered yet!")
                return

            # Clear conversation and show error log
            self.conversation.clear()
            self.conversation.write(
                "❌ ERROR LOG - Press 'q' to return to main view\n\n"
            )

            for i, error in enumerate(self.error_log, 1):
                timestamp = error["timestamp"].strftime("%H:%M:%S")
                error_type = error["type"].upper()
                message = error["message"]

                self.conversation.write(
                    f"[{i:2d}] {timestamp} [{error_type}]: {message}\n"
                )

            self.conversation.write(f"\n📊 Total Errors: {len(self.error_log)}\n")

        except Exception as e:
            with open("/tmp/tui_debug.log", "a") as f:
                f.write(f"Show errors UI call error: {e}\n")

    def action_quit(self):
        """Quit the application."""
        if self.agent_process:
            # Terminate the agent process if it's still running
            try:
                if self.agent_process.returncode is None:  # Process is still running
                    self.agent_process.terminate()
                    # Wait for a short time for graceful termination
                    self.set_timer(2.0, self._final_exit)
                    return
            except Exception as e:
                with open("/tmp/tui_debug.log", "a") as f:
                    f.write(f"Error terminating agent process: {e}\n")

        # Close log file if open
        if self.log_handle:
            try:
                self.log_handle.write(
                    f"\n=== Data Scout TUI Session Ended at {datetime.now().isoformat()} ===\n"
                )
                self.log_handle.close()
            except Exception as e:
                with open("/tmp/tui_debug.log", "a") as f:
                    f.write(f"Log file close error: {e}\n")

        self.exit()


# CLI Interface
cli_app = typer.Typer()


def _redirect_stdout_to_file(log_file: Optional[str]):
    """Context manager to redirect stdout to a log file during setup."""
    if not log_file:
        # If no log file specified, redirect to /dev/null to suppress output
        return contextlib.redirect_stdout(io.StringIO())
    else:
        # Redirect to the specified log file
        log_handle = open(log_file, "a")
        log_handle.write(
            f"\n=== Data Scout Setup Started at {datetime.now().isoformat()} ===\n"
        )
        return contextlib.redirect_stdout(log_handle)


class DataScoutLauncher(App):
    """Launcher app that handles mission selection and then launches the main TUI."""

    def __init__(
        self,
        mission_plan_path: str,
        stub: bool = False,
        log_file: Optional[str] = None,
        target: int = 1200,
    ):
        super().__init__()
        self.mission_plan_path = mission_plan_path
        self.stub = stub
        self.log_file = log_file
        self.target = target
        self.available_missions = []
        self.chosen_mission = None
        self.chosen_mission_target = None

    def on_mount(self):
        """Load missions and show selector when app starts."""
        # Import get_mission_names silently
        with _redirect_stdout_to_file(self.log_file):
            from aclarai_claimify.scout.scout_utils import (
                get_mission_details_from_file,
            )

            mission_details = get_mission_details_from_file(self.mission_plan_path)

        if not mission_details or not mission_details["mission_names"]:
            self.exit(message=f"❌ No missions found in {self.mission_plan_path}")
            return

        self.available_missions = mission_details["mission_names"]
        self.mission_targets = mission_details["mission_targets"]

        # Show mission selector modal
        self.push_screen(
            MissionSelector(self.available_missions), self.on_mission_selected
        )

    def on_mission_selected(self, mission_name: str):
        """Called when a mission is selected from the modal."""
        if not mission_name:
            self.exit(message="❌ No mission selected")
            return

        self.chosen_mission = mission_name
        self.chosen_mission_target = self.mission_targets.get(mission_name, 1200)

        # Exit this launcher and start the main TUI
        self.exit()

    def compose(self) -> ComposeResult:
        """Simple compose - the modal will handle the UI."""
        yield Static("Loading missions...", id="loading")


@cli_app.command()
def generate(
    mission: Optional[str] = typer.Argument(None, help="Path to mission plan YAML"),
    target: int = typer.Option(1200, help="Target number of samples to generate"),
    stub: bool = typer.Option(False, help="Run in stub/demo mode"),
    log: Optional[str] = typer.Option(
        None, "--log", help="Log file to save terminal output for debugging"
    ),
    resume_from: Optional[str] = typer.Option(
        None, "--resume-from", help="The thread ID to resume from."
    ),
):
    """Start the Data Scout Agent TUI for sample generation."""

    mission_plan_path = mission or "settings/scout_mission.yaml"

    # Validate mission file exists
    if not os.path.exists(mission_plan_path):
        typer.echo(f"❌ Mission file not found: {mission_plan_path}", err=True)
        raise typer.Exit(1)

    # Check for Ollama (silently)
    if not stub:
        import subprocess

        try:
            subprocess.run(["ollama", "list"], capture_output=True, check=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            typer.echo(
                "❌ Ollama not found. Please install Ollama or use --stub mode.",
                err=True,
            )
            raise typer.Exit(1)

    # Create output directories (silently)
    os.makedirs("examples/data/datasets/tier1", exist_ok=True)
    os.makedirs("examples/data/datasets/tier2", exist_ok=True)
    os.makedirs("examples", exist_ok=True)

    # Step 1: Launch mission selector TUI (skip in stub mode)
    if not stub:
        launcher = DataScoutLauncher(mission_plan_path, stub, log, target)
        launcher.run()

        chosen_mission_name = launcher.chosen_mission
        total_samples_target = launcher.chosen_mission_target
        if not chosen_mission_name:
            typer.echo("❌ No mission selected.", err=True)
            raise typer.Exit(1)
    else:
        # In stub mode, use a default mission name
        chosen_mission_name = "research_dataset"  # or whatever default makes sense for stub
        total_samples_target = 1200

    # Step 2: Load configuration and setup (with silent patches)
    with _redirect_stdout_to_file(log):
        # Import and apply patches silently - this is where all the patch logs go
        import aclarai_claimify.scout.patch  # This triggers all the patches

        # Load configuration
        config = load_claimify_config()

        # Calculate recursion limit
        STEPS_PER_SAMPLE = 4
        RETRY_MARGIN = 20
        OVERHEAD_FACTOR = 1.2
        recursion_limit = int(
            STEPS_PER_SAMPLE * (total_samples_target + RETRY_MARGIN) * OVERHEAD_FACTOR
        )

    # Step 3: Determine initial prompt (interactively if needed)
    final_prompt = ""
    if not stub and config.scout_agent and config.scout_agent.initial_prompt:
        # Use prompt from config file
        final_prompt = config.scout_agent.initial_prompt
    elif not stub:
        # Prompt the user interactively if not in config and not in stub mode
        final_prompt = typer.prompt("🚀 Please enter the initial prompt for the agent")

    # Step 4: Launch the main TUI (cleanly, with no setup noise)
    app = DataScoutTUI(
        initial_prompt=final_prompt,
        mission_path=mission_plan_path,
        stub=stub,
        log_file=log,
        recursion_limit=recursion_limit,
        mission_name=chosen_mission_name,
        total_samples_target=total_samples_target,
        resume_from=resume_from,
    )
    app.run()


if __name__ == "__main__":
    if len(sys.argv) == 1:
        # Default to generate command if no arguments
        generate()
    else:
        cli_app()
