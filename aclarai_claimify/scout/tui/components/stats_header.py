from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import Optional

from textual.widgets import Static


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
        
        # Debug: Log stats header updates
        with open("/tmp/tui_debug.log", "a") as f:
            f.write(f"STATS HEADER UPDATE: {status_line}\n")
            
        self.update(status_line)
