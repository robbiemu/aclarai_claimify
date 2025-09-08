from textual.app import ComposeResult
from textual.containers import Vertical
from textual.widgets import ProgressBar, Static

from .stats_header import GenerationStats


class ProgressPanel(Static):
    """Left panel with progress bar and recent samples."""

    def __init__(self):
        super().__init__(id="progress-panel")
        self.progress_bar = None
        self.recent_samples_widget = None
        self.recent_samples = []

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Static("📈 Generation Progress", classes="panel-title")
            self.progress_bar = ProgressBar(
                total=100, show_percentage=True, id="main-progress"
            )
            yield self.progress_bar
            self.recent_samples_widget = Static("Recent samples will appear here...", id="recent-samples")
            yield self.recent_samples_widget

    def update_progress(self, stats: GenerationStats):
        if self.progress_bar:
            progress_pct = int(100 * stats.completed / max(1, stats.target))
            self.progress_bar.update(progress=progress_pct)
    
    def add_sample(self, sample_number: int, description: str):
        """Add a recent sample to the display."""
        # Format the sample entry with proper wrapping for long descriptions
        if len(description) > 80:
            # For long excerpts, show on multiple lines with proper indentation
            sample_entry = f"#{sample_number}:\n  {description}"
        else:
            # For short descriptions, keep on one line
            sample_entry = f"#{sample_number}: {description}"
        
        # Keep only the 3 most recent samples
        self.recent_samples.append(sample_entry)
        if len(self.recent_samples) > 3:
            self.recent_samples = self.recent_samples[-3:]
        
        # Update the widget
        if self.recent_samples_widget:
            if self.recent_samples:
                content = "\n\n".join(self.recent_samples)  # Extra spacing for readability
            else:
                content = "Recent samples will appear here..."
            self.recent_samples_widget.update(content)
