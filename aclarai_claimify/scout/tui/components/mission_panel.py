from textual.app import ComposeResult
from textual.containers import Vertical
from textual.widgets import Static

from ....config import ClaimifyConfig


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
        self.current_status = "Initializing..."
        self.mission_info_widget = None

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Static("🎯 Mission Status", classes="panel-title")
            mission_info = (
                f"Mission: {self.mission_name}\n"
                f"Target: {self.total_samples_target} samples\n"
                f"Model: {self.config.scout_agent.mission_plan.nodes[0].model}\n"
                f"Provider: {self.config.scout_agent.search_provider}\n"
                f"Status: {self.current_status}"
            )
            self.mission_info_widget = Static(mission_info, id="mission-info")
            yield self.mission_info_widget
    
    def update_status(self, new_status: str):
        """Update the mission status displayed in the panel."""
        self.current_status = new_status
        if self.mission_info_widget:
            mission_info = (
                f"Mission: {self.mission_name}\n"
                f"Target: {self.total_samples_target} samples\n"
                f"Model: {self.config.scout_agent.mission_plan.nodes[0].model}\n"
                f"Provider: {self.config.scout_agent.search_provider}\n"
                f"Status: {self.current_status}"
            )
            self.mission_info_widget.update(mission_info)
