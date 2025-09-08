from textual.app import ComposeResult
from textual.screen import ModalScreen
from textual.widgets import Button, Static

class MissionSelector(ModalScreen[str]):
    """A modal screen to select a mission."""

    def __init__(self, missions: list[str]):
        super().__init__()
        self.missions = missions

    def compose(self) -> ComposeResult:
        yield Static("🚀 Please select a mission to run:", id="mission-title")
        for i, mission in enumerate(self.missions):
            yield Button(f"{i + 1}. {mission}", id=mission, variant="primary", classes="mission-button")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        self.dismiss(event.button.id)
