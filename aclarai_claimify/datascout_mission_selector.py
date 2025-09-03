
from textual.app import ComposeResult
from textual.screen import ModalScreen
from textual.widgets import Button, Static
from textual.containers import Vertical, Container

class MissionSelector(ModalScreen[str]):
    """A modal screen to select a mission."""
    
    CSS = """
    MissionSelector {
        align: center middle;
    }
    
    #mission-dialog {
        width: 60;
        height: auto;
        background: $surface;
        border: thick $primary;
        padding: 2;
        margin: 1;
    }
    
    #mission-title {
        text-align: center;
        text-style: bold;
        color: $accent;
        margin-bottom: 1;
    }
    
    #mission-buttons {
        height: auto;
        align: center middle;
    }
    
    .mission-button {
        width: 100%;
        margin: 1 0;
    }
    """

    def __init__(self, missions: list[str]):
        super().__init__()
        self.missions = missions

    def compose(self) -> ComposeResult:
        with Container(id="mission-dialog"):
            yield Static("🚀 Please select a mission to run:", id="mission-title")
            with Vertical(id="mission-buttons"):
                for i, mission in enumerate(self.missions):
                    yield Button(f"{i + 1}. {mission}", id=mission, variant="primary", classes="mission-button")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        self.dismiss(event.button.id)
