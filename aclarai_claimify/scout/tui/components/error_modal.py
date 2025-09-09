from textual.screen import ModalScreen
from textual.widgets import RichLog, Button
from textual.containers import Container
from textual.app import ComposeResult
from textual.binding import Binding
from typing import List


class ErrorModal(ModalScreen):
    """Modal screen for displaying error messages."""

    BINDINGS = [
        Binding("escape", "close_modal", "Close", show=False),
        Binding("e", "close_modal", "Close", show=False),
    ]

    def __init__(self, error_messages: List[str], **kwargs):
        super().__init__(**kwargs)
        self.error_messages = error_messages

    def compose(self) -> ComposeResult:
        yield Container(
            RichLog(id="error-log", wrap=True),
            Button("Close", variant="primary", id="close-errors"),
            id="error-modal-container",
        )

    def on_mount(self) -> None:
        error_log = self.query_one(RichLog)
        error_log.write("## Error Log")
        for msg in self.error_messages:
            error_log.write(msg)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "close-errors":
            self.dismiss()

    def action_close_modal(self) -> None:
        """Close the modal screen."""
        self.dismiss()