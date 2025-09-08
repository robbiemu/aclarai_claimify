import re
from dataclasses import dataclass
from typing import Generator, Optional


@dataclass
class ProgressUpdate:
    completed: int
    target: int


@dataclass
class NewMessage:
    role: str
    content: str


@dataclass
class ErrorMessage:
    message: str


class AgentOutputParser:
    """Parses agent log output and emits structured events."""

    def __init__(self):
        self.ansi_escape = re.compile(r"\x1B(?:[@-Z\\-_]|\\[0-?]*[ -/]*[@-~])")

    def parse_line(
        self,
        line: str,
    ) -> Generator[ProgressUpdate | NewMessage | ErrorMessage, None, None]:
        """Parses a single line of agent output."""
        clean_line = self.ansi_escape.sub("", line).strip()
        if not clean_line:
            return

        # 1. Progress patterns: "📊 Mission Progress: 123/1200 samples (10.3%)"
        progress_match = re.search(
            r"📊 Mission Progress: (\d+)/(\d+) samples \((\d+\.\d+)%\)", clean_line
        )
        if progress_match:
            yield ProgressUpdate(
                completed=int(progress_match.group(1)),
                target=int(progress_match.group(2)),
            )
            return

        # 2. Alternative progress pattern: "📊 Progress: 123/1200 samples (10.3%)"
        progress_match2 = re.search(
            r"📊 Progress: (\d+)/(\d+) samples \((\d+\.\d+)%\)", clean_line
        )
        if progress_match2:
            yield ProgressUpdate(
                completed=int(progress_match2.group(1)),
                target=int(progress_match2.group(2)),
            )
            return

        # 3. Sample archived: "📊 Sample #123 archived (10.3% complete)"
        sample_match = re.search(r"📊 Sample #(\d+) archived", clean_line)
        if sample_match:
            completed = int(sample_match.group(1))
            yield ProgressUpdate(completed=completed, target=completed)  # Update with current progress
            yield NewMessage(role="assistant", content=clean_line)
            return

        # 4. Mission plan info: "📊 Mission Plan: Targeting 1200 samples"
        if clean_line.startswith("📊 Mission Plan:"):
            target_match = re.search(r"Targeting (\d+) samples", clean_line)
            if target_match:
                target = int(target_match.group(1))
                yield ProgressUpdate(completed=0, target=target)
            yield NewMessage(role="info", content=clean_line)
            return

        # 5. Agent status: "🤖 Data Scout Agent is ready..."
        if clean_line.startswith("🤖 "):
            yield NewMessage(role="system", content=clean_line)
            return

        # 6. Supervisor messages: "🔍 Supervisor: ..." (excluding end decisions)
        if clean_line.startswith("🔍 Supervisor:") and "Decided on 'end'" not in clean_line:
            yield NewMessage(role="assistant", content=clean_line)
            return

        # 7. Warning/error messages: "⚠️ Supervisor: ..."
        if clean_line.startswith("⚠️ "):
            yield ErrorMessage(message=clean_line)
            return

        # 8. Success messages: "✅ Supervisor: ..." (excluding end decisions)
        if clean_line.startswith("✅ Supervisor:") and "Decided on 'end'" not in clean_line:
            yield NewMessage(role="assistant", content=clean_line)
            return

        # 9. Research node messages: "🔍 RESEARCH NODE DEBUG:"
        if "RESEARCH NODE DEBUG:" in clean_line or "RESEARCH NODE" in clean_line:
            yield NewMessage(role="tool", content=clean_line)
            return

        # 10. Node execution: "-> Executing Node: RESEARCH"
        if clean_line.startswith("-> Executing Node:"):
            yield NewMessage(role="system", content=clean_line)
            return

        # 11. Tool calls: "- Decided to call tools: web_search(...)"
        if "Decided to call tools:" in clean_line:
            yield NewMessage(role="tool", content=clean_line)
            return

        # 12. Agent responses: "- Responded: ..."
        if clean_line.strip().startswith("- Responded:"):
            yield NewMessage(role="assistant", content=clean_line)
            return

        # 13. Final completion: "🏁 Agent has finished the task."
        if clean_line.startswith("🏁 "):
            yield NewMessage(role="system", content=clean_line)
            return

        # 14. State creation messages
        if "Creating new state for this thread" in clean_line:
            yield NewMessage(role="system", content=clean_line)
            return

        # 15. Warning messages (without emoji)
        if clean_line.startswith("Warning:"):
            yield NewMessage(role="debug", content=clean_line)
            return

        # 16. Error patterns
        clean_lower = clean_line.lower()
        if any(keyword in clean_lower for keyword in ["error", "failed", "exception", "traceback"]):
            yield ErrorMessage(message=clean_line)
            return

        # 17. Tool operation messages
        if any(keyword in clean_lower for keyword in ["search", "found", "fetching", "downloading", "crawling"]):
            yield NewMessage(role="tool", content=clean_line)
            return

        # 18. Creation/generation messages
        if any(keyword in clean_lower for keyword in ["generating", "created", "completed", "writing", "saved"]):
            yield NewMessage(role="assistant", content=clean_line)
            return

        # 19. Thread/connection messages
        if any(keyword in clean_line for keyword in ["thread", "Thread", "config"]):
            yield NewMessage(role="system", content=clean_line)
            return

        # 20. Supervisor end decision - detect when agent decides to end
        if "Decided on 'end'" in clean_line or "Routing to END" in clean_line:
            yield NewMessage(role="system", content="Agent decided to end conversation")
            return

        # 21. Default: system message for anything else
        yield NewMessage(role="system", content=clean_line)
