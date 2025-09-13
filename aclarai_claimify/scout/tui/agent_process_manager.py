import asyncio
import os
from typing import Optional


class AgentProcessManager:
    """Manages the lifecycle of the agent subprocess."""

    def __init__(self, mission_name: str, recursion_limit: int):
        self.mission_name = mission_name
        self.recursion_limit = recursion_limit
        self.process: Optional[asyncio.subprocess.Process] = None

    async def start(self) -> asyncio.subprocess.Process:
        """Starts the agent subprocess."""
        command = [
            "aclarai-claimify-scout",
            "--mission",
            self.mission_name,
            "--recursion-limit",
            str(self.recursion_limit),
        ]
        
        # Use the current environment without forcing any variables
        env = os.environ.copy()
        
        self.process = await asyncio.create_subprocess_exec(
            *command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            cwd=os.getcwd(),
            env=env,
        )
        return self.process

    def terminate(self):
        """Terminates the agent subprocess."""
        if self.process and self.process.returncode is None:
            self.process.terminate()
