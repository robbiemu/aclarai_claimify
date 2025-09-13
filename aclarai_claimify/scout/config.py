"""
Configuration loader for the Data Scout Agent.
Handles loading mission-specific configuration from separate config files.
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional
import yaml

from .models import ScoutAgentMissionPlanToolConfig

logger = logging.getLogger(__name__)

# Global variable to store the use_robots setting
_global_use_robots = True

# Active configuration instance set at application startup
_active_scout_config: Optional["StructuredScoutConfig"] = None


def set_global_use_robots(use_robots: bool):
    """Set the global use_robots setting."""
    global _global_use_robots
    _global_use_robots = use_robots


def get_global_use_robots() -> bool:
    """Get the global use_robots setting."""
    global _global_use_robots
    return _global_use_robots


def set_active_scout_config(config: "StructuredScoutConfig") -> None:
    """Set the process-wide active scout configuration instance."""
    global _active_scout_config
    _active_scout_config = config


def get_active_scout_config() -> "StructuredScoutConfig":
    """Get the active scout configuration, loading defaults if not yet set."""
    global _active_scout_config
    if _active_scout_config is None:
        # Fall back to loading with current global use_robots
        _active_scout_config = load_scout_config(use_robots=get_global_use_robots())
    return _active_scout_config


class StructuredScoutConfig:
    """Structured configuration wrapper that provides object-oriented access to scout config."""
    
    def __init__(self, config_dict: Dict[str, Any]):
        self._raw_config = config_dict
        
        # Extract mission plan tools if they exist
        self._tools = {}
        mission_plan = self._raw_config.get("mission_plan", {})
        if isinstance(mission_plan, dict) and "tools" in mission_plan:
            tools_config = mission_plan["tools"]
            if isinstance(tools_config, dict):
                for tool_name, tool_config in tools_config.items():
                    if isinstance(tool_config, dict):
                        self._tools[tool_name] = ScoutAgentMissionPlanToolConfig(**tool_config)
    
    def __getattr__(self, name: str) -> Any:
        """Delegate attribute access to the raw config dictionary."""
        return self._raw_config.get(name)
    
    def __getitem__(self, key: str) -> Any:
        """Dictionary-style access to config values."""
        return self._raw_config.get(key)
    
    def __contains__(self, key: str) -> bool:
        """Check if a key exists in the config."""
        return key in self._raw_config
    
    def get(self, key: str, default: Any = None) -> Any:
        """Dictionary-style access to config values with default."""
        return self._raw_config.get(key, default)
    
    def keys(self) -> Any:
        """Return the keys of the config dictionary."""
        return self._raw_config.keys()
    
    def values(self) -> Any:
        """Return the values of the config dictionary."""
        return self._raw_config.values()
    
    def items(self) -> Any:
        """Return the items of the config dictionary."""
        return self._raw_config.items()
    
    def __iter__(self):
        """Make the config iterable like a dictionary."""
        return iter(self._raw_config)
    
    def get_tool_config(self, tool_name: str) -> Optional[ScoutAgentMissionPlanToolConfig]:
        """Retrieve the configuration for a specific tool by name."""
        return self._tools.get(tool_name)
    
    def to_dict(self) -> Dict[str, Any]:
        """Return the raw configuration dictionary."""
        return self._raw_config.copy()


def load_scout_config(
    config_path: Optional[str] = None, use_robots: Optional[bool] = None
) -> StructuredScoutConfig:
    """
    Load mission configuration for the scout agent from a separate config file.

    Args:
        config_path: Optional path to the scout config file.
                    If None, defaults to 'scout_config.yaml' in current directory.
        use_robots: Whether to respect robots.txt rules. If None, uses global setting.

    Returns:
        StructuredScoutConfig object containing the mission configuration.
    """
    # Default to scout_config.yaml in current directory
    if config_path is None:
        config_path = "scout_config.yaml"

    # Use global setting if use_robots is not explicitly provided
    if use_robots is None:
        use_robots = get_global_use_robots()

    config_file = Path(config_path)

    if not config_file.exists():
        logger.error(f"Scout configuration file not found: {config_path}")
        # Return a minimal default configuration
        default_config = {
            "search_provider": "duckduckgo/search",
            "recursion_per_sample": 30,
            "writer": {
                "tier1_path": "examples/data/datasets/tier1",
                "tier2_path": "examples/data/datasets/tier2",
                "audit_trail_path": "examples/PEDIGREE.md",
            },
            "checkpointer_path": ".checkpointer.sqlite",
            "nodes": {"research": {"max_iterations": 7}},
            "use_robots": use_robots,
        }
        return StructuredScoutConfig(default_config)

    try:
        with open(config_file, "r") as f:
            config_data = yaml.safe_load(f) or {}

        # Add use_robots to the config
        config_data["use_robots"] = use_robots

        logger.debug(f"Loaded scout configuration from {config_path}")
        return StructuredScoutConfig(config_data)

    except Exception as e:
        logger.error(f"Failed to load scout configuration from {config_path}: {e}")
        # Return minimal default on error
        default_config = {
            "search_provider": "duckduckgo/search",
            "recursion_per_sample": 30,
            "writer": {
                "tier1_path": "examples/data/datasets/tier1",
                "tier2_path": "examples/data/datasets/tier2",
                "audit_trail_path": "examples/PEDIGREE.md",
            },
            "checkpointer_path": ".checkpointer.sqlite",
            "use_robots": use_robots,
        }
        return StructuredScoutConfig(default_config)
