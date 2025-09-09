"""
Configuration loader for the Data Scout Agent.
Handles loading mission-specific configuration from separate config files.
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional
import yaml

logger = logging.getLogger(__name__)


def load_scout_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load mission configuration for the scout agent from a separate config file.

    Args:
        config_path: Optional path to the scout config file.
                    If None, defaults to 'scout_config.yaml' in current directory.

    Returns:
        Dictionary containing the mission configuration for the scout agent.
    """
    # Default to scout_config.yaml in current directory
    if config_path is None:
        config_path = "scout_config.yaml"

    config_file = Path(config_path)

    if not config_file.exists():
        logger.error(f"Scout configuration file not found: {config_path}")
        # Return a minimal default configuration
        return {
            "search_provider": "duckduckgo/search",
            "recursion_per_sample": 30,
            "writer": {
                "tier1_path": "examples/data/datasets/tier1",
                "tier2_path": "examples/data/datasets/tier2",
                "audit_trail_path": "examples/PEDIGREE.md",
            },
            "checkpointer_path": ".checkpointer.sqlite",
            "nodes": {"research": {"max_iterations": 7}},
        }

    try:
        with open(config_file, "r") as f:
            config_data = yaml.safe_load(f) or {}

        logger.info(f"Loaded scout configuration from {config_path}")
        return config_data

    except Exception as e:
        logger.error(f"Failed to load scout configuration from {config_path}: {e}")
        # Return minimal default on error
        return {
            "search_provider": "duckduckgo/search",
            "recursion_per_sample": 30,
            "writer": {
                "tier1_path": "examples/data/datasets/tier1",
                "tier2_path": "examples/data/datasets/tier2",
                "audit_trail_path": "examples/PEDIGREE.md",
            },
            "checkpointer_path": ".checkpointer.sqlite",
        }
