"""
Configuration management for the Claimify pipeline.
Handles loading and managing configuration from default and user-provided files.
"""
import importlib.resources as resources
import logging
from pathlib import Path
from typing import Any, Dict, MutableMapping, Optional

import yaml
from pydantic import ValidationError

from .data_models import ClaimifyConfig, OptimizationConfig

logger = logging.getLogger(__name__)


def deep_merge(
    base: Dict[str, Any], override: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Recursively merge two dictionaries.
    Values in `override` take precedence over `base`.
    """
    merged = base.copy()
    for key, value in override.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, dict)
        ):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _parse_claimify_config_data(
    config_data: Dict[str, Any]
) -> Optional[ClaimifyConfig]:
    """
    Parse configuration data from dictionary into ClaimifyConfig.
    """
    try:
        # Extract nested structures for Pydantic model
        window_config = config_data.get("window", {}).get("claimify", {})
        model_config = config_data.get("model", {}).get("claimify", {})
        processing_config = config_data.get("processing", {}).get("claimify", {})
        logging_config = processing_config.get("logging", {})

        flat_config = {
            "context_window_p": window_config.get("p"),
            "context_window_f": window_config.get("f"),
            "agents": config_data.get("agents", {}),
            "selection_model": model_config.get("selection"),
            "disambiguation_model": model_config.get("disambiguation"),
            "decomposition_model": model_config.get("decomposition"),
            "default_model": model_config.get("default"),
            "max_retries": processing_config.get("max_retries"),
            "timeout_seconds": config_data.get("processing", {}).get(
                "timeout_seconds"
            ),
            "temperature": config_data.get("processing", {}).get("temperature"),
            "max_tokens": config_data.get("processing", {}).get("max_tokens"),
            "log_decisions": logging_config.get("log_decisions"),
            "log_transformations": logging_config.get("log_transformations"),
            "log_timing": logging_config.get("log_timing"),
            "generate_dataset": config_data.get("generate_dataset", {}),
            "scout_agent": config_data.get("scout_agent"),
        }

        # Filter out None values so Pydantic uses defaults
        filtered_config = {k: v for k, v in flat_config.items() if v is not None}

        return ClaimifyConfig(**filtered_config)
    except ValidationError as e:
        logger.error(f"Configuration validation error: {e}")
        return None
    except Exception as e:
        logger.error(f"Failed to parse config data: {e}")
        return None


def load_claimify_config(
    override_path: Optional[str] = None,
) -> ClaimifyConfig:
    """
    Load Claimify configuration with a cascading override system.
    1. Load the default `config.yaml` from within the package.
    2. Load the user's `config.yaml` from `override_path` (if found)
       and merge it on top of the base.
    """
    try:
        # 1. Load base config from package resources
        with resources.open_text(
            "aclarai_claimify.settings", "config.yaml"
        ) as f:
            base_config_data = yaml.safe_load(f) or {}
    except FileNotFoundError:
        logger.error("Default 'config.yaml' not found in package.")
        base_config_data = {}
    except Exception as e:
        logger.error(f"Error loading default config: {e}")
        base_config_data = {}

    # 2. Load override config if path is provided
    override_config_data = {}
    if override_path and Path(override_path).exists():
        try:
            with open(override_path, "r") as f:
                override_config_data = yaml.safe_load(f) or {}
        except Exception as e:
            logger.error(f"Failed to load override config from {override_path}: {e}")

    # 3. Deep merge configs
    merged_config_data = deep_merge(base_config_data, override_config_data)

    # 4. Validate and return Pydantic model
    config = _parse_claimify_config_data(merged_config_data)
    if not config:
        logger.warning(
            "Could not create a valid configuration. "
            "Falling back to default Pydantic model."
        )
        return ClaimifyConfig()
    return config


def load_optimization_config(
    override_path: Optional[str] = None,
) -> OptimizationConfig:
    """
    Load optimization configuration with a cascading override system.
    1. Load the default `optimization.yaml` from within the package.
    2. Load the user's `optimization.yaml` from `override_path` (if found)
       and merge it on top of the base.
    """
    try:
        # 1. Load base config from package resources
        with resources.open_text(
            "aclarai_claimify.settings", "optimization.yaml"
        ) as f:
            base_config_data = yaml.safe_load(f) or {}
    except FileNotFoundError:
        logger.error("Default 'optimization.yaml' not found in package.")
        base_config_data = {}
    except Exception as e:
        logger.error(f"Error loading default optimization config: {e}")
        base_config_data = {}

    # 2. Load override config if path is provided
    override_config_data = {}
    if override_path and Path(override_path).exists():
        try:
            with open(override_path, "r") as f:
                override_config_data = yaml.safe_load(f) or {}
        except Exception as e:
            logger.error(
                f"Failed to load override optimization config from {override_path}: {e}"
            )

    # 3. Deep merge configs
    merged_config_data = deep_merge(base_config_data, override_config_data)

    # 4. Validate and return Pydantic model
    try:
        return OptimizationConfig(**merged_config_data)
    except ValidationError as e:
        logger.error(f"Optimization configuration validation error: {e}")
        logger.warning(
            "Could not create a valid optimization configuration. "
            "Falling back to an empty config."
        )
        return OptimizationConfig(optimizer_name="unknown")


def load_prompt_template(stage: str) -> Optional[Dict[str, Any]]:
    """
    Load prompt template for a specific stage using importlib.resources.
    """
    prompt_files = {
        "selection": "claimify_selection.yaml",
        "disambiguation": "claimify_disambiguation.yaml",
        "decomposition": "claimify_decomposition.yaml",
    }

    if stage not in prompt_files:
        logger.error(f"Unknown prompt stage: {stage}")
        return None

    prompt_file = prompt_files[stage]

    try:
        with resources.open_text("aclarai_claimify.prompts", prompt_file) as f:
            return yaml.safe_load(f)
    except Exception as e:
        logger.error(f"Failed to load prompt template for {stage}: {e}")
        return None
