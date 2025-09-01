"""
Configuration management for the Claimify pipeline.
Handles loading and managing configuration independently of external systems.
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from .data_models import ClaimifyConfig

logger = logging.getLogger(__name__)


def load_claimify_config() -> ClaimifyConfig:
    """
    Load Claimify configuration from various sources.
    Priority order:
    1. Environment variables
    2. Local config file (claimify.config.yaml)
    3. Default configuration
    
    Returns:
        ClaimifyConfig instance with loaded settings
    """
    # Start with default configuration
    config = ClaimifyConfig()
    
    # Try to load from environment variables
    config = _load_from_env(config)
    
    # Try to load from config file
    config_file_config = _load_from_config_file()
    if config_file_config:
        # Merge file config with existing config (file overrides defaults)
        config = _merge_configs(config, config_file_config)
    
    return config


def _load_from_env(config: ClaimifyConfig) -> ClaimifyConfig:
    """
    Load configuration from environment variables.
    Args:
        config: Base configuration to update
    Returns:
        Updated configuration
    """
    # Context window settings
    context_window_p = os.getenv("CLAIMIFY_CONTEXT_WINDOW_P")
    if context_window_p is not None:
        config.context_window_p = int(context_window_p)
        
    context_window_f = os.getenv("CLAIMIFY_CONTEXT_WINDOW_F")
    if context_window_f is not None:
        config.context_window_f = int(context_window_f)
    
    # Model settings
    selection_model = os.getenv("CLAIMIFY_SELECTION_MODEL")
    if selection_model:
        config.selection_model = selection_model
        
    disambiguation_model = os.getenv("CLAIMIFY_DISAMBIGUATION_MODEL")
    if disambiguation_model:
        config.disambiguation_model = disambiguation_model
        
    decomposition_model = os.getenv("CLAIMIFY_DECOMPOSITION_MODEL")
    if decomposition_model:
        config.decomposition_model = decomposition_model
        
    default_model = os.getenv("CLAIMIFY_DEFAULT_MODEL")
    if default_model:
        config.default_model = default_model
    
    # Processing settings
    max_retries = os.getenv("CLAIMIFY_MAX_RETRIES")
    if max_retries is not None:
        config.max_retries = int(max_retries)
        
    timeout_seconds = os.getenv("CLAIMIFY_TIMEOUT_SECONDS")
    if timeout_seconds is not None:
        config.timeout_seconds = int(timeout_seconds)
        
    temperature = os.getenv("CLAIMIFY_TEMPERATURE")
    if temperature is not None:
        config.temperature = float(temperature)
        
    max_tokens = os.getenv("CLAIMIFY_MAX_TOKENS")
    if max_tokens is not None:
        config.max_tokens = int(max_tokens)
    
    # Threshold settings
    selection_threshold = os.getenv("CLAIMIFY_SELECTION_CONFIDENCE_THRESHOLD")
    if selection_threshold is not None:
        config.selection_confidence_threshold = float(selection_threshold)
        
    disambiguation_threshold = os.getenv("CLAIMIFY_DISAMBIGUATION_CONFIDENCE_THRESHOLD")
    if disambiguation_threshold is not None:
        config.disambiguation_confidence_threshold = float(disambiguation_threshold)
        
    decomposition_threshold = os.getenv("CLAIMIFY_DECOMPOSITION_CONFIDENCE_THRESHOLD")
    if decomposition_threshold is not None:
        config.decomposition_confidence_threshold = float(decomposition_threshold)
    
    # Logging settings
    log_decisions = os.getenv("CLAIMIFY_LOG_DECISIONS")
    if log_decisions is not None:
        config.log_decisions = log_decisions.lower() in ("true", "1", "yes")
        
    log_transformations = os.getenv("CLAIMIFY_LOG_TRANSFORMATIONS")
    if log_transformations is not None:
        config.log_transformations = log_transformations.lower() in ("true", "1", "yes")
        
    log_timing = os.getenv("CLAIMIFY_LOG_TIMING")
    if log_timing is not None:
        config.log_timing = log_timing.lower() in ("true", "1", "yes")
    
    return config


def _load_from_config_file() -> Optional[ClaimifyConfig]:
    """
    Load configuration from YAML config file.
    Returns:
        ClaimifyConfig instance or None if no config file found
    """
    # Look for config file in common locations
    config_file = _find_config_file()
    if not config_file:
        return None
    
    try:
        with open(config_file, "r") as f:
            config_data = yaml.safe_load(f) or {}
        return _parse_config_data(config_data)
    except Exception as e:
        logger.error(f"[config._load_from_config_file] Failed to load config from {config_file}: {e}")
        return None


def _find_config_file() -> Optional[str]:
    """
    Find claimify config file in common locations.
    Returns:
        Path to config file or None if not found
    """
    current_path = Path.cwd()
    search_paths = []
    
    # Priority 1: settings directory
    for path in [current_path] + list(current_path.parents):
        search_paths.append(path / "settings" / "claimify.config.yaml")
    
    # Priority 2: root level in current and parent directories
    for path in [current_path] + list(current_path.parents):
        search_paths.append(path / "claimify.config.yaml")
    
    for config_path in search_paths:
        if config_path.exists():
            return str(config_path)
    
    return None


def _parse_config_data(config_data: Dict[str, Any]) -> Optional[ClaimifyConfig]:
    """
    Parse configuration data from dictionary.
    Args:
        config_data: Dictionary containing configuration data
    Returns:
        ClaimifyConfig instance or None if parsing failed
    """
    try:
        # Context window settings
        window_config = config_data.get("window", {})
        claimify_window = window_config.get("claimify", {})
        context_window_p = claimify_window.get("p", 3)
        context_window_f = claimify_window.get("f", 1)
        
        # Model settings
        model_config = config_data.get("model", {})
        claimify_models = model_config.get("claimify", {})
        selection_model = claimify_models.get("selection")
        disambiguation_model = claimify_models.get("disambiguation")
        decomposition_model = claimify_models.get("decomposition")
        claimify_default = claimify_models.get("default")
        # Fall back to global processing config if claimify-specific values not set
        if not claimify_default:
            claimify_default = model_config.get("fallback_plugin", "gpt-3.5-turbo")
        
        # Processing settings
        processing_config = config_data.get("processing", {})
        claimify_processing = processing_config.get("claimify", {})
        max_retries = claimify_processing.get("max_retries", 3)
        timeout_seconds = processing_config.get("timeout_seconds", 30)
        temperature = processing_config.get("temperature", 0.1)
        max_tokens = processing_config.get("max_tokens", 1000)
        
        # Threshold settings
        selection_confidence_threshold = 0.5
        disambiguation_confidence_threshold = 0.5
        decomposition_confidence_threshold = 0.5
        
        # Logging settings
        logging_config = claimify_processing.get("logging", {})
        log_decisions = logging_config.get("log_decisions", True)
        log_transformations = logging_config.get("log_transformations", True)
        log_timing = logging_config.get("log_timing", True)
        
        return ClaimifyConfig(
            context_window_p=context_window_p,
            context_window_f=context_window_f,
            selection_model=selection_model,
            disambiguation_model=disambiguation_model,
            decomposition_model=decomposition_model,
            default_model=claimify_default,
            max_retries=max_retries,
            timeout_seconds=timeout_seconds,
            temperature=temperature,
            max_tokens=max_tokens,
            selection_confidence_threshold=selection_confidence_threshold,
            disambiguation_confidence_threshold=disambiguation_confidence_threshold,
            decomposition_confidence_threshold=decomposition_confidence_threshold,
            log_decisions=log_decisions,
            log_transformations=log_transformations,
            log_timing=log_timing,
        )
    except Exception as e:
        logger.error(f"[config._parse_config_data] Failed to parse config data: {e}")
        return None


def _merge_configs(base_config: ClaimifyConfig, override_config: ClaimifyConfig) -> ClaimifyConfig:
    """
    Merge two configurations, with override_config taking precedence.
    Args:
        base_config: Base configuration
        override_config: Configuration with override values
    Returns:
        Merged configuration
    """
    # For now, we'll just return the override config since it already has all the values
    # In a more complex scenario, we might want to merge specific fields
    return override_config


def load_prompt_template(stage: str) -> Optional[Dict[str, Any]]:
    """
    Load prompt template for a specific stage using importlib.resources.
    Args:
        stage: Stage name ("selection", "disambiguation", "decomposition")
    Returns:
        Dictionary with prompt template data or None if loading failed
    """
    try:
        import importlib.resources as resources
    except ImportError:
        # Fallback for older Python versions
        import importlib_resources as resources
    
    prompt_files = {
        "selection": "claimify_selection.yaml",
        "disambiguation": "claimify_disambiguation.yaml",
        "decomposition": "claimify_decomposition.yaml"
    }
    
    if stage not in prompt_files:
        logger.error(f"[config.load_prompt_template] Unknown stage: {stage}")
        return None
    
    prompt_file = prompt_files[stage]
    
    try:
        # Try to load from the package resources
        with resources.open_text("aclarai_claimify.prompts", prompt_file) as f:
            prompt_data = yaml.safe_load(f)
        return prompt_data
    except Exception as e:
        logger.error(f"[config.load_prompt_template] Failed to load prompt template for {stage}: {e}")
        return None