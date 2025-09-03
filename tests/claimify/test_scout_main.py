"""
Tests for the Data Scout Agent main functionality.
"""
import pytest
from unittest.mock import patch, MagicMock
from aclarai_claimify.scout.main import run
from aclarai_claimify.config import load_claimify_config


def test_scout_agent_main_function_startup():
    """Test that the scout agent main function can start up correctly."""
    # Load the configuration to ensure it's valid
    config = load_claimify_config()
    
    # Check that scout_agent configuration is present
    assert config.scout_agent is not None
    assert config.scout_agent.mission_plan is not None
    assert config.scout_agent.mission_plan.goal is not None
    
    # Check that all required fields are present
    assert len(config.scout_agent.mission_plan.goal) > 0
    assert config.scout_agent.mission_plan.max_iterations >= 1
    assert len(config.scout_agent.mission_plan.nodes) > 0
    
    # Check that the checkpointer path is set
    assert config.scout_agent.checkpointer_path is not None


def test_scout_agent_config_inclusion_in_flat_config():
    """Test that scout_agent is included in the flat_config dictionary during parsing."""
    from aclarai_claimify.config import _parse_claimify_config_data
    import importlib.resources as resources
    import yaml
    
    # Load the default config
    with resources.open_text("aclarai_claimify.settings", "config.yaml") as f:
        config_data = yaml.safe_load(f)
    
    # Check that scout_agent is in the raw config data
    assert "scout_agent" in config_data
    assert config_data["scout_agent"] is not None
    
    # Parse the configuration
    config = _parse_claimify_config_data(config_data)
    
    # Check that scout_agent is present in the parsed config
    assert config is not None
    assert config.scout_agent is not None


@patch('aclarai_claimify.scout.main.load_claimify_config')
def test_scout_agent_main_handles_missing_config(mock_load_config):
    """Test that the main function handles missing scout_agent configuration gracefully."""
    # Mock the config to return a config without scout_agent
    mock_config = MagicMock()
    mock_config.scout_agent = None
    mock_load_config.return_value = mock_config
    
    # Run the main function - it should not crash
    run()
    
    # Verify that load_claimify_config was called
    mock_load_config.assert_called_once()


def test_scout_agent_node_config_validation():
    """Test that all node configurations have valid parameters."""
    config = load_claimify_config()
    
    # Check that we have scout_agent config
    assert config.scout_agent is not None
    
    # Check each node
    for node in config.scout_agent.mission_plan.nodes:
        # Check that name is valid
        assert node.name is not None
        assert len(node.name) > 0
        
        # Check that model is valid
        assert node.model is not None
        assert len(node.model) > 0
        
        # Check that temperature is in valid range
        assert 0.0 <= node.temperature <= 2.0
        
        # Check that max_tokens is valid (this is the key test for our bug)
        assert node.max_tokens >= 1, f"Node {node.name} has invalid max_tokens: {node.max_tokens}"