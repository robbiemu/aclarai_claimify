"""
Tests for the Claimify configuration system.
"""
import pytest
import importlib.resources as resources
import yaml
from aclarai_claimify.config import _parse_claimify_config_data, load_claimify_config
from aclarai_claimify.data_models import ClaimifyConfig


def test_scout_agent_config_parsing():
    """Test that the scout_agent configuration is correctly parsed from YAML."""
    # Load the default config from the package
    with resources.open_text("aclarai_claimify.settings", "config.yaml") as f:
        config_data = yaml.safe_load(f)
    
    # Parse the configuration
    config = _parse_claimify_config_data(config_data)
    
    # Check that the config is valid
    assert config is not None
    assert isinstance(config, ClaimifyConfig)
    
    # Check that scout_agent is present and correctly configured
    assert config.scout_agent is not None
    assert config.scout_agent.mission_plan is not None
    assert config.scout_agent.mission_plan.goal is not None
    assert len(config.scout_agent.mission_plan.nodes) > 0
    
    # Check that all nodes have valid max_tokens values (>= 1)
    for node in config.scout_agent.mission_plan.nodes:
        assert node.max_tokens >= 1, f"Node {node.name} has invalid max_tokens: {node.max_tokens}"


def test_scout_agent_config_validation():
    """Test that the scout_agent configuration validates correctly."""
    # Load the default config from the package
    with resources.open_text("aclarai_claimify.settings", "config.yaml") as f:
        config_data = yaml.safe_load(f)
    
    # Parse the configuration
    config = _parse_claimify_config_data(config_data)
    
    # The config should be valid
    assert config is not None
    
    # Validate that the scout_agent configuration is complete
    scout_agent = config.scout_agent
    assert scout_agent is not None
    
    # Check mission_plan
    mission_plan = scout_agent.mission_plan
    assert mission_plan.goal is not None and len(mission_plan.goal) > 0
    assert mission_plan.max_iterations >= 1
    assert len(mission_plan.nodes) > 0
    
    # Check nodes
    for node in mission_plan.nodes:
        assert node.name is not None and len(node.name) > 0
        assert node.model is not None and len(node.model) > 0
        assert 0.0 <= node.temperature <= 2.0
        assert node.max_tokens >= 1
    
    # Check writer config
    writer = scout_agent.writer
    assert writer.tier1_path is not None
    assert writer.tier2_path is not None
    assert writer.audit_trail_path is not None
    
    # Check checkpointer path
    assert scout_agent.checkpointer_path is not None


def test_scout_agent_config_loading():
    """Test that the scout_agent configuration loads correctly."""
    # Load the configuration
    config = load_claimify_config()
    
    # Check that scout_agent is present
    assert config.scout_agent is not None
    
    # Check mission_plan
    mission_plan = config.scout_agent.mission_plan
    assert mission_plan.goal is not None
    assert isinstance(mission_plan.max_iterations, int)
    assert len(mission_plan.nodes) > 0
    
    # Check that we have the expected nodes
    node_names = [node.name for node in mission_plan.nodes]
    expected_nodes = ["DeconstructGoalNode", "PlanNode", "WebSearchNode", "FitnessCheckNode", "ArchivingNode"]
    for expected_node in expected_nodes:
        assert expected_node in node_names, f"Expected node {expected_node} not found in {node_names}"


def test_scout_agent_node_config_access():
    """Test that we can access node configurations correctly."""
    from aclarai_claimify.scout.nodes import get_node_config
    
    # Test that we can get each node configuration
    node_names = ["DeconstructGoalNode", "PlanNode", "WebSearchNode", "FitnessCheckNode", "ArchivingNode"]
    for node_name in node_names:
        node_config = get_node_config(node_name)
        assert node_config is not None, f"Could not get config for node {node_name}"
        assert node_config.name == node_name
        assert node_config.model is not None
        assert 0.0 <= node_config.temperature <= 2.0
        assert node_config.max_tokens >= 1


def test_scout_agent_config_zero_max_tokens_validation():
    """Test that zero max_tokens values are rejected by validation."""
    # Create a config with invalid max_tokens (0)
    invalid_config_data = {
        "scout_agent": {
            "mission_plan": {
                "goal": "Test goal",
                "max_iterations": 5,
                "nodes": [
                    {
                        "name": "TestNode",
                        "model": "test-model",
                        "temperature": 0.7,
                        "max_tokens": 0  # This should be rejected
                    }
                ]
            },
            "writer": {
                "tier1_path": "./test_tier1",
                "tier2_path": "./test_tier2",
                "audit_trail_path": "./test_pedigree.md"
            },
            "checkpointer_path": "./test_checkpointer.sqlite"
        }
    }
    
    # Try to parse the configuration - this should fail validation
    config = _parse_claimify_config_data(invalid_config_data)
    
    # The config should be None because of validation failure
    # OR if it falls back to default, scout_agent should be None
    if config is not None:
        assert config.scout_agent is None or any(node.max_tokens >= 1 for node in config.scout_agent.mission_plan.nodes)


def test_scout_agent_config_missing_field_validation():
    """Test that missing required fields are handled correctly."""
    # Create a config with missing required fields
    incomplete_config_data = {
        "scout_agent": {
            # Missing mission_plan.goal
            "mission_plan": {
                "max_iterations": 5,
                "nodes": []
            },
            "writer": {
                "tier1_path": "./test_tier1",
                "tier2_path": "./test_tier2",
                "audit_trail_path": "./test_pedigree.md"
            },
            "checkpointer_path": "./test_checkpointer.sqlite"
        }
    }
    
    # Try to parse the configuration
    config = _parse_claimify_config_data(incomplete_config_data)
    
    # The config should be None or scout_agent should be None due to validation failure
    if config is not None:
        assert config.scout_agent is None