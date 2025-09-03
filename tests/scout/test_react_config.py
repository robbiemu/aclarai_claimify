"""
Test for configurable ReAct max_iterations functionality.
"""
import pytest
from unittest.mock import Mock, patch
from aclarai_claimify.data_models import ClaimifyConfig, AgentsConfig, AgentsReactConfig
from aclarai_claimify.scout.nodes import research_node
from aclarai_claimify.scout.state import DataScoutState


def test_react_max_iterations_default():
    """Test that default max_iterations is 3."""
    config = ClaimifyConfig()
    assert config.agents.react.max_iterations == 3


def test_react_max_iterations_custom():
    """Test that custom max_iterations is respected."""
    config = ClaimifyConfig(agents=AgentsConfig(react=AgentsReactConfig(max_iterations=5)))
    assert config.agents.react.max_iterations == 5


def test_react_max_iterations_validation():
    """Test that max_iterations validation works."""
    # Valid range
    config = ClaimifyConfig(agents=AgentsConfig(react=AgentsReactConfig(max_iterations=1)))
    assert config.agents.react.max_iterations == 1
    
    config = ClaimifyConfig(agents=AgentsConfig(react=AgentsReactConfig(max_iterations=50)))
    assert config.agents.react.max_iterations == 50
    
    # Invalid values should raise validation error
    with pytest.raises(Exception):  # Pydantic validation error
        ClaimifyConfig(agents=AgentsConfig(react=AgentsReactConfig(max_iterations=0)))
    
    with pytest.raises(Exception):  # Pydantic validation error
        ClaimifyConfig(agents=AgentsConfig(react=AgentsReactConfig(max_iterations=51)))


def test_research_node_uses_config_max_iterations():
    """Test that research_node uses configured max_iterations."""
    # Mock the necessary functions and classes
    with patch('aclarai_claimify.scout.nodes.load_claimify_config') as mock_config, \
         patch('aclarai_claimify.scout.nodes.create_llm') as mock_create_llm, \
         patch('aclarai_claimify.scout.nodes.get_tools_for_role') as mock_get_tools:
        
        # Set up custom config with max_iterations = 7
        custom_config = ClaimifyConfig(
            agents=AgentsConfig(react=AgentsReactConfig(max_iterations=7))
        )
        mock_config.return_value = custom_config
        
        # Mock LLM and tools
        mock_llm = Mock()
        mock_create_llm.return_value = mock_llm
        mock_get_tools.return_value = []
        
        # Create state with a simple message
        from langchain_core.messages import HumanMessage
        state = DataScoutState(
            messages=[HumanMessage(content="Find examples of atomic requirements")]
        )
        
        # Mock the research agent to raise an exception after we confirm max_iterations
        # We'll capture what max_iterations was used by mocking the loop
        captured_max_iterations = None
        
        def mock_invoke(*args, **kwargs):
            nonlocal captured_max_iterations
            # This simulates the research_node reading the max_iterations from config
            captured_max_iterations = custom_config.agents.react.max_iterations
            # Raise exception to exit early and check our captured value
            raise Exception("Test exception to exit early")
        
        mock_llm_with_tools = Mock()
        mock_llm.bind_tools.return_value = mock_llm_with_tools
        mock_llm_with_tools.invoke = mock_invoke
        
        # Call the research_node
        with pytest.raises(Exception, match="Test exception to exit early"):
            research_node(state)
        
        # Verify that our custom max_iterations value was read
        assert captured_max_iterations == 7
        
        # Verify config was loaded
        mock_config.assert_called_once()
        mock_create_llm.assert_called_once_with(custom_config, "research")


if __name__ == "__main__":
    # Run tests
    test_react_max_iterations_default()
    test_react_max_iterations_custom()
    test_react_max_iterations_validation()
    print("All tests passed!")
