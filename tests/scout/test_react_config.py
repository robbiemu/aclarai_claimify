"""
Test for configurable ReAct max_iterations functionality.
"""
import pytest
from unittest.mock import Mock, patch
from aclarai_claimify.data_models import ClaimifyConfig, ScoutAgentConfig, ScoutAgentNodesConfig, ScoutAgentResearchNodeConfig, ScoutAgentMissionPlanConfig, ScoutAgentWriterConfig
from aclarai_claimify.scout.nodes import research_node
from aclarai_claimify.scout.state import DataScoutState
from langchain_core.messages import HumanMessage, AIMessage


def test_research_node_max_iterations_default():
    """Test that the default max_iterations for the research node is 7."""
    config = ScoutAgentNodesConfig()
    assert config.research.max_iterations == 7


def test_research_node_max_iterations_custom():
    """Test that custom max_iterations for the research node is respected."""
    config = ScoutAgentNodesConfig(research=ScoutAgentResearchNodeConfig(max_iterations=5))
    assert config.research.max_iterations == 5


def test_research_node_max_iterations_validation():
    """Test that research node max_iterations validation works."""
    # Valid range
    ScoutAgentResearchNodeConfig(max_iterations=1)
    ScoutAgentResearchNodeConfig(max_iterations=50)
    
    # Invalid values should raise validation error
    with pytest.raises(Exception):
        ScoutAgentResearchNodeConfig(max_iterations=0)
    with pytest.raises(Exception):
        ScoutAgentResearchNodeConfig(max_iterations=51)


@patch('aclarai_claimify.scout.nodes.load_claimify_config')
@patch('aclarai_claimify.scout.nodes.ChatLiteLLM')
@patch('aclarai_claimify.scout.nodes.get_tools_for_role')
def test_research_node_uses_config_max_iterations(mock_get_tools, mock_chat_llm, mock_load_config):
    """Test that research_node uses configured max_iterations."""
    # --- Arrange ---
    # Set up custom config with max_iterations = 5
    mock_research_config = ScoutAgentResearchNodeConfig(max_iterations=5)
    mock_scout_config = ScoutAgentConfig(
        mission_plan=ScoutAgentMissionPlanConfig(goal="Test Goal", nodes=[], tools={}),
        writer=ScoutAgentWriterConfig(),
        nodes=ScoutAgentNodesConfig(research=mock_research_config)
    )
    custom_config = ClaimifyConfig(scout_agent=mock_scout_config)
    mock_load_config.return_value = custom_config

    # Mock LLM and tools
    mock_llm_instance = Mock()
    mock_chat_llm.return_value = mock_llm_instance
    mock_get_tools.return_value = []

    # Mock LLM to return a final report on the last iteration
    final_report = AIMessage(content="# Data Prospecting Report\nSuccess")
    # Simulate the loop by having invoke return non-reports until the last call
    mock_llm_instance.invoke.side_effect = [AIMessage(content="Thinking...")] * 4 + [final_report]

    # Create state
    state = DataScoutState(messages=[HumanMessage(content="Find stuff")])

    # --- Act ---
    research_node(state)

    # --- Assert ---
    # The research_node's internal loop should run `max_iterations` times.
    # So, the LLM should be invoked 5 times.
    assert mock_llm_instance.invoke.call_count == 5
