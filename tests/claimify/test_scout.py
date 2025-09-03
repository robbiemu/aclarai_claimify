from unittest.mock import patch, MagicMock

import pytest

pytest.importorskip("langsmith")

from aclarai_claimify.scout.main import run
from aclarai_claimify.data_models import (
    ClaimifyConfig,
    ScoutAgentConfig,
    ScoutAgentMissionPlanConfig,
    ScoutAgentWriterConfig,
)

@patch('aclarai_claimify.scout.main.SqliteSaver')
@patch('aclarai_claimify.scout.main.create_graph')
@patch('aclarai_claimify.scout.main.load_claimify_config')
def test_scout_agent_runnable_new_mission(mock_load_config, mock_create_graph, mock_sqlite_saver):
    """
    Tests that the scout agent can start a new mission.
    """
    # Mock the configuration
    mock_mission_plan = ScoutAgentMissionPlanConfig(
        goal="Test Goal", max_iterations=1, nodes=[]
    )
    mock_writer_config = ScoutAgentWriterConfig()
    mock_scout_config = ScoutAgentConfig(
        mission_plan=mock_mission_plan, writer=mock_writer_config
    )
    mock_claimify_config = MagicMock(spec=ClaimifyConfig)
    mock_claimify_config.scout_agent = mock_scout_config
    mock_load_config.return_value = mock_claimify_config

    # Mock the checkpointer context manager
    mock_checkpointer = MagicMock()
    mock_checkpointer.get.return_value = None
    mock_sqlite_saver.from_conn_string.return_value.__enter__.return_value = mock_checkpointer

    # Mock the graph compilation and execution
    mock_graph = mock_create_graph.return_value
    mock_graph.stream.return_value = iter([])

    # We expect this to run without errors
    run()

    # Check that the config was loaded and the graph was created and executed
    mock_load_config.assert_called_once()
    mock_sqlite_saver.from_conn_string.assert_called_once_with(".checkpointer.sqlite")
    mock_checkpointer.get.assert_called_once()
    mock_create_graph.assert_called_once()
    mock_graph.stream.assert_called_once()


@patch('aclarai_claimify.scout.main.SqliteSaver')
@patch('aclarai_claimify.scout.main.create_graph')
@patch('aclarai_claimify.scout.main.load_claimify_config')
def test_scout_agent_runnable_resume_mission(mock_load_config, mock_create_graph, mock_sqlite_saver):
    """
    Tests that the scout agent can resume a mission.
    """
    # Mock the configuration
    mock_mission_plan = ScoutAgentMissionPlanConfig(
        goal="Test Goal", max_iterations=1, nodes=[]
    )
    mock_writer_config = ScoutAgentWriterConfig()
    mock_scout_config = ScoutAgentConfig(
        mission_plan=mock_mission_plan, writer=mock_writer_config
    )
    mock_claimify_config = MagicMock(spec=ClaimifyConfig)
    mock_claimify_config.scout_agent = mock_scout_config
    mock_load_config.return_value = mock_claimify_config

    # Mock the checkpointer context manager
    mock_checkpointer = MagicMock()
    mock_checkpointer.get.return_value = {"some": "state"}
    mock_sqlite_saver.from_conn_string.return_value.__enter__.return_value = mock_checkpointer

    # Mock the graph compilation and execution
    mock_graph = mock_create_graph.return_value
    mock_graph.stream.return_value = iter([])

    # We expect this to run without errors
    run()

    # Check that the config was loaded and the graph was created and executed
    mock_load_config.assert_called_once()
    mock_sqlite_saver.from_conn_string.assert_called_once_with(".checkpointer.sqlite")
    mock_checkpointer.get.assert_called_once()
    mock_create_graph.assert_called_once()
    mock_graph.stream.assert_called_once()
