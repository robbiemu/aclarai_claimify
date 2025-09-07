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

@patch('aclarai_claimify.scout.main.input')
@patch('aclarai_claimify.scout.main.MissionRunner')
@patch('aclarai_claimify.scout.main.load_claimify_config')
def test_scout_agent_runnable_new_mission(mock_load_config, mock_mission_runner, mock_input):
    """
    Tests that the scout agent can start a new mission.
    """
    # Mock the input to exit immediately
    mock_input.return_value = "exit"

    # Mock the MissionRunner
    mock_runner_instance = mock_mission_runner.return_value
    mock_runner_instance.get_mission_names.return_value = ['test_mission']
    mock_runner_instance.start_new_mission.return_value = 'new-thread-id'
    mock_runner_instance.get_progress.return_value = {
        "samples_generated": 0, "total_target": 10, "progress_pct": 0.0
    }

    # Mock the configuration
    mock_mission_plan = ScoutAgentMissionPlanConfig(goal="Test Goal", nodes=[], tools={})
    mock_writer_config = ScoutAgentWriterConfig()
    mock_scout_config = ScoutAgentConfig(
        mission_plan=mock_mission_plan, writer=mock_writer_config
    )
    mock_claimify_config = MagicMock(spec=ClaimifyConfig)
    mock_claimify_config.scout_agent = mock_scout_config
    mock_load_config.return_value = mock_claimify_config

    # We expect this to run without errors
    run(mission_name='test_mission', resume_from=None)

    # Check that the config was loaded and MissionRunner was used
    mock_load_config.assert_called_once()
    mock_mission_runner.assert_called_once_with("settings/scout_mission.yaml")
    mock_runner_instance.start_new_mission.assert_called_once()


@patch('aclarai_claimify.scout.main.input')
@patch('aclarai_claimify.scout.main.MissionRunner')
@patch('aclarai_claimify.scout.main.load_claimify_config')
def test_scout_agent_runnable_resume_mission(mock_load_config, mock_mission_runner, mock_input):
    """
    Tests that the scout agent can resume a mission.
    """
    # Mock the input to exit immediately
    mock_input.return_value = "exit"

    # Mock the MissionRunner
    mock_runner_instance = mock_mission_runner.return_value
    mock_runner_instance.get_mission_names.return_value = ['test_mission']
    mock_runner_instance.resume_mission.return_value = True
    mock_runner_instance.get_progress.return_value = {
        "samples_generated": 5, "total_target": 10, "progress_pct": 50.0
    }

    # Mock the configuration
    mock_mission_plan = ScoutAgentMissionPlanConfig(goal="Test Goal", nodes=[], tools={})
    mock_writer_config = ScoutAgentWriterConfig()
    mock_scout_config = ScoutAgentConfig(
        mission_plan=mock_mission_plan, writer=mock_writer_config
    )
    mock_claimify_config = MagicMock(spec=ClaimifyConfig)
    mock_claimify_config.scout_agent = mock_scout_config
    mock_load_config.return_value = mock_claimify_config

    # We expect this to run without errors
    run(mission_name='test_mission', resume_from='existing-thread-id')

    # Check that the config was loaded and MissionRunner was used
    mock_load_config.assert_called_once()
    mock_mission_runner.assert_called_once_with("settings/scout_mission.yaml")
    mock_runner_instance.resume_mission.assert_called_once()