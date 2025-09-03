import pytest
from unittest.mock import patch, MagicMock
from aclarai_claimify.scout.nodes import (
    supervisor_node,
    research_node,
    archive_node,
    fitness_node,
    synthetic_node
)
from aclarai_claimify.scout.state import DataScoutState

class TestNodes:
    """Test suite for the Data Scout Agent nodes."""
    
    def create_test_state(self):
        """Create a test state object."""
        return {
            "messages": [],
            "task_queue": [],
            "research_findings": [],
            "pedigree_path": "test_pedigree.md",
            "decision_history": [],
            "tool_execution_failures": 0,
            "research_attempts": 0,
            "samples_generated": 0,
            "total_samples_target": 1200,
            "current_mission": "test_mission",
            "consecutive_failures": 0,
            "last_action_status": "success",
            "last_action_agent": ""
        }
    
    @patch('aclarai_claimify.scout.nodes.load_claimify_config')
    @patch('aclarai_claimify.scout.nodes.ChatLiteLLM')
    def test_supervisor_node_research_decision(self, mock_chat_llm, mock_load_config):
        """Test supervisor node making a research decision."""
        # Mock config and LLM
        mock_config = MagicMock()
        mock_config.temperature = 0.1
        mock_config.max_tokens = 2000
        mock_config.scout_agent = None
        mock_load_config.return_value = mock_config
        
        mock_llm_instance = MagicMock()
        mock_llm_instance.model = "ollama/gpt-oss:20b"
        mock_chat_llm.return_value = mock_llm_instance
        
        # Mock structured output
        mock_structured_llm = MagicMock()
        mock_decision = MagicMock()
        mock_decision.next_agent = "research"
        mock_structured_llm.invoke.return_value = mock_decision
        mock_llm_instance.with_structured_output.return_value = mock_structured_llm
        
        # Create test state
        state = self.create_test_state()
        
        # Call the node
        result = supervisor_node(state)
        
        # Verify the result
        assert result["next_agent"] == "research"
        assert "research" in result["decision_history"]
        assert result["tool_execution_failures"] == 0
    
    @patch('aclarai_claimify.scout.nodes.load_claimify_config')
    @patch('aclarai_claimify.scout.nodes.ChatLiteLLM')
    def test_supervisor_node_synthetic_fallback(self, mock_chat_llm, mock_load_config):
        """Test supervisor node falling back to synthetic agent after failures."""
        # Mock config and LLM
        mock_config = MagicMock()
        mock_config.temperature = 0.1
        mock_config.max_tokens = 2000
        mock_config.scout_agent = None
        mock_load_config.return_value = mock_config
        
        mock_llm_instance = MagicMock()
        mock_llm_instance.model = "ollama/gpt-oss:20b"
        mock_chat_llm.return_value = mock_llm_instance
        
        # Mock structured output
        mock_structured_llm = MagicMock()
        mock_decision = MagicMock()
        mock_decision.next_agent = "synthetic"
        mock_structured_llm.invoke.return_value = mock_decision
        mock_llm_instance.with_structured_output.return_value = mock_structured_llm
        
        # Create test state with failures
        state = self.create_test_state()
        state["consecutive_failures"] = 3  # Trigger synthetic fallback
        
        # Call the node
        result = supervisor_node(state)
        
        # Verify the result
        assert result["next_agent"] == "synthetic"
        assert "synthetic" in result["decision_history"]
    
    @patch('aclarai_claimify.scout.nodes.load_claimify_config')
    @patch('aclarai_claimify.scout.nodes.ChatLiteLLM')
    @patch('aclarai_claimify.scout.nodes.get_tools_for_role')
    def test_research_node_success(self, mock_get_tools, mock_chat_llm, mock_load_config):
        """Test research node successful operation."""
        # Mock config and LLM
        mock_config = MagicMock()
        mock_config.temperature = 0.1
        mock_config.max_tokens = 2000
        mock_config.scout_agent = None
        mock_load_config.return_value = mock_config
        
        mock_llm_instance = MagicMock()
        mock_llm_instance.model = "ollama/gpt-oss:20b"
        mock_chat_llm.return_value = mock_llm_instance
        
        # Mock tools
        mock_get_tools.return_value = []
        
        # Create test state with a user message
        state = self.create_test_state()
        from langchain_core.messages import HumanMessage
        state["messages"] = [HumanMessage(content="Find information about LangGraph")]
        
        # Mock LLM response
        mock_response = MagicMock()
        mock_response.content = "Here is information about LangGraph..."
        mock_response.tool_calls = []
        mock_llm_instance.bind_tools.return_value = mock_llm_instance
        mock_llm_instance.invoke.return_value = mock_response
        
        # Call the node
        result = research_node(state)
        
        # Verify the result
        assert "messages" in result
        assert len(result["messages"]) == 1
        assert "LangGraph" in result["messages"][0].content
    
    @patch('aclarai_claimify.scout.nodes.load_claimify_config')
    @patch('aclarai_claimify.scout.nodes.ChatLiteLLM')
    def test_fitness_node_evaluation(self, mock_chat_llm, mock_load_config):
        """Test fitness node evaluation."""
        # Mock config and LLM
        mock_config = MagicMock()
        mock_config.temperature = 0.1
        mock_config.max_tokens = 2000
        mock_config.scout_agent = None
        mock_load_config.return_value = mock_config
        
        mock_llm_instance = MagicMock()
        mock_llm_instance.model = "ollama/gpt-oss:20b"
        mock_chat_llm.return_value = mock_llm_instance
        
        # Mock LLM response
        mock_response = MagicMock()
        mock_response.content = "The research findings are good and meet the criteria."
        mock_llm_instance.invoke.return_value = mock_response
        
        # Create test state
        state = self.create_test_state()
        
        # Call the node
        result = fitness_node(state)
        
        # Verify the result
        assert "messages" in result
        assert len(result["messages"]) == 1
        assert "research findings" in result["messages"][0].content
    
    @patch('aclarai_claimify.scout.nodes.load_claimify_config')
    @patch('aclarai_claimify.scout.nodes.ChatLiteLLM')
    def test_synthetic_node_generation(self, mock_chat_llm, mock_load_config):
        """Test synthetic node generation."""
        # Mock config and LLM
        mock_config = MagicMock()
        mock_config.temperature = 0.1
        mock_config.max_tokens = 2000
        mock_config.scout_agent = None
        mock_load_config.return_value = mock_config
        
        mock_llm_instance = MagicMock()
        mock_llm_instance.model = "ollama/gpt-oss:20b"
        mock_chat_llm.return_value = mock_llm_instance
        
        # Mock LLM response
        mock_response = MagicMock()
        mock_response.content = "Generating synthetic examples..."
        mock_response.tool_calls = []
        mock_llm_instance.bind_tools.return_value = mock_llm_instance
        mock_llm_instance.invoke.return_value = mock_response
        
        # Create test state
        state = self.create_test_state()
        
        # Call the node
        result = synthetic_node(state)
        
        # Verify the result
        assert "messages" in result
        assert len(result["messages"]) == 1
        assert "synthetic" in result["messages"][0].content

    @patch('aclarai_claimify.scout.nodes.load_claimify_config')
    @patch('aclarai_claimify.scout.nodes.create_llm')
    @patch('aclarai_claimify.scout.nodes._get_next_research_task')
    @patch('aclarai_claimify.scout.nodes.write_file')
    @patch('aclarai_claimify.scout.nodes.append_to_pedigree')
    @patch('time.strftime')
    def test_archive_node_filepath_generation(self, mock_strftime, mock_append_to_pedigree, mock_write_file, mock_get_next_research_task, mock_create_llm, mock_load_claimify_config):
        """Test that the archive_node generates a deterministic filepath."""
        # Arrange
        # Mock configs and LLMs
        mock_load_claimify_config.return_value = MagicMock()
        mock_create_llm.return_value = MagicMock()

        # Mock the LLM response to only return the markdown
        mock_llm_result = MagicMock()
        mock_llm_result.content = '{"entry_markdown": "### 2025-09-04 — Sample Archived..."}'
        mock_create_llm.return_value.invoke.return_value = mock_llm_result

        # Mock the task queue to provide characteristic and topic
        mock_get_next_research_task.return_value = {
            "characteristic": "Test Characteristic",
            "topic": "Test Topic"
        }

        # Mock time to get a predictable timestamp
        mock_strftime.return_value = "20250904123456"

        # Mock the file writing and pedigree appending to be successful
        mock_write_file.invoke.return_value = {"status": "ok"}
        mock_append_to_pedigree.return_value = None

        # Create a state with a dummy report
        from langchain_core.messages import AIMessage
        state = self.create_test_state()
        state["messages"] = [AIMessage(content="# Data Prospecting Report\nThis is a test report.")]
        state["task_queue"] = ["dummy_task"]

        # Act
        result = archive_node(state)

        # Assert
        # 1. Check that the filepath was constructed as expected
        expected_filename = "test_characteristic_test_topic_20250904123456.md"
        expected_filepath = f"output/approved_books/{expected_filename}"
        
        # 2. Verify that write_file was called with the correct path and content
        mock_write_file.invoke.assert_called_once_with({
            "filepath": expected_filepath,
            "content": "# Data Prospecting Report\nThis is a test report."
        })

        # 3. Verify that the pedigree was updated
        mock_append_to_pedigree.assert_called_once()

        # 4. Check the confirmation message
        assert "Successfully archived document" in result["messages"][-1].content
        assert expected_filepath in result["messages"][-1].content
