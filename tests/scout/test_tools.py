import pytest
from unittest.mock import patch, MagicMock
from aclarai_claimify.scout.tools import (
    create_web_search_tool,
    url_to_markdown,
    _safe_request_get,
    get_tools_for_role,
)

# Create the web_search tool for testing
web_search = create_web_search_tool()


class TestTools:
    """Test suite for the Data Scout Agent tools."""

    def test_get_tools_for_role(self):
        """Test that tools are correctly assigned to roles."""
        # Test research role
        research_tools = get_tools_for_role("research")
        tool_names = [tool.name for tool in research_tools]
        assert "web_search" in tool_names
        assert "url_to_markdown" in tool_names

        # Test synthetic role
        # we have no tools to test

        # Test archive role
        archive_tools = get_tools_for_role("archive")
        tool_names = [tool.name for tool in archive_tools]
        assert "write_file" in tool_names
        assert "append_to_pedigree" in tool_names

    @patch("aclarai_claimify.scout.tools.SearchProviderProxy")
    def test_web_search_success(self, mock_search_proxy):
        """Test successful web search."""
        # Mock the search provider
        mock_proxy_instance = MagicMock()
        mock_proxy_instance.run.return_value = "Test search results"
        mock_search_proxy.return_value = mock_proxy_instance

        # Call the tool
        result = web_search("test query")

        # Verify the result
        assert result["status"] == "ok"
        assert result["query"] == "test query"
        assert result["results"] == "Test search results"
        # The provider should now be the configured provider, not the default
        # Since we're not mocking the config, it will fall back to the default
        assert result["provider"] == "google/search"

    @patch("aclarai_claimify.scout.tools.SearchProviderProxy")
    def test_web_search_failure(self, mock_search_proxy):
        """Test failed web search."""
        # Mock the search provider to raise an exception
        mock_proxy_instance = MagicMock()
        mock_proxy_instance.run.side_effect = Exception("Search failed")
        mock_search_proxy.return_value = mock_proxy_instance

        # Call the tool
        result = web_search("test query")

        # Verify the result
        assert result["status"] == "error"
        assert result["query"] == "test query"
        assert result["results"] is None
        assert "Search failed" in result["error"]

    @patch("aclarai_claimify.scout.tools._safe_request_get")
    def test_url_to_markdown_success(self, mock_safe_request):
        """Test successful URL to markdown conversion."""
        # Mock the HTTP response
        mock_response = MagicMock()
        mock_response.text = "<html><head><title>Test Title</title></head><body><p>Test content</p></body></html>"
        mock_safe_request.return_value = mock_response

        # Call the tool
        result = url_to_markdown("http://example.com")

        # Verify the result
        assert result["status"] == "ok"
        assert result["url"] == "http://example.com"
        assert "Test Title" in result["markdown"]
        assert "Test content" in result["markdown"]
        assert result["title"] == "Test Title"

    @patch("aclarai_claimify.scout.tools._safe_request_get")
    def test_url_to_markdown_failure(self, mock_safe_request):
        """Test failed URL to markdown conversion."""
        # Mock the HTTP request to raise an exception
        mock_safe_request.side_effect = Exception("Network error")

        # Call the tool
        result = url_to_markdown("http://example.com")

        # Verify the result
        assert result["status"] == "error"
        assert result["url"] == "http://example.com"
        assert result["markdown"] == ""
        assert result["title"] == ""
        assert "Network error" in result["error"]

    @patch("aclarai_claimify.scout.tools._run_async_safely")
    def test_safe_request_get_success(self, mock_run_async):
        """Test successful safe request."""
        # Mock the async response
        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None
        mock_run_async.return_value = mock_response

        # Call the function
        result = _safe_request_get("http://example.com")

        # Verify the result
        assert result == mock_response

    @patch("aclarai_claimify.scout.tools._run_async_safely")
    def test_safe_request_get_failure(self, mock_run_async):
        """Test failed safe request."""
        # Mock the async request to raise an exception
        mock_run_async.side_effect = Exception("Network error")

        # Call the function and expect it to raise
        with pytest.raises(Exception):
            _safe_request_get("http://example.com", max_retries=0)
