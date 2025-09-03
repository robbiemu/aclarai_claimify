"""
Unit tests for supervisor decision logic and research cache improvements.
Tests the fixes for duplicate consecutive agent calls and research result caching.
"""

import pytest
from unittest.mock import Mock
from aclarai_claimify.scout.nodes import (
    should_retry_agent_with_reason,
    choose_fallback_agent,
)
from aclarai_claimify.scout.utils import (
    normalize_url,
    normalize_title,
    create_dedupe_hash,
    calculate_result_score,
    dedupe_and_rank_results,
)


class TestSupervisorHelpers:
    """Test the supervisor helper functions for retry logic and fallback selection."""

    def test_should_retry_agent_with_reason_exceeds_max_retries(self):
        """Test that agents are not retried when max retries exceeded."""
        config = Mock()
        config.agents.supervisor.agent_max_retries = {"research": 2, "fitness": 1}

        context = {"research_consecutive_count": 3}
        should_retry, reason = should_retry_agent_with_reason(
            "research", context, config
        )

        assert not should_retry
        assert "exceeded_max_retries_2" in reason

    def test_should_retry_agent_with_reason_transient_error(self):
        """Test that agents are retried on transient errors."""
        config = Mock()
        config.agents.supervisor.agent_max_retries = {"research": 2}

        context = {
            "research_consecutive_count": 1,
            "last_message_content": "Connection timeout occurred",
        }
        should_retry, reason = should_retry_agent_with_reason(
            "research", context, config
        )

        assert should_retry
        assert reason == "transient_error_detected"

    def test_should_retry_agent_with_reason_query_changed(self):
        """Test that research agent can retry when query changes."""
        config = Mock()
        config.agents.supervisor.agent_max_retries = {"research": 2}

        context = {
            "research_consecutive_count": 1,
            "current_query": "new query",
            "last_query_hash": hash("old query"),
            "last_message_content": "",
        }
        should_retry, reason = should_retry_agent_with_reason(
            "research", context, config
        )

        assert should_retry
        assert reason == "query_changed"

    def test_should_retry_agent_with_reason_no_valid_reason(self):
        """Test that agents are not retried without valid reasons."""
        config = Mock()
        config.agents.supervisor.agent_max_retries = {"fitness": 1}

        context = {
            "fitness_consecutive_count": 0,
            "last_message_content": "Normal response",
            "research_findings": [],
        }
        should_retry, reason = should_retry_agent_with_reason(
            "fitness", context, config
        )

        assert not should_retry
        assert reason == "no_valid_retry_reason"

    def test_choose_fallback_agent_research_to_synthetic(self):
        """Test fallback from research to synthetic when budget available."""
        config = Mock()
        config.agents.supervisor.fallback_chain = {"research": "synthetic"}

        context = {"synthetic_budget_available": True}
        fallback = choose_fallback_agent("research", context, config)

        assert fallback == "synthetic"

    def test_choose_fallback_agent_research_to_end(self):
        """Test fallback from research to end when no synthetic budget."""
        config = Mock()
        config.agents.supervisor.fallback_chain = {"research": "synthetic"}

        context = {"synthetic_budget_available": False}
        fallback = choose_fallback_agent("research", context, config)

        assert fallback == "end"

    def test_choose_fallback_agent_fitness_to_research(self):
        """Test fallback from fitness to research."""
        config = Mock()
        config.agents.supervisor.fallback_chain = {"fitness": "research"}

        context = {}
        fallback = choose_fallback_agent("fitness", context, config)

        assert fallback == "research"


class TestResearchUtils:
    """Test the research utility functions for URL/title normalization and scoring."""

    def test_normalize_url_removes_tracking(self):
        """Test that URL normalization removes tracking parameters."""
        url = "https://example.com/page?utm_source=google&utm_medium=cpc&id=123"
        normalized = normalize_url(url)
        assert "utm_source" not in normalized
        assert "utm_medium" not in normalized
        assert "id=123" in normalized  # Keep non-tracking params

    def test_normalize_url_strips_fragment(self):
        """Test that URL normalization removes fragments."""
        url = "https://example.com/page#section1"
        normalized = normalize_url(url)
        assert "#section1" not in normalized

    def test_normalize_title_basic_cleanup(self):
        """Test basic title normalization."""
        title = "  EXAMPLE TITLE - Site Name  "
        normalized = normalize_title(title)
        assert normalized == "example title"

    def test_normalize_title_removes_site_suffix(self):
        """Test that site suffixes are removed from titles."""
        title = "Article Title | Wikipedia"
        normalized = normalize_title(title)
        assert normalized == "article title"

    def test_create_dedupe_hash_consistency(self):
        """Test that result hashing is consistent."""
        result1 = {
            "url": "https://example.com/page",
            "title": "Example Page",
            "content": "Some content here",
        }
        result2 = {
            "url": "https://example.com/page",
            "title": "Example Page",
            "content": "Some content here",
        }

        hash1 = create_dedupe_hash(result1)
        hash2 = create_dedupe_hash(result2)
        assert hash1 == hash2

    def test_calculate_result_score_provider_trust(self):
        """Test that scoring considers provider trust weights."""
        result = {
            "source": "url_content",
            "content": "This is high quality content with verifiable atomic requirements.",
            "url": "https://example.edu/page",
        }

        provider_weights = {"url_content": 1.5, "web_search": 1.0}

        score = calculate_result_score(result, "atomic requirements", provider_weights)

        assert score > 0
        # Should get reasonable score for url_content provider
        assert score > 0.3

    def test_dedupe_and_rank_results_removes_duplicates(self):
        """Test that deduplication removes duplicate results."""
        raw_results = [
            {
                "url": "https://example.com/page1",
                "title": "Example Page",
                "content": "Content about atomic requirements",
                "source": "web_search",
            },
            {
                "url": "https://example.com/page1",  # Duplicate URL
                "title": "Example Page",
                "content": "Content about atomic requirements",
                "source": "url_content",
            },
            {
                "url": "https://example.com/page2",
                "title": "Different Page",
                "content": "More content about requirements",
                "source": "web_search",
            },
        ]

        final_results = dedupe_and_rank_results(
            raw_results, "atomic requirements", max_results=5, max_per_domain=3
        )

        # Should keep 2 unique results (duplicate removed)
        assert len(final_results) == 2
        urls = [r["url"] for r in final_results]
        assert len(set(urls)) == 2  # All unique URLs

    def test_dedupe_and_rank_results_enforces_domain_limit(self):
        """Test that domain limits are enforced."""
        raw_results = []
        # Create 5 results from the same domain
        for i in range(5):
            raw_results.append(
                {
                    "url": f"https://example.com/page{i}",
                    "title": f"Page {i}",
                    "content": "Content about requirements",
                    "source": "web_search",
                }
            )

        final_results = dedupe_and_rank_results(
            raw_results, "requirements", max_results=10, max_per_domain=2
        )

        # Should limit to 2 results per domain
        assert len(final_results) <= 2

    def test_dedupe_and_rank_results_respects_max_results(self):
        """Test that max results limit is respected."""
        raw_results = []
        # Create 10 results from different domains
        for i in range(10):
            raw_results.append(
                {
                    "url": f"https://example{i}.com/page",
                    "title": f"Page {i}",
                    "content": "Content about requirements",
                    "source": "web_search",
                }
            )

        final_results = dedupe_and_rank_results(
            raw_results, "requirements", max_results=5, max_per_domain=3
        )

        # Should limit to max 5 results
        assert len(final_results) <= 5

    def test_dedupe_and_rank_results_adds_scores(self):
        """Test that results get scored and sorted."""
        raw_results = [
            {
                "url": "https://example.edu/page1",  # edu domain should score higher
                "title": "Academic Page",
                "content": "Detailed content about atomic requirements",
                "source": "url_content",
            },
            {
                "url": "https://example.com/page2",
                "title": "Commercial Page",
                "content": "Brief content",
                "source": "web_search",
            },
        ]

        final_results = dedupe_and_rank_results(
            raw_results, "atomic requirements", max_results=5, max_per_domain=3
        )

        # All results should have scores
        assert all("_score" in result for result in final_results)

        # Results should be sorted by score (descending)
        scores = [result["_score"] for result in final_results]
        assert scores == sorted(scores, reverse=True)


class TestIntegration:
    """Integration tests for the complete flow."""

    def test_supervisor_prevents_duplicate_consecutive_calls(self):
        """Integration test that supervisor prevents duplicate consecutive agent calls."""
        # This would need to be implemented with proper mocks of the supervisor_node
        # For now, we'll verify the logic exists in the helper functions
        config = Mock()
        config.agents.supervisor.agent_max_retries = {"fitness": 1}

        # Simulate fitness agent being called consecutively without valid retry reason
        context = {
            "fitness_consecutive_count": 1,
            "last_message_content": "Normal fitness evaluation",
            "research_findings": [],  # Same findings as before
        }

        should_retry, reason = should_retry_agent_with_reason(
            "fitness", context, config
        )
        assert not should_retry

        # Should choose fallback instead
        config.agents.supervisor.fallback_chain = {"fitness": "research"}
        fallback = choose_fallback_agent("fitness", context, config)
        assert fallback == "research"


if __name__ == "__main__":
    pytest.main([__file__])
