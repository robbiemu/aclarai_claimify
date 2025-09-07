#!/usr/bin/env python3
"""
Test script to verify the truncation functionality.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))

from unittest.mock import patch
from aclarai_claimify.scout.tools import _truncate_response_for_role

def test_truncation_with_mocked_config():
    """Test the truncation function with mocked configuration."""
    print("Testing truncation function with mocked config...")
    
    # Mock the config to return a small max_tokens value to trigger truncation
    class MockNodeConfig:
        max_tokens = 100  # Very small value to trigger truncation
    
    class MockMissionPlan:
        def get_node_config(self, role):
            return MockNodeConfig()
    
    class MockScoutAgent:
        mission_plan = MockMissionPlan()
    
    class MockConfig:
        scout_agent = MockScoutAgent()
        max_tokens = 100
    
    # Test with a response that should be truncated
    large_content = "A" * 1000  # 1000 characters, should be truncated to 400 (100 tokens * 4)
    
    response = {
        "markdown": large_content,
        "status": "ok"
    }
    
    print(f"Original content length: {len(response['markdown'])}")
    
    # Mock the config loading
    with patch('aclarai_claimify.scout.tools.load_claimify_config', return_value=MockConfig()):
        truncated_response = _truncate_response_for_role(response, "research")
    
    truncated_length = len(truncated_response['markdown'])
    print(f"Truncated content length: {truncated_length}")
    
    # Check if truncation occurred
    if truncated_length < 1000 and "Response truncated" in truncated_response['markdown']:
        print("✅ Truncation working correctly!")
        print(f"Content preview: {truncated_response['markdown'][:100]}...")
        return True
    else:
        print("❌ Truncation not working!")
        return False

if __name__ == "__main__":
    success = test_truncation_with_mocked_config()
    sys.exit(0 if success else 1)