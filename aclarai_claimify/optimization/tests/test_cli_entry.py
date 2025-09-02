"""Tests for the CLI entry point registration."""

import subprocess
import sys
from pathlib import Path
import pytest


def test_cli_entry_point_help():
    """Test that the CLI entry point is registered and shows help."""
    # Run the CLI with no arguments to show help
    result = subprocess.run(
        [sys.executable, "-m", "aclarai_claimify.cli"],
        capture_output=True,
        text=True,
        cwd=Path(__file__).parent.parent.parent
    )
    
    # Should exit with code 0 and show help text
    assert result.returncode == 0
    assert "usage:" in result.stdout.lower() or "usage:" in result.stderr.lower()
    assert "dspy" in result.stdout.lower() or "dspy" in result.stderr.lower()


def test_cli_entry_point_via_console_script():
    """Test that the console script entry point works."""
    # This test requires the package to be installed, so we'll skip if not available
    try:
        result = subprocess.run(
            ["aclarai-claimify"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent.parent
        )
        
        # Should exit with code 0 and show help text
        assert result.returncode == 0
        assert "usage:" in result.stdout.lower() or "usage:" in result.stderr.lower()
    except FileNotFoundError:
        # Console script not available (not installed), which is expected in dev environment
        pytest.skip("Console script not available in development environment")