"""Tests for configuration loading helpers."""

from __future__ import annotations

import textwrap

from aclarai_claimify.config import load_optimization_config


def test_load_optimization_config_replaces_params_for_gepa(tmp_path):
    """Ensure GEPA overrides do not inherit BootstrapFewShot parameters."""

    override_path = tmp_path / "gepa.yaml"
    override_path.write_text(
        textwrap.dedent(
            """
            optimizer_name: "gepa"
            params:
              auto: "light"
              reflection_minibatch_size: 5
              candidate_selection_strategy: "pareto"
              seed: 99
            """
        ).strip()
    )

    config = load_optimization_config(override_path=str(override_path))

    assert config.optimizer_name == "gepa"
    assert config.params == {
        "auto": "light",
        "reflection_minibatch_size": 5,
        "candidate_selection_strategy": "pareto",
        "seed": 99,
    }
    assert "max_bootstrapped_demos" not in config.params
