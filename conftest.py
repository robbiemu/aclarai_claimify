from __future__ import annotations

import os
import tempfile
from pathlib import Path


def pytest_configure(config):
    """Set up DSPy caching for tests before modules import DSPy."""

    os.environ.setdefault("DSPY_DISABLE_CACHE", "1")
    cache_dir = Path(
        os.environ.get(
            "DSPY_CACHEDIR",
            Path(tempfile.gettempdir()) / "aclarai_claimify_dspy_cache",
        )
    ).resolve()
    os.environ["DSPY_CACHEDIR"] = str(cache_dir)
    os.environ.setdefault("DSPY_CACHE_LIMIT", "104857600")  # 100 MB cap
    cache_dir.mkdir(parents=True, exist_ok=True)

    try:
        from dspy.clients import configure_cache  # type: ignore
    except Exception:  # pragma: no cover - DSPy not available
        return

    configure_cache(
        enable_disk_cache=False,
        enable_memory_cache=True,
        disk_cache_dir=str(cache_dir),
        disk_size_limit_bytes=0,
    )
