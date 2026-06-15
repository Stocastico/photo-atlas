"""Run the front-end URL/history-state harness under Node.

The web UI is vanilla JS with no browser test runner, so this drives the pure
query (de)serialisation helpers (``buildQuery`` / ``applyQuery`` / ``syncURL``)
through a tiny fake-DOM harness executed by Node. It **skips** (never fails)
when Node isn't installed, the same way the deep-faces test skips when models
can't be fetched.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

HARNESS = Path(__file__).parent / "js" / "url_state_harness.mjs"


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js not installed")
def test_web_url_state_roundtrip():
    result = subprocess.run(
        ["node", str(HARNESS)],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "failed" in result.stdout and ", 0 failed" in result.stdout
