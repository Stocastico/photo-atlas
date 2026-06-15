"""Run the front-end JS harnesses under Node.

The web UI is vanilla JS with no browser test runner, so the pure helpers
(URL/history serialisation and the virtualised-grid math) are exercised through
tiny fake-DOM harnesses in ``tests/js`` executed by Node. Each harness prints
``"<n> passed, <m> failed"`` and exits non-zero on failure. These **skip**
(never fail) when Node isn't installed, like the deep-faces test does offline.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

HARNESSES = sorted((Path(__file__).parent / "js").glob("*_harness.mjs"))


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js not installed")
@pytest.mark.parametrize("harness", HARNESSES, ids=lambda p: p.stem)
def test_web_js_harness(harness):
    result = subprocess.run(
        ["node", str(harness)],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert ", 0 failed" in result.stdout
