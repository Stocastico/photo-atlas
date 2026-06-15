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
    combined = result.stdout + result.stderr
    # The harness prints "<n> passed, <m> failed" and exits 1 if anything failed.
    # Trust that stdout verdict rather than the exit code: some Node builds
    # occasionally SIGSEGV during V8 teardown *after* a fully successful run
    # (returncode -11 with a clean "..., 0 failed" on stdout), and that exit-time
    # crash is not a test failure. A real failure still shows "<m> failed" (m>0)
    # and the harness's own non-zero (positive) exit.
    assert ", 0 failed" in result.stdout, combined
    assert "0 passed" not in result.stdout, combined  # guard: the harness ran
    assert result.returncode <= 0, combined  # allow signal kills, reject exit(1)
