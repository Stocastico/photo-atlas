"""End-to-end smoke test of the web UI in a real browser (Chromium via Playwright).

Starts the app on a free local port against the indexed demo library and runs
``tests/js/browser_smoke.cjs``. Skips (never fails) when Node, Playwright or its
Chromium aren't available, like the other front-end tests.
"""

from __future__ import annotations

import os
import shutil
import socket
import subprocess
import threading
import time
from pathlib import Path

import pytest

SMOKE = Path(__file__).parent / "js" / "browser_smoke.cjs"


def _playwright_available() -> bool:
    node = shutil.which("node")
    if node is None:
        return False
    probe = subprocess.run(
        [node, "-e", "require('playwright')"], capture_output=True, timeout=30
    )
    return probe.returncode == 0


def test_ui_smoke_in_real_browser(indexed):
    if not _playwright_available():
        # CI sets REQUIRE_BROWSER_TESTS so a missing browser can't skip silently.
        if os.environ.get("REQUIRE_BROWSER_TESTS"):
            pytest.fail("REQUIRE_BROWSER_TESTS is set but Node + Playwright are unavailable")
        pytest.skip("Node + Playwright not installed")
    import uvicorn

    from photo_atlas.api import create_app

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    server = uvicorn.Server(
        uvicorn.Config(create_app(indexed), host="127.0.0.1", port=port, log_level="warning")
    )
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    try:
        for _ in range(100):
            if server.started:
                break
            time.sleep(0.1)
        assert server.started, "server did not start"
        result = subprocess.run(
            ["node", str(SMOKE), f"http://127.0.0.1:{port}"],
            capture_output=True, text=True, timeout=120,
        )
    finally:
        server.should_exit = True
        thread.join(timeout=10)
    combined = result.stdout + result.stderr
    assert ", 0 failed" in result.stdout, combined
    assert "0 passed" not in result.stdout, combined
