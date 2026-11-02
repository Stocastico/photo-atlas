"""File-system helpers that behave on Windows, where open files can't be replaced/deleted."""

from __future__ import annotations

import os
from pathlib import Path


def replace_atomic(tmp: Path, dest: Path) -> None:
    """``os.replace(tmp, dest)``, tolerating a *locked* destination.

    On Windows replacing a file another handle has open (a thumbnail being served)
    raises ``PermissionError``. Our derivatives are content-addressed, so an existing
    ``dest`` already holds the same bytes: keep it and discard ``tmp``. Re-raises when
    there is no existing file to fall back on.
    """

    try:
        os.replace(tmp, dest)
    except PermissionError:
        if not Path(dest).exists():
            raise
        Path(tmp).unlink(missing_ok=True)


def unlink_quietly(path: Path) -> bool:
    """Delete ``path`` if present; ``False`` (not an exception) if it's locked or busy."""

    try:
        Path(path).unlink(missing_ok=True)
        return True
    except OSError:
        return False
