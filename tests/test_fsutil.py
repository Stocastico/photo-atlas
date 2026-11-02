"""File operations that must tolerate Windows' open-file locking."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from PIL import Image

from photo_atlas import fsutil, metadata


def _deny_replace_over_existing(monkeypatch):
    real = os.replace

    def fake(src, dst):
        if Path(dst).exists():
            raise PermissionError(5, "Access is denied (file in use)")
        return real(src, dst)

    monkeypatch.setattr(os, "replace", fake)


def test_replace_atomic_moves_when_free(tmp_path):
    src, dst = tmp_path / "a.part", tmp_path / "a"
    src.write_text("new")
    fsutil.replace_atomic(src, dst)
    assert dst.read_text() == "new" and not src.exists()


def test_replace_atomic_keeps_a_locked_content_addressed_file(tmp_path, monkeypatch):
    _deny_replace_over_existing(monkeypatch)
    src, dst = tmp_path / "a.part", tmp_path / "a"
    src.write_text("same-bytes")
    dst.write_text("same-bytes")  # e.g. a thumbnail currently being served on Windows
    fsutil.replace_atomic(src, dst)
    assert dst.read_text() == "same-bytes" and not src.exists()  # temp cleaned up


def test_replace_atomic_still_raises_when_there_is_nothing_to_keep(tmp_path, monkeypatch):
    def deny(src, dst):
        raise PermissionError(13, "nope")

    monkeypatch.setattr(os, "replace", deny)
    src = tmp_path / "a.part"
    src.write_text("x")
    with pytest.raises(PermissionError):
        fsutil.replace_atomic(src, tmp_path / "missing")


def test_cached_resized_survives_losing_the_race_on_windows(tmp_path, monkeypatch):
    src = tmp_path / "p.jpg"
    Image.new("RGB", (64, 64), (1, 2, 3)).save(src)

    def other_request_won(tmp, dest):
        Path(dest).write_bytes(Path(tmp).read_bytes())  # same bytes, now held open elsewhere
        raise PermissionError(32, "The process cannot access the file")

    monkeypatch.setattr(os, "replace", other_request_won)
    out = metadata.cached_resized(tmp_path / "c", src, "ab" * 20, 32)
    assert out.exists() and out.stat().st_size > 0
    assert not list(out.parent.glob("*.part"))


def test_unlink_quietly_swallows_locks_and_reports(tmp_path, monkeypatch):
    f = tmp_path / "x"
    f.write_text("1")
    assert fsutil.unlink_quietly(f) is True and not f.exists()
    assert fsutil.unlink_quietly(f) is True  # already gone is fine

    def locked(self, missing_ok=False):
        raise PermissionError(32, "in use")

    g = tmp_path / "y"
    g.write_text("1")
    monkeypatch.setattr(Path, "unlink", locked)
    assert fsutil.unlink_quietly(g) is False
