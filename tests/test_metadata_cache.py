"""cached_resized must generate derivatives atomically (no partial files)."""

from __future__ import annotations

import pytest
from PIL import Image

from photo_atlas import metadata


def _src(tmp_path):
    p = tmp_path / "src.jpg"
    Image.new("RGB", (200, 150), (40, 90, 160)).save(p, "JPEG")
    return p


def test_cached_resized_generates_valid_derivative(tmp_path):
    dest = metadata.cached_resized(tmp_path / "cache", _src(tmp_path), "a" * 40, 64)
    assert dest.exists()
    with Image.open(dest) as img:
        assert max(img.size) == 64


def test_failed_write_leaves_no_partial_dest(tmp_path, monkeypatch):
    src = _src(tmp_path)
    cache = tmp_path / "cache"

    def boom(path, dest, size, quality):
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"partial-junk")  # a half-written file at the work path
        raise RuntimeError("disk full")

    monkeypatch.setattr(metadata, "_write_resized", boom)
    with pytest.raises(RuntimeError):
        metadata.cached_resized(cache, src, "b" * 40, 64)

    # The destination must not exist (write went to a temp path), and no stray
    # temp/partial files are left behind in the cache shard.
    expected = cache / "bb" / "bb_no.jpg"  # not the real name; just assert clean dir
    assert not expected.exists()
    leftovers = list(cache.rglob("*")) if cache.exists() else []
    files = [p for p in leftovers if p.is_file()]
    assert files == [], f"unexpected leftover files: {files}"
