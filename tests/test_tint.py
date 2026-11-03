"""Blur-up placeholder colour (photos.tint): computed at index, backfilled, listed."""

from __future__ import annotations

from PIL import Image

from photo_atlas import db, indexer, metadata, search
from photo_atlas.config import AtlasConfig


def test_tint_of_solid_image():
    assert metadata.tint_of(Image.new("RGB", (40, 30), (255, 0, 0))) == "ff0000"
    assert metadata.tint_of(Image.new("L", (8, 8), 128)) == "808080"


def _lib(tmp_path):
    d = tmp_path / "photos"
    d.mkdir()
    Image.new("RGB", (64, 48), (200, 30, 30)).save(d / "a.jpg")
    Image.new("RGB", (48, 64), (10, 120, 220)).save(d / "b.jpg")
    config = AtlasConfig(home=tmp_path / "lib").ensure_dirs()
    indexer.index_path(config, d, backend_name="none", geocode=False)
    return config


def test_schema_has_tint_column(tmp_path):
    config = _lib(tmp_path)
    conn = db.connect(config.db_path)
    try:
        assert db.SCHEMA_VERSION >= 4
        assert "tint" in {r["name"] for r in conn.execute("PRAGMA table_info(photos)")}
    finally:
        conn.close()


def test_index_stores_tint_and_list_exposes_it(tmp_path):
    config = _lib(tmp_path)
    conn = db.connect(config.db_path)
    try:
        photos, _ = search.search_photos(conn, {}, limit=10)
        tints = {p["filename"]: p["tint"] for p in photos}
        assert abs(int(tints["a.jpg"][:2], 16) - 200) <= 6  # reddish
        assert len(tints["b.jpg"]) == 6
    finally:
        conn.close()


def test_backfill_tints_from_thumbnails(tmp_path):
    config = _lib(tmp_path)
    conn = db.connect(config.db_path)
    conn.execute("UPDATE photos SET tint=NULL")
    conn.commit()
    conn.close()
    assert indexer.backfill_tints(config) == 2
    assert indexer.backfill_tints(config) == 0  # idempotent
    assert indexer.backfill_tints(config, recompute=True) == 2
    conn = db.connect(config.db_path)
    try:
        assert all(len(r[0]) == 6 for r in conn.execute("SELECT tint FROM photos"))
    finally:
        conn.close()
