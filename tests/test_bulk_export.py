"""Bulk export: copy a selection's original files to a destination folder.

A multi-select bulk action that copies the chosen photos' source files (originals
preserved, never moved) into a target directory, so a filtered/selected set can be
pulled out of the library. Offline: pure filesystem copy.
"""

from __future__ import annotations

from PIL import Image

from photo_atlas import db, indexer
from photo_atlas.config import AtlasConfig


def _library(tmp_path):
    config = AtlasConfig(home=tmp_path / "lib").ensure_dirs()
    src = tmp_path / "src"
    src.mkdir()
    conn = db.connect(config.db_path)
    ids = {}
    for name in ("a.jpg", "b.jpg"):
        p = src / name
        Image.new("RGB", (8, 8), (10, 20, 30)).save(p)
        ids[name] = db.upsert_photo(conn, {"path": str(p), "filename": name})
    # A row whose source file is missing on disk.
    ids["gone.jpg"] = db.upsert_photo(
        conn, {"path": str(src / "gone.jpg"), "filename": "gone.jpg"}
    )
    conn.commit()
    conn.close()
    return config, src, ids


def test_export_copies_originals(tmp_path):
    config, _src, ids = _library(tmp_path)
    dest = tmp_path / "out"
    result = indexer.export_photos(config, [ids["a.jpg"], ids["b.jpg"]], dest)
    assert result["copied"] == 2 and result["missing"] == 0
    assert (dest / "a.jpg").exists() and (dest / "b.jpg").exists()


def test_export_counts_missing_sources(tmp_path):
    config, _src, ids = _library(tmp_path)
    dest = tmp_path / "out"
    result = indexer.export_photos(config, [ids["a.jpg"], ids["gone.jpg"], 99999], dest)
    # a.jpg copies; the deleted file and the unknown id are both "missing".
    assert result["copied"] == 1 and result["missing"] == 2
    assert (dest / "a.jpg").exists()


def test_export_disambiguates_name_collisions(tmp_path):
    # Two different photos sharing a basename must not clobber each other.
    config = AtlasConfig(home=tmp_path / "lib").ensure_dirs()
    conn = db.connect(config.db_path)
    ids = []
    for sub in ("one", "two"):
        d = tmp_path / sub
        d.mkdir()
        p = d / "IMG_001.jpg"
        Image.new("RGB", (8, 8)).save(p)
        ids.append(db.upsert_photo(conn, {"path": str(p), "filename": "IMG_001.jpg"}))
    conn.commit()
    conn.close()

    dest = tmp_path / "out"
    result = indexer.export_photos(config, ids, dest)
    assert result["copied"] == 2
    # Both landed: the second got an id-suffixed name rather than overwriting.
    assert (dest / "IMG_001.jpg").exists()
    assert len(list(dest.glob("IMG_001*.jpg"))) == 2


def test_export_empty_is_noop(tmp_path):
    config, _src, _ids = _library(tmp_path)
    result = indexer.export_photos(config, [], tmp_path / "out")
    assert result == {"requested": 0, "copied": 0, "missing": 0}


# -- API --------------------------------------------------------------------
def _client(config):
    from fastapi.testclient import TestClient

    from photo_atlas.api import create_app

    return TestClient(create_app(config))


def test_api_export(tmp_path):
    config, _src, ids = _library(tmp_path)
    dest = tmp_path / "out"
    resp = _client(config).post(
        "/api/photos/export", json={"ids": [ids["a.jpg"]], "dest": str(dest)}
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] and body["copied"] == 1
    assert (dest / "a.jpg").exists()


def test_api_export_rejects_empty_dest(tmp_path):
    config, _src, ids = _library(tmp_path)
    resp = _client(config).post(
        "/api/photos/export", json={"ids": [ids["a.jpg"]], "dest": "   "}
    )
    assert resp.status_code == 400
