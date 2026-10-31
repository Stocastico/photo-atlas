"""Restoring trashed originals back to where they came from."""

from __future__ import annotations

import json

from PIL import Image

from photo_atlas import cli, db, indexer
from photo_atlas.config import AtlasConfig


def _library(tmp_path):
    config = AtlasConfig(home=tmp_path / "lib").ensure_dirs()
    photos = tmp_path / "photos"
    photos.mkdir()
    Image.new("RGB", (64, 64), (200, 30, 30)).save(photos / "a.jpg")
    Image.new("RGB", (64, 64), (30, 200, 30)).save(photos / "b.jpg")
    indexer.index_path(config, photos, backend_name="none", geocode=False)
    conn = db.connect(config.db_path)
    ids = {r["filename"]: r["id"] for r in conn.execute("SELECT id, filename FROM photos")}
    conn.close()
    return config, photos, ids


def _manifest(config):
    path = config.trash_dir / "manifest.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def test_restore_puts_files_back_and_clears_manifest(tmp_path):
    config, photos, ids = _library(tmp_path)
    indexer.delete_photos(config, [ids["a.jpg"], ids["b.jpg"]])
    assert not (photos / "a.jpg").exists() and not (photos / "b.jpg").exists()

    result = indexer.restore_trash(config)
    assert result == {"restored": 2, "missing": 0, "conflicts": 0}
    assert (photos / "a.jpg").exists() and (photos / "b.jpg").exists()
    assert _manifest(config) == []
    assert not [p for p in config.trash_dir.iterdir() if p.name != "manifest.jsonl"]


def test_restore_only_selected_ids(tmp_path):
    config, photos, ids = _library(tmp_path)
    indexer.delete_photos(config, [ids["a.jpg"], ids["b.jpg"]])
    result = indexer.restore_trash(config, [ids["a.jpg"]])
    assert result["restored"] == 1
    assert (photos / "a.jpg").exists() and not (photos / "b.jpg").exists()
    assert [e["id"] for e in _manifest(config)] == [ids["b.jpg"]]  # b stays recoverable


def test_restore_never_overwrites_an_existing_file(tmp_path):
    config, photos, ids = _library(tmp_path)
    indexer.delete_photos(config, [ids["a.jpg"]])
    Image.new("RGB", (8, 8), (1, 2, 3)).save(photos / "a.jpg")  # new file took the name
    before = (photos / "a.jpg").read_bytes()

    result = indexer.restore_trash(config)
    assert result == {"restored": 0, "missing": 0, "conflicts": 1}
    assert (photos / "a.jpg").read_bytes() == before
    assert len(_manifest(config)) == 1  # still in the trash, still listed


def test_restore_skips_entries_whose_trashed_file_vanished(tmp_path):
    config, _photos, ids = _library(tmp_path)
    indexer.delete_photos(config, [ids["a.jpg"]])
    for p in config.trash_dir.iterdir():
        if p.name != "manifest.jsonl":
            p.unlink()
    assert indexer.restore_trash(config) == {"restored": 0, "missing": 1, "conflicts": 0}
    assert _manifest(config) == []  # nothing left to restore, so the stale entry goes


def test_restore_recreates_missing_parent_directories(tmp_path):
    import shutil

    config, photos, ids = _library(tmp_path)
    indexer.delete_photos(config, [ids["a.jpg"]])
    shutil.rmtree(photos)
    assert indexer.restore_trash(config)["restored"] == 1
    assert (photos / "a.jpg").exists()


def test_restore_with_no_trash_is_a_noop(tmp_path):
    config = AtlasConfig(home=tmp_path / "lib").ensure_dirs()
    assert indexer.restore_trash(config) == {"restored": 0, "missing": 0, "conflicts": 0}


def test_cli_restore_reports_and_hints_reindex(tmp_path, capsys):
    config, photos, ids = _library(tmp_path)
    indexer.delete_photos(config, [ids["a.jpg"]])
    assert cli.main(["--home", str(config.home), "restore"]) == 0
    out = capsys.readouterr().out
    assert "Restored 1" in out and "photo-atlas index" in out
    assert (photos / "a.jpg").exists()


def test_cli_restore_by_id_and_conflict_exit_code(tmp_path, capsys):
    config, photos, ids = _library(tmp_path)
    indexer.delete_photos(config, [ids["a.jpg"]])
    Image.new("RGB", (8, 8)).save(photos / "a.jpg")
    assert cli.main(["--home", str(config.home), "restore", str(ids["a.jpg"])]) == 1
    assert "skipped 1" in capsys.readouterr().out
