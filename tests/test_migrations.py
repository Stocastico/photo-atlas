"""Versioned catalog migrations (`PRAGMA user_version`)."""

from __future__ import annotations

import sqlite3

import pytest

from photo_atlas import db


def _version(path) -> int:
    raw = sqlite3.connect(path)
    try:
        return raw.execute("PRAGMA user_version").fetchone()[0]
    finally:
        raw.close()


def test_fresh_catalog_is_stamped_with_current_version(tmp_path):
    db.connect(tmp_path / "a.db").close()
    assert _version(tmp_path / "a.db") == db.SCHEMA_VERSION


def test_legacy_unversioned_catalog_is_migrated_and_stamped(tmp_path):
    path = tmp_path / "old.db"
    raw = sqlite3.connect(path)
    raw.executescript(
        "CREATE TABLE photos (id INTEGER PRIMARY KEY, path TEXT UNIQUE, filename TEXT, "
        "taken_at TEXT, scene_type TEXT, folder_place TEXT, place_country TEXT, "
        "place_city TEXT, camera_model TEXT, indexed_at TEXT);"
        "CREATE TABLE faces (id INTEGER PRIMARY KEY, photo_id INTEGER, "
        "person_id INTEGER, cluster_id INTEGER);"
    )
    raw.commit()
    raw.close()
    assert _version(path) == 0
    conn = db.connect(path)
    cols = {r["name"] for r in conn.execute("PRAGMA table_info(photos)")}
    conn.close()
    assert {"embedding", "phash", "hidden"} <= cols
    assert _version(path) == db.SCHEMA_VERSION


def test_catalog_from_a_newer_version_is_refused(tmp_path):
    path = tmp_path / "new.db"
    raw = sqlite3.connect(path)
    raw.execute(f"PRAGMA user_version = {db.SCHEMA_VERSION + 1}")
    raw.close()
    with pytest.raises(RuntimeError, match="newer"):
        db.connect(path)


def test_later_migrations_run_once_in_order(tmp_path, monkeypatch):
    ran: list[int] = []
    new = (db.SCHEMA_VERSION + 1, lambda c: ran.append(1))
    monkeypatch.setattr(db, "MIGRATIONS", [*db.MIGRATIONS, new])
    monkeypatch.setattr(db, "SCHEMA_VERSION", db.SCHEMA_VERSION + 1)
    db.connect(tmp_path / "m.db").close()
    db.connect(tmp_path / "m.db").close()
    assert ran == [1]
    assert _version(tmp_path / "m.db") == db.SCHEMA_VERSION
