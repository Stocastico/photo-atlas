"""Multi-select bulk actions + the user-hidden flag.

Bulk favorite/hide flips the 0/1 flags for a whole selection; ``hidden`` is a
tri-state filter (absent → no clause so ``_where({})`` stays empty, ``False`` →
exclude — the browsing default the API sets — ``True`` → only hidden).
"""

from __future__ import annotations

import itertools

import pytest

from photo_atlas import db, search
from photo_atlas.config import AtlasConfig
from photo_atlas.search import _where

_paths = itertools.count()


def _photo(conn, **cols):
    cols.setdefault("filename", "p.jpg")
    return db.upsert_photo(conn, {"path": f"/p{next(_paths)}.jpg", **cols})


# -- _where hidden tri-state ------------------------------------------------
def test_where_empty_stays_empty():
    assert _where({}) == ("", [])  # the documented contract is preserved


def test_where_hidden_tristate():
    assert "p.hidden = 0" in _where({"hidden": False})[0]
    assert "p.hidden = 1" in _where({"hidden": True})[0]
    assert "hidden" not in _where({"scene": "food"})[0]


# -- DB bulk helpers --------------------------------------------------------
def test_bulk_flags_update_and_count(tmp_path):
    conn = db.connect(tmp_path / "a.db")
    try:
        ids = [_photo(conn) for _ in range(3)]
        assert db.set_favorite_bulk(conn, ids, True) == 3
        assert db.set_hidden_bulk(conn, ids[:2], True) == 2
        favs = conn.execute("SELECT COUNT(*) FROM photos WHERE favorite=1").fetchone()[0]
        hid = conn.execute("SELECT COUNT(*) FROM photos WHERE hidden=1").fetchone()[0]
        assert favs == 3 and hid == 2
        assert db.set_favorite_bulk(conn, [], True) == 0  # empty selection is a no-op
    finally:
        conn.close()


def test_bulk_flag_rejects_unknown_column(tmp_path):
    conn = db.connect(tmp_path / "b.db")
    try:
        with pytest.raises(ValueError):
            db._set_flag_bulk(conn, "path", [1], True)  # guards the f-string column
    finally:
        conn.close()


# -- search excludes hidden by default --------------------------------------
def test_search_and_facets_exclude_hidden_by_default(tmp_path):
    conn = db.connect(tmp_path / "c.db")
    try:
        visible = [_photo(conn, taken_at="2021-01-01") for _ in range(3)]
        hidden = _photo(conn, taken_at="2021-01-02")
        db.set_hidden_bulk(conn, [hidden], True)

        rows, total = search.search_photos(conn, {"hidden": False})
        ids = {r["id"] for r in rows}
        assert total == 3 and hidden not in ids and set(visible) <= ids

        # Only-hidden view.
        rows, total = search.search_photos(conn, {"hidden": True})
        assert total == 1 and rows[0]["id"] == hidden

        # Facets reflect the default exclusion + expose a hidden count.
        f = search.facets(conn, {"hidden": False})
        assert f["hidden"] == 1
    finally:
        conn.close()


# -- API --------------------------------------------------------------------
def _client(tmp_path, n=3):
    config = AtlasConfig(home=tmp_path / "lib").ensure_dirs()
    conn = db.connect(config.db_path)
    ids = [_photo(conn, taken_at=f"2021-01-0{i + 1}") for i in range(n)]
    conn.commit()
    conn.close()
    from fastapi.testclient import TestClient

    from photo_atlas.api import create_app

    return TestClient(create_app(config)), ids


def test_api_bulk_hide_then_unhide(tmp_path):
    client, ids = _client(tmp_path, n=3)
    assert client.get("/api/photos").json()["total"] == 3

    r = client.post("/api/photos/bulk", json={"ids": ids[:2], "action": "hide"})
    assert r.status_code == 200 and r.json()["updated"] == 2
    # Hidden photos leave the default grid…
    assert client.get("/api/photos").json()["total"] == 1
    # …but are reachable via the hidden view, and counted in facets.
    assert client.get("/api/photos", params={"hidden": "true"}).json()["total"] == 2
    assert client.get("/api/facets").json()["hidden"] == 2

    client.post("/api/photos/bulk", json={"ids": ids[:2], "action": "unhide"})
    assert client.get("/api/photos").json()["total"] == 3


def test_api_bulk_favorite(tmp_path):
    client, ids = _client(tmp_path, n=3)
    client.post("/api/photos/bulk", json={"ids": ids, "action": "favorite"})
    assert client.get("/api/photos", params={"favorite": "true"}).json()["total"] == 3
    assert client.get("/api/facets").json()["favorites"] == 3
    client.post("/api/photos/bulk", json={"ids": ids[:1], "action": "unfavorite"})
    assert client.get("/api/photos", params={"favorite": "true"}).json()["total"] == 2


def test_api_bulk_unknown_action_is_400(tmp_path):
    client, ids = _client(tmp_path, n=1)
    assert client.post("/api/photos/bulk", json={"ids": ids, "action": "zap"}).status_code == 400


def test_api_photos_returns_hidden_flag(tmp_path):
    client, ids = _client(tmp_path, n=1)
    assert client.get("/api/photos").json()["photos"][0]["hidden"] == 0
