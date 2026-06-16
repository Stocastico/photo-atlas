"""Smart Albums — persist a named filter set (saved search) and restore it.

A saved search is just a name + the querystring of filters to re-apply, so the
storage is a tiny ``saved_searches`` table with create (upsert-by-name) / list /
delete helpers, surfaced over ``/api/albums``.
"""

from __future__ import annotations

from photo_atlas import db
from photo_atlas.config import AtlasConfig


# -- db round-trip ----------------------------------------------------------
def test_saved_search_create_list_delete(tmp_path):
    conn = db.connect(tmp_path / "s.db")
    try:
        a = db.create_saved_search(conn, "Beach trips", "scene=landscape&country=Italy")
        b = db.create_saved_search(conn, "Favourites", "favorite=true")

        albums = db.list_saved_searches(conn)
        # Ordered by name (case-insensitive): "Beach trips" before "Favourites".
        assert [x["name"] for x in albums] == ["Beach trips", "Favourites"]
        assert albums[0]["query"] == "scene=landscape&country=Italy"
        assert all(x["created_at"] for x in albums)

        assert db.delete_saved_search(conn, a) is True
        assert [x["id"] for x in db.list_saved_searches(conn)] == [b]
        # Deleting an unknown id is a no-op (False), not an error.
        assert db.delete_saved_search(conn, 9999) is False
    finally:
        conn.close()


def test_saved_search_upserts_by_name(tmp_path):
    conn = db.connect(tmp_path / "s.db")
    try:
        first = db.create_saved_search(conn, "Trips", "year=2020")
        # Saving again under the same name overwrites the query (no duplicate row,
        # no IntegrityError) and keeps the same id.
        second = db.create_saved_search(conn, "Trips", "year=2021")
        assert first == second
        albums = db.list_saved_searches(conn)
        assert len(albums) == 1 and albums[0]["query"] == "year=2021"
    finally:
        conn.close()


def test_saved_searches_table_is_created_on_existing_catalog(tmp_path):
    path = tmp_path / "old.db"
    conn = db.connect(path)
    conn.execute("DROP TABLE saved_searches")
    conn.commit()
    conn.close()

    conn = db.connect(path)  # re-open runs the schema script (CREATE IF NOT EXISTS)
    try:
        tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        assert "saved_searches" in tables
    finally:
        conn.close()


# -- API --------------------------------------------------------------------
def _client(tmp_path):
    config = AtlasConfig(home=tmp_path / "lib").ensure_dirs()
    db.connect(config.db_path).close()
    from fastapi.testclient import TestClient

    from photo_atlas.api import create_app

    return TestClient(create_app(config))


def test_api_albums_crud(tmp_path):
    client = _client(tmp_path)
    assert client.get("/api/albums").json() == {"albums": []}

    created = client.post("/api/albums", json={"name": "Trips", "query": "year=2020"})
    assert created.status_code == 200 and created.json()["ok"] is True
    album_id = created.json()["id"]

    albums = client.get("/api/albums").json()["albums"]
    assert len(albums) == 1
    assert albums[0]["name"] == "Trips" and albums[0]["query"] == "year=2020"

    assert client.delete(f"/api/albums/{album_id}").status_code == 200
    assert client.get("/api/albums").json() == {"albums": []}


def test_api_album_empty_name_rejected(tmp_path):
    client = _client(tmp_path)
    assert client.post("/api/albums", json={"name": "   ", "query": "x=1"}).status_code == 400
