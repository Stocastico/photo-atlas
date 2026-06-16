"""Favorites: the ``favorite`` star column, its filter + facet, the toggle API,
and the guarantee that a re-index never clears a user's star.
"""

from __future__ import annotations

from photo_atlas import db, indexer, search


def _insert(conn, path, **cols):
    cols["path"] = path
    cols.setdefault("filename", path.rsplit("/", 1)[-1])
    return db.upsert_photo(conn, cols)


# -- db helper -------------------------------------------------------------
def test_set_favorite_toggles_and_reports_existence(tmp_path):
    conn = db.connect(tmp_path / "f.db")
    try:
        pid = _insert(conn, "/a.jpg")
        assert db.set_favorite(conn, pid, True) is True
        assert conn.execute("SELECT favorite FROM photos WHERE id=?", (pid,)).fetchone()[0] == 1
        assert db.set_favorite(conn, pid, False) is True
        assert conn.execute("SELECT favorite FROM photos WHERE id=?", (pid,)).fetchone()[0] == 0
        # A missing photo reports "not updated" (the API maps this to 404).
        assert db.set_favorite(conn, 999999, True) is False
    finally:
        conn.close()


# -- search filter + facet -------------------------------------------------
def test_favorite_filter_and_facet(tmp_path):
    conn = db.connect(tmp_path / "f.db")
    try:
        a = _insert(conn, "/a.jpg", scene_type="food")
        _insert(conn, "/b.jpg", scene_type="food")
        c = _insert(conn, "/c.jpg", scene_type="landscape")
        db.set_favorite(conn, a, True)
        db.set_favorite(conn, c, True)

        _, total = search.search_photos(conn, {"favorite": True})
        assert total == 2
        # Falsy favorite is a no-op filter (shows everything).
        _, all_total = search.search_photos(conn, {"favorite": False})
        assert all_total == 3

        facets = search.facets(conn)
        assert facets["favorites"] == 2
        # Filter-aware: favourites count reflects *other* active filters.
        assert search.facets(conn, {"scene": "food"})["favorites"] == 1
    finally:
        conn.close()


def test_favorite_present_in_list_payload(tmp_path):
    conn = db.connect(tmp_path / "f.db")
    try:
        a = _insert(conn, "/a.jpg")
        db.set_favorite(conn, a, True)
        rows, _ = search.search_photos(conn, {})
        assert rows[0]["favorite"] == 1  # the grid needs it to render the star
    finally:
        conn.close()


def test_favorite_survives_reindex(tmp_path):
    """``favorite`` is outside PHOTO_COLUMNS, so re-upserting the same path (a
    re-index) must not reset the user's star."""
    conn = db.connect(tmp_path / "f.db")
    try:
        pid = _insert(conn, "/a.jpg", scene_type="food")
        db.set_favorite(conn, pid, True)
        # Re-index: same path, fresh metadata, no 'favorite' key in the record.
        again = _insert(conn, "/a.jpg", scene_type="landscape")
        assert again == pid
        assert conn.execute("SELECT favorite FROM photos WHERE id=?", (pid,)).fetchone()[0] == 1
    finally:
        conn.close()


# -- API -------------------------------------------------------------------
def _first_photo_id(client):
    return client.get("/api/photos?limit=1").json()["photos"][0]["id"]


def test_favorite_api_toggle_filter_and_facet(client):
    pid = _first_photo_id(client)
    assert client.get(f"/api/photos/{pid}").json()["favorite"] == 0

    r = client.put(f"/api/photos/{pid}/favorite", json={"favorite": True})
    assert r.status_code == 200 and r.json() == {"ok": True, "favorite": True}
    assert client.get(f"/api/photos/{pid}").json()["favorite"] == 1

    # The favourite filter and facet both see exactly the one starred photo.
    listed = client.get("/api/photos?favorite=true").json()
    assert listed["total"] == 1 and listed["photos"][0]["id"] == pid
    assert client.get("/api/facets").json()["favorites"] == 1

    # Un-star.
    assert client.put(f"/api/photos/{pid}/favorite", json={"favorite": False}).json()[
        "favorite"
    ] is False
    assert client.get("/api/photos?favorite=true").json()["total"] == 0


def test_favorite_api_404_for_unknown_photo(client):
    assert (
        client.put("/api/photos/999999/favorite", json={"favorite": True}).status_code == 404
    )


def test_favorite_api_blocks_cross_origin_write(client):
    pid = _first_photo_id(client)
    r = client.put(
        f"/api/photos/{pid}/favorite",
        json={"favorite": True},
        headers={"Origin": "http://evil.example"},
    )
    assert r.status_code == 403
    # Same-origin write is allowed.
    ok = client.put(
        f"/api/photos/{pid}/favorite",
        json={"favorite": True},
        headers={"Origin": "http://testserver"},
    )
    assert ok.status_code == 200


def test_favorite_survives_reindex_end_to_end(client, indexed, tmp_path):
    """Star a photo through the API, re-run the indexer over the same tree, and
    confirm the star is still set."""
    from photo_atlas import demo

    pid = _first_photo_id(client)
    client.put(f"/api/photos/{pid}/favorite", json={"favorite": True})

    # Re-index the demo photos already present under the library.
    photos_dir = tmp_path / "photos"
    if not photos_dir.exists():  # conftest's indexed fixture builds this tree
        demo.generate(photos_dir, count=20, seed=7)
    indexer.index_path(indexed, photos_dir, backend_name="synthetic", geocode=True)

    assert client.get(f"/api/photos/{pid}").json()["favorite"] == 1
