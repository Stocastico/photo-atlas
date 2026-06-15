"""DB-backed tests for search behaviours that need real rows: LIKE-wildcard
escaping in free-text search, exact camera matching, and the map endpoint query.
"""

from __future__ import annotations

import sqlite3

import pytest

from photo_atlas import db, search


def test_connect_ensure_schema_false_skips_ddl(tmp_path):
    """A request-time connection can skip the (idempotent but not free) schema
    create/migrate script; on a fresh DB that means the tables don't exist yet."""

    path = tmp_path / "fresh.db"
    conn = db.connect(path, ensure_schema=False)
    try:
        with pytest.raises(sqlite3.OperationalError):
            conn.execute("SELECT COUNT(*) FROM photos").fetchone()
    finally:
        conn.close()

    # The default path still builds the schema, and a later skip-schema connection
    # then sees the tables created by the first.
    db.connect(path).close()
    conn = db.connect(path, ensure_schema=False)
    try:
        assert conn.execute("SELECT COUNT(*) FROM photos").fetchone()[0] == 0
    finally:
        conn.close()


def _insert(conn, **cols):
    cols.setdefault("filename", cols["path"].rsplit("/", 1)[-1])
    return db.upsert_photo(conn, cols)


def test_search_photos_omits_scene_scores_but_detail_keeps_it(tmp_path):
    """The grid/list payload drops the unused ``scene_scores`` JSON blob; the
    single-photo detail (which the lightbox uses) still carries it."""

    conn = db.connect(tmp_path / "s.db")
    try:
        pid = _insert(
            conn, path="/a/x.jpg", scene_type="people",
            scene_scores='{"people": 1.0}', width=4, height=3,
        )
        rows, _ = search.search_photos(conn, {})
        assert "scene_scores" not in rows[0]
        # Other columns the grid/lightbox-from-list rely on are still present.
        for col in ("id", "filename", "scene_type", "width", "height", "taken_at"):
            assert col in rows[0]

        detail = search.photo_detail(conn, pid)
        assert detail["scene_scores"] == '{"people": 1.0}'
    finally:
        conn.close()


def test_search_skips_count_when_not_requested(tmp_path):
    """``count=False`` (used for every infinite-scroll page after the first)
    returns a sentinel total of -1 and still pages correctly."""

    conn = db.connect(tmp_path / "s.db")
    try:
        for i in range(5):
            _insert(conn, path=f"/a/{i}.jpg", taken_at=f"2020-01-0{i + 1}")
        rows, total = search.search_photos(conn, {}, limit=2, offset=0)
        assert total == 5 and len(rows) == 2
        page2, total2 = search.search_photos(conn, {}, limit=2, offset=2, count=False)
        assert total2 == -1 and len(page2) == 2
        # The page is still correct, just without re-counting.
        assert page2[0]["path"] != rows[0]["path"]
    finally:
        conn.close()


def test_q_escapes_like_wildcards(tmp_path):
    conn = db.connect(tmp_path / "s.db")
    try:
        _insert(conn, path="/a/report_v2.jpg")   # literal underscore
        _insert(conn, path="/a/reportXv2.jpg")    # X where the _ is
        # Without escaping, "_" is a single-char wildcard and matches both.
        rows, total = search.search_photos(conn, {"q": "report_v2"})
        assert total == 1
        assert rows[0]["filename"] == "report_v2.jpg"
    finally:
        conn.close()


def test_q_percent_is_literal(tmp_path):
    conn = db.connect(tmp_path / "s.db")
    try:
        _insert(conn, path="/a/100%done.jpg")
        _insert(conn, path="/a/100kdone.jpg")
        rows, total = search.search_photos(conn, {"q": "100%done"})
        assert total == 1 and rows[0]["filename"] == "100%done.jpg"
    finally:
        conn.close()


def test_camera_filter_is_exact_not_substring(tmp_path):
    conn = db.connect(tmp_path / "s.db")
    try:
        _insert(conn, path="/a/1.jpg", camera_model="Canon EOS 5D")
        _insert(conn, path="/a/2.jpg", camera_model="Canon EOS 5D Mark II")
        rows, total = search.search_photos(conn, {"camera": "Canon EOS 5D"})
        # Exact match: the "Mark II" superstring is NOT included (matches facet).
        assert total == 1
        assert rows[0]["camera_model"] == "Canon EOS 5D"
    finally:
        conn.close()


def test_map_points_only_returns_geotagged_and_respects_filters(tmp_path):
    conn = db.connect(tmp_path / "s.db")
    try:
        _insert(conn, path="/a/geo.jpg", lat=41.9, lon=12.5, scene_type="food",
                taken_at="2020-01-01T00:00:00")
        _insert(conn, path="/a/nogeo.jpg", scene_type="food")  # no coords
        _insert(conn, path="/a/other.jpg", lat=1.0, lon=2.0, scene_type="landscape")

        pts = search.map_points(conn, {})
        assert len(pts) == 2  # only the two with coordinates
        assert all(p["lat"] is not None and p["lon"] is not None for p in pts)

        food = search.map_points(conn, {"scene": "food"})
        assert len(food) == 1 and food[0]["year"] == "2020"
    finally:
        conn.close()
