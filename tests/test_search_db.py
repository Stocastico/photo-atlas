"""DB-backed tests for search behaviours that need real rows: LIKE-wildcard
escaping in free-text search, exact camera matching, and the map endpoint query.
"""

from __future__ import annotations

from photo_atlas import db, search


def _insert(conn, **cols):
    cols.setdefault("filename", cols["path"].rsplit("/", 1)[-1])
    return db.upsert_photo(conn, cols)


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
