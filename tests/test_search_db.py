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


def test_people_count_filter_and_facet(tmp_path):
    """Number-of-people buckets filter on face_count and the facet counts agree."""

    conn = db.connect(tmp_path / "s.db")
    try:
        for i, fc in enumerate([0, 1, 1, 3, 7]):
            _insert(conn, path=f"/a/{i}.jpg", face_count=fc)

        # Portrait = exactly one face.
        _, portraits = search.search_photos(conn, {"people": ["1"]})
        assert portraits == 2
        # Groups = 2-4 OR 5+ (OR within the facet).
        _, groups = search.search_photos(conn, {"people": ["2-4", "5+"]})
        assert groups == 2
        # No-people bucket.
        _, none = search.search_photos(conn, {"people": ["0"]})
        assert none == 1

        buckets = {b["value"]: b["count"] for b in search.facets(conn)["people"]}
        assert buckets == {"0": 1, "1": 2, "2-4": 1, "5+": 1}
    finally:
        conn.close()


def test_known_people_filter_and_facet(tmp_path):
    """`known` buckets photos by how many of their faces are assigned to a person."""

    conn = db.connect(tmp_path / "s.db")
    try:
        p1 = db.get_or_create_person(conn, "P1")
        p2 = db.get_or_create_person(conn, "P2")
        # A: 2 named; B: 1 named + 1 unknown; C: all unknown; D: no faces at all.
        pa = _insert(conn, path="/a.jpg", face_count=2)
        pb = _insert(conn, path="/b.jpg", face_count=2)
        pc = _insert(conn, path="/c.jpg", face_count=1)
        _insert(conn, path="/d.jpg", face_count=0)

        def add_face(photo_id, person_id):
            conn.execute(
                "INSERT INTO faces (photo_id, person_id) VALUES (?, ?)",
                (photo_id, person_id),
            )

        add_face(pa, p1)
        add_face(pa, p2)
        add_face(pb, p1)
        add_face(pb, None)
        add_face(pc, None)
        conn.commit()

        _, none = search.search_photos(conn, {"known": ["0"]})
        assert none == 2  # C (unknown only) and D (no faces)
        _, one = search.search_photos(conn, {"known": ["1"]})
        assert one == 1  # B
        _, two_plus = search.search_photos(conn, {"known": ["2+"]})
        assert two_plus == 1  # A

        buckets = {b["value"]: b["count"] for b in search.facets(conn)["known"]}
        assert buckets == {"0": 2, "1": 1, "2+": 1}
    finally:
        conn.close()


def test_people_and_or_mode(tmp_path):
    """`person_mode='all'` requires every selected person; the default matches any."""

    conn = db.connect(tmp_path / "s.db")
    try:
        p1 = db.get_or_create_person(conn, "P1")
        p2 = db.get_or_create_person(conn, "P2")
        pa = _insert(conn, path="/a.jpg")  # persons p1 AND p2
        pb = _insert(conn, path="/b.jpg")  # person p1 only
        pc = _insert(conn, path="/c.jpg")  # person p2 only

        def add_face(photo_id, person_id):
            conn.execute(
                "INSERT INTO faces (photo_id, person_id) VALUES (?, ?)",
                (photo_id, person_id),
            )

        add_face(pa, p1)
        add_face(pa, p2)
        add_face(pb, p1)
        add_face(pc, p2)
        conn.commit()

        # Default (any-of): all three photos contain person p1 or p2.
        _, any_total = search.search_photos(conn, {"person_id": [p1, p2]})
        assert any_total == 3
        # All-of: only photo A contains both.
        _, all_total = search.search_photos(
            conn, {"person_id": [p1, p2], "person_mode": "all"}
        )
        assert all_total == 1
        # A single person behaves the same under either mode.
        _, one = search.search_photos(conn, {"person_id": [p1], "person_mode": "all"})
        assert one == 2
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


def test_facet_date_bounds_are_filter_aware(tmp_path):
    """The date slider's min/max are a facet too: they reflect the other active
    filters but not the date filter itself (honouring the filter-aware docstring)."""

    conn = db.connect(tmp_path / "s.db")
    try:
        _insert(conn, path="/a.jpg", scene_type="food", taken_at="2018-05-01T00:00:00")
        _insert(conn, path="/b.jpg", scene_type="food", taken_at="2020-07-01T00:00:00")
        _insert(conn, path="/c.jpg", scene_type="landscape", taken_at="2010-01-01T00:00:00")

        f = search.facets(conn)
        assert f["date_min"] == "2010-01-01" and f["date_max"] == "2020-07-01"

        # Filtering to food narrows the bounds to just the food photos.
        ff = search.facets(conn, {"scene": "food"})
        assert ff["date_min"] == "2018-05-01" and ff["date_max"] == "2020-07-01"

        # The date filter must NOT shrink its own bounds (own-dimension rule).
        fd = search.facets(conn, {"date_from": "2019-01-01"})
        assert fd["date_min"] == "2010-01-01" and fd["date_max"] == "2020-07-01"
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
