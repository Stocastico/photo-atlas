"""The denormalised ``photos.named_face_count`` must stay exact across every
write path (it backs the Known-people facet, replacing a per-row subquery).

The column is maintained by SQLite triggers, so these tests drive the real
``library``/``db`` operations and assert the column equals a fresh recount.
"""

from __future__ import annotations

import sqlite3

from photo_atlas import db, library, search


def _conn(tmp_path):
    return db.connect(tmp_path / "n.db")


def _insert(conn, path, **cols):
    cols["path"] = path
    cols.setdefault("filename", path.rsplit("/", 1)[-1])
    return db.upsert_photo(conn, cols)


def _add_face(conn, photo_id, person_id=None):
    cur = conn.execute(
        "INSERT INTO faces (photo_id, person_id) VALUES (?, ?)", (photo_id, person_id)
    )
    conn.commit()
    return int(cur.lastrowid)


def _named(conn, photo_id):
    return conn.execute(
        "SELECT named_face_count FROM photos WHERE id=?", (photo_id,)
    ).fetchone()[0]


def _recount(conn, photo_id):
    return conn.execute(
        "SELECT COUNT(*) FROM faces WHERE photo_id=? AND person_id IS NOT NULL",
        (photo_id,),
    ).fetchone()[0]


def _assert_consistent(conn):
    """Every photo's stored count matches a live recount of its named faces."""
    for row in conn.execute("SELECT id FROM photos"):
        pid = row["id"]
        assert _named(conn, pid) == _recount(conn, pid), f"photo {pid} drifted"


def test_insert_named_face_increments(tmp_path):
    conn = _conn(tmp_path)
    try:
        p = db.get_or_create_person(conn, "P")
        photo = _insert(conn, "/a.jpg")
        assert _named(conn, photo) == 0
        _add_face(conn, photo, p)
        assert _named(conn, photo) == 1
        # An unknown face does not bump the named count.
        _add_face(conn, photo, None)
        assert _named(conn, photo) == 1
        _assert_consistent(conn)
    finally:
        conn.close()


def test_assign_and_unassign_face(tmp_path):
    conn = _conn(tmp_path)
    try:
        p = db.get_or_create_person(conn, "P")
        photo = _insert(conn, "/a.jpg")
        fid = _add_face(conn, photo, None)
        assert _named(conn, photo) == 0
        library.assign_face(conn, fid, person_id=p)
        assert _named(conn, photo) == 1
        # Re-assigning to a *different* named person leaves the count unchanged.
        p2 = db.get_or_create_person(conn, "Q")
        library.assign_face(conn, fid, person_id=p2)
        assert _named(conn, photo) == 1
        library.unassign_face(conn, fid)
        assert _named(conn, photo) == 0
        _assert_consistent(conn)
    finally:
        conn.close()


def test_assign_cluster_bumps_all_member_photos(tmp_path):
    conn = _conn(tmp_path)
    try:
        a = _insert(conn, "/a.jpg")
        b = _insert(conn, "/b.jpg")
        for photo in (a, b):
            conn.execute(
                "INSERT INTO faces (photo_id, person_id, cluster_id) VALUES (?, NULL, 7)",
                (photo,),
            )
        conn.commit()
        assert _named(conn, a) == 0 and _named(conn, b) == 0
        library.assign_cluster(conn, 7, name="Group")
        assert _named(conn, a) == 1 and _named(conn, b) == 1
        _assert_consistent(conn)
    finally:
        conn.close()


def test_merge_persons_keeps_count(tmp_path):
    conn = _conn(tmp_path)
    try:
        p1 = db.get_or_create_person(conn, "P1")
        p2 = db.get_or_create_person(conn, "P2")
        # One photo with both people: 2 named faces.
        photo = _insert(conn, "/a.jpg")
        _add_face(conn, photo, p1)
        _add_face(conn, photo, p2)
        assert _named(conn, photo) == 2
        # Merging p2 into p1 reassigns named->named: the count must not change.
        library.merge_persons(conn, p2, p1)
        assert _named(conn, photo) == 2
        _assert_consistent(conn)
    finally:
        conn.close()


def test_delete_person_detaches_and_decrements(tmp_path):
    conn = _conn(tmp_path)
    try:
        p = db.get_or_create_person(conn, "P")
        photo = _insert(conn, "/a.jpg")
        _add_face(conn, photo, p)
        _add_face(conn, photo, p)
        assert _named(conn, photo) == 2
        library.delete_person(conn, p)  # faces detach to person_id=NULL
        assert _named(conn, photo) == 0
        _assert_consistent(conn)
    finally:
        conn.close()


def test_replace_faces_recomputes(tmp_path):
    conn = _conn(tmp_path)
    try:
        p = db.get_or_create_person(conn, "P")
        photo = _insert(conn, "/a.jpg")
        _add_face(conn, photo, p)
        _add_face(conn, photo, p)
        assert _named(conn, photo) == 2
        # Re-index the photo with a single unknown face (DELETE then INSERT).
        db.replace_faces(conn, photo, [{"person_id": None}])
        conn.commit()
        assert _named(conn, photo) == 0
        _assert_consistent(conn)
    finally:
        conn.close()


def test_photo_delete_cascade_does_not_leave_drift(tmp_path):
    conn = _conn(tmp_path)
    try:
        p = db.get_or_create_person(conn, "P")
        a = _insert(conn, "/a.jpg")
        b = _insert(conn, "/b.jpg")
        _add_face(conn, a, p)
        _add_face(conn, b, p)
        # Deleting a photo cascades to its faces; the *other* photo stays correct.
        conn.execute("DELETE FROM photos WHERE id=?", (a,))
        conn.commit()
        assert _named(conn, b) == 1
        _assert_consistent(conn)
    finally:
        conn.close()


def test_known_facet_matches_results_after_assignments(tmp_path):
    """End-to-end: the Known facet counts (now column-backed) equal the filtered
    result counts after a realistic mix of assignments."""
    conn = _conn(tmp_path)
    try:
        p1 = db.get_or_create_person(conn, "P1")
        p2 = db.get_or_create_person(conn, "P2")
        a = _insert(conn, "/a.jpg")  # 2 named
        b = _insert(conn, "/b.jpg")  # 1 named + 1 unknown
        _insert(conn, "/c.jpg")      # no faces
        _add_face(conn, a, p1)
        _add_face(conn, a, p2)
        _add_face(conn, b, p1)
        _add_face(conn, b, None)

        buckets = {x["value"]: x["count"] for x in search.facets(conn)["known"]}
        for tok in ("0", "1", "2+"):
            _, total = search.search_photos(conn, {"known": [tok]})
            assert buckets.get(tok, 0) == total
        assert buckets == {"0": 1, "1": 1, "2+": 1}
    finally:
        conn.close()


def test_migration_backfills_named_face_count(tmp_path):
    """An old catalog (no column) gains it correctly backfilled on open."""
    db_path = tmp_path / "old.db"
    raw = sqlite3.connect(db_path)
    raw.executescript(
        "CREATE TABLE photos (id INTEGER PRIMARY KEY, path TEXT UNIQUE, filename TEXT, "
        "taken_at TEXT, scene_type TEXT, folder_place TEXT, place_country TEXT, "
        "place_city TEXT, camera_model TEXT, indexed_at TEXT);"
        "CREATE TABLE faces (id INTEGER PRIMARY KEY, photo_id INTEGER, "
        "person_id INTEGER, cluster_id INTEGER);"
        "INSERT INTO photos (id, path, filename) VALUES (1,'/a.jpg','a.jpg'),(2,'/b.jpg','b.jpg');"
        "INSERT INTO faces (photo_id, person_id) VALUES (1,5),(1,6),(2,NULL);"
    )
    raw.commit()
    raw.close()

    conn = db.connect(db_path)
    try:
        cols = {r["name"] for r in conn.execute("PRAGMA table_info(photos)")}
        assert "named_face_count" in cols
        assert _named(conn, 1) == 2 and _named(conn, 2) == 0
    finally:
        conn.close()
