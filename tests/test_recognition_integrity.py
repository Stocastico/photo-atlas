"""Human-confirmed labels are the only ground truth for recognition."""

from __future__ import annotations

import numpy as np

from photo_atlas import db, indexer, library


def _face(conn, person=None, cluster=None, confidence=0.0, vec=(1.0, 0.0, 0.0)):
    photo = db.upsert_photo(conn, {"path": f"/p{conn.total_changes}_{np.random.rand()}.jpg",
                                   "filename": "x.jpg"})
    db.replace_faces(conn, photo, [{
        "person_id": person, "cluster_id": cluster, "confidence": confidence,
        "embedding": db.embedding_to_blob(np.array(vec, dtype=np.float32)), "dim": len(vec),
    }])
    return conn.execute("SELECT id FROM faces WHERE photo_id=?", (photo,)).fetchone()["id"]


def test_assign_cluster_records_a_human_label(tmp_path):
    conn = db.connect(tmp_path / "a.db")
    try:
        ids = [_face(conn, cluster=5) for _ in range(3)]
        other = _face(conn, cluster=6)
        person = library.assign_cluster(conn, 5, name="Anna")
        rows = {r["id"]: r for r in conn.execute("SELECT * FROM faces")}
        for i in ids:
            assert rows[i]["person_id"] == person and rows[i]["confidence"] == 1.0
            assert rows[i]["cluster_id"] is None
        assert rows[other]["person_id"] is None and rows[other]["confidence"] == 0.0
    finally:
        conn.close()


def test_assign_cluster_clears_stale_negatives_against_that_person(tmp_path):
    conn = db.connect(tmp_path / "a.db")
    try:
        fid = _face(conn, cluster=5)
        anna = db.get_or_create_person(conn, "Anna")
        db.add_face_negative(conn, fid, anna)  # earlier "this is not Anna"
        library.assign_cluster(conn, 5, person_id=anna)  # now the user says it is
        assert conn.execute("SELECT COUNT(*) FROM face_negatives").fetchone()[0] == 0
    finally:
        conn.close()


def test_migration_v2_promotes_cluster_named_faces(tmp_path):
    path = tmp_path / "old.db"
    conn = db.connect(path)
    try:
        anna = db.get_or_create_person(conn, "Anna")
        cluster_named = _face(conn, person=anna, confidence=0.0)   # old assign_cluster
        null_conf = _face(conn, person=anna, confidence=None)      # ditto, never set
        auto_tag = _face(conn, person=anna, confidence=0.72)       # a machine guess
        unknown = _face(conn, person=None, confidence=0.0)         # nobody
        conn.execute("PRAGMA user_version = 1")  # a catalog from before this fix
        conn.commit()
    finally:
        conn.close()

    conn = db.connect(path)  # reopening runs the pending migration
    try:
        conf = {r["id"]: r["confidence"] for r in conn.execute("SELECT id, confidence FROM faces")}
        assert conf[cluster_named] == 1.0 and conf[null_conf] == 1.0
        assert conf[auto_tag] == 0.72 and conf[unknown] == 0.0
        assert conn.execute("PRAGMA user_version").fetchone()[0] == db.SCHEMA_VERSION >= 2
    finally:
        conn.close()


def test_enrollment_excludes_unconfirmed_auto_tags(tmp_path):
    conn = db.connect(tmp_path / "a.db")
    try:
        anna = db.get_or_create_person(conn, "Anna")
        bob = db.get_or_create_person(conn, "Bob")
        _face(conn, person=anna, confidence=1.0, vec=(1, 0, 0))
        _face(conn, person=bob, confidence=0.55, vec=(0, 1, 0))  # a guess, not ground truth
        enrollment = indexer._load_enrollment(conn)
        assert enrollment.person_ids.tolist() == [anna]
    finally:
        conn.close()


def test_recompute_keeps_names_given_through_clusters(tmp_path):
    """The _carry_human_labels guard (confidence>=1.0) now also covers cluster naming."""
    conn = db.connect(tmp_path / "a.db")
    try:
        fid = _face(conn, cluster=3)
        library.assign_cluster(conn, 3, name="Anna")
        row = conn.execute("SELECT confidence FROM faces WHERE id=?", (fid,)).fetchone()
        assert row["confidence"] >= 1.0  # what _carry_human_labels checks
    finally:
        conn.close()

