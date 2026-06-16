"""Face active-learning: negative-aware k-NN + the review/correction flow.

When the user corrects an auto-tag (un/reassigns a face away from a person) we
record a "not this person" negative; those negatives penalise that identity in
future k-NN votes, and low-confidence guesses are surfaced for review.
"""

from __future__ import annotations

import itertools

import numpy as np

from photo_atlas import db, library
from photo_atlas.config import AtlasConfig
from photo_atlas.faces import Enrollment, knn_person_match


# -- negative-aware k-NN ----------------------------------------------------
def test_knn_without_negatives_matches_as_before():
    enr = Enrollment.from_pairs([(1, np.array([1.0, 0.0])), (1, np.array([0.98, 0.02]))])
    pid, conf = knn_person_match(np.array([1.0, 0.0]), enr, k=5, threshold=0.5)
    assert pid == 1 and conf > 0.5


def test_knn_single_negative_vetoes_single_vote():
    enr = Enrollment.from_pairs([(1, np.array([1.0, 0.0]))], [(1, np.array([1.0, 0.0]))])
    pid, _ = knn_person_match(np.array([1.0, 0.0]), enr, k=5, threshold=0.5)
    assert pid is None  # 1 positive − 1 negative = 0 → vetoed


def test_knn_majority_survives_a_single_negative():
    pos = [(1, np.array([1.0, 0.0])), (1, np.array([0.99, 0.01])), (1, np.array([0.98, 0.02]))]
    enr = Enrollment.from_pairs(pos, [(1, np.array([1.0, 0.0]))])
    pid, _ = knn_person_match(np.array([1.0, 0.0]), enr, k=5, threshold=0.5)
    assert pid == 1  # 3 − 1 = 2 net, still wins


def test_knn_negative_lets_runner_up_win():
    pos = [(1, np.array([1.0, 0.0])), (2, np.array([0.9, 0.2]))]
    enr = Enrollment.from_pairs(pos, [(1, np.array([1.0, 0.0]))])
    pid, _ = knn_person_match(np.array([1.0, 0.0]), enr, k=5, threshold=0.5)
    assert pid == 2  # person 1 vetoed (net 0); person 2 wins on its own vote


# -- DB helpers -------------------------------------------------------------
_paths = itertools.count()  # unique photo path per inserted face


def _face(conn, person_id=None, confidence=0.5, crop="/c.jpg", vec=(1.0, 0.0, 0.0)):
    photo_id = db.upsert_photo(conn, {"path": f"/p{next(_paths)}.jpg", "filename": "p.jpg"})
    db.replace_faces(conn, photo_id, [{
        "person_id": person_id, "cluster_id": None,
        "bbox_x": 0, "bbox_y": 0, "bbox_w": 1, "bbox_h": 1, "dim": len(vec),
        "embedding": db.embedding_to_blob(np.array(vec, dtype=np.float32)),
        "crop_path": crop, "confidence": confidence,
    }])
    return conn.execute("SELECT id FROM faces WHERE photo_id=?", (photo_id,)).fetchone()["id"]


def test_negatives_add_idempotent_load_and_cascade(tmp_path):
    conn = db.connect(tmp_path / "a.db")
    try:
        pid = db.get_or_create_person(conn, "Alice")
        fid = _face(conn, person_id=pid)
        db.add_face_negative(conn, fid, pid)
        db.add_face_negative(conn, fid, pid)  # idempotent (UNIQUE)
        assert conn.execute("SELECT COUNT(*) FROM face_negatives").fetchone()[0] == 1
        negs = db.load_negatives(conn)
        assert len(negs) == 1 and negs[0][0] == pid and negs[0][1].shape == (3,)
        # Deleting the person cascades its negatives away.
        conn.execute("DELETE FROM persons WHERE id=?", (pid,))
        conn.commit()
        assert conn.execute("SELECT COUNT(*) FROM face_negatives").fetchone()[0] == 0
    finally:
        conn.close()


# -- library correction flow ------------------------------------------------
def test_unassign_records_a_negative(tmp_path):
    conn = db.connect(tmp_path / "b.db")
    try:
        pid = db.get_or_create_person(conn, "Bob")
        fid = _face(conn, person_id=pid, confidence=0.5)
        library.unassign_face(conn, fid)
        assert conn.execute("SELECT person_id FROM faces WHERE id=?", (fid,)).fetchone()[0] is None
        neg = conn.execute(
            "SELECT person_id FROM face_negatives WHERE face_id=?", (fid,)
        ).fetchone()
        assert neg["person_id"] == pid
    finally:
        conn.close()


def test_reassign_records_negative_for_old_and_sets_confidence(tmp_path):
    conn = db.connect(tmp_path / "c.db")
    try:
        a, b = db.get_or_create_person(conn, "A"), db.get_or_create_person(conn, "B")
        fid = _face(conn, person_id=a, confidence=0.5)
        library.assign_face(conn, fid, person_id=b)
        row = conn.execute("SELECT person_id, confidence FROM faces WHERE id=?", (fid,)).fetchone()
        assert row["person_id"] == b and row["confidence"] == 1.0  # human label is certain
        negs = [r["person_id"] for r in conn.execute(
            "SELECT person_id FROM face_negatives WHERE face_id=?", (fid,)
        )]
        assert negs == [a]  # "not A"; nothing recorded against B
    finally:
        conn.close()


def test_confirming_clears_a_prior_negative(tmp_path):
    conn = db.connect(tmp_path / "d.db")
    try:
        a = db.get_or_create_person(conn, "A")
        fid = _face(conn, person_id=None, confidence=0.0)
        db.add_face_negative(conn, fid, a)
        library.assign_face(conn, fid, person_id=a)  # user says it *is* A
        assert conn.execute(
            "SELECT COUNT(*) FROM face_negatives WHERE face_id=? AND person_id=?", (fid, a)
        ).fetchone()[0] == 0
    finally:
        conn.close()


def test_low_confidence_faces_filters(tmp_path):
    conn = db.connect(tmp_path / "e.db")
    try:
        a = db.get_or_create_person(conn, "A")
        keep = _face(conn, person_id=a, confidence=0.55)        # auto + low → kept
        _face(conn, person_id=a, confidence=0.9)                # confident → excluded
        _face(conn, person_id=a, confidence=1.0)                # manual → excluded
        _face(conn, person_id=a, confidence=0.55, crop=None)    # no crop → excluded
        _face(conn, person_id=None, confidence=0.55)            # unassigned → excluded
        rows = library.low_confidence_faces(conn, max_confidence=0.6)
        assert [r["id"] for r in rows] == [keep]
        assert rows[0]["person_name"] == "A"
    finally:
        conn.close()


# -- API --------------------------------------------------------------------
def _client(tmp_path):
    config = AtlasConfig(home=tmp_path / "lib").ensure_dirs()
    conn = db.connect(config.db_path)
    a = db.get_or_create_person(conn, "Ann")
    fid = _face(conn, person_id=a, confidence=0.52)
    conn.commit()
    conn.close()
    from fastapi.testclient import TestClient

    from photo_atlas.api import create_app

    return TestClient(create_app(config)), config, fid, a


def test_api_review_lists_low_confidence_then_reject_records_negative(tmp_path):
    client, config, fid, a = _client(tmp_path)
    review = client.get("/api/faces/review").json()
    assert review["count"] == 1 and review["faces"][0]["id"] == fid

    # Reject → unassign + a recorded negative; the guess drops out of review.
    assert client.post(f"/api/faces/{fid}/unassign").status_code == 200
    assert client.get("/api/faces/review").json()["count"] == 0
    conn = db.connect(config.db_path)
    try:
        assert conn.execute(
            "SELECT person_id FROM face_negatives WHERE face_id=?", (fid,)
        ).fetchone()["person_id"] == a
    finally:
        conn.close()


def test_api_confirm_promotes_to_full_confidence(tmp_path):
    client, config, fid, a = _client(tmp_path)
    assert client.post(f"/api/faces/{fid}/assign", json={"person_id": a}).status_code == 200
    assert client.get("/api/faces/review").json()["count"] == 0  # confidence now 1.0
