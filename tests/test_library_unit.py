"""Direct tests for the person/cluster management error branches."""

from __future__ import annotations

import pytest

from photo_atlas import db, library


def _conn(tmp_path):
    return db.connect(tmp_path / "lib.db")


def test_assign_cluster_requires_name_or_id(tmp_path):
    conn = _conn(tmp_path)
    try:
        with pytest.raises(ValueError):
            library.assign_cluster(conn, 1)
    finally:
        conn.close()


def test_assign_face_requires_name_or_id(tmp_path):
    conn = _conn(tmp_path)
    try:
        with pytest.raises(ValueError):
            library.assign_face(conn, 1)
    finally:
        conn.close()


def test_set_cover_face_not_found(tmp_path):
    conn = _conn(tmp_path)
    try:
        pid = db.get_or_create_person(conn, "Ada")
        with pytest.raises(ValueError, match="face not found"):
            library.set_cover_face(conn, pid, 12345)
    finally:
        conn.close()


def test_merge_into_self_rejected(tmp_path):
    conn = _conn(tmp_path)
    try:
        pid = db.get_or_create_person(conn, "Ada")
        with pytest.raises(ValueError):
            library.merge_persons(conn, pid, pid)
    finally:
        conn.close()


def test_assign_then_unassign_face_roundtrip(tmp_path):
    conn = _conn(tmp_path)
    try:
        photo_id = db.upsert_photo(conn, {"path": "/x.jpg", "filename": "x.jpg"})
        db.replace_faces(conn, photo_id, [{"bbox_x": 0, "bbox_y": 0, "bbox_w": 1, "bbox_h": 1}])
        face_id = conn.execute("SELECT id FROM faces").fetchone()["id"]

        pid = library.assign_face(conn, face_id, name="Grace")
        assert conn.execute(
            "SELECT person_id FROM faces WHERE id=?", (face_id,)
        ).fetchone()["person_id"] == pid

        library.unassign_face(conn, face_id)
        assert conn.execute(
            "SELECT person_id FROM faces WHERE id=?", (face_id,)
        ).fetchone()["person_id"] is None
    finally:
        conn.close()
