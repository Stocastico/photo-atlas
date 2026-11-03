"""Undoing a cluster naming (quick-review 'Z')."""

from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient

from photo_atlas import db, library
from photo_atlas.api import create_app
from photo_atlas.config import AtlasConfig


def _face(conn, cluster):
    photo = db.upsert_photo(conn, {"path": f"/p/{conn.total_changes}.jpg", "filename": "x.jpg"})
    v = np.array([1, 0, 0], dtype=np.float32)
    db.replace_faces(conn, photo, [{
        "person_id": None, "confidence": 0.0, "cluster_id": cluster,
        "embedding": db.embedding_to_blob(v), "dim": 3, "crop_path": "c.jpg",
    }])
    return conn.execute("SELECT id FROM faces WHERE photo_id=?", (photo,)).fetchone()["id"]


@pytest.fixture
def conn(tmp_path):
    c = db.connect(tmp_path / "c.db")
    yield c
    c.close()


def test_unassign_cluster_restores_the_group_and_drops_the_empty_person(conn):
    ids = [_face(conn, 7) for _ in range(3)]
    other = _face(conn, 8)
    conn.commit()
    pid = library.assign_cluster(conn, 7, name="Zed")
    n = library.unassign_cluster(conn, 7, ids, pid)
    assert n == 3
    rows = conn.execute("SELECT id, person_id, cluster_id, confidence FROM faces").fetchall()
    by = {r["id"]: r for r in rows}
    assert all(by[i]["person_id"] is None and by[i]["cluster_id"] == 7 for i in ids)
    assert all((by[i]["confidence"] or 0) < 1.0 for i in ids)
    assert by[other]["cluster_id"] == 8
    assert conn.execute("SELECT COUNT(*) FROM persons WHERE id=?", (pid,)).fetchone()[0] == 0


def test_unassign_keeps_a_person_who_still_has_other_faces(conn):
    ids = [_face(conn, 7) for _ in range(2)]
    keep = _face(conn, 9)
    conn.commit()
    pid = library.assign_cluster(conn, 9, name="Ann")
    pid2 = library.assign_cluster(conn, 7, person_id=pid)
    library.unassign_cluster(conn, 7, ids, pid2)
    assert conn.execute("SELECT person_id FROM faces WHERE id=?", (keep,)).fetchone()[0] == pid
    assert conn.execute("SELECT COUNT(*) FROM persons WHERE id=?", (pid,)).fetchone()[0] == 1


def test_unassign_only_touches_the_given_faces(conn):
    a = _face(conn, 7)
    b = _face(conn, 7)
    conn.commit()
    pid = library.assign_cluster(conn, 7, name="Zed")
    assert library.unassign_cluster(conn, 7, [a], pid) == 1
    assert conn.execute("SELECT person_id FROM faces WHERE id=?", (b,)).fetchone()[0] == pid


def test_api_assign_returns_face_ids_and_unassign_undoes(tmp_path):
    config = AtlasConfig(home=tmp_path / "lib").ensure_dirs()
    c = db.connect(config.db_path)
    ids = [_face(c, 5), _face(c, 5)]
    c.commit()
    c.close()
    client = TestClient(create_app(config))
    r = client.post("/api/clusters/5/assign", json={"name": "Zed"}).json()
    assert sorted(r["face_ids"]) == sorted(ids)
    u = client.post("/api/clusters/5/unassign",
                    json={"face_ids": r["face_ids"], "person_id": r["person_id"]})
    assert u.status_code == 200 and u.json()["restored"] == 2
    assert [c_["cluster_id"] for c_ in client.get("/api/clusters").json()["clusters"]] == [5]
    assert client.get("/api/persons").json()["persons"] == []
