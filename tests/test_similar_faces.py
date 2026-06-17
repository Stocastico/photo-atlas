"""'More like this person' — rank photos by SFace face-embedding similarity.

Complements the named-person filter: for an *unnamed* face (no person assigned,
so not filterable) this gathers other photos of the same face from the stored
SFace embeddings — no model download, a pure cosine ranking deduped to photos.
"""

from __future__ import annotations

import numpy as np

from photo_atlas import db, search
from photo_atlas.config import AtlasConfig


def _unit(*vals: float) -> np.ndarray:
    v = np.array(vals, dtype=np.float32)
    return v / np.linalg.norm(v)


def _photo(conn, path, **cols):
    cols.setdefault("filename", path.rsplit("/", 1)[-1])
    return db.upsert_photo(conn, {"path": path, **cols})


def _face(conn, photo_id, vec):
    db.replace_faces(
        conn, photo_id,
        [{"embedding": db.embedding_to_blob(vec), "dim": int(vec.shape[0])}],
    )
    return conn.execute(
        "SELECT id FROM faces WHERE photo_id=? ORDER BY id DESC", (photo_id,)
    ).fetchone()[0]


# -- FaceIndex --------------------------------------------------------------
def test_face_index_vector_and_photo_lookup(tmp_path):
    conn = db.connect(tmp_path / "s.db")
    try:
        pa = _photo(conn, "/a.jpg")
        fa = _face(conn, pa, _unit(1, 0, 0))
        index = search.FaceIndex.load(conn)
        vec = index.vector_for(fa)
        assert vec is not None and np.allclose(vec, _unit(1, 0, 0), atol=1e-6)
        assert index.photo_for(fa) == pa
        # Unknown ids are None, not an IndexError.
        assert index.vector_for(999) is None
        assert index.photo_for(999) is None
    finally:
        conn.close()


def test_face_index_empty_is_safe(tmp_path):
    conn = db.connect(tmp_path / "s.db")
    try:
        idx = search.FaceIndex.load(conn)
        assert idx.size == 0
        assert idx.vector_for(1) is None
    finally:
        conn.close()


# -- similar_faces ----------------------------------------------------------
def test_similar_faces_orders_by_cosine_and_dedupes_to_photos(tmp_path):
    conn = db.connect(tmp_path / "s.db")
    try:
        pa = _photo(conn, "/a.jpg")
        fa = _face(conn, pa, _unit(1, 0, 0))
        pb = _photo(conn, "/b.jpg")
        _face(conn, pb, _unit(0.9, 0.1, 0))
        pc = _photo(conn, "/c.jpg")
        _face(conn, pc, _unit(0.2, 0.98, 0))
        index = search.FaceIndex.load(conn)

        rows, total = search.similar_faces(conn, fa, index, limit=10)
        ids = [r["id"] for r in rows]
        assert pa not in ids  # the source face's own photo is excluded
        assert ids == [pb, pc]  # nearest first
        assert total == 2
        assert "score" in rows[0] and "scene_scores" not in rows[0]
        assert rows[0]["score"] >= rows[1]["score"]
    finally:
        conn.close()


def test_similar_faces_excludes_source_photo_even_with_other_faces(tmp_path):
    conn = db.connect(tmp_path / "s.db")
    try:
        pa = _photo(conn, "/a.jpg")
        # Two faces in the *same* photo: the source and a near-identical one.
        db.replace_faces(conn, pa, [
            {"embedding": db.embedding_to_blob(_unit(1, 0, 0)), "dim": 3},
            {"embedding": db.embedding_to_blob(_unit(0.99, 0.01, 0)), "dim": 3},
        ])
        fa = conn.execute("SELECT id FROM faces WHERE photo_id=? ORDER BY id", (pa,)).fetchone()[0]
        pb = _photo(conn, "/b.jpg")
        _face(conn, pb, _unit(0.5, 0.5, 0))
        index = search.FaceIndex.load(conn)

        rows, total = search.similar_faces(conn, fa, index, limit=10)
        # Only photo B comes back — the source photo is never a match for itself,
        # even though it holds a second very-similar face.
        assert [r["id"] for r in rows] == [pb]
        assert total == 1
    finally:
        conn.close()


def test_similar_faces_keeps_best_score_per_photo(tmp_path):
    conn = db.connect(tmp_path / "s.db")
    try:
        pa = _photo(conn, "/a.jpg")
        fa = _face(conn, pa, _unit(1, 0, 0))
        # Photo B has two faces: one far, one near. The photo should rank by its
        # *best* (nearest) face.
        pb = _photo(conn, "/b.jpg")
        db.replace_faces(conn, pb, [
            {"embedding": db.embedding_to_blob(_unit(0.1, 0.99, 0)), "dim": 3},
            {"embedding": db.embedding_to_blob(_unit(0.98, 0.2, 0)), "dim": 3},
        ])
        index = search.FaceIndex.load(conn)
        rows, total = search.similar_faces(conn, fa, index, limit=10)
        assert total == 1 and rows[0]["id"] == pb
        assert rows[0]["score"] > 0.9  # the near face, not the far one
    finally:
        conn.close()


def test_similar_faces_paginates_and_top_k_caps(tmp_path):
    conn = db.connect(tmp_path / "s.db")
    try:
        pa = _photo(conn, "/a.jpg")
        fa = _face(conn, pa, _unit(1, 0, 0))
        others = []
        for i in range(1, 6):
            p = _photo(conn, f"/p{i}.jpg")
            _face(conn, p, _unit(1, i * 0.01, 0))
            others.append(p)
        index = search.FaceIndex.load(conn)

        page1, total = search.similar_faces(conn, fa, index, limit=2, offset=0)
        page2, _ = search.similar_faces(conn, fa, index, limit=2, offset=2)
        assert total == 5
        assert len(page1) == 2 and len(page2) == 2
        assert page1[0]["id"] == others[0]  # nearest neighbour leads

        _, capped = search.similar_faces(conn, fa, index, top_k=3)
        assert capped == 3
    finally:
        conn.close()


def test_similar_faces_source_without_embedding_is_empty(tmp_path):
    conn = db.connect(tmp_path / "s.db")
    try:
        pa = _photo(conn, "/a.jpg")
        db.replace_faces(conn, pa, [{"bbox_x": 1}])  # a face row with no embedding
        fa = conn.execute("SELECT id FROM faces WHERE photo_id=?", (pa,)).fetchone()[0]
        pb = _photo(conn, "/b.jpg")
        _face(conn, pb, _unit(1, 0, 0))
        index = search.FaceIndex.load(conn)
        assert search.similar_faces(conn, fa, index) == ([], 0)
    finally:
        conn.close()


# -- API --------------------------------------------------------------------
def _app(tmp_path):
    config = AtlasConfig(home=tmp_path / "lib").ensure_dirs()
    conn = db.connect(config.db_path)
    pa = _photo(conn, "/a.jpg")
    fa = _face(conn, pa, _unit(1, 0, 0))
    pb = _photo(conn, "/b.jpg")
    _face(conn, pb, _unit(0.9, 0.1, 0))
    pc = _photo(conn, "/c.jpg")
    _face(conn, pc, _unit(0.1, 1, 0))
    conn.commit()
    conn.close()
    from fastapi.testclient import TestClient

    from photo_atlas.api import create_app

    return TestClient(create_app(config)), fa, pb, pc


def test_api_similar_faces_orders_by_relevance(tmp_path):
    client, fa, pb, pc = _app(tmp_path)
    data = client.get(f"/api/faces/{fa}/similar").json()
    assert data["total"] == 2
    assert [p["id"] for p in data["photos"]] == [pb, pc]
    assert "score" in data["photos"][0]


def test_api_similar_faces_404_for_unknown_face(tmp_path):
    client, _fa, _pb, _pc = _app(tmp_path)
    assert client.get("/api/faces/99999/similar").status_code == 404


def test_api_similar_faces_409_without_embeddings(tmp_path):
    config = AtlasConfig(home=tmp_path / "lib").ensure_dirs()
    conn = db.connect(config.db_path)
    pa = _photo(conn, "/p.jpg")
    db.replace_faces(conn, pa, [{"bbox_x": 1}])  # a face but no embedding anywhere
    fa = conn.execute("SELECT id FROM faces WHERE photo_id=?", (pa,)).fetchone()[0]
    conn.commit()
    conn.close()
    from fastapi.testclient import TestClient

    from photo_atlas.api import create_app

    client = TestClient(create_app(config))
    assert client.get(f"/api/faces/{fa}/similar").status_code == 409
