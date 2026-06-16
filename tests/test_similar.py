"""'More like this' — rank photos by SigLIP image-embedding similarity.

Reuses the persisted per-photo embeddings (the same matrix semantic search uses),
so it needs no text encoder or model download: a similar-photos request is a pure
cosine ranking of the target photo's own vector against the rest of the library.
"""

from __future__ import annotations

import numpy as np

from photo_atlas import db, search
from photo_atlas.config import AtlasConfig


def _unit(*vals: float) -> np.ndarray:
    v = np.array(vals, dtype=np.float32)
    return v / np.linalg.norm(v)


def _insert(conn, path, vec=None, **cols):
    cols.setdefault("filename", path.rsplit("/", 1)[-1])
    pid = db.upsert_photo(conn, {"path": path, **cols})
    if vec is not None:
        db.set_photo_embedding(conn, pid, vec)
    return pid


# -- SemanticIndex.vector_for ----------------------------------------------
def test_vector_for_returns_embedding_or_none(tmp_path):
    conn = db.connect(tmp_path / "s.db")
    try:
        a = _insert(conn, "/a.jpg", vec=_unit(1, 0, 0))
        none_id = _insert(conn, "/b.jpg")  # no embedding
        index = search.SemanticIndex.load(conn)

        vec = index.vector_for(a)
        assert vec is not None and np.allclose(vec, _unit(1, 0, 0), atol=1e-6)
        # A photo with no embedding isn't in the matrix.
        assert index.vector_for(none_id) is None
        # An unknown id is None, not an IndexError.
        assert index.vector_for(999) is None
    finally:
        conn.close()


def test_vector_for_empty_index_is_safe(tmp_path):
    conn = db.connect(tmp_path / "s.db")
    try:
        assert search.SemanticIndex.load(conn).vector_for(1) is None
    finally:
        conn.close()


# -- similar_photos ---------------------------------------------------------
def test_similar_photos_orders_by_cosine_and_excludes_self(tmp_path):
    conn = db.connect(tmp_path / "s.db")
    try:
        a = _insert(conn, "/a.jpg", vec=_unit(1, 0, 0))
        b = _insert(conn, "/b.jpg", vec=_unit(0.9, 0.1, 0))
        c = _insert(conn, "/c.jpg", vec=_unit(0.2, 0.98, 0))
        index = search.SemanticIndex.load(conn)

        rows, total = search.similar_photos(conn, a, index, limit=10)
        ids = [r["id"] for r in rows]
        # The target is excluded; the nearest neighbour (b) leads, then c.
        assert a not in ids
        assert ids == [b, c]
        assert total == 2
        # Rows carry the similarity score and drop the heavy scene_scores blob.
        assert "score" in rows[0] and "scene_scores" not in rows[0]
        assert rows[0]["score"] >= rows[1]["score"]
    finally:
        conn.close()


def test_similar_photos_paginates(tmp_path):
    conn = db.connect(tmp_path / "s.db")
    try:
        a = _insert(conn, "/a.jpg", vec=_unit(1, 0, 0))
        b = _insert(conn, "/b.jpg", vec=_unit(0.9, 0.1, 0))
        c = _insert(conn, "/c.jpg", vec=_unit(0.8, 0.2, 0))
        index = search.SemanticIndex.load(conn)

        page1, total = search.similar_photos(conn, a, index, limit=1, offset=0)
        page2, total2 = search.similar_photos(conn, a, index, limit=1, offset=1)
        assert total == total2 == 2
        assert [r["id"] for r in page1] == [b]
        assert [r["id"] for r in page2] == [c]
    finally:
        conn.close()


def test_similar_photos_top_k_caps_total(tmp_path):
    conn = db.connect(tmp_path / "s.db")
    try:
        a = _insert(conn, "/a.jpg", vec=_unit(1, 0, 0))
        for i in range(1, 6):
            _insert(conn, f"/p{i}.jpg", vec=_unit(1, i * 0.01, 0))
        index = search.SemanticIndex.load(conn)
        # 5 neighbours exist, but top_k caps the candidate set (self excluded).
        _, total = search.similar_photos(conn, a, index, top_k=3)
        assert total == 3
    finally:
        conn.close()


def test_similar_photos_target_without_embedding_is_empty(tmp_path):
    conn = db.connect(tmp_path / "s.db")
    try:
        _insert(conn, "/a.jpg", vec=_unit(1, 0, 0))
        no_vec = _insert(conn, "/b.jpg")  # no embedding -> nothing to compare
        index = search.SemanticIndex.load(conn)
        assert search.similar_photos(conn, no_vec, index) == ([], 0)
    finally:
        conn.close()


# -- API endpoint -----------------------------------------------------------
def _app_with_embeddings(tmp_path, vecs):
    config = AtlasConfig(home=tmp_path / "lib").ensure_dirs()
    conn = db.connect(config.db_path)
    ids = [_insert(conn, f"/p{i}.jpg", vec=vec) for i, vec in enumerate(vecs)]
    conn.commit()
    conn.close()
    from fastapi.testclient import TestClient

    from photo_atlas.api import create_app

    return config, ids, TestClient(create_app(config))


def test_api_similar_orders_by_relevance(tmp_path):
    _, ids, client = _app_with_embeddings(
        tmp_path, [_unit(1, 0, 0), _unit(0.9, 0.1, 0), _unit(0.1, 1, 0)]
    )
    data = client.get(f"/api/photos/{ids[0]}/similar").json()
    assert data["total"] == 2
    # Nearest first, target excluded.
    assert [p["id"] for p in data["photos"]] == [ids[1], ids[2]]
    assert "score" in data["photos"][0]


def test_api_similar_404_for_unknown_photo(tmp_path):
    _, _ids, client = _app_with_embeddings(tmp_path, [_unit(1, 0, 0)])
    assert client.get("/api/photos/99999/similar").status_code == 404


def test_api_similar_409_without_embeddings(tmp_path):
    config = AtlasConfig(home=tmp_path / "lib").ensure_dirs()
    conn = db.connect(config.db_path)
    pid = _insert(conn, "/p.jpg")  # exists but no embedding, library has none
    conn.commit()
    conn.close()
    from fastapi.testclient import TestClient

    from photo_atlas.api import create_app

    client = TestClient(create_app(config))
    assert client.get(f"/api/photos/{pid}/similar").status_code == 409
