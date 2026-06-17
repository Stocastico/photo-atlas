"""Embedding-cache freshness: a `meta` version bumped on embedding writes.

The web layer caches the in-memory `SemanticIndex` and keys it on a signature.
`(count, max_id)` alone can't detect an *in-place* re-embed (`embed --recompute`
rewrites the BLOBs without changing either), so a running server would serve a
stale index until restart. A monotonic `meta['embeddings_version']`, bumped by
`db.set_photo_embedding`, closes that gap.
"""

from __future__ import annotations

import numpy as np

from photo_atlas import db
from photo_atlas.config import AtlasConfig


def _unit(*vals: float) -> np.ndarray:
    v = np.array(vals, dtype=np.float32)
    return v / np.linalg.norm(v)


# -- db meta version --------------------------------------------------------
def test_bump_meta_increments_and_get_reads(tmp_path):
    conn = db.connect(tmp_path / "s.db")
    try:
        assert db.get_meta(conn, "embeddings_version") is None
        assert db.bump_meta(conn, "embeddings_version") == 1
        assert db.bump_meta(conn, "embeddings_version") == 2
        assert db.get_meta(conn, "embeddings_version") == "2"
    finally:
        conn.close()


def test_set_photo_embedding_bumps_version(tmp_path):
    conn = db.connect(tmp_path / "s.db")
    try:
        pid = db.upsert_photo(conn, {"path": "/a.jpg", "filename": "a.jpg"})
        assert db.get_meta(conn, "embeddings_version") is None
        db.set_photo_embedding(conn, pid, _unit(1, 0, 0))
        assert db.get_meta(conn, "embeddings_version") == "1"
        # Clearing an embedding is still a change → bumps again.
        db.set_photo_embedding(conn, pid, None)
        assert db.get_meta(conn, "embeddings_version") == "2"
    finally:
        conn.close()


def test_set_photo_embedding_blob_bumps_version(tmp_path):
    # The index-time write path stores a *pre-serialised* blob (computed in a
    # worker) rather than an ndarray, but it must bump the version exactly like
    # set_photo_embedding so `index --embed --recompute` invalidates a live cache.
    conn = db.connect(tmp_path / "s.db")
    try:
        pid = db.upsert_photo(conn, {"path": "/a.jpg", "filename": "a.jpg"})
        blob = db.embedding_to_blob(_unit(1, 0, 0))
        db.set_photo_embedding_blob(conn, pid, blob, 3)
        assert db.get_meta(conn, "embeddings_version") == "1"
        row = conn.execute(
            "SELECT embedding, embed_dim FROM photos WHERE id=?", (pid,)
        ).fetchone()
        assert row["embed_dim"] == 3
        assert db.blob_to_embedding(row["embedding"]) is not None
        # A second write (an in-place recompute) bumps again.
        db.set_photo_embedding_blob(conn, pid, blob, 3)
        assert db.get_meta(conn, "embeddings_version") == "2"
    finally:
        conn.close()


# -- API: a running server reflects an in-place recompute -------------------
def _photo(conn, path, vec):
    pid = db.upsert_photo(conn, {"path": path, "filename": path.rsplit("/", 1)[-1]})
    db.set_photo_embedding(conn, pid, vec)
    return pid


def test_similar_reflects_in_place_recompute(tmp_path):
    config = AtlasConfig(home=tmp_path / "lib").ensure_dirs()
    conn = db.connect(config.db_path)
    a = _photo(conn, "/a.jpg", _unit(1, 0, 0))
    b = _photo(conn, "/b.jpg", _unit(0.9, 0.1, 0))
    c = _photo(conn, "/c.jpg", _unit(0.2, 0.98, 0))
    conn.commit()
    conn.close()

    from fastapi.testclient import TestClient

    from photo_atlas.api import create_app

    client = TestClient(create_app(config))
    first = client.get(f"/api/photos/{a}/similar").json()
    assert [p["id"] for p in first["photos"]] == [b, c]  # b nearer than c

    # Re-embed c in place to be the *nearest* neighbour. Count and max id are
    # unchanged, so only the version bump can invalidate the cached index.
    conn = db.connect(config.db_path)
    db.set_photo_embedding(conn, c, _unit(0.95, 0.05, 0))
    conn.commit()
    conn.close()

    second = client.get(f"/api/photos/{a}/similar").json()
    assert [p["id"] for p in second["photos"]] == [c, b]  # cache reloaded
