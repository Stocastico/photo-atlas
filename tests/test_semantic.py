"""Natural-language semantic search.

The ranking math, the DB embedding round-trip, the filter-ANDed
``semantic_search`` and the ``/api/photos?text=`` endpoint are all covered
offline: image embeddings are written directly and a stub text encoder stands in
for SigLIP, so the suite needs no model download or the ``scene`` extra.
"""

from __future__ import annotations

import numpy as np

from photo_atlas import db, embed, indexer, search
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


# -- DB round-trip ----------------------------------------------------------
def test_set_photo_embedding_round_trip(tmp_path):
    conn = db.connect(tmp_path / "s.db")
    try:
        vec = _unit(1, 2, 3, 4)
        pid = _insert(conn, "/a/x.jpg", vec=vec)
        row = conn.execute(
            "SELECT embedding, embed_dim FROM photos WHERE id=?", (pid,)
        ).fetchone()
        stored = db.blob_to_embedding(row["embedding"])
        assert row["embed_dim"] == 4
        assert np.allclose(stored, vec, atol=1e-6)

        # Clearing sets both columns back to NULL.
        db.set_photo_embedding(conn, pid, None)
        row = conn.execute(
            "SELECT embedding, embed_dim FROM photos WHERE id=?", (pid,)
        ).fetchone()
        assert row["embedding"] is None and row["embed_dim"] is None
    finally:
        conn.close()


def test_embedding_columns_are_migrated_onto_old_catalogs(tmp_path):
    """A catalog created before the embedding columns gains them via _migrate."""

    path = tmp_path / "old.db"
    conn = db.connect(path)
    conn.execute("ALTER TABLE photos DROP COLUMN embedding")
    conn.execute("ALTER TABLE photos DROP COLUMN embed_dim")
    conn.commit()
    conn.close()

    conn = db.connect(path)  # re-open runs _migrate first
    try:
        cols = {r["name"] for r in conn.execute("PRAGMA table_info(photos)")}
        assert {"embedding", "embed_dim"} <= cols
    finally:
        conn.close()


# -- ranking ----------------------------------------------------------------
def test_semantic_index_ranks_by_cosine(tmp_path):
    conn = db.connect(tmp_path / "s.db")
    try:
        a = _insert(conn, "/a.jpg", vec=_unit(1, 0, 0))
        b = _insert(conn, "/b.jpg", vec=_unit(0, 1, 0))
        _insert(conn, "/noembed.jpg")  # excluded: no embedding
        index = search.SemanticIndex.load(conn)
        assert index.size == 2  # the un-embedded photo is not in the matrix

        ranked = index.rank(_unit(0.9, 0.1, 0))
        assert [pid for pid, _ in ranked] == [a, b]
        assert ranked[0][1] > ranked[1][1]
    finally:
        conn.close()


def test_semantic_index_respects_allowed_and_top_k(tmp_path):
    conn = db.connect(tmp_path / "s.db")
    try:
        a = _insert(conn, "/a.jpg", vec=_unit(1, 0, 0))
        b = _insert(conn, "/b.jpg", vec=_unit(0.9, 0.1, 0))
        _insert(conn, "/c.jpg", vec=_unit(0.8, 0.2, 0))
        index = search.SemanticIndex.load(conn)

        # allowed_ids masks out everything but a + b.
        ranked = index.rank(_unit(1, 0, 0), allowed_ids={a, b})
        assert {pid for pid, _ in ranked} == {a, b}
        # top_k caps the result length.
        assert len(index.rank(_unit(1, 0, 0), top_k=1)) == 1
    finally:
        conn.close()


def test_semantic_index_empty_is_safe(tmp_path):
    conn = db.connect(tmp_path / "s.db")
    try:
        index = search.SemanticIndex.load(conn)
        assert index.size == 0
        assert index.rank(_unit(1, 0, 0)) == []
    finally:
        conn.close()


# -- semantic_search (filters ANDed, paginated) -----------------------------
def test_semantic_search_ands_filters_and_paginates(tmp_path):
    conn = db.connect(tmp_path / "s.db")
    try:
        # Three "beach-ish" photos, one tagged food. A scene filter must mask the
        # food one out before relevance ordering is applied.
        a = _insert(conn, "/a.jpg", vec=_unit(1, 0, 0), scene_type="landscape")
        b = _insert(conn, "/b.jpg", vec=_unit(0.9, 0.1, 0), scene_type="landscape")
        _insert(conn, "/c.jpg", vec=_unit(0.95, 0.05, 0), scene_type="food")
        index = search.SemanticIndex.load(conn)

        rows, total = search.semantic_search(
            conn, {"scene": ["landscape"]}, _unit(1, 0, 0), index, limit=10
        )
        assert total == 2
        assert [r["id"] for r in rows] == [a, b]
        # Each row carries its relevance score and drops the heavy scene_scores blob.
        assert "score" in rows[0] and "scene_scores" not in rows[0]

        # Pagination over the ranked list.
        page2, total2 = search.semantic_search(
            conn, {"scene": ["landscape"]}, _unit(1, 0, 0), index, limit=1, offset=1
        )
        assert total2 == 2 and [r["id"] for r in page2] == [b]
    finally:
        conn.close()


def test_semantic_search_top_k_caps_total(tmp_path):
    conn = db.connect(tmp_path / "s.db")
    try:
        for i in range(5):
            _insert(conn, f"/p{i}.jpg", vec=_unit(1, i * 0.01, 0))
        index = search.SemanticIndex.load(conn)
        _, total = search.semantic_search(conn, {}, _unit(1, 0, 0), index, top_k=3)
        assert total == 3
    finally:
        conn.close()


# -- embed_library backfill (with a stub encoder, no ONNX) ------------------
class _StubImageEncoder:
    """Maps any image to a fixed embedding (deterministic, model-free)."""

    def __init__(self, vec):
        self._vec = np.asarray(vec, dtype=np.float32)

    def embed_image(self, _img):
        return self._vec


def test_embed_library_backfills_only_missing(tmp_path):
    config = AtlasConfig(home=tmp_path / "lib").ensure_dirs()
    # Two real (tiny) images on disk to decode.
    from PIL import Image

    conn = db.connect(config.db_path)
    paths = []
    for i in range(2):
        p = tmp_path / f"img{i}.jpg"
        Image.new("RGB", (8, 8), (i * 40, 0, 0)).save(p)
        paths.append(str(p))
        _insert(conn, str(p))
    conn.commit()
    conn.close()

    n = indexer.embed_library(config, image_encoder=_StubImageEncoder(_unit(1, 0)))
    assert n == 2

    conn = db.connect(config.db_path)
    try:
        embedded = conn.execute(
            "SELECT COUNT(*) FROM photos WHERE embedding IS NOT NULL"
        ).fetchone()[0]
        assert embedded == 2
        # A second pass (no --recompute) skips the already-embedded photos.
        assert indexer.embed_library(config, image_encoder=_StubImageEncoder(_unit(0, 1))) == 0
    finally:
        conn.close()


# -- API endpoint -----------------------------------------------------------
class _StubTextEncoder:
    def __init__(self, vec):
        self._vec = np.asarray(vec, dtype=np.float32)

    def embed_text(self, _text):
        return self._vec


def _app_with_embeddings(tmp_path, vecs):
    config = AtlasConfig(home=tmp_path / "lib").ensure_dirs()
    conn = db.connect(config.db_path)
    ids = []
    for i, vec in enumerate(vecs):
        ids.append(_insert(conn, f"/p{i}.jpg", vec=vec))
    conn.commit()
    conn.close()
    from fastapi.testclient import TestClient

    from photo_atlas.api import create_app

    return config, ids, TestClient(create_app(config))


def test_api_semantic_search_orders_by_relevance(tmp_path, monkeypatch):
    _, ids, client = _app_with_embeddings(tmp_path, [_unit(1, 0, 0), _unit(0, 1, 0)])
    # Stand in for the SigLIP text encoder so no model/extra is needed.
    monkeypatch.setattr(
        embed.SigLipTextEncoder, "from_config",
        classmethod(lambda cls, c: _StubTextEncoder(_unit(0, 1, 0))),
    )
    data = client.get("/api/photos", params={"text": "a green field"}).json()
    assert data["total"] == 2
    # The query vector points at the second photo, so it ranks first.
    assert [p["id"] for p in data["photos"]] == [ids[1], ids[0]]
    assert "score" in data["photos"][0]


def test_api_semantic_search_409_without_embeddings(tmp_path):
    # An empty library has no embeddings -> a clean 409, not a 500.
    config = AtlasConfig(home=tmp_path / "lib").ensure_dirs()
    db.connect(config.db_path).close()
    from fastapi.testclient import TestClient

    from photo_atlas.api import create_app

    client = TestClient(create_app(config))
    assert client.get("/api/photos", params={"text": "anything"}).status_code == 409


def test_api_semantic_search_501_without_encoder(tmp_path, monkeypatch):
    _, _ids, client = _app_with_embeddings(tmp_path, [_unit(1, 0, 0)])

    def _boom(cls, c):
        raise RuntimeError("no onnxruntime")

    monkeypatch.setattr(embed.SigLipTextEncoder, "from_config", classmethod(_boom))
    assert client.get("/api/photos", params={"text": "x"}).status_code == 501


def test_api_capabilities_reports_semantic(tmp_path):
    _, _ids, client = _app_with_embeddings(tmp_path, [_unit(1, 0, 0)])
    caps = client.get("/api/capabilities").json()
    # Truthy iff embeddings exist AND the runtime libs are importable; the value
    # depends on whether the optional extra is installed, so just assert the shape.
    assert set(caps) == {"semantic"} and isinstance(caps["semantic"], bool)
