"""Per-person semantic grounding.

For a hybrid query like "Stefano eating food" the whole-image embedding only
says the photo *contains* Stefano and *looks like* eating food. Grounding stores
a SigLIP embedding of the **region around each face** and ranks a named person's
photos by how well *their* region matches the residual visual query, so the score
is about the person rather than the whole frame.

This file covers the storage, geometry, ranking, indexing and API legs.
"""

from __future__ import annotations

import numpy as np
import pytest

from photo_atlas import db, indexer, search
from photo_atlas.config import AtlasConfig
from photo_atlas.indexer import region_box


def _unit(*vals: float) -> np.ndarray:
    v = np.array(vals, dtype=np.float32)
    return v / np.linalg.norm(v)


class _StubImageEncoder:
    """Maps any image to a fixed embedding (deterministic, model-free)."""

    def __init__(self, vec):
        self._vec = np.asarray(vec, dtype=np.float32)

    def embed_image(self, _img):
        return self._vec


# -- DB storage: faces.region_embedding ------------------------------------
def test_region_columns_present_and_round_trip(tmp_path):
    conn = db.connect(tmp_path / "s.db")
    try:
        cols = {r["name"] for r in conn.execute("PRAGMA table_info(faces)")}
        assert {"region_embedding", "region_dim"} <= cols

        pid = db.upsert_photo(conn, {"path": "/a.jpg", "filename": "a.jpg"})
        db.replace_faces(
            conn,
            pid,
            [{"bbox_x": 0, "bbox_y": 0, "bbox_w": 4, "bbox_h": 4,
              "region_embedding": db.embedding_to_blob(_unit(1, 0, 0)),
              "region_dim": 3}],
        )
        row = conn.execute(
            "SELECT region_embedding, region_dim FROM faces WHERE photo_id=?", (pid,)
        ).fetchone()
        assert row["region_dim"] == 3
        assert np.allclose(db.blob_to_embedding(row["region_embedding"]), _unit(1, 0, 0))
    finally:
        conn.close()


def test_set_face_region_embedding_bumps_version(tmp_path):
    conn = db.connect(tmp_path / "s.db")
    try:
        pid = db.upsert_photo(conn, {"path": "/a.jpg", "filename": "a.jpg"})
        db.replace_faces(conn, pid, [{"bbox_x": 0, "bbox_y": 0, "bbox_w": 4, "bbox_h": 4}])
        fid = conn.execute("SELECT id FROM faces WHERE photo_id=?", (pid,)).fetchone()["id"]

        before = db.get_meta(conn, "face_regions_version")
        db.set_face_region_embedding(conn, fid, _unit(0, 1, 0))
        after = db.get_meta(conn, "face_regions_version")
        assert before != after

        row = conn.execute(
            "SELECT region_embedding, region_dim FROM faces WHERE id=?", (fid,)
        ).fetchone()
        assert row["region_dim"] == 3
        assert np.allclose(db.blob_to_embedding(row["region_embedding"]), _unit(0, 1, 0))

        # Clearing it is allowed (None -> NULL/NULL).
        db.set_face_region_embedding(conn, fid, None)
        row = conn.execute(
            "SELECT region_embedding, region_dim FROM faces WHERE id=?", (fid,)
        ).fetchone()
        assert row["region_embedding"] is None and row["region_dim"] is None
    finally:
        conn.close()


# -- region geometry --------------------------------------------------------
def test_region_box_expands_around_face_with_torso_bias():
    # A face well inside a large frame: the region grows sideways and (more)
    # downward to take in the torso, and never leaves the image.
    x, y, w, h = region_box((400, 300, 100, 100), 1000, 1000)
    # Wider than the face and biased so the box extends below it.
    assert w > 100 and h > 100
    assert x < 400 and x + w > 500
    assert y <= 300
    assert y + h > 400 + 100  # reaches well below the face (torso)
    # Stays in bounds.
    assert 0 <= x and 0 <= y and x + w <= 1000 and y + h <= 1000


def test_region_box_clamps_at_edges():
    # A face in the top-left corner can't grow off the image.
    x, y, w, h = region_box((0, 0, 50, 50), 200, 200)
    assert x == 0 and y == 0
    assert x + w <= 200 and y + h <= 200

    # A face in the bottom-right corner likewise stays inside.
    x, y, w, h = region_box((150, 150, 50, 50), 200, 200)
    assert x >= 0 and y >= 0
    assert x + w <= 200 and y + h <= 200
    assert w > 0 and h > 0


def test_region_box_full_frame_face_is_the_whole_image():
    # A face filling the frame yields the whole image (everything clamps).
    assert region_box((0, 0, 64, 64), 64, 64) == (0, 0, 64, 64)


# -- RegionIndex + grounded_search -----------------------------------------
def _add_face(conn, photo_id, person_id, region_vec=None, **bbox):
    bbox.setdefault("bbox_x", 0)
    bbox.setdefault("bbox_y", 0)
    bbox.setdefault("bbox_w", 4)
    bbox.setdefault("bbox_h", 4)
    cur = conn.execute(
        "INSERT INTO faces (photo_id, person_id, bbox_x, bbox_y, bbox_w, bbox_h, "
        "region_embedding, region_dim) VALUES (?,?,?,?,?,?,?,?)",
        (
            photo_id, person_id, bbox["bbox_x"], bbox["bbox_y"],
            bbox["bbox_w"], bbox["bbox_h"],
            db.embedding_to_blob(region_vec) if region_vec is not None else None,
            None if region_vec is None else int(np.asarray(region_vec).reshape(-1).shape[0]),
        ),
    )
    return int(cur.lastrowid)


def _grounding_library(tmp_path):
    """Stefano in two photos with opposite region embeddings; Bob in a third."""
    conn = db.connect(tmp_path / "g.db")
    stefano = db.get_or_create_person(conn, "Stefano")
    bob = db.get_or_create_person(conn, "Bob")

    def photo(path, vec, **cols):
        pid = db.upsert_photo(conn, {"path": path, "filename": path.lstrip("/"), **cols})
        if vec is not None:
            db.set_photo_embedding(conn, pid, vec)
        return pid

    # Whole-image embeddings make B look most "eating"; region embeddings flip it.
    a = photo("/a.jpg", _unit(0.3, 1, 0))  # Stefano's region = eating
    b = photo("/b.jpg", _unit(1, 0.1, 0))  # Stefano's region = beach
    c = photo("/c.jpg", _unit(1, 0, 0))    # Bob only
    _add_face(conn, a, stefano, _unit(1, 0, 0))    # eating
    _add_face(conn, b, stefano, _unit(0, 1, 0))    # beach
    _add_face(conn, c, bob, _unit(1, 0, 0))        # eating, but Bob
    conn.commit()
    return conn, {"stefano": stefano, "bob": bob, "a": a, "b": b, "c": c}


def test_region_index_loads_named_faces_only(tmp_path):
    conn, ids = _grounding_library(tmp_path)
    try:
        # An unnamed face with a region embedding is not grounding material.
        _add_face(conn, ids["a"], None, _unit(0, 0, 1))
        conn.commit()
        index = search.RegionIndex.load(conn)
        # Three named faces carry region embeddings (a, b, c); the unnamed one drops.
        assert index.size == 3
        assert index.has_any({ids["stefano"]})
        assert not index.has_any({999})
    finally:
        conn.close()


def test_grounded_search_ranks_by_the_person_region(tmp_path):
    conn, ids = _grounding_library(tmp_path)
    try:
        index = search.RegionIndex.load(conn)
        # "eating" query, grounded on Stefano: photo A (his region = eating) wins
        # over B even though B's *whole-image* embedding is the more "eating" one.
        filters = {"person_id": [ids["stefano"]]}
        rows, total = search.grounded_search(
            conn, filters, _unit(1, 0, 0), index, [ids["stefano"]]
        )
        assert total == 2  # Bob's photo is excluded by the person filter
        assert [r["id"] for r in rows] == [ids["a"], ids["b"]]
        assert "score" in rows[0] and rows[0]["score"] >= rows[1]["score"]
    finally:
        conn.close()


def test_grounded_search_takes_best_region_per_photo(tmp_path):
    conn, ids = _grounding_library(tmp_path)
    try:
        # A second Stefano face in photo B whose region matches "eating" lifts B.
        _add_face(conn, ids["b"], ids["stefano"], _unit(1, 0, 0))
        conn.commit()
        index = search.RegionIndex.load(conn)
        rows, _ = search.grounded_search(
            conn, {"person_id": [ids["stefano"]]}, _unit(1, 0, 0), index, [ids["stefano"]]
        )
        # Both photos now have an "eating" region, so they tie at the top; the page
        # still contains both (a stable sort keeps a deterministic order).
        assert {r["id"] for r in rows} == {ids["a"], ids["b"]}
    finally:
        conn.close()


def test_grounded_search_honours_structured_filters(tmp_path):
    conn, ids = _grounding_library(tmp_path)
    try:
        # Pin photo A to 2010 and restrict the query to 2010 — only A survives.
        conn.execute("UPDATE photos SET taken_at='2010-05-01T00:00:00' WHERE id=?", (ids["a"],))
        conn.execute("UPDATE photos SET taken_at='2021-05-01T00:00:00' WHERE id=?", (ids["b"],))
        conn.commit()
        index = search.RegionIndex.load(conn)
        filters = {"person_id": [ids["stefano"]], "year": "2010"}
        rows, total = search.grounded_search(
            conn, filters, _unit(1, 0, 0), index, [ids["stefano"]]
        )
        assert total == 1 and [r["id"] for r in rows] == [ids["a"]]
    finally:
        conn.close()


def test_grounded_search_paginates(tmp_path):
    conn, ids = _grounding_library(tmp_path)
    try:
        index = search.RegionIndex.load(conn)
        f = {"person_id": [ids["stefano"]]}
        p1, total = search.grounded_search(conn, f, _unit(1, 0, 0), index, [ids["stefano"]],
                                           limit=1, offset=0)
        p2, total2 = search.grounded_search(conn, f, _unit(1, 0, 0), index, [ids["stefano"]],
                                            limit=1, offset=1)
        assert total == total2 == 2
        assert [r["id"] for r in p1] == [ids["a"]]
        assert [r["id"] for r in p2] == [ids["b"]]
    finally:
        conn.close()


def test_grounded_search_empty_index_is_safe(tmp_path):
    conn = db.connect(tmp_path / "e.db")
    try:
        index = search.RegionIndex.load(conn)
        assert index.size == 0 and not index.has_any({1})
        assert search.grounded_search(conn, {}, _unit(1, 0, 0), index, [1]) == ([], 0)
    finally:
        conn.close()


# -- index-time region embedding -------------------------------------------
def _demo_photo_with_face(config, tmp_path, encoder):
    from scene_stub import StubTagger

    from photo_atlas import demo
    from photo_atlas.faces import SyntheticFaceBackend

    photos = tmp_path / "photos"
    demo.generate(photos, count=8, seed=3)
    backend = SyntheticFaceBackend()
    for i, p in enumerate(sorted(photos.glob("*"))):
        prepared = indexer._prepare_photo(
            config, p, backend=backend, tagger=StubTagger(), enrollment=None,
            sha1=f"deadbeef{i:08d}", image_encoder=encoder,
        )
        if prepared.faces:
            return prepared
    pytest.fail("the synthetic demo produced no detectable faces")


def test_prepare_photo_computes_region_embeddings(config, tmp_path):
    enc = _StubImageEncoder(_unit(1, 0, 0, 0))
    prepared = _demo_photo_with_face(config, tmp_path, enc)
    # Every detected face carries a region embedding in the SigLIP space.
    assert all(f.region_embedding_blob is not None for f in prepared.faces)
    assert all(f.region_dim == 4 for f in prepared.faces)
    region = db.blob_to_embedding(prepared.faces[0].region_embedding_blob)
    assert np.allclose(region, _unit(1, 0, 0, 0))
    # The whole-image embedding is still computed too.
    assert prepared.embedding_blob is not None


def test_prepare_photo_no_encoder_means_no_region_embedding(config, tmp_path):
    from scene_stub import StubTagger

    from photo_atlas import demo
    from photo_atlas.faces import SyntheticFaceBackend

    photos = tmp_path / "photos"
    demo.generate(photos, count=8, seed=3)
    backend = SyntheticFaceBackend()
    for i, p in enumerate(sorted(photos.glob("*"))):
        prepared = indexer._prepare_photo(
            config, p, backend=backend, tagger=StubTagger(), enrollment=None,
            sha1=f"cafe{i:08d}", image_encoder=None,
        )
        if prepared.faces:
            assert all(f.region_embedding_blob is None for f in prepared.faces)
            return
    pytest.fail("the synthetic demo produced no detectable faces")


def test_commit_prepared_persists_region_embeddings(config, tmp_path):
    enc = _StubImageEncoder(_unit(0, 1, 0, 0))
    prepared = _demo_photo_with_face(config, tmp_path, enc)
    conn = db.connect(config.db_path)
    try:
        indexer._commit_prepared(conn, config, prepared, None, None)
        rows = conn.execute(
            "SELECT region_embedding, region_dim FROM faces WHERE region_embedding IS NOT NULL"
        ).fetchall()
        assert rows, "region embeddings should have been written for the faces"
        assert all(r["region_dim"] == 4 for r in rows)
        assert np.allclose(db.blob_to_embedding(rows[0]["region_embedding"]), _unit(0, 1, 0, 0))
    finally:
        conn.close()


# -- embed_face_regions backfill -------------------------------------------
def _library_with_unembedded_faces(tmp_path):
    from PIL import Image

    config = AtlasConfig(home=tmp_path / "lib").ensure_dirs()
    conn = db.connect(config.db_path)
    src = tmp_path / "img.jpg"
    Image.new("RGB", (20, 20), (10, 20, 30)).save(src)
    pid = db.upsert_photo(conn, {"path": str(src), "filename": "img.jpg"})
    db.replace_faces(
        conn,
        pid,
        [
            {"bbox_x": 2, "bbox_y": 2, "bbox_w": 6, "bbox_h": 6},
            {"bbox_x": 10, "bbox_y": 10, "bbox_w": 5, "bbox_h": 5},
        ],
    )
    conn.commit()
    conn.close()
    return config


def test_embed_face_regions_backfills_only_missing(tmp_path):
    config = _library_with_unembedded_faces(tmp_path)
    n = indexer.embed_face_regions(config, image_encoder=_StubImageEncoder(_unit(1, 0, 0, 0)))
    assert n == 2

    conn = db.connect(config.db_path)
    try:
        cnt = conn.execute(
            "SELECT COUNT(*) FROM faces WHERE region_embedding IS NOT NULL"
        ).fetchone()[0]
        assert cnt == 2
    finally:
        conn.close()

    # A second pass (no recompute) skips the already-embedded faces.
    assert indexer.embed_face_regions(
        config, image_encoder=_StubImageEncoder(_unit(0, 1, 0, 0))
    ) == 0


def test_embed_face_regions_recompute_overwrites(tmp_path):
    config = _library_with_unembedded_faces(tmp_path)
    indexer.embed_face_regions(config, image_encoder=_StubImageEncoder(_unit(1, 0, 0, 0)))
    assert indexer.embed_face_regions(
        config, image_encoder=_StubImageEncoder(_unit(0, 1, 0, 0)), recompute=True
    ) == 2
    conn = db.connect(config.db_path)
    try:
        vec = db.blob_to_embedding(
            conn.execute("SELECT region_embedding FROM faces LIMIT 1").fetchone()[0]
        )
        assert np.allclose(vec, _unit(0, 1, 0, 0))
    finally:
        conn.close()


def test_embed_face_regions_skips_missing_source(tmp_path):
    config = AtlasConfig(home=tmp_path / "lib").ensure_dirs()
    conn = db.connect(config.db_path)
    pid = db.upsert_photo(conn, {"path": str(tmp_path / "gone.jpg"), "filename": "gone.jpg"})
    db.replace_faces(conn, pid, [{"bbox_x": 0, "bbox_y": 0, "bbox_w": 4, "bbox_h": 4}])
    conn.commit()
    conn.close()
    # Source file doesn't exist -> nothing embedded, no crash.
    assert indexer.embed_face_regions(config, image_encoder=_StubImageEncoder(_unit(1, 0))) == 0


def test_region_columns_added_to_legacy_faces_table(tmp_path):
    # A catalog whose faces table predates the region columns gains them on connect.
    import sqlite3

    path = tmp_path / "old.db"
    raw = sqlite3.connect(path)
    # The original faces table, before the region columns existed.
    raw.execute(
        "CREATE TABLE faces (id INTEGER PRIMARY KEY AUTOINCREMENT, photo_id INTEGER, "
        "person_id INTEGER, cluster_id INTEGER, bbox_x INTEGER, bbox_y INTEGER, "
        "bbox_w INTEGER, bbox_h INTEGER, dim INTEGER, embedding BLOB, "
        "crop_path TEXT, confidence REAL)"
    )
    raw.commit()
    raw.close()

    conn = db.connect(path)
    try:
        cols = {r["name"] for r in conn.execute("PRAGMA table_info(faces)")}
        assert {"region_embedding", "region_dim"} <= cols
    finally:
        conn.close()
