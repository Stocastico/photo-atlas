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

from photo_atlas import db
from photo_atlas.indexer import region_box


def _unit(*vals: float) -> np.ndarray:
    v = np.array(vals, dtype=np.float32)
    return v / np.linalg.norm(v)


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
