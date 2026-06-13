"""SQLite catalog for Photo Atlas.

The schema is intentionally small and denormalised for fast filtering over a
large library:

``photos``   one row per image with metadata, place and scene tags.
``persons``  named identities created by the user.
``faces``    detected faces, each optionally linked to a person / cluster.

Embeddings are stored as raw ``float32`` bytes in a BLOB column; helpers in
this module convert to and from :class:`numpy.ndarray`.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Iterable

import numpy as np

SCHEMA = """
CREATE TABLE IF NOT EXISTS photos (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    path         TEXT UNIQUE NOT NULL,
    filename     TEXT NOT NULL,
    sha1         TEXT,
    width        INTEGER,
    height       INTEGER,
    bytes        INTEGER,
    taken_at     TEXT,          -- ISO 8601, best available timestamp
    taken_source TEXT,          -- 'exif' | 'mtime'
    camera_make  TEXT,
    camera_model TEXT,
    lat          REAL,
    lon          REAL,
    place_city   TEXT,
    place_country TEXT,
    place_label  TEXT,
    scene_type   TEXT,
    scene_scores TEXT,          -- JSON map label -> score
    face_count   INTEGER DEFAULT 0,
    thumb_path   TEXT,
    indexed_at   TEXT
);

CREATE TABLE IF NOT EXISTS persons (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    name          TEXT UNIQUE NOT NULL,
    cover_face_id INTEGER,
    created_at    TEXT
);

CREATE TABLE IF NOT EXISTS faces (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    photo_id    INTEGER NOT NULL REFERENCES photos(id) ON DELETE CASCADE,
    person_id   INTEGER REFERENCES persons(id) ON DELETE SET NULL,
    cluster_id  INTEGER,
    bbox_x      INTEGER,
    bbox_y      INTEGER,
    bbox_w      INTEGER,
    bbox_h      INTEGER,
    dim         INTEGER,
    embedding   BLOB,
    crop_path   TEXT,
    confidence  REAL
);

CREATE INDEX IF NOT EXISTS idx_photos_taken   ON photos(taken_at);
CREATE INDEX IF NOT EXISTS idx_photos_scene   ON photos(scene_type);
CREATE INDEX IF NOT EXISTS idx_photos_country ON photos(place_country);
CREATE INDEX IF NOT EXISTS idx_photos_city    ON photos(place_city);
CREATE INDEX IF NOT EXISTS idx_faces_photo    ON faces(photo_id);
CREATE INDEX IF NOT EXISTS idx_faces_person   ON faces(person_id);
CREATE INDEX IF NOT EXISTS idx_faces_cluster  ON faces(cluster_id);
"""


def connect(db_path: Path) -> sqlite3.Connection:
    """Open the catalog, creating the schema on first use."""

    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(SCHEMA)
    return conn


# -- embedding (de)serialisation ------------------------------------------
def embedding_to_blob(vector: np.ndarray | None) -> bytes | None:
    if vector is None:
        return None
    return np.asarray(vector, dtype=np.float32).tobytes()


def blob_to_embedding(blob: bytes | None) -> np.ndarray | None:
    if blob is None:
        return None
    return np.frombuffer(blob, dtype=np.float32)


# -- small repository helpers ---------------------------------------------
def upsert_photo(conn: sqlite3.Connection, record: dict) -> int:
    """Insert (or replace) a photo row keyed by ``path`` and return its id."""

    columns = [
        "path", "filename", "sha1", "width", "height", "bytes", "taken_at",
        "taken_source", "camera_make", "camera_model", "lat", "lon",
        "place_city", "place_country", "place_label", "scene_type",
        "scene_scores", "face_count", "thumb_path", "indexed_at",
    ]
    values = [record.get(c) for c in columns]
    placeholders = ", ".join(["?"] * len(columns))
    collist = ", ".join(columns)
    updates = ", ".join(f"{c}=excluded.{c}" for c in columns if c != "path")
    conn.execute(
        f"INSERT INTO photos ({collist}) VALUES ({placeholders}) "
        f"ON CONFLICT(path) DO UPDATE SET {updates}",
        values,
    )
    # ``path`` is UNIQUE, so resolving by it returns the right id on both the
    # insert and the update branch (lastrowid is unreliable after an UPSERT).
    row = conn.execute("SELECT id FROM photos WHERE path=?", (record["path"],)).fetchone()
    return int(row["id"])


def replace_faces(conn: sqlite3.Connection, photo_id: int, faces: Iterable[dict]) -> None:
    """Replace all faces for a photo (used when re-indexing)."""

    conn.execute("DELETE FROM faces WHERE photo_id=?", (photo_id,))
    rows = list(faces)
    for f in rows:
        conn.execute(
            "INSERT INTO faces (photo_id, person_id, cluster_id, bbox_x, bbox_y, "
            "bbox_w, bbox_h, dim, embedding, crop_path, confidence) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (
                photo_id,
                f.get("person_id"),
                f.get("cluster_id"),
                f.get("bbox_x"), f.get("bbox_y"), f.get("bbox_w"), f.get("bbox_h"),
                f.get("dim"),
                f.get("embedding"),
                f.get("crop_path"),
                f.get("confidence"),
            ),
        )
    conn.execute("UPDATE photos SET face_count=? WHERE id=?", (len(rows), photo_id))


def get_or_create_person(conn: sqlite3.Connection, name: str) -> int:
    name = name.strip()
    row = conn.execute("SELECT id FROM persons WHERE name=?", (name,)).fetchone()
    if row:
        return int(row["id"])
    from datetime import datetime, timezone

    cur = conn.execute(
        "INSERT INTO persons (name, created_at) VALUES (?, ?)",
        (name, datetime.now(timezone.utc).isoformat(timespec="seconds")),
    )
    return int(cur.lastrowid)
