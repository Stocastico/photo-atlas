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
from collections.abc import Iterable
from datetime import UTC
from pathlib import Path

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
    taken_source TEXT,          -- 'exif' | 'folder' | 'mtime'
    camera_make  TEXT,
    camera_model TEXT,
    lat          REAL,
    lon          REAL,
    place_city   TEXT,
    place_country TEXT,
    place_label  TEXT,
    folder_place TEXT,          -- trip/region label mined from the folder name
    scene_type   TEXT,
    scene_scores TEXT,          -- JSON map label -> score
    face_count   INTEGER DEFAULT 0,
    named_face_count INTEGER NOT NULL DEFAULT 0,  -- faces assigned to a named person (trigger-kept)
    favorite     INTEGER NOT NULL DEFAULT 0,      -- user star (0/1); preserved across re-index
    is_video     INTEGER NOT NULL DEFAULT 0,      -- 1 for video rows (poster-frame thumbnail)
    hidden       INTEGER NOT NULL DEFAULT 0,      -- user-hidden (0/1); excluded from browsing
    thumb_path   TEXT,
    embedding    BLOB,          -- SigLIP image embedding (float32) for semantic search
    embed_dim    INTEGER,       -- length of ``embedding`` (NULL when not embedded)
    phash        TEXT,          -- perceptual hash (dHash, hex) for near-duplicate grouping
    indexed_at   TEXT
);

CREATE TABLE IF NOT EXISTS persons (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    name          TEXT UNIQUE NOT NULL,
    cover_face_id INTEGER,
    created_at    TEXT
);

-- Smart Albums: a saved search is a user-named filter set, stored as the
-- querystring of filters to re-apply. Independent of the photo tables, so it's
-- created via CREATE IF NOT EXISTS (no _migrate entry needed) and survives a
-- re-index untouched.
CREATE TABLE IF NOT EXISTS saved_searches (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    name       TEXT UNIQUE NOT NULL,
    query      TEXT NOT NULL,
    created_at TEXT
);

-- Small key/value store for catalog-wide counters. Used for an
-- ``embeddings_version`` that bumps whenever an image embedding is written, so a
-- running server can detect an *in-place* re-embed (which leaves the row count and
-- max id unchanged) and reload its cached semantic index. CREATE IF NOT EXISTS, so
-- no _migrate entry needed.
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT
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

-- Face active-learning: a "not this person" negative recorded when the user
-- corrects an auto-tag (un/reassigns a face away from a person). Fed into the
-- k-NN vote to penalise that identity for similar future faces. Both FKs cascade
-- (the negative is meaningless once the face or person is gone); UNIQUE keeps a
-- repeated correction idempotent. CREATE IF NOT EXISTS, so no _migrate entry.
CREATE TABLE IF NOT EXISTS face_negatives (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    face_id    INTEGER NOT NULL REFERENCES faces(id) ON DELETE CASCADE,
    person_id  INTEGER NOT NULL REFERENCES persons(id) ON DELETE CASCADE,
    created_at TEXT,
    UNIQUE(face_id, person_id)
);

CREATE INDEX IF NOT EXISTS idx_photos_taken   ON photos(taken_at);
CREATE INDEX IF NOT EXISTS idx_photos_indexed ON photos(indexed_at);
CREATE INDEX IF NOT EXISTS idx_photos_country ON photos(place_country);
CREATE INDEX IF NOT EXISTS idx_photos_city    ON photos(place_city);
CREATE INDEX IF NOT EXISTS idx_photos_camera  ON photos(camera_model);
CREATE INDEX IF NOT EXISTS idx_faces_photo    ON faces(photo_id);
CREATE INDEX IF NOT EXISTS idx_faces_cluster  ON faces(cluster_id);
-- Composite indexes for the real browse/filter access patterns: filter on a
-- facet column and sort by capture time (the default ``taken_at DESC`` order),
-- so SQLite can satisfy the WHERE + ORDER BY from one index without a sort.
-- Their leading column supersedes the old single-column scene/folder indexes
-- (dropped in ``_migrate``). The person filter is an EXISTS into ``faces``
-- correlated on ``photo_id``; ``(person_id, photo_id)`` seeks both at once
-- (taken_at lives on ``photos``, so it can't join this cross-table index).
CREATE INDEX IF NOT EXISTS idx_photos_scene_taken  ON photos(scene_type, taken_at);
CREATE INDEX IF NOT EXISTS idx_photos_folder_taken ON photos(folder_place, taken_at);
CREATE INDEX IF NOT EXISTS idx_faces_person_photo  ON faces(person_id, photo_id);
CREATE INDEX IF NOT EXISTS idx_photos_favorite     ON photos(favorite);

-- ``photos.named_face_count`` denormalises "how many of this photo's faces are
-- assigned to a named person" so the Known-people facet is a plain column read
-- instead of a per-row correlated subquery. Triggers keep it exact across every
-- write path (auto-recognition at index time, assign/unassign, merge, delete,
-- re-index via replace_faces, prune's FK cascade) — no Python call site can
-- forget to update it. They touch ``photos`` only, so no recursive firing.
CREATE TRIGGER IF NOT EXISTS trg_faces_named_insert
AFTER INSERT ON faces WHEN NEW.person_id IS NOT NULL
BEGIN
    UPDATE photos SET named_face_count = named_face_count + 1 WHERE id = NEW.photo_id;
END;
CREATE TRIGGER IF NOT EXISTS trg_faces_named_delete
AFTER DELETE ON faces WHEN OLD.person_id IS NOT NULL
BEGIN
    UPDATE photos SET named_face_count = named_face_count - 1 WHERE id = OLD.photo_id;
END;
CREATE TRIGGER IF NOT EXISTS trg_faces_named_update
AFTER UPDATE OF person_id ON faces
WHEN (OLD.person_id IS NULL) <> (NEW.person_id IS NULL)
BEGIN
    UPDATE photos
       SET named_face_count =
           named_face_count + (CASE WHEN NEW.person_id IS NULL THEN -1 ELSE 1 END)
     WHERE id = NEW.photo_id;
END;
"""


def _migrate(conn: sqlite3.Connection) -> None:
    """Apply additive migrations to catalogs created by older versions.

    Runs before the schema script so that columns referenced by ``CREATE INDEX``
    statements (e.g. ``folder_place``) exist on pre-existing tables.
    """

    tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    if "photos" in tables:
        cols = {r["name"] for r in conn.execute("PRAGMA table_info(photos)")}
        if "folder_place" not in cols:
            conn.execute("ALTER TABLE photos ADD COLUMN folder_place TEXT")
        # Semantic-search embedding columns (added 2026-06): additive, so older
        # catalogs gain them empty and are filled by `index --embed` / `embed`.
        if "embedding" not in cols:
            conn.execute("ALTER TABLE photos ADD COLUMN embedding BLOB")
        if "embed_dim" not in cols:
            conn.execute("ALTER TABLE photos ADD COLUMN embed_dim INTEGER")
        # User favorites (added 2026-06): a 0/1 star, additive so older catalogs
        # gain it un-starred. Kept out of ``PHOTO_COLUMNS`` so a re-index never
        # clobbers it.
        if "favorite" not in cols:
            conn.execute("ALTER TABLE photos ADD COLUMN favorite INTEGER NOT NULL DEFAULT 0")
        # Video rows (added 2026-06): a 0/1 flag marking poster-frame video entries,
        # additive so older catalogs gain it as 0 (all-photos) until a re-index.
        if "is_video" not in cols:
            conn.execute("ALTER TABLE photos ADD COLUMN is_video INTEGER NOT NULL DEFAULT 0")
        # User-hidden rows (added 2026-06): excluded from browsing by default;
        # additive so older catalogs gain it un-hidden. Kept out of PHOTO_COLUMNS
        # (like favorite) so a re-index never un-hides a photo.
        if "hidden" not in cols:
            conn.execute("ALTER TABLE photos ADD COLUMN hidden INTEGER NOT NULL DEFAULT 0")
        # Perceptual hash (added 2026-06): a dHash hex string for near-duplicate /
        # burst grouping. Additive, so older catalogs gain it empty and fill it in
        # via a re-index or the `photo-atlas dedup` backfill.
        if "phash" not in cols:
            conn.execute("ALTER TABLE photos ADD COLUMN phash TEXT")
        # Denormalised named-face count (added 2026-06): add the column, then
        # backfill it once from the faces table before the maintenance triggers
        # (created by the schema script below) take over for future writes.
        if "named_face_count" not in cols:
            conn.execute(
                "ALTER TABLE photos ADD COLUMN named_face_count INTEGER NOT NULL DEFAULT 0"
            )
            if "faces" in tables:
                conn.execute(
                    "UPDATE photos SET named_face_count = (SELECT COUNT(*) FROM faces f "
                    "WHERE f.photo_id = photos.id AND f.person_id IS NOT NULL)"
                )
    # Drop single-column indexes that the new composite indexes (added to the
    # schema below) fully supersede on their leading column, so existing catalogs
    # don't carry redundant indexes that only cost extra on every write.
    for name in ("idx_photos_scene", "idx_photos_folder", "idx_faces_person"):
        conn.execute(f"DROP INDEX IF EXISTS {name}")


def connect(db_path: Path, *, ensure_schema: bool = True) -> sqlite3.Connection:
    """Open the catalog.

    The per-connection PRAGMAs are always applied. ``ensure_schema`` controls the
    one-time-ish cost of running the migration + ``CREATE TABLE/INDEX`` script:
    it is idempotent but still parses and checks ~12 objects against
    ``sqlite_master`` on every call. The web server creates the schema once at
    startup and opens its request connections with ``ensure_schema=False`` so a
    single filter toggle (which fans out to several endpoints) doesn't re-run the
    DDL each time.
    """

    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    # WAL lets a reader (the web UI) and a writer (a concurrent `index` run)
    # coexist without "database is locked"; busy_timeout retries briefly instead
    # of failing immediately when they do contend.
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA busy_timeout = 5000")
    if ensure_schema:
        _migrate(conn)
        conn.executescript(SCHEMA)
    return conn


# -- meta key/value counters ----------------------------------------------
def get_meta(conn: sqlite3.Connection, key: str) -> str | None:
    """Return the stored ``meta`` value for ``key`` (or ``None`` if unset)."""

    row = conn.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
    return None if row is None else str(row["value"])


def bump_meta(conn: sqlite3.Connection, key: str) -> int:
    """Increment the integer counter at ``key`` (starting from 0) and return it."""

    current = get_meta(conn, key)
    nxt = (int(current) if current is not None else 0) + 1
    conn.execute(
        "INSERT INTO meta (key, value) VALUES (?, ?) "
        "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
        (key, str(nxt)),
    )
    return nxt


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
# The writable photo columns (everything but the autoincrement ``id``), in a
# single place so the writer (upsert) and readers (explicit SELECT lists) can't
# drift. ``id`` is added by readers that need it.
PHOTO_COLUMNS = [
    "path", "filename", "sha1", "width", "height", "bytes", "taken_at",
    "taken_source", "camera_make", "camera_model", "lat", "lon",
    "place_city", "place_country", "place_label", "folder_place",
    "scene_type", "scene_scores", "face_count", "thumb_path", "indexed_at",
]


def upsert_photo(conn: sqlite3.Connection, record: dict) -> int:
    """Insert (or replace) a photo row keyed by ``path`` and return its id."""

    columns = PHOTO_COLUMNS
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


def set_photo_embedding(
    conn: sqlite3.Connection, photo_id: int, vector: np.ndarray | None
) -> None:
    """Persist (or clear) a photo's SigLIP image embedding for semantic search.

    Kept out of :data:`PHOTO_COLUMNS` (and thus out of the grid/list SELECT) so the
    multi-kilobyte BLOB is never shipped to the browser for every card; it is
    written separately, only when an embedding was computed.
    """

    blob = embedding_to_blob(vector)
    dim = None if vector is None else int(np.asarray(vector).reshape(-1).shape[0])
    conn.execute(
        "UPDATE photos SET embedding=?, embed_dim=? WHERE id=?", (blob, dim, photo_id)
    )
    # Bump the version so a cached semantic index reloads even on an in-place
    # re-embed (same row count + max id), which the (count, max_id) signature misses.
    bump_meta(conn, "embeddings_version")


def set_phash(conn: sqlite3.Connection, photo_id: int, phash: str | None) -> None:
    """Persist (or clear) a photo's perceptual hash for near-duplicate grouping.

    Kept out of :data:`PHOTO_COLUMNS` (like the embedding) so the grid/list SELECT
    never ships it; it's written here, only when a hash was computed. A re-index
    still refreshes it because the commit path always calls this with the freshly
    derived value.
    """

    conn.execute("UPDATE photos SET phash=? WHERE id=?", (phash, photo_id))


def delete_photos(conn: sqlite3.Connection, ids: list[int]) -> int:
    """Delete photo rows (faces cascade); return the number removed.

    The catalog half of a hard delete — removing the source files + derivatives
    on disk is the caller's job (see ``indexer.delete_photos``).
    """

    clean = [int(i) for i in ids]
    if not clean:
        return 0
    placeholders = ", ".join(["?"] * len(clean))
    cur = conn.execute(f"DELETE FROM photos WHERE id IN ({placeholders})", clean)
    conn.commit()
    return cur.rowcount


def set_favorite(conn: sqlite3.Connection, photo_id: int, favorite: bool) -> bool:
    """Star/un-star a photo. Returns ``True`` if the photo exists (was updated).

    ``favorite`` lives outside :data:`PHOTO_COLUMNS` so a re-index never resets a
    user's star; it's written only through here (and the API).
    """

    cur = conn.execute(
        "UPDATE photos SET favorite=? WHERE id=?", (1 if favorite else 0, photo_id)
    )
    conn.commit()
    return cur.rowcount > 0


def _set_flag_bulk(
    conn: sqlite3.Connection, column: str, ids: list[int], value: bool
) -> int:
    """Set a 0/1 photo flag (``favorite``/``hidden``) for many photos at once.

    Both flags live outside :data:`PHOTO_COLUMNS` so a re-index never clears them;
    they're written only through these helpers (and the per-photo ``set_favorite``).
    """

    if column not in ("favorite", "hidden"):  # guard the f-string interpolation
        raise ValueError(f"not a bulk-settable flag: {column}")
    clean = [int(i) for i in ids]
    if not clean:
        return 0
    placeholders = ", ".join(["?"] * len(clean))
    cur = conn.execute(
        f"UPDATE photos SET {column}=? WHERE id IN ({placeholders})",
        [1 if value else 0, *clean],
    )
    conn.commit()
    return cur.rowcount


def set_favorite_bulk(conn: sqlite3.Connection, ids: list[int], favorite: bool) -> int:
    return _set_flag_bulk(conn, "favorite", ids, favorite)


def set_hidden_bulk(conn: sqlite3.Connection, ids: list[int], hidden: bool) -> int:
    return _set_flag_bulk(conn, "hidden", ids, hidden)


# -- face active-learning negatives ----------------------------------------
def add_face_negative(conn: sqlite3.Connection, face_id: int, person_id: int) -> None:
    """Record that ``face_id`` is *not* ``person_id`` (idempotent)."""

    from datetime import datetime

    conn.execute(
        "INSERT OR IGNORE INTO face_negatives (face_id, person_id, created_at) "
        "VALUES (?, ?, ?)",
        (face_id, person_id, datetime.now(UTC).isoformat(timespec="seconds")),
    )


def remove_face_negative(conn: sqlite3.Connection, face_id: int, person_id: int) -> None:
    """Drop a negative — e.g. the user just confirmed this face *is* that person."""

    conn.execute(
        "DELETE FROM face_negatives WHERE face_id=? AND person_id=?", (face_id, person_id)
    )


def load_negatives(conn: sqlite3.Connection) -> list[tuple[int, np.ndarray]]:
    """Every ``(person_id, embedding)`` negative, for negative-aware recognition."""

    rows = conn.execute(
        "SELECT n.person_id, f.embedding FROM face_negatives n "
        "JOIN faces f ON f.id = n.face_id WHERE f.embedding IS NOT NULL"
    ).fetchall()
    out: list[tuple[int, np.ndarray]] = []
    for r in rows:
        vec = blob_to_embedding(r["embedding"])
        if vec is not None:
            out.append((int(r["person_id"]), vec))
    return out


# -- saved searches (Smart Albums) ----------------------------------------
def create_saved_search(conn: sqlite3.Connection, name: str, query: str) -> int:
    """Create (or overwrite by name) a saved search and return its id.

    Upserts on ``name`` so re-saving an album under the same name updates its
    stored query in place — no duplicate row, no ``IntegrityError``.
    """

    from datetime import datetime

    name = name.strip()
    now = datetime.now(UTC).isoformat(timespec="seconds")
    conn.execute(
        "INSERT INTO saved_searches (name, query, created_at) VALUES (?, ?, ?) "
        "ON CONFLICT(name) DO UPDATE SET query=excluded.query",
        (name, query, now),
    )
    conn.commit()
    row = conn.execute("SELECT id FROM saved_searches WHERE name=?", (name,)).fetchone()
    return int(row["id"])


def list_saved_searches(conn: sqlite3.Connection) -> list[dict]:
    rows = conn.execute(
        "SELECT id, name, query, created_at FROM saved_searches "
        "ORDER BY name COLLATE NOCASE"
    ).fetchall()
    return [dict(r) for r in rows]


def delete_saved_search(conn: sqlite3.Connection, search_id: int) -> bool:
    """Delete a saved search. Returns ``True`` if a row was removed."""

    cur = conn.execute("DELETE FROM saved_searches WHERE id=?", (search_id,))
    conn.commit()
    return cur.rowcount > 0


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
    from datetime import datetime

    cur = conn.execute(
        "INSERT INTO persons (name, created_at) VALUES (?, ?)",
        (name, datetime.now(UTC).isoformat(timespec="seconds")),
    )
    assert cur.lastrowid is not None  # row was just inserted
    return int(cur.lastrowid)
