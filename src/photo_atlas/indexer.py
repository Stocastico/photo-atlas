"""Ingest a directory tree of photos into the catalog.

For every supported image the indexer:

1. reads metadata (dimensions, capture time, camera, GPS),
2. reverse-geocodes GPS into a city / country label,
3. generates a thumbnail,
4. detects faces, stores crops + embeddings, and auto-recognises people that
   have already been named,
5. derives a coarse scene tag,
6. upserts everything into SQLite.

A second pass, :func:`cluster_library`, groups the still-unnamed faces into
clusters so the user can name a whole group at once.
"""

from __future__ import annotations

import sqlite3
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np
from PIL import Image

from . import db
from .classify import SceneTagger
from .config import AtlasConfig
from .faces import FaceBackend, best_person_match, cluster_embeddings, get_backend
from .geocode import Geocoder
from .metadata import extract_meta, is_supported, make_thumbnail, sha1_of
import json


@dataclass
class IndexStats:
    scanned: int = 0
    indexed: int = 0
    skipped: int = 0
    failed: int = 0
    faces: int = 0
    recognized: int = 0


def _person_centroids(conn: sqlite3.Connection) -> dict[int, np.ndarray]:
    """Average embedding per named person, for auto-recognition."""

    rows = conn.execute(
        "SELECT person_id, embedding, dim FROM faces WHERE person_id IS NOT NULL AND embedding IS NOT NULL"
    ).fetchall()
    buckets: dict[int, list[np.ndarray]] = defaultdict(list)
    for row in rows:
        vec = db.blob_to_embedding(row["embedding"])
        if vec is not None:
            buckets[int(row["person_id"])].append(vec)
    return {pid: np.mean(np.vstack(vs), axis=0) for pid, vs in buckets.items() if vs}


def thumb_path_for(config: AtlasConfig, sha1: str) -> Path:
    """Content-addressed thumbnail path.

    Deriving the name from the file's SHA-1 (not ``hash()``, which is salted per
    process) keeps it stable across runs, so re-indexing reuses the same file
    instead of orphaning the previous thumbnail.
    """

    return config.thumbs_dir / sha1[:2] / f"{sha1}.jpg"


def _save_face_crop(src: Path, bbox: tuple[int, int, int, int], dest: Path) -> None:
    x, y, w, h = bbox
    dest.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(src) as img:
        # Detection runs on the non-transposed image, so crop without transpose.
        crop = img.convert("RGB").crop((x, y, x + w, y + h))
        crop.save(dest, "JPEG", quality=85)


def index_file(
    conn: sqlite3.Connection,
    config: AtlasConfig,
    path: Path,
    *,
    backend: FaceBackend | None,
    geocoder: Geocoder | None,
    tagger: SceneTagger,
    centroids: dict[int, np.ndarray] | None = None,
    stats: IndexStats | None = None,
) -> int:
    """Index a single image file and return its photo id."""

    path = Path(path)
    meta = extract_meta(path)
    sha1 = sha1_of(path)

    place = None
    if geocoder is not None:
        place = geocoder.lookup(meta.lat, meta.lon)

    # Detect faces first so the scene tagger can use the count.
    observations = backend.detect(path) if backend is not None else []

    thumb_path = thumb_path_for(config, sha1)
    make_thumbnail(path, thumb_path, size=config.thumb_size)

    scene_label, scene_scores = tagger.tag(path, face_count=len(observations))

    record = {
        "path": str(path.resolve()),
        "filename": path.name,
        "sha1": sha1,
        "width": meta.width,
        "height": meta.height,
        "bytes": path.stat().st_size,
        "taken_at": meta.taken_at,
        "taken_source": meta.taken_source,
        "camera_make": meta.camera_make,
        "camera_model": meta.camera_model,
        "lat": meta.lat,
        "lon": meta.lon,
        "place_city": place.city if place else None,
        "place_country": place.country if place else None,
        "place_label": place.label if place else None,
        "scene_type": scene_label,
        "scene_scores": json.dumps(scene_scores),
        "face_count": len(observations),
        "thumb_path": str(thumb_path),
        "indexed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    photo_id = db.upsert_photo(conn, record)

    face_rows = []
    for i, obs in enumerate(observations):
        crop_path = config.faces_dir / f"{photo_id}" / f"face_{i}.jpg"
        try:
            _save_face_crop(path, obs.bbox, crop_path)
        except Exception:
            crop_path = None

        person_id, confidence = (None, 0.0)
        if centroids:
            person_id, confidence = best_person_match(
                obs.embedding, centroids, config.face_match_threshold
            )
            if person_id is not None and stats is not None:
                stats.recognized += 1

        x, y, w, h = obs.bbox
        face_rows.append(
            {
                "person_id": person_id,
                "cluster_id": None,
                "bbox_x": x, "bbox_y": y, "bbox_w": w, "bbox_h": h,
                "dim": int(obs.embedding.shape[0]),
                "embedding": db.embedding_to_blob(obs.embedding),
                "crop_path": str(crop_path) if crop_path else None,
                "confidence": confidence,
            }
        )

    db.replace_faces(conn, photo_id, face_rows)
    if stats is not None:
        stats.faces += len(face_rows)
    return photo_id


def iter_images(root: Path):
    root = Path(root)
    if root.is_file():
        if is_supported(root):
            yield root
        return
    for path in sorted(root.rglob("*")):
        if path.is_file() and is_supported(path):
            yield path


def index_path(
    config: AtlasConfig,
    root: Path,
    *,
    backend_name: str = "auto",
    geocode: bool = True,
    recompute: bool = False,
    progress: Callable[[Path, IndexStats], None] | None = None,
) -> IndexStats:
    """Index every supported image under ``root`` into the library."""

    config.ensure_dirs()
    conn = db.connect(config.db_path)
    backend = (
        get_backend(backend_name, model_dir=config.models_dir)
        if backend_name != "none"
        else None
    )
    if backend_name not in ("none",) and backend is None:
        import sys

        print(
            f"warning: face backend '{backend_name}' unavailable "
            "(missing models or OpenCV DNN support); indexing without faces.",
            file=sys.stderr,
        )
    geocoder = Geocoder() if geocode else None
    tagger = SceneTagger()
    stats = IndexStats()

    try:
        centroids = _person_centroids(conn)
        existing = {
            r["path"] for r in conn.execute("SELECT path FROM photos").fetchall()
        }
        for path in iter_images(root):
            stats.scanned += 1
            if not recompute and str(path.resolve()) in existing:
                stats.skipped += 1
                continue
            try:
                index_file(
                    conn, config, path,
                    backend=backend, geocoder=geocoder, tagger=tagger,
                    centroids=centroids, stats=stats,
                )
                stats.indexed += 1
                conn.commit()
            except Exception:
                stats.failed += 1
            if progress is not None:
                progress(path, stats)
        conn.commit()
    finally:
        conn.close()
    return stats


def cluster_library(config: AtlasConfig) -> dict[str, int]:
    """Cluster all unnamed faces so groups can be labelled in one go."""

    conn = db.connect(config.db_path)
    try:
        rows = conn.execute(
            "SELECT id, embedding FROM faces WHERE person_id IS NULL AND embedding IS NOT NULL"
        ).fetchall()
        ids = [int(r["id"]) for r in rows]
        embeddings = [db.blob_to_embedding(r["embedding"]) for r in rows]
        labels = cluster_embeddings(
            embeddings, eps=config.cluster_eps, min_samples=config.cluster_min_samples
        )

        # Reset previous clustering for unnamed faces, then assign new labels.
        conn.execute("UPDATE faces SET cluster_id=NULL WHERE person_id IS NULL")
        n_clusters = 0
        seen: set[int] = set()
        for face_id, label in zip(ids, labels):
            if label < 0:
                continue
            conn.execute("UPDATE faces SET cluster_id=? WHERE id=?", (label, face_id))
            if label not in seen:
                seen.add(label)
                n_clusters += 1
        conn.commit()
        return {"faces": len(ids), "clusters": n_clusters}
    finally:
        conn.close()
