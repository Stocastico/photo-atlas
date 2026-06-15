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

import json
import os
import sqlite3
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import cast

import numpy as np
from PIL import Image

from . import db
from .classify import SceneTagger
from .config import AtlasConfig
from .faces import (
    Enrollment,
    FaceBackend,
    cluster_embeddings,
    get_backend,
    knn_person_match,
    pil_to_bgr,
)
from .folder_meta import extract_folder_meta
from .geocode import Geocoder
from .metadata import (
    extract_meta_from_image,
    is_supported,
    is_video,
    make_thumbnail_from_image,
    sha1_of,
)


@dataclass
class IndexStats:
    scanned: int = 0
    indexed: int = 0
    skipped: int = 0
    failed: int = 0
    faces: int = 0
    recognized: int = 0
    #: Video files seen during the walk. Not indexed (no still-image pipeline),
    #: but counted so they're reported instead of silently dropped.
    videos: int = 0
    #: Files skipped because a byte-identical copy (same SHA-1) was already
    #: indexed under a different path.
    duplicates: int = 0
    #: First few "<path>: <error>" strings for files that failed to index, so a
    #: bad file is diagnosable instead of vanishing into the ``failed`` count.
    #: Capped (see ``_MAX_ERRORS``) to stay bounded on huge libraries.
    errors: list[str] = field(default_factory=list)


_MAX_ERRORS = 50


def _load_enrollment(conn: sqlite3.Connection) -> Enrollment:
    """Collect every named face for k-NN auto-recognition.

    Unlike the old per-person centroid, this keeps each enrolled face as its own
    vector, so recognition matches the nearest individual examples rather than an
    average that blurs a person's look across years.
    """

    rows = conn.execute(
        "SELECT person_id, embedding FROM faces "
        "WHERE person_id IS NOT NULL AND embedding IS NOT NULL"
    ).fetchall()
    pairs: list[tuple[int, np.ndarray]] = []
    for row in rows:
        vec = db.blob_to_embedding(row["embedding"])
        if vec is not None:
            pairs.append((int(row["person_id"]), vec))
    return Enrollment.from_pairs(pairs)


def thumb_path_for(config: AtlasConfig, sha1: str) -> Path:
    """Content-addressed thumbnail path.

    Deriving the name from the file's SHA-1 (not ``hash()``, which is salted per
    process) keeps it stable across runs, so re-indexing reuses the same file
    instead of orphaning the previous thumbnail.
    """

    return config.thumbs_dir / sha1[:2] / f"{sha1}.jpg"


@dataclass
class _PreparedFace:
    """A detected face reduced to picklable, DB-ready primitives."""

    bbox: tuple[int, int, int, int]
    dim: int
    embedding_blob: bytes | None
    #: The cropped face encoded as JPEG bytes. Carried in-memory (not written to
    #: disk yet) because its final path depends on the photo id, which only the
    #: main process knows after the DB insert.
    crop_jpeg: bytes | None
    person_id: int | None
    confidence: float


@dataclass
class _PreparedPhoto:
    """Everything derived from one decoded image, ready to persist.

    Holds only picklable primitives/bytes — no open :class:`PIL.Image.Image`, no
    numpy arrays, no SQLite handle — so it can cross a process boundary when
    indexing in parallel. The thumbnail is already written to its content-addressed
    path (safe across processes: the name is the file's SHA-1); face crops travel
    as encoded bytes and are written by the main process once the photo id exists.
    """

    path: str
    filename: str
    sha1: str
    width: int | None
    height: int | None
    bytes: int
    taken_at: str | None
    taken_source: str
    camera_make: str | None
    camera_model: str | None
    lat: float | None
    lon: float | None
    folder_place: str | None
    scene_type: str
    scene_scores: dict
    thumb_path: str
    faces: list[_PreparedFace]


def _encode_face_crop(img: Image.Image, bbox: tuple[int, int, int, int]) -> bytes | None:
    """Crop ``bbox`` from the open image and return JPEG bytes (or ``None``)."""

    x, y, w, h = bbox
    try:
        # Detection runs on the non-transposed image, so crop without transpose.
        crop = img.convert("RGB").crop((x, y, x + w, y + h))
        buf = BytesIO()
        crop.save(buf, "JPEG", quality=85)
        return buf.getvalue()
    except Exception:
        return None


def _prepare_photo(
    config: AtlasConfig,
    path: Path,
    *,
    backend: FaceBackend | None,
    tagger: SceneTagger,
    enrollment: Enrollment | None,
    sha1: str,
) -> _PreparedPhoto:
    """Decode one image exactly once and derive everything but the DB write.

    Geocoding and persistence are left to the caller so this can run in a worker
    process (it touches neither the SQLite handle nor a shared geocoder table) and
    is reused unchanged by the serial path. Decoding the file a single time and
    reusing the one Pillow image across metadata, faces, thumbnail and scene tag
    (was 4+ decodes per file) is the core per-file speed-up.
    """

    path = Path(path)
    with Image.open(path) as img:
        img.load()
        meta = extract_meta_from_image(img, path)

        # Folder names (e.g. 2012/2012_05_Sardegna) often carry a year/month/place
        # the file's EXIF lacks. Use them only to fill gaps: a folder date replaces
        # the filesystem-mtime fallback but never a real EXIF capture time.
        folder = extract_folder_meta(path)
        if meta.taken_source != "exif" and folder.year is not None:
            synthesized = datetime(folder.year, folder.month or 1, 1)
            meta.taken_at = synthesized.isoformat(timespec="seconds")
            meta.taken_source = "folder"

        # Detect faces first so the scene tagger can use the count. The backend
        # gets the already-decoded BGR array, so it never re-reads the file.
        bgr = pil_to_bgr(img)
        observations = backend.detect(path, image=bgr) if backend is not None else []

        thumb_path = thumb_path_for(config, sha1)
        make_thumbnail_from_image(img, thumb_path, size=config.thumb_size)

        scene_label, scene_scores = tagger.tag_image(img, face_count=len(observations))

        faces: list[_PreparedFace] = []
        for obs in observations:
            person_id: int | None = None
            confidence = 0.0
            if enrollment is not None and not enrollment.is_empty:
                person_id, confidence = knn_person_match(
                    obs.embedding, enrollment,
                    k=config.recognition_k, threshold=config.face_match_threshold,
                )
            x, y, w, h = obs.bbox
            faces.append(
                _PreparedFace(
                    bbox=(int(x), int(y), int(w), int(h)),
                    dim=int(obs.embedding.shape[0]),
                    embedding_blob=db.embedding_to_blob(obs.embedding),
                    crop_jpeg=_encode_face_crop(img, obs.bbox),
                    person_id=person_id,
                    confidence=confidence,
                )
            )

    return _PreparedPhoto(
        path=str(path.resolve()),
        filename=path.name,
        sha1=sha1,
        width=meta.width,
        height=meta.height,
        bytes=path.stat().st_size,
        taken_at=meta.taken_at,
        taken_source=meta.taken_source,
        camera_make=meta.camera_make,
        camera_model=meta.camera_model,
        lat=meta.lat,
        lon=meta.lon,
        folder_place=folder.place,
        scene_type=scene_label,
        scene_scores=scene_scores,
        thumb_path=str(thumb_path),
        faces=faces,
    )


def _commit_prepared(
    conn: sqlite3.Connection,
    config: AtlasConfig,
    prepared: _PreparedPhoto,
    place,
    stats: IndexStats | None,
) -> int:
    """Persist one prepared photo and its faces; return the photo id.

    This is the only DB-touching half of indexing, so in parallel mode every
    SQLite write still funnels through the single main-process connection.
    """

    record = {
        "path": prepared.path,
        "filename": prepared.filename,
        "sha1": prepared.sha1,
        "width": prepared.width,
        "height": prepared.height,
        "bytes": prepared.bytes,
        "taken_at": prepared.taken_at,
        "taken_source": prepared.taken_source,
        "camera_make": prepared.camera_make,
        "camera_model": prepared.camera_model,
        "lat": prepared.lat,
        "lon": prepared.lon,
        "place_city": place.city if place else None,
        "place_country": place.country if place else None,
        "place_label": place.label if place else None,
        "folder_place": prepared.folder_place,
        "scene_type": prepared.scene_type,
        "scene_scores": json.dumps(prepared.scene_scores),
        "face_count": len(prepared.faces),
        "thumb_path": prepared.thumb_path,
        "indexed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    photo_id = db.upsert_photo(conn, record)

    face_rows = []
    for i, face in enumerate(prepared.faces):
        crop_path = config.faces_dir / f"{photo_id}" / f"face_{i}.jpg"
        crop_saved = False
        if face.crop_jpeg is not None:
            try:
                crop_path.parent.mkdir(parents=True, exist_ok=True)
                crop_path.write_bytes(face.crop_jpeg)
                crop_saved = True
            except Exception:
                crop_saved = False
        if face.person_id is not None and stats is not None:
            stats.recognized += 1
        x, y, w, h = face.bbox
        face_rows.append(
            {
                "person_id": face.person_id,
                "cluster_id": None,
                "bbox_x": x, "bbox_y": y, "bbox_w": w, "bbox_h": h,
                "dim": face.dim,
                "embedding": face.embedding_blob,
                "crop_path": str(crop_path) if crop_saved else None,
                "confidence": face.confidence,
            }
        )

    db.replace_faces(conn, photo_id, face_rows)
    if stats is not None:
        stats.faces += len(face_rows)
    return photo_id


def index_file(
    conn: sqlite3.Connection,
    config: AtlasConfig,
    path: Path,
    *,
    backend: FaceBackend | None,
    geocoder: Geocoder | None,
    tagger: SceneTagger,
    enrollment: Enrollment | None = None,
    stats: IndexStats | None = None,
    sha1: str | None = None,
) -> int:
    """Index a single image file and return its photo id."""

    path = Path(path)
    if sha1 is None:
        sha1 = sha1_of(path)
    prepared = _prepare_photo(
        config, path, backend=backend, tagger=tagger, enrollment=enrollment, sha1=sha1
    )
    place = geocoder.lookup(prepared.lat, prepared.lon) if geocoder is not None else None
    return _commit_prepared(conn, config, prepared, place, stats)


def iter_files(root: Path):
    """Yield every file under ``root`` in a deterministic order.

    Uses :func:`os.walk` and sorts each directory level in place rather than
    materialising and sorting the *entire* tree up front, so memory stays flat
    on very large libraries (years of folders, 100k+ files).
    """

    root = Path(root)
    if root.is_file():
        yield root
        return
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames.sort()  # deterministic descent into subfolders
        for name in sorted(filenames):
            yield Path(dirpath) / name


def iter_images(root: Path):
    """Yield supported image files under ``root`` in a deterministic order."""

    for path in iter_files(root):
        if is_supported(path):
            yield path


# -- parallel worker plumbing ---------------------------------------------
#: Per-process state for the worker pool. Built once by :func:`_worker_init`
#: (the heavy ONNX backend, the scene tagger, the read-only enrollment) and
#: reused across every file that worker handles, so the models load once per
#: process rather than once per image.
_WORKER_STATE: dict = {}


def _worker_init(
    backend_name: str,
    model_dir: Path,
    config: AtlasConfig,
    enrollment: Enrollment | None,
) -> None:
    """Initialise a pool worker with its own backend / tagger (called once)."""

    _WORKER_STATE["backend"] = (
        get_backend(backend_name, model_dir=model_dir) if backend_name != "none" else None
    )
    _WORKER_STATE["tagger"] = SceneTagger()
    _WORKER_STATE["config"] = config
    _WORKER_STATE["enrollment"] = enrollment


def _worker_prepare(task: tuple[str, str]) -> tuple[bool, object]:
    """Prepare one file in a worker; return ``(ok, prepared_or_error_message)``."""

    path_str, sha1 = task
    try:
        prepared = _prepare_photo(
            _WORKER_STATE["config"], Path(path_str),
            backend=_WORKER_STATE["backend"], tagger=_WORKER_STATE["tagger"],
            enrollment=_WORKER_STATE["enrollment"], sha1=sha1,
        )
        return True, prepared
    except Exception as exc:  # pragma: no cover - hit via the broken-file test
        return False, f"{type(exc).__name__}: {exc}"


def _index_parallel(
    conn: sqlite3.Connection,
    config: AtlasConfig,
    tasks: Iterator[tuple[str, str]],
    *,
    backend_name: str,
    enrollment: Enrollment | None,
    geocoder: Geocoder | None,
    stats: IndexStats,
    workers: int,
    progress: Callable[[Path, IndexStats], None] | None,
) -> None:
    """Fan the per-file decode/detect/thumbnail work out over a process pool.

    Workers do the CPU-bound preparation; the main process keeps the single
    SQLite connection and performs every write. Only ``workers * 4`` files are
    ever in flight, so memory stays bounded regardless of library size, and
    commits are batched rather than per-file.
    """

    import multiprocessing as mp
    from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait

    # Make sure the ONNX weights are present before fanning out, so N workers
    # don't race to download the same files into the shared model cache.
    if backend_name in ("auto", "yunet"):
        try:
            from .models import ensure_models

            ensure_models(config.models_dir, download=True)
        except Exception:  # pragma: no cover - worker surfaces a clearer error
            pass

    ctx = mp.get_context("spawn")  # clean workers; safe for OpenCV/ONNX native libs
    max_inflight = workers * 4
    commit_every = 64
    since_commit = 0

    with ProcessPoolExecutor(
        max_workers=workers, mp_context=ctx,
        initializer=_worker_init,
        initargs=(backend_name, config.models_dir, config, enrollment),
    ) as pool:
        inflight: dict = {}

        def submit_one() -> bool:
            for task in tasks:
                inflight[pool.submit(_worker_prepare, task)] = task
                return True
            return False

        while len(inflight) < max_inflight and submit_one():
            pass

        while inflight:
            done, _ = wait(inflight, return_when=FIRST_COMPLETED)
            for fut in done:
                path_str, _sha1 = inflight.pop(fut)
                ok, payload = fut.result()
                if ok:
                    prepared = cast(_PreparedPhoto, payload)
                    place = (
                        geocoder.lookup(prepared.lat, prepared.lon)
                        if geocoder is not None else None
                    )
                    _commit_prepared(conn, config, prepared, place, stats)
                    stats.indexed += 1
                    since_commit += 1
                    if since_commit >= commit_every:
                        conn.commit()
                        since_commit = 0
                else:
                    stats.failed += 1
                    if len(stats.errors) < _MAX_ERRORS:
                        stats.errors.append(f"{path_str}: {payload}")
                if progress is not None:
                    progress(Path(path_str), stats)
                submit_one()
    conn.commit()


def index_path(
    config: AtlasConfig,
    root: Path,
    *,
    backend_name: str = "auto",
    geocode: bool = True,
    recompute: bool = False,
    workers: int | None = None,
    progress: Callable[[Path, IndexStats], None] | None = None,
) -> IndexStats:
    """Index every supported image under ``root`` into the library.

    ``workers`` controls fan-out: ``None``/``1`` keeps the in-process path; a
    larger value decodes and detects across that many worker processes (DB writes
    still funnel through the single main connection).
    """

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
    if geocoder is not None and not geocoder.high_resolution:
        import sys

        print(
            "warning: reverse geocoding is using the bundled ~120-city table, so "
            "city/country labels will be coarse and often wrong. Install the "
            "high-resolution backend with `uv sync --extra geo` (reverse_geocoder).",
            file=sys.stderr,
        )
    tagger = SceneTagger()
    stats = IndexStats()

    try:
        enrollment = _load_enrollment(conn)
        existing = {
            r["path"] for r in conn.execute("SELECT path FROM photos").fetchall()
        }
        seen_sha1 = {
            r["sha1"]
            for r in conn.execute("SELECT sha1 FROM photos WHERE sha1 IS NOT NULL")
        }
        # Never ingest our own generated derivatives (thumbs / face crops /
        # previews / models) if a library dir happens to sit inside the indexed
        # tree. Scoped to those dirs so e.g. the demo's photos under home still index.
        derived = tuple(
            d.resolve()
            for d in (config.thumbs_dir, config.faces_dir, config.previews_dir, config.models_dir)
        )

        def iter_tasks() -> Iterator[tuple[str, str]]:
            """Walk the tree, filtering + deduping, and yield ``(path, sha1)``.

            All bookkeeping the parallel path can't do safely from a worker —
            scan/skip/duplicate/video counting and SHA-1 dedup against the
            catalog — happens here in the main process before a file is handed off.
            """

            for path in iter_files(root):
                resolved = path.resolve()
                if any(resolved == d or d in resolved.parents for d in derived):
                    continue
                if is_video(path):
                    stats.videos += 1
                    continue
                if not is_supported(path):
                    continue
                stats.scanned += 1
                if not recompute and str(resolved) in existing:
                    stats.skipped += 1
                    continue
                try:
                    sha1 = sha1_of(path)
                except Exception as exc:
                    stats.failed += 1
                    if len(stats.errors) < _MAX_ERRORS:
                        stats.errors.append(f"{path}: {type(exc).__name__}: {exc}")
                    continue
                # A byte-identical copy already in the catalog (same photo in two
                # folders, a re-export, etc.) is skipped rather than duplicated.
                if not recompute and sha1 in seen_sha1:
                    stats.duplicates += 1
                    continue
                seen_sha1.add(sha1)
                yield str(path), sha1

        if workers is not None and workers > 1:
            _index_parallel(
                conn, config, iter_tasks(),
                backend_name=backend_name, enrollment=enrollment, geocoder=geocoder,
                stats=stats, workers=workers, progress=progress,
            )
        else:
            for path_str, sha1 in iter_tasks():
                path = Path(path_str)
                try:
                    index_file(
                        conn, config, path,
                        backend=backend, geocoder=geocoder, tagger=tagger,
                        enrollment=enrollment, stats=stats, sha1=sha1,
                    )
                    stats.indexed += 1
                    conn.commit()
                except Exception as exc:
                    stats.failed += 1
                    if len(stats.errors) < _MAX_ERRORS:
                        stats.errors.append(f"{path}: {type(exc).__name__}: {exc}")
                if progress is not None:
                    progress(path, stats)
            conn.commit()
    finally:
        conn.close()
    return stats


def prune_library(config: AtlasConfig) -> dict[str, int]:
    """Drop catalog rows whose source file no longer exists on disk.

    Indexing only ever adds or updates rows, so moved/deleted photos linger as
    dead entries that 404 in the UI. ``prune`` reconciles the catalog with the
    filesystem: for each missing file it removes the photo row (its faces cascade
    away) and deletes the now-orphaned thumbnail and face crops.
    """

    import shutil

    conn = db.connect(config.db_path)
    removed = kept = 0
    try:
        rows = conn.execute("SELECT id, path, thumb_path FROM photos").fetchall()
        for row in rows:
            if Path(row["path"]).exists():
                kept += 1
                continue
            if row["thumb_path"]:
                Path(row["thumb_path"]).unlink(missing_ok=True)
            crop_dir = config.faces_dir / str(row["id"])
            if crop_dir.exists():
                shutil.rmtree(crop_dir, ignore_errors=True)
            conn.execute("DELETE FROM photos WHERE id=?", (row["id"],))
            removed += 1
        conn.commit()
    finally:
        conn.close()
    return {"removed": removed, "kept": kept}


def cluster_library(config: AtlasConfig) -> dict[str, int]:
    """Cluster all unnamed faces so groups can be labelled in one go."""

    conn = db.connect(config.db_path)
    try:
        rows = conn.execute(
            "SELECT id, embedding FROM faces WHERE person_id IS NULL AND embedding IS NOT NULL"
        ).fetchall()
        ids = [int(r["id"]) for r in rows]
        # The WHERE clause guarantees non-NULL embeddings, so each decodes to an
        # array (never None); cast to keep the list type aligned with ``ids``.
        embeddings = [cast(np.ndarray, db.blob_to_embedding(r["embedding"])) for r in rows]
        labels = cluster_embeddings(
            embeddings, eps=config.cluster_eps, min_samples=config.cluster_min_samples
        )

        # Reset previous clustering for unnamed faces, then assign new labels.
        conn.execute("UPDATE faces SET cluster_id=NULL WHERE person_id IS NULL")
        n_clusters = 0
        seen: set[int] = set()
        for face_id, label in zip(ids, labels, strict=True):
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
