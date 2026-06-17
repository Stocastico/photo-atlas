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
import sys
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from datetime import UTC, datetime
from io import BytesIO
from pathlib import Path
from typing import cast

import numpy as np
from PIL import Image, ImageOps

from . import db, video
from .classify import Tagger, ZeroShotSceneTagger, get_tagger
from .config import AtlasConfig
from .embed import SigLipImageEncoder
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
    dhash,
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

#: Builtin ``bytes`` under an alias, so ``_PreparedPhoto``'s ``bytes`` field (the
#: file size) doesn't shadow the type in that dataclass's annotations.
_Bytes = bytes


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
    # "Not this person" negatives from active learning penalise the k-NN vote.
    return Enrollment.from_pairs(pairs, db.load_negatives(conn))


def thumb_path_for(config: AtlasConfig, sha1: str) -> Path:
    """Content-addressed thumbnail path.

    Deriving the name from the file's SHA-1 (not ``hash()``, which is salted per
    process) keeps it stable across runs, so re-indexing reuses the same file
    instead of orphaning the previous thumbnail.
    """

    return config.thumbs_dir / sha1[:2] / f"{sha1}.jpg"


def poster_path_for(config: AtlasConfig, sha1: str) -> Path:
    """Content-addressed full-resolution video poster-frame path."""

    return config.posters_dir / sha1[:2] / f"{sha1}.jpg"


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
    #: SigLIP image embedding bytes for semantic search (``None`` unless embeddings
    #: were requested). Carried as raw float32 bytes so it crosses a process boundary.
    #: ``_Bytes`` is the builtin ``bytes`` aliased so the ``bytes`` field above
    #: doesn't shadow it as a type within this class body.
    embedding_blob: _Bytes | None = None
    embed_dim: int | None = None
    #: Perceptual hash (dHash, hex) for near-duplicate / burst grouping. Always
    #: computed (it's cheap), so even non-embedded libraries get duplicate detection.
    phash: str | None = None


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


def regenerate_face_crop(
    conn: sqlite3.Connection, config: AtlasConfig, face_id: int
) -> str | None:
    """Re-create a face's crop JPEG from its source photo and persist the path.

    This is the retry path for a crop that failed to write at index time (the
    face row keeps its embedding + bbox but ``crop_path`` is ``NULL``, so
    ``/api/face/{id}`` 404s forever otherwise) or whose file was later deleted.
    Returns the new crop path on success, or ``None`` if the face/source is gone
    or the crop still can't be written.
    """

    row = conn.execute(
        "SELECT f.photo_id, f.bbox_x, f.bbox_y, f.bbox_w, f.bbox_h, p.path "
        "FROM faces f JOIN photos p ON p.id = f.photo_id WHERE f.id=?",
        (face_id,),
    ).fetchone()
    if row is None:
        return None
    src = row["path"]
    bbox = (row["bbox_x"], row["bbox_y"], row["bbox_w"], row["bbox_h"])
    if not src or not Path(src).exists() or any(v is None for v in bbox):
        return None
    try:
        with Image.open(src) as raw:
            raw.load()
            # Match the index-time pixels: detection ran on the EXIF-transposed
            # (upright) image, so the stored bbox indexes the upright frame.
            img = ImageOps.exif_transpose(raw) or raw
            crop_jpeg = _encode_face_crop(img, cast(tuple[int, int, int, int], bbox))
    except Exception:
        return None
    if crop_jpeg is None:
        return None
    crop_path = config.faces_dir / str(row["photo_id"]) / f"face_{face_id}.jpg"
    try:
        crop_path.parent.mkdir(parents=True, exist_ok=True)
        crop_path.write_bytes(crop_jpeg)
    except Exception:
        return None
    conn.execute("UPDATE faces SET crop_path=? WHERE id=?", (str(crop_path), face_id))
    conn.commit()
    return str(crop_path)


def _build_encoders(
    config: AtlasConfig, *, embed: bool
) -> tuple[SigLipImageEncoder | None, Tagger]:
    """Return ``(image_encoder, tagger)`` for one index/embed run.

    The SigLIP zero-shot tagger already holds the vision encoder, so when semantic
    embeddings are requested it's reused for them — one ONNX session, one inference
    per photo, both jobs done. ``image_encoder`` is ``None`` when ``embed`` is off.
    """

    tagger = get_tagger(config)
    image_encoder: SigLipImageEncoder | None = None
    if embed:
        # The tagger is always the zero-shot one in production; a test may inject a
        # stub tagger without an encoder, in which case load a standalone one.
        image_encoder = (
            tagger.encoder
            if isinstance(tagger, ZeroShotSceneTagger)
            else SigLipImageEncoder.from_config(config)
        )
    return image_encoder, tagger


def _prepare_photo(
    config: AtlasConfig,
    path: Path,
    *,
    backend: FaceBackend | None,
    tagger: Tagger,
    enrollment: Enrollment | None,
    sha1: str,
    image_encoder: SigLipImageEncoder | None = None,
) -> _PreparedPhoto:
    """Decode one image exactly once and derive everything but the DB write.

    Geocoding and persistence are left to the caller so this can run in a worker
    process (it touches neither the SQLite handle nor a shared geocoder table) and
    is reused unchanged by the serial path. Decoding the file a single time and
    reusing the one Pillow image across metadata, faces, thumbnail and scene tag
    (was 4+ decodes per file) is the core per-file speed-up.
    """

    path = Path(path)
    with Image.open(path) as raw:
        raw.load()
        # Read EXIF (capture time, camera, GPS) from the raw image, then bake in
        # the EXIF orientation once so every derived artefact — face detection,
        # crops, thumbnail and scene tag — works on the same *upright* pixels.
        # Previously the thumbnail was transposed but detection + crops used the
        # raw image, so face crops from portrait-orientation photos came out
        # rotated. ``exif_transpose`` drops the orientation tag, so downstream
        # transposes (e.g. in ``resize_image_to``) become no-ops.
        meta = extract_meta_from_image(raw, path)
        img = ImageOps.exif_transpose(raw) or raw
        # Width/height describe the displayed (upright) image, so the grid's
        # intrinsic-size hints match the transposed thumbnail.
        meta.width, meta.height = img.size

        # Folder names (e.g. 2012/2012_05_Sardegna) often carry a year/month/place
        # the file's EXIF lacks. Use them only to fill gaps: a folder date replaces
        # the filesystem-mtime fallback but never a real EXIF capture time.
        folder = extract_folder_meta(path)
        if meta.taken_source != "exif" and folder.year is not None:
            synthesized = datetime(folder.year, folder.month or 1, 1)
            meta.taken_at = synthesized.isoformat(timespec="seconds")
            meta.taken_source = "folder"

        # Detect faces first so the scene tagger can use the count. The backend
        # gets the already-decoded (upright) BGR array, so it never re-reads the file.
        bgr = pil_to_bgr(img)
        observations = backend.detect(path, image=bgr) if backend is not None else []

        thumb_path = thumb_path_for(config, sha1)
        make_thumbnail_from_image(img, thumb_path, size=config.thumb_size)

        # Perceptual hash off the same upright pixels (cheap), for burst/dupe grouping.
        phash = dhash(img)

        # Compute the SigLIP image embedding once (if requested) and reuse it for
        # the zero-shot scene tag, so the vision tower runs a single time per photo.
        embedding = image_encoder.embed_image(img) if image_encoder is not None else None
        if embedding is not None and isinstance(tagger, ZeroShotSceneTagger):
            scene_label, scene_scores = tagger.tag_embedding(
                embedding, face_count=len(observations)
            )
        else:
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
        embedding_blob=db.embedding_to_blob(embedding),
        embed_dim=None if embedding is None else int(embedding.shape[0]),
        phash=phash,
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
        "indexed_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    photo_id = db.upsert_photo(conn, record)
    if prepared.embedding_blob is not None:
        conn.execute(
            "UPDATE photos SET embedding=?, embed_dim=? WHERE id=?",
            (prepared.embedding_blob, prepared.embed_dim, photo_id),
        )
    # Always refresh the perceptual hash (kept out of PHOTO_COLUMNS, like the
    # embedding), so a re-index keeps it current and burst grouping stays accurate.
    db.set_phash(conn, photo_id, prepared.phash)

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
    tagger: Tagger,
    enrollment: Enrollment | None = None,
    stats: IndexStats | None = None,
    sha1: str | None = None,
    image_encoder: SigLipImageEncoder | None = None,
) -> int:
    """Index a single image file and return its photo id."""

    path = Path(path)
    if sha1 is None:
        sha1 = sha1_of(path)
    prepared = _prepare_photo(
        config, path, backend=backend, tagger=tagger, enrollment=enrollment, sha1=sha1,
        image_encoder=image_encoder,
    )
    place = geocoder.lookup(prepared.lat, prepared.lon) if geocoder is not None else None
    return _commit_prepared(conn, config, prepared, place, stats)


def index_video(
    conn: sqlite3.Connection,
    config: AtlasConfig,
    path: Path,
    *,
    sha1: str,
    geocoder: Geocoder | None = None,
    stats: IndexStats | None = None,
    extract_poster: Callable[[Path, Path], Path] | None = None,
    probe_metadata: Callable[[Path], video.VideoMeta] | None = None,
) -> int:
    """Index one video as a poster-frame row and return its photo id.

    Extracts a full-resolution poster frame (content-addressed by ``sha1``), reads
    the capture date / GPS from the container, and stores a ``photos`` row with
    ``is_video=1`` whose ``path`` is the video itself (so it streams for playback)
    and whose thumbnail is built from the poster. No face/scene pass is run on
    videos. ``extract_poster``/``probe_metadata`` are injectable so the suite can
    exercise this offline without ffmpeg.
    """

    # Resolved at call time (not as defaults) so the suite can monkeypatch the
    # ``video`` module to exercise the whole indexing walk without ffmpeg.
    extract_poster = extract_poster or video.extract_poster
    probe_metadata = probe_metadata or video.probe_metadata

    path = Path(path)
    poster = poster_path_for(config, sha1)
    if not poster.exists():
        extract_poster(path, poster)
    meta = probe_metadata(path)

    with Image.open(poster) as pimg:
        poster_w, poster_h = pimg.size
        thumb_path = thumb_path_for(config, sha1)
        make_thumbnail_from_image(pimg, thumb_path, size=config.thumb_size)

    # Capture time: container metadata → folder hint → filesystem mtime.
    folder = extract_folder_meta(path)
    taken_at, taken_source = meta.taken_at, "video"
    if taken_at is None and folder.year is not None:
        taken_at = datetime(folder.year, folder.month or 1, 1).isoformat(timespec="seconds")
        taken_source = "folder"
    if taken_at is None:
        taken_at = datetime.fromtimestamp(path.stat().st_mtime, tz=UTC).isoformat(
            timespec="seconds"
        )
        taken_source = "mtime"

    place = geocoder.lookup(meta.lat, meta.lon) if geocoder is not None else None
    record = {
        "path": str(path.resolve()),
        "filename": path.name,
        "sha1": sha1,
        "width": meta.width or poster_w,
        "height": meta.height or poster_h,
        "bytes": path.stat().st_size,
        "taken_at": taken_at,
        "taken_source": taken_source,
        "camera_make": None,
        "camera_model": None,
        "lat": meta.lat,
        "lon": meta.lon,
        "place_city": place.city if place else None,
        "place_country": place.country if place else None,
        "place_label": place.label if place else None,
        "folder_place": folder.place,
        "scene_type": None,
        "scene_scores": json.dumps({}),
        "face_count": 0,
        "thumb_path": str(thumb_path),
        "indexed_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    photo_id = db.upsert_photo(conn, record)
    # ``is_video`` lives outside PHOTO_COLUMNS (like ``favorite``) so a photo
    # re-index never touches it; set it explicitly for the video row here.
    conn.execute("UPDATE photos SET is_video=1 WHERE id=?", (photo_id,))
    db.replace_faces(conn, photo_id, [])  # a re-index of a video clears any stale faces
    if stats is not None:
        stats.indexed += 1
    return photo_id


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
    embed: bool,
    tagger: Tagger | None = None,
) -> None:
    """Initialise a pool worker with its own backend / tagger (called once).

    ``tagger`` may be an injected (picklable) tagger — used by tests to keep the
    suite offline; production passes ``None`` and the worker builds the SigLIP one.
    """

    if tagger is None:
        image_encoder, tagger = _build_encoders(config, embed=embed)
    else:
        image_encoder = None
    _WORKER_STATE["backend"] = (
        get_backend(backend_name, model_dir=model_dir) if backend_name != "none" else None
    )
    _WORKER_STATE["tagger"] = tagger
    _WORKER_STATE["image_encoder"] = image_encoder
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
            image_encoder=_WORKER_STATE["image_encoder"],
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
    embed: bool,
    progress: Callable[[Path, IndexStats], None] | None,
    tagger: Tagger | None = None,
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
    # Likewise pre-fetch the SigLIP vision model so N workers don't race to
    # download it — it's needed for every index run now (the only scene tagger is
    # zero-shot) and additionally for embeddings. Skipped when a tagger is injected
    # (tests), since the worker then never builds the real one.
    if tagger is None:
        try:
            from .models import ensure_scene_model

            ensure_scene_model(config.models_dir, download=True)
        except Exception:  # pragma: no cover - worker surfaces a clearer error
            pass

    ctx = mp.get_context("spawn")  # clean workers; safe for OpenCV/ONNX native libs
    max_inflight = workers * 4
    commit_every = 64
    since_commit = 0

    with ProcessPoolExecutor(
        max_workers=workers, mp_context=ctx,
        initializer=_worker_init,
        initargs=(backend_name, config.models_dir, config, enrollment, embed, tagger),
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
    embed: bool = False,
    progress: Callable[[Path, IndexStats], None] | None = None,
    tagger: Tagger | None = None,
) -> IndexStats:
    """Index every supported image under ``root`` into the library.

    ``workers`` controls fan-out: ``None``/``1`` keeps the in-process path; a
    larger value decodes and detects across that many worker processes (DB writes
    still funnel through the single main connection). ``embed`` additionally stores
    a SigLIP image embedding per photo for natural-language semantic search.

    ``tagger`` injects a scene tagger instead of building the SigLIP one; it's the
    dependency-injection seam the offline test suite uses (a picklable stub tagger),
    and it's passed through to the worker processes unchanged. Production leaves it
    ``None`` and every path builds the zero-shot tagger.
    """

    config.ensure_dirs()
    conn = db.connect(config.db_path)
    backend = (
        get_backend(backend_name, model_dir=config.models_dir)
        if backend_name != "none"
        else None
    )
    if backend_name not in ("none",) and backend is None:
        print(
            f"warning: face backend '{backend_name}' unavailable "
            "(missing models or OpenCV DNN support); indexing without faces.",
            file=sys.stderr,
        )
    geocoder = Geocoder() if geocode else None
    if geocoder is not None and not geocoder.high_resolution:
        print(
            "warning: reverse geocoding is using the bundled ~120-city table, so "
            "city/country labels will be coarse and often wrong. Install the "
            "high-resolution backend with `uv sync --extra geo` (reverse_geocoder).",
            file=sys.stderr,
        )
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
            for d in (
                config.thumbs_dir, config.faces_dir, config.previews_dir,
                config.posters_dir, config.models_dir,
            )
        )

        # Videos are collected during the walk and ingested after the photo pass
        # (poster-frame extraction is main-process/serial, not part of the worker
        # fan-out). Skip them entirely when ffmpeg isn't available.
        ffmpeg_ok = video.ffmpeg_available()
        videos_to_index: list[tuple[str, str]] = []

        def _dedup_ok(path: Path, resolved: Path) -> str | None:
            """Path-skip + SHA-1 dedup bookkeeping shared by photos and videos.

            Returns the file's SHA-1 when it should be ingested, or ``None`` when it
            was skipped/duplicated/failed (counters are bumped as a side effect).
            """

            if not recompute and str(resolved) in existing:
                stats.skipped += 1
                return None
            try:
                sha1 = sha1_of(path)
            except Exception as exc:
                stats.failed += 1
                if len(stats.errors) < _MAX_ERRORS:
                    stats.errors.append(f"{path}: {type(exc).__name__}: {exc}")
                return None
            if not recompute and sha1 in seen_sha1:
                stats.duplicates += 1
                return None
            seen_sha1.add(sha1)
            return sha1

        def iter_tasks() -> Iterator[tuple[str, str]]:
            """Walk the tree, filtering + deduping, and yield ``(path, sha1)``.

            All bookkeeping the parallel path can't do safely from a worker —
            scan/skip/duplicate/video counting and SHA-1 dedup against the
            catalog — happens here in the main process before a file is handed off.
            Videos are siphoned off into ``videos_to_index`` rather than yielded.
            """

            for path in iter_files(root):
                resolved = path.resolve()
                if any(resolved == d or d in resolved.parents for d in derived):
                    continue
                if is_video(path):
                    stats.videos += 1
                    if not ffmpeg_ok:
                        continue  # no ffmpeg: keep counting videos, don't ingest
                    sha1 = _dedup_ok(path, resolved)
                    if sha1 is not None:
                        videos_to_index.append((str(path), sha1))
                    continue
                if not is_supported(path):
                    continue
                stats.scanned += 1
                sha1 = _dedup_ok(path, resolved)
                if sha1 is None:
                    continue
                yield str(path), sha1

        if workers is not None and workers > 1:
            # Workers build their own encoders/tagger (the ONNX sessions aren't
            # picklable), so the main process doesn't load them here.
            _index_parallel(
                conn, config, iter_tasks(),
                backend_name=backend_name, enrollment=enrollment, geocoder=geocoder,
                stats=stats, workers=workers, embed=embed, progress=progress,
                tagger=tagger,
            )
        else:
            if tagger is None:
                image_encoder, tagger = _build_encoders(config, embed=embed)
            else:
                image_encoder = None
            for path_str, sha1 in iter_tasks():
                path = Path(path_str)
                try:
                    index_file(
                        conn, config, path,
                        backend=backend, geocoder=geocoder, tagger=tagger,
                        enrollment=enrollment, stats=stats, sha1=sha1,
                        image_encoder=image_encoder,
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

        # Video pass: poster-frame ingest happens here in the main process (after
        # the photo fan-out), so it's the same code whether or not workers were used.
        for path_str, sha1 in videos_to_index:
            path = Path(path_str)
            try:
                index_video(conn, config, path, sha1=sha1, geocoder=geocoder, stats=stats)
                conn.commit()
            except Exception as exc:
                stats.failed += 1
                if len(stats.errors) < _MAX_ERRORS:
                    stats.errors.append(f"{path}: {type(exc).__name__}: {exc}")
            if progress is not None:
                progress(path, stats)
        if stats.videos and not ffmpeg_ok:
            print(
                f"note: found {stats.videos} video(s) but ffmpeg/ffprobe aren't on "
                "PATH, so they were counted but not indexed. Install ffmpeg to get "
                "poster-frame thumbnails + capture dates for videos.",
                file=sys.stderr,
            )
    finally:
        conn.close()
    return stats


def sweep_orphan_derivatives(config: AtlasConfig) -> int:
    """Delete derivative files no catalog row references; return how many were removed.

    Thumbnails and preview/retina variants are content-addressed by the source
    photo's SHA-1 (``{sha1}.jpg`` / ``{sha1}_{size}.jpg``), and face crops live
    under ``faces_dir/<photo_id>/``. A crash mid-index, or a source photo whose
    bytes (and thus SHA-1) changed, can strand such files with no owning row. This
    reclaims them: any thumb/preview whose SHA-1 isn't in the catalog, any leftover
    ``.part`` temp from an interrupted write, and any face-crop dir for a photo id
    that no longer exists. Referenced files are always kept.
    """

    import shutil

    conn = db.connect(config.db_path, ensure_schema=False)
    try:
        live_sha1 = {
            r[0] for r in conn.execute("SELECT sha1 FROM photos WHERE sha1 IS NOT NULL")
        }
        live_ids = {str(r[0]) for r in conn.execute("SELECT id FROM photos")}
    finally:
        conn.close()

    removed = 0
    # Content-addressed thumbnails, preview/retina variants and video posters.
    for cache_dir in (config.thumbs_dir, config.previews_dir, config.posters_dir):
        if not cache_dir.exists():
            continue
        for f in cache_dir.rglob("*"):
            if not f.is_file():
                continue
            # A leftover atomic-write temp is an orphan regardless of name.
            if f.name.endswith(".part"):
                f.unlink(missing_ok=True)
                removed += 1
                continue
            # Filenames are ``{sha1}.jpg`` or ``{sha1}_{size}.jpg``; pull the sha1.
            sha1 = f.name.split(".", 1)[0].split("_", 1)[0]
            if sha1 not in live_sha1:
                f.unlink(missing_ok=True)
                removed += 1
    # Face-crop directories keyed by photo id.
    if config.faces_dir.exists():
        for sub in config.faces_dir.iterdir():
            if sub.is_dir() and sub.name not in live_ids:
                shutil.rmtree(sub, ignore_errors=True)
                removed += 1
    return removed


def prune_library(config: AtlasConfig) -> dict[str, int]:
    """Reconcile the catalog with the filesystem (rows *and* derivative files).

    Indexing only ever adds or updates rows, so moved/deleted photos linger as
    dead entries that 404 in the UI. ``prune`` removes the photo row for each
    missing source file (its faces cascade away) plus that row's thumbnail and
    face crops, then sweeps any remaining orphaned derivative files
    (:func:`sweep_orphan_derivatives`) — e.g. preview/retina variants of the
    removed photos, or files stranded by an interrupted index. Returns
    ``{removed, kept, orphans}``.
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
    # Now that dead rows are gone, reclaim any derivative files they (or a crash)
    # left without an owning row.
    orphans = sweep_orphan_derivatives(config)
    return {"removed": removed, "kept": kept, "orphans": orphans}


def retag_scenes(
    config: AtlasConfig,
    *,
    tagger: Tagger | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> int:
    """Recompute every photo's scene tag in place, without a full re-index.

    Scene tags are independent of faces/thumbnails, so switching tagger
    (heuristic <-> zero-shot) or tuning its hyperparameters shouldn't require
    re-decoding + re-detecting the whole library. This decodes each still-present
    source file once, re-runs the configured tagger (reusing the stored
    ``face_count`` so the people bonus still applies), and upserts only the
    ``scene_type``/``scene_scores`` columns. Returns the number of photos retagged.
    """

    tagger = tagger or get_tagger(config)
    conn = db.connect(config.db_path)
    retagged = 0
    try:
        rows = conn.execute("SELECT id, path, face_count FROM photos").fetchall()
        total = len(rows)
        for i, row in enumerate(rows):
            path = Path(row["path"])
            if not path.exists():
                continue
            try:
                with Image.open(path) as raw:
                    raw.load()
                    img = ImageOps.exif_transpose(raw) or raw
                    label, scores = tagger.tag_image(img, face_count=row["face_count"] or 0)
            except Exception as exc:
                # Skip an unreadable/corrupt file but don't fail silently — report
                # which photo so a partial result is explainable.
                print(f"warning: could not retag {path}: {exc}", file=sys.stderr)
                continue
            conn.execute(
                "UPDATE photos SET scene_type=?, scene_scores=? WHERE id=?",
                (label, json.dumps(scores), row["id"]),
            )
            retagged += 1
            if progress is not None:
                progress(i + 1, total)
        conn.commit()
    finally:
        conn.close()
    return retagged


def embed_library(
    config: AtlasConfig,
    *,
    image_encoder: SigLipImageEncoder | None = None,
    recompute: bool = False,
    progress: Callable[[int, int], None] | None = None,
) -> int:
    """Backfill SigLIP image embeddings for semantic search, without a re-index.

    Embeddings are independent of faces/thumbnails, so an already-indexed library
    can gain semantic search by decoding each still-present photo once and storing
    only its image embedding — no face re-detection. By default only photos that
    don't already have an embedding are processed (``recompute`` re-embeds all).
    Returns the number of photos embedded.
    """

    encoder = image_encoder or SigLipImageEncoder.from_config(config)
    conn = db.connect(config.db_path)
    embedded = 0
    try:
        where = "" if recompute else " WHERE embedding IS NULL"
        rows = conn.execute(f"SELECT id, path FROM photos{where}").fetchall()
        total = len(rows)
        for i, row in enumerate(rows):
            path = Path(row["path"])
            if not path.exists():
                continue
            try:
                with Image.open(path) as raw:
                    raw.load()
                    img = ImageOps.exif_transpose(raw) or raw
                    vector = encoder.embed_image(img)
            except Exception as exc:
                print(f"warning: could not embed {path}: {exc}", file=sys.stderr)
                continue
            db.set_photo_embedding(conn, int(row["id"]), vector)
            embedded += 1
            if progress is not None:
                progress(i + 1, total)
        conn.commit()
    finally:
        conn.close()
    return embedded


def backfill_phashes(
    config: AtlasConfig,
    *,
    recompute: bool = False,
    progress: Callable[[int, int], None] | None = None,
) -> int:
    """Backfill perceptual hashes for near-duplicate / burst grouping, no re-index.

    New indexes compute the dHash for free, but a library indexed before the column
    existed has none. This decodes each still-present photo once and stores only its
    ``phash`` — no face re-detection, no embedding. By default only photos missing a
    hash are processed (``recompute`` re-hashes all). Videos (no still pixels) are
    skipped. Returns the number of photos hashed.
    """

    conn = db.connect(config.db_path)
    hashed = 0
    try:
        where = "WHERE is_video = 0" if recompute else "WHERE phash IS NULL AND is_video = 0"
        rows = conn.execute(f"SELECT id, path FROM photos {where}").fetchall()
        total = len(rows)
        for i, row in enumerate(rows):
            path = Path(row["path"])
            if not path.exists():
                continue
            try:
                with Image.open(path) as raw:
                    raw.load()
                    img = ImageOps.exif_transpose(raw) or raw
                    value = dhash(img)
            except Exception as exc:
                print(f"warning: could not hash {path}: {exc}", file=sys.stderr)
                continue
            db.set_phash(conn, int(row["id"]), value)
            hashed += 1
            if progress is not None:
                progress(i + 1, total)
        conn.commit()
    finally:
        conn.close()
    return hashed


def delete_photos(config: AtlasConfig, ids: list[int]) -> dict[str, int]:
    """Hard-delete photos: their catalog rows *and* source files + derivatives.

    Irreversible. For each id this removes the source file on disk, the photo's
    thumbnail and face-crop directory, then the catalog row (faces cascade); a
    final sweep reclaims any remaining content-addressed derivatives (preview /
    retina variants, video posters) the deleted rows owned. Returns
    ``{rows, files, orphans}`` — rows removed from the catalog, source files
    unlinked from disk, and orphaned derivative files swept.
    """

    import shutil

    clean = [int(i) for i in ids]
    conn = db.connect(config.db_path)
    files_removed = 0
    try:
        if clean:
            placeholders = ", ".join(["?"] * len(clean))
            rows = conn.execute(
                f"SELECT id, path, thumb_path FROM photos WHERE id IN ({placeholders})",
                clean,
            ).fetchall()
            for row in rows:
                src = Path(row["path"]) if row["path"] else None
                if src is not None and src.exists():
                    try:
                        src.unlink()
                        files_removed += 1
                    except OSError as exc:  # surface, don't crash the whole batch
                        print(f"warning: could not delete {src}: {exc}", file=sys.stderr)
                if row["thumb_path"]:
                    Path(row["thumb_path"]).unlink(missing_ok=True)
                crop_dir = config.faces_dir / str(row["id"])
                if crop_dir.exists():
                    shutil.rmtree(crop_dir, ignore_errors=True)
        removed = db.delete_photos(conn, clean)
    finally:
        conn.close()
    orphans = sweep_orphan_derivatives(config)
    return {"rows": removed, "files": files_removed, "orphans": orphans}


def export_photos(config: AtlasConfig, ids: list[int], dest: Path | str) -> dict[str, int]:
    """Copy the chosen photos' original files into ``dest``; never moves them.

    A multi-select bulk action: pull a filtered/selected set out of the library as
    plain files. Originals are copied (with metadata, ``copy2``), so the catalog is
    untouched. A row whose source file is gone — or an unknown id — counts toward
    ``missing``. Basename collisions (two photos sharing a filename from different
    folders) are disambiguated with an id suffix rather than clobbering. Returns
    ``{requested, copied, missing}``.
    """

    import shutil

    dest = Path(dest).expanduser()
    clean = [int(i) for i in ids]
    copied = missing = 0
    if clean:
        dest.mkdir(parents=True, exist_ok=True)
        conn = db.connect(config.db_path, ensure_schema=False)
        try:
            placeholders = ", ".join(["?"] * len(clean))
            found = {
                int(r["id"]): r
                for r in conn.execute(
                    f"SELECT id, path, filename FROM photos WHERE id IN ({placeholders})",
                    clean,
                ).fetchall()
            }
        finally:
            conn.close()
        for i in clean:
            row = found.get(i)
            if row is None:
                missing += 1
                continue
            src = Path(row["path"]) if row["path"] else None
            if src is None or not src.exists():
                missing += 1
                continue
            target = dest / (row["filename"] or src.name)
            if target.exists():  # don't clobber a same-named file from another folder
                target = dest / f"{src.stem}_{i}{src.suffix}"
            try:
                shutil.copy2(src, target)
                copied += 1
            except OSError as exc:
                print(f"warning: could not export {src}: {exc}", file=sys.stderr)
                missing += 1
    return {"requested": len(clean), "copied": copied, "missing": missing}


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
