"""Image metadata extraction and thumbnail generation.

Only Pillow is required. We pull the most useful EXIF fields for a photo
library -- capture time, camera make/model and GPS coordinates -- and fall
back to filesystem timestamps when EXIF is missing (common for scans and
messaging-app exports).
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image, ImageOps
from PIL.ExifTags import GPSTAGS, TAGS

# HEIC/HEIF (the default iPhone format) needs the optional ``pillow-heif``
# plugin; register it when present so those files decode instead of failing.
# Install with ``uv sync --extra heic`` (or ``pip install 'photo-atlas[heic]'``).
try:  # pragma: no cover - depends on an optional dependency being installed
    from pillow_heif import register_heif_opener

    register_heif_opener()
    _HEIF_AVAILABLE = True
except Exception:  # pragma: no cover - plugin absent is the common case
    _HEIF_AVAILABLE = False

SUPPORTED_SUFFIXES = {
    ".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tif", ".tiff", ".webp", ".heic",
}

# Video files aren't indexed (no still-image pipeline), but we recognise them so
# the indexer can *report* how many were skipped instead of dropping them
# silently — a real library is typically a few percent video.
VIDEO_SUFFIXES = {
    ".mov", ".mp4", ".m4v", ".avi", ".3gp", ".mkv", ".webm", ".mts", ".wmv",
}


@dataclass
class PhotoMeta:
    width: int | None = None
    height: int | None = None
    taken_at: str | None = None
    taken_source: str = "mtime"
    camera_make: str | None = None
    camera_model: str | None = None
    lat: float | None = None
    lon: float | None = None


def is_supported(path: Path) -> bool:
    return path.suffix.lower() in SUPPORTED_SUFFIXES


def is_video(path: Path) -> bool:
    return path.suffix.lower() in VIDEO_SUFFIXES


def sha1_of(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def _exif_dict(img: Image.Image) -> dict:
    raw = img.getexif()
    if not raw:
        return {}
    out = {}
    for tag_id, value in raw.items():
        out[TAGS.get(tag_id, tag_id)] = value
    # The canonical capture time (DateTimeOriginal / DateTimeDigitized) lives in
    # the Exif sub-IFD (0x8769), not the base IFD that ``getexif()`` returns, so
    # merge it in. Base-IFD tags win on the rare key collision.
    try:
        exif_ifd = raw.get_ifd(0x8769)
    except Exception:  # pragma: no cover - defensive
        exif_ifd = {}
    for tag_id, value in (exif_ifd or {}).items():
        out.setdefault(TAGS.get(tag_id, tag_id), value)
    return out


def _parse_exif_datetime(value: str) -> str | None:
    # EXIF format: "YYYY:MM:DD HH:MM:SS"
    for fmt in ("%Y:%m:%d %H:%M:%S", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(value.strip(), fmt).isoformat(timespec="seconds")
        except (ValueError, AttributeError):
            continue
    return None


def _ratio(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        try:
            return value[0] / value[1]
        except Exception:  # pragma: no cover - defensive
            return 0.0


def _dms_to_decimal(dms, ref) -> float | None:
    try:
        degrees = _ratio(dms[0])
        minutes = _ratio(dms[1])
        seconds = _ratio(dms[2])
    except (TypeError, IndexError):
        return None
    dec = degrees + minutes / 60.0 + seconds / 3600.0
    if ref in ("S", "W"):
        dec = -dec
    return round(dec, 6)


def _extract_gps(img: Image.Image) -> tuple[float | None, float | None]:
    exif = img.getexif()
    if not exif:
        return None, None
    try:
        gps_ifd = exif.get_ifd(0x8825)
    except Exception:  # pragma: no cover - defensive
        return None, None
    if not gps_ifd:
        return None, None
    gps = {GPSTAGS.get(k, k): v for k, v in gps_ifd.items()}
    lat = lon = None
    if "GPSLatitude" in gps and "GPSLatitudeRef" in gps:
        lat = _dms_to_decimal(gps["GPSLatitude"], gps["GPSLatitudeRef"])
    if "GPSLongitude" in gps and "GPSLongitudeRef" in gps:
        lon = _dms_to_decimal(gps["GPSLongitude"], gps["GPSLongitudeRef"])
    return lat, lon


def extract_meta_from_image(img: Image.Image, path: Path) -> PhotoMeta:
    """Read dimensions, capture time, camera and GPS from an already-open image.

    Split from :func:`extract_meta` so the indexer can decode a file once and
    reuse that single :class:`PIL.Image.Image` across metadata, thumbnail, scene
    tagging and face crops instead of re-opening it for each stage. ``path`` is
    still needed for the filesystem-mtime fallback when EXIF carries no date.
    """

    meta = PhotoMeta()
    meta.width, meta.height = img.size
    exif = _exif_dict(img)

    for key in ("DateTimeOriginal", "DateTimeDigitized", "DateTime"):
        if key in exif:
            parsed = _parse_exif_datetime(str(exif[key]))
            if parsed:
                meta.taken_at = parsed
                meta.taken_source = "exif"
                break

    make = exif.get("Make")
    model = exif.get("Model")
    meta.camera_make = str(make).strip("\x00 ").strip() or None if make else None
    meta.camera_model = str(model).strip("\x00 ").strip() or None if model else None

    meta.lat, meta.lon = _extract_gps(img)

    if meta.taken_at is None:
        ts = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
        meta.taken_at = ts.isoformat(timespec="seconds")
        meta.taken_source = "mtime"

    return meta


def extract_meta(path: Path) -> PhotoMeta:
    """Read dimensions, capture time, camera and GPS from an image file."""

    with Image.open(path) as img:
        return extract_meta_from_image(img, path)


def resize_image_to(img: Image.Image, dest: Path, size: int, quality: int) -> Path:
    """Write an orientation-corrected, downscaled JPEG from an open image.

    Never upscales (Pillow's ``thumbnail`` only shrinks). Shared by the path-based
    helpers and the indexer's decode-once path, which already holds the image.
    """

    dest.parent.mkdir(parents=True, exist_ok=True)
    out = ImageOps.exif_transpose(img)
    out = out.convert("RGB")
    out.thumbnail((size, size))
    out.save(dest, "JPEG", quality=quality)
    return dest


def _write_resized(path: Path, dest: Path, size: int, quality: int) -> Path:
    """Open ``path`` and write an orientation-corrected, downscaled JPEG."""

    with Image.open(path) as img:
        return resize_image_to(img, dest, size, quality)


def make_thumbnail_from_image(img: Image.Image, dest: Path, size: int = 320) -> Path:
    """Write a JPEG thumbnail from an already-open image (decode-once path)."""

    return resize_image_to(img, dest, size, quality=82)


def make_thumbnail(path: Path, dest: Path, size: int = 320) -> Path:
    """Write an orientation-corrected JPEG thumbnail and return its path."""

    return _write_resized(path, dest, size, quality=82)


def cached_resized(
    cache_dir: Path, src: Path, sha1: str, size: int, *, quality: int = 82
) -> Path:
    """Return a content-addressed, on-demand JPEG derivative of ``src``.

    The file is named ``{sha1}_{size}.jpg`` and generated on first request, so
    repeat requests (and re-indexing) reuse it. Shared by the lightbox preview
    and the retina (2x) thumbnail ``srcset`` variants.
    """

    dest = Path(cache_dir) / sha1[:2] / f"{sha1}_{size}.jpg"
    if dest.exists():
        return dest
    # Write to a unique temp file and atomically rename into place. Two requests
    # racing on the same first-time derivative (or one interrupted mid-write) can
    # then never leave a half-written/corrupt file at ``dest`` — the loser's temp
    # is simply discarded and the rename is atomic on POSIX.
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(f"{dest.name}.{os.getpid()}.part")
    try:
        _write_resized(src, tmp, size, quality)
        os.replace(tmp, dest)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    return dest
