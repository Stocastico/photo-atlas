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
from datetime import UTC, datetime
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


def _ratio(value) -> float | None:
    """A single EXIF rational as a float, or ``None`` if it can't be parsed.

    Returning ``None`` (rather than the old ``0.0``) matters: a malformed or
    zero-denominator component must *invalidate* the coordinate, not silently
    contribute a 0 that drags the location to Null Island (0°,0°).
    """

    try:
        return float(value)
    except (TypeError, ValueError):
        try:
            num, den = value[0], value[1]
            return num / den if den else None
        except Exception:  # pragma: no cover - defensive
            return None


def _dms_to_decimal(dms, ref) -> float | None:
    try:
        degrees, minutes, seconds = _ratio(dms[0]), _ratio(dms[1]), _ratio(dms[2])
    except (TypeError, IndexError):
        return None
    # Any unparseable component invalidates the whole coordinate.
    if degrees is None or minutes is None or seconds is None:
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
    # Reject impossible coordinates (corrupt EXIF can yield out-of-range values),
    # so the geocoder and map are never fed garbage.
    if lat is not None and not -90.0 <= lat <= 90.0:
        lat = None
    if lon is not None and not -180.0 <= lon <= 180.0:
        lon = None
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
        ts = datetime.fromtimestamp(path.stat().st_mtime, tz=UTC)
        meta.taken_at = ts.isoformat(timespec="seconds")
        meta.taken_source = "mtime"

    return meta


def extract_meta(path: Path) -> PhotoMeta:
    """Read dimensions, capture time, camera and GPS from an image file."""

    with Image.open(path) as img:
        return extract_meta_from_image(img, path)


def _trim(value: float, places: int = 1) -> str:
    """Format a float with up to ``places`` decimals, dropping trailing zeros."""

    return f"{value:.{places}f}".rstrip("0").rstrip(".")


def _format_aperture(value) -> str | None:
    f = _ratio(value)
    if f is None or f <= 0:
        return None
    return f"ƒ/{_trim(f)}"  # ƒ/2.8, ƒ/8


def _format_shutter(value) -> str | None:
    t = _ratio(value)
    if t is None or t <= 0:
        return None
    if t >= 1:
        return f"{_trim(t)}s"  # 2s, 1.3s
    return f"1/{round(1 / t)}s"  # 1/250s


def _format_focal(value) -> str | None:
    f = _ratio(value)
    if f is None or f <= 0:
        return None
    return f"{_trim(f)}mm"


def _format_iso(value) -> str | None:
    if isinstance(value, (tuple, list)):
        value = value[0] if value else None
    try:
        iso = int(value)
    except (TypeError, ValueError):
        return None
    if iso <= 0:
        return None
    return f"ISO {iso}"


def exif_settings(img: Image.Image) -> dict[str, str]:
    """Pull human-readable capture settings (ƒ/ISO/shutter/lens) from EXIF.

    Returns only the fields actually present, each already formatted for display
    (e.g. ``{"aperture": "ƒ/2.8", "shutter": "1/250s", "iso": "ISO 200"}``).
    These aren't stored in the catalog — they're read on demand for the lightbox
    info panel — so adding them needs no schema change or re-index.
    """

    exif = _exif_dict(img)
    out: dict[str, str] = {}
    if (ap := _format_aperture(exif.get("FNumber"))) is not None:
        out["aperture"] = ap
    if (sh := _format_shutter(exif.get("ExposureTime"))) is not None:
        out["shutter"] = sh
    iso_raw = exif.get("ISOSpeedRatings", exif.get("PhotographicSensitivity"))
    if (iso := _format_iso(iso_raw)) is not None:
        out["iso"] = iso
    if (fl := _format_focal(exif.get("FocalLength"))) is not None:
        out["focal_length"] = fl
    lens = exif.get("LensModel") or exif.get("LensMake")
    if lens and (lens := str(lens).strip("\x00 ").strip()):
        out["lens"] = lens
    return out


def read_exif_settings(path: Path) -> dict[str, str]:
    """Open ``path`` and return its formatted EXIF capture settings."""

    with Image.open(path) as img:
        return exif_settings(img)


def dhash(img: Image.Image, hash_size: int = 8) -> str:
    """Return the difference-hash (dHash) of an image as a hex string.

    dHash is a tiny, robust perceptual fingerprint: downscale to greyscale
    ``(hash_size+1) x hash_size``, then emit one bit per adjacent-pixel pair
    (1 when the left pixel is brighter). Near-identical shots — a camera burst,
    a re-saved/lightly-edited copy — differ in only a handful of bits, so a small
    Hamming distance between two dHashes flags them as near-duplicates.

    Stored as a ``hash_size*hash_size``-bit value rendered hex (16 chars for the
    default 64-bit hash) rather than an INTEGER: a 64-bit unsigned value doesn't
    fit SQLite's signed-64 INTEGER, and grouping compares hashes in Python anyway.
    """

    small = img.convert("L").resize((hash_size + 1, hash_size), Image.Resampling.LANCZOS)
    pixels = list(small.tobytes())  # one byte per pixel (mode "L"), row-major
    row_stride = hash_size + 1
    bits = 0
    for row in range(hash_size):
        base = row * row_stride
        for col in range(hash_size):
            left = pixels[base + col]
            right = pixels[base + col + 1]
            bits = (bits << 1) | (1 if left > right else 0)
    width = (hash_size * hash_size + 3) // 4  # hex digits needed
    return f"{bits:0{width}x}"


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
