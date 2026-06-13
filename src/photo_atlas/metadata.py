"""Image metadata extraction and thumbnail generation.

Only Pillow is required. We pull the most useful EXIF fields for a photo
library -- capture time, camera make/model and GPS coordinates -- and fall
back to filesystem timestamps when EXIF is missing (common for scans and
messaging-app exports).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image, ImageOps
from PIL.ExifTags import GPSTAGS, TAGS

SUPPORTED_SUFFIXES = {
    ".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tif", ".tiff", ".webp", ".heic",
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


def extract_meta(path: Path) -> PhotoMeta:
    """Read dimensions, capture time, camera and GPS from an image file."""

    meta = PhotoMeta()
    with Image.open(path) as img:
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


def make_thumbnail(path: Path, dest: Path, size: int = 320) -> Path:
    """Write an orientation-corrected JPEG thumbnail and return its path."""

    dest.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(path) as img:
        img = ImageOps.exif_transpose(img)
        img = img.convert("RGB")
        img.thumbnail((size, size))
        img.save(dest, "JPEG", quality=82)
    return dest
