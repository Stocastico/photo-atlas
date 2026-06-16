"""Poster-frame and metadata extraction for video files (optional, ffmpeg-based).

Videos aren't run through the still-image pipeline, but we can make them
browsable by pulling a single **poster frame** plus a capture date / GPS from
their container metadata. Both steps shell out to ``ffmpeg`` / ``ffprobe``; when
those binaries aren't on ``PATH`` the indexer simply keeps counting videos
without ingesting them (the previous behaviour).

The pure parsing of an ``ffprobe`` JSON document (:func:`_parse_probe`) is split
out from the subprocess call so it can be unit-tested offline with canned data.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

#: Override the binaries for an unusual install (e.g. a vendored static build).
FFMPEG = os.environ.get("PHOTO_ATLAS_FFMPEG", "ffmpeg")
FFPROBE = os.environ.get("PHOTO_ATLAS_FFPROBE", "ffprobe")


@dataclass
class VideoMeta:
    taken_at: str | None = None
    lat: float | None = None
    lon: float | None = None
    width: int | None = None
    height: int | None = None
    duration: float | None = None


def ffmpeg_available() -> bool:
    """True when both ``ffmpeg`` and ``ffprobe`` are resolvable on ``PATH``."""

    return shutil.which(FFMPEG) is not None and shutil.which(FFPROBE) is not None


def extract_poster(video: Path, dest: Path, *, at: float = 1.0, timeout: int = 120) -> Path:
    """Write a full-resolution poster JPEG ~``at`` seconds into ``video``.

    Uses an atomic temp-then-replace write so a crash mid-encode can't leave a
    half-written poster at ``dest``. Falls back to the very first frame for clips
    shorter than ``at``.
    """

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(f"{dest.name}.{os.getpid()}.part")

    def _run(seek: float | None) -> None:
        cmd = [FFMPEG, "-y", "-loglevel", "error"]
        if seek:
            cmd += ["-ss", str(seek)]  # input-side seek: fast and frame-accurate enough
        cmd += ["-i", str(video), "-frames:v", "1", "-q:v", "3", str(tmp)]
        subprocess.run(cmd, check=True, capture_output=True, timeout=timeout)

    try:
        _run(at)
        if not tmp.exists() or tmp.stat().st_size == 0:
            raise RuntimeError("empty poster")
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, RuntimeError):
        _run(None)  # clip shorter than `at`: grab the opening frame instead
    os.replace(tmp, dest)
    return dest


def probe_metadata(video: Path, *, timeout: int = 60) -> VideoMeta:
    """Read capture date, GPS and dimensions from ``video``'s container metadata."""

    out = subprocess.run(
        [FFPROBE, "-v", "quiet", "-print_format", "json",
         "-show_format", "-show_streams", str(video)],
        check=True, capture_output=True, text=True, timeout=timeout,
    ).stdout
    return _parse_probe(json.loads(out))


def _parse_probe(data: dict) -> VideoMeta:
    meta = VideoMeta()
    fmt_tags = {k.lower(): v for k, v in (data.get("format", {}).get("tags") or {}).items()}

    streams = data.get("streams") or []
    video_streams = [s for s in streams if s.get("codec_type") == "video"]
    stream_tags: dict = {}
    for s in video_streams:
        if meta.width is None and s.get("width"):
            meta.width, meta.height = int(s["width"]), int(s.get("height") or 0) or None
        stream_tags.update({k.lower(): v for k, v in (s.get("tags") or {}).items()})

    tags = {**stream_tags, **fmt_tags}  # format-level tags win on a collision

    for key in ("creation_time", "com.apple.quicktime.creationdate", "date"):
        if tags.get(key) and (ts := _parse_creation_time(str(tags[key]))) is not None:
            meta.taken_at = ts
            break

    for key in ("com.apple.quicktime.location.iso6709", "location", "location-eng"):
        if tags.get(key):
            lat, lon = _parse_iso6709(str(tags[key]))
            if lat is not None:
                meta.lat, meta.lon = lat, lon
                break

    duration = data.get("format", {}).get("duration")
    if duration is not None:
        try:
            meta.duration = float(duration)
        except (TypeError, ValueError):
            meta.duration = None
    return meta


def _parse_creation_time(value: str) -> str | None:
    """Normalise an ISO-ish container timestamp to ``YYYY-MM-DDTHH:MM:SS``."""

    value = value.strip().replace("Z", "")
    if "+" in value:  # drop a trailing timezone offset
        value = value.split("+")[0]
    if "." in value:  # drop fractional seconds
        value = value.split(".")[0]
    for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(value, fmt).isoformat(timespec="seconds")
        except ValueError:
            continue
    return None


_ISO6709 = re.compile(r"([+-]\d+(?:\.\d+)?)([+-]\d+(?:\.\d+)?)")


def _parse_iso6709(value: str) -> tuple[float | None, float | None]:
    """Parse an ISO 6709 ``±DD.DDDD±DDD.DDDD…`` location string to (lat, lon)."""

    m = _ISO6709.match(value.strip())
    if not m:
        return None, None
    lat, lon = float(m.group(1)), float(m.group(2))
    if not (-90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0):
        return None, None
    return lat, lon
