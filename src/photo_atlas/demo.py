"""Generate a synthetic photo library for trying Photo Atlas end to end.

Real photos are not always at hand (and never in CI), so this module paints a
handful of JPEGs that exercise every part of the pipeline:

* EXIF capture date spread across multiple years,
* EXIF GPS coordinates near several real cities (so reverse geocoding labels
  them), and
* a few simple drawn faces with recurring "identities" (same colour palette per
  person) so detection, clustering and naming have something to chew on.
"""

from __future__ import annotations

import random
from datetime import datetime
from pathlib import Path

from PIL import Image, ImageDraw
from PIL.TiffImagePlugin import IFDRational

# (city, lat, lon) anchors near bundled cities.
_LOCATIONS = [
    ("Rome", 41.9028, 12.4964),
    ("Paris", 48.8566, 2.3522),
    ("Tokyo", 35.6762, 139.6503),
    ("New York", 40.7128, -74.0060),
    ("Barcelona", 41.3851, 2.1734),
]

# Each "person" is a distinctly-hued face colour. They are deliberately not
# photo-realistic: the synthetic detector separates identities by hue, so the
# three palettes sit far apart in colour space (warm tan, periwinkle, mint).
_PEOPLE = [
    (230, 170, 120),
    (170, 150, 225),
    (150, 210, 175),
]


def _deg_to_dms_rational(value: float) -> tuple:
    value = abs(value)
    d = int(value)
    m = int((value - d) * 60)
    s = round((value - d - m / 60) * 3600, 2)
    return (IFDRational(d, 1), IFDRational(m, 1), IFDRational(int(s * 100), 100))


def _build_exif(when: datetime, lat: float, lon: float) -> Image.Exif:
    exif = Image.Exif()
    exif[0x010F] = "PhotoAtlas"          # Make
    exif[0x0110] = "DemoCam 1.0"         # Model
    exif[0x0132] = when.strftime("%Y:%m:%d %H:%M:%S")  # DateTime (main IFD)
    gps = {
        1: "N" if lat >= 0 else "S",
        2: _deg_to_dms_rational(lat),
        3: "E" if lon >= 0 else "W",
        4: _deg_to_dms_rational(lon),
    }
    exif[0x8825] = gps
    return exif


def _draw_face(draw: ImageDraw.ImageDraw, cx: int, cy: int, size: int, skin) -> None:
    r = size // 2
    draw.ellipse([cx - r, cy - int(r * 1.15), cx + r, cy - r], fill=(40, 30, 25))  # hair
    draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=skin)                       # face
    eye = max(2, size // 12)
    ey = cy - size // 8
    draw.ellipse([cx - r // 2 - eye, ey - eye, cx - r // 2 + eye, ey + eye], fill=(30, 30, 30))
    draw.ellipse([cx + r // 2 - eye, ey - eye, cx + r // 2 + eye, ey + eye], fill=(30, 30, 30))
    draw.arc([cx - r // 2, cy, cx + r // 2, cy + size // 4], 200, 340, fill=(120, 40, 40), width=2)


def _paint_scene(kind: str, people_idx: list[int]) -> Image.Image:
    w, h = 640, 480
    img = Image.new("RGB", (w, h))
    draw = ImageDraw.Draw(img)

    if kind == "landscape":
        for y in range(h):
            t = y / h
            draw.line([(0, y), (w, y)], fill=(int(120 + 80 * t), int(170 + 40 * t), 230))
        draw.rectangle([0, int(h * 0.7), w, h], fill=(70, 130, 60))  # ground
    elif kind == "food":
        draw.rectangle([0, 0, w, h], fill=(235, 225, 210))
        draw.ellipse([w // 4, h // 4, 3 * w // 4, 3 * h // 4], fill=(255, 90, 10))  # vivid
        draw.ellipse([w // 3, h // 3, 2 * w // 3, 2 * h // 3], fill=(220, 40, 20))
    elif kind == "document":
        draw.rectangle([0, 0, w, h], fill=(250, 250, 248))
        for i in range(8):
            y = 60 + i * 45
            draw.line([(60, y), (w - 60, y)], fill=(120, 120, 120), width=3)
    else:  # people
        draw.rectangle([0, 0, w, h], fill=(150, 160, 175))

    # Draw the recurring faces (large enough for the synthetic skin-blob detector).
    for i, pidx in enumerate(people_idx):
        cx = (i + 1) * w // (len(people_idx) + 1)
        _draw_face(draw, cx, h // 2, 150, _PEOPLE[pidx % len(_PEOPLE)])
    return img


def generate(dest: Path, count: int = 24, seed: int = 7) -> list[Path]:
    """Write ``count`` demo JPEGs under ``dest`` and return their paths."""

    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    scenes = ["people", "people", "people", "landscape", "food", "document"]
    written: list[Path] = []

    for i in range(count):
        kind = rng.choice(scenes)
        city, lat, lon = rng.choice(_LOCATIONS)
        year = rng.randint(2010, 2024)
        when = datetime(year, rng.randint(1, 12), rng.randint(1, 28),
                        rng.randint(8, 20), rng.randint(0, 59), rng.randint(0, 59))

        if kind == "people":
            people_idx = rng.sample(range(len(_PEOPLE)), k=rng.randint(1, 2))
        else:
            people_idx = []

        jitter_lat = lat + rng.uniform(-0.03, 0.03)
        jitter_lon = lon + rng.uniform(-0.03, 0.03)
        img = _paint_scene(kind, people_idx)
        exif = _build_exif(when, jitter_lat, jitter_lon)

        path = dest / f"{when:%Y%m%d}_{i:03d}_{kind}.jpg"
        img.save(path, "JPEG", quality=90, exif=exif)
        written.append(path)

    return written
