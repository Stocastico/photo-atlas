"""Stored width/height must match the displayed (EXIF-transposed) orientation."""

from __future__ import annotations

from PIL import Image

from photo_atlas.metadata import extract_meta_from_image


def _oriented(tmp_path, orientation: int):
    p = tmp_path / "o.jpg"
    img = Image.new("RGB", (400, 200), "red")
    exif = img.getexif()
    exif[274] = orientation
    img.save(p, exif=exif)
    return Image.open(p), p


def test_rotated_exif_swaps_dimensions(tmp_path):
    img, p = _oriented(tmp_path, 6)  # 90° CW: displayed as portrait
    meta = extract_meta_from_image(img, p)
    assert (meta.width, meta.height) == (200, 400)


def test_upright_exif_keeps_dimensions(tmp_path):
    img, p = _oriented(tmp_path, 1)
    meta = extract_meta_from_image(img, p)
    assert (meta.width, meta.height) == (400, 200)
