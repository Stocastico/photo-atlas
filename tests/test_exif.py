"""EXIF capture-settings extraction for the lightbox info panel.

These fields (ƒ/ISO/shutter/lens) aren't stored in the catalog — they're read on
demand from the source file — so the coverage here is the formatting logic and a
real save/reload round-trip rather than an indexing path.
"""

from __future__ import annotations

from PIL import Image

from photo_atlas import metadata


def _jpeg_with_exif(path, tags):
    exif = Image.Exif()
    exif[0x8769] = tags  # the Exif sub-IFD, where cameras store capture settings
    Image.new("RGB", (16, 16), (90, 90, 90)).save(path, "JPEG", exif=exif)
    return path


def test_read_exif_settings_formats_all_fields(tmp_path):
    path = _jpeg_with_exif(
        tmp_path / "shot.jpg",
        {
            0x829D: 2.8,  # FNumber
            0x829A: 1 / 250,  # ExposureTime
            0x8827: 200,  # ISOSpeedRatings
            0x920A: 50.0,  # FocalLength
            0xA434: "Test 50mm f/1.8",  # LensModel
        },
    )
    s = metadata.read_exif_settings(path)
    assert s["aperture"] == "ƒ/2.8"
    assert s["shutter"] == "1/250s"
    assert s["iso"] == "ISO 200"
    assert s["focal_length"] == "50mm"
    assert s["lens"] == "Test 50mm f/1.8"


def test_read_exif_settings_empty_without_tags(tmp_path):
    path = tmp_path / "plain.jpg"
    Image.new("RGB", (16, 16)).save(path, "JPEG")
    assert metadata.read_exif_settings(path) == {}


def test_aperture_formatting():
    assert metadata._format_aperture(8.0) == "ƒ/8"  # whole stop drops the .0
    assert metadata._format_aperture(1.8) == "ƒ/1.8"
    assert metadata._format_aperture(0) is None  # zero/garbage invalidates
    assert metadata._format_aperture("nope") is None


def test_shutter_formatting():
    assert metadata._format_shutter(1 / 60) == "1/60s"  # sub-second → 1/x
    assert metadata._format_shutter(2.0) == "2s"  # long exposure
    assert metadata._format_shutter(1.3) == "1.3s"
    assert metadata._format_shutter(0) is None


def test_focal_and_iso_formatting():
    assert metadata._format_focal(50.0) == "50mm"
    assert metadata._format_focal(10.5) == "10.5mm"  # keep a real fraction
    assert metadata._format_focal(None) is None
    assert metadata._format_iso(400) == "ISO 400"
    assert metadata._format_iso([800, 800]) == "ISO 800"  # legacy list form
    assert metadata._format_iso(0) is None
    assert metadata._format_iso("x") is None
