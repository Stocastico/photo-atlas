"""Small focused unit tests for db helpers, config, metadata and models."""

from __future__ import annotations

import numpy as np
import pytest

from photo_atlas import db, metadata, models
from photo_atlas.config import AtlasConfig, default_home
from photo_atlas.geocode import Geocoder


def test_geocoder_high_resolution_flag_reflects_backend():
    # Forcing the external backend off must report low resolution (bundled table).
    assert Geocoder(prefer_external=False).high_resolution is False


# -- db embedding (de)serialisation ----------------------------------------
def test_embedding_blob_roundtrip_and_none():
    assert db.embedding_to_blob(None) is None
    assert db.blob_to_embedding(None) is None
    vec = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    back = db.blob_to_embedding(db.embedding_to_blob(vec))
    assert np.allclose(back, vec)


def test_get_or_create_person_is_idempotent(tmp_path):
    conn = db.connect(tmp_path / "x.db")
    try:
        a = db.get_or_create_person(conn, "Ada")
        b = db.get_or_create_person(conn, "  Ada  ")  # trimmed -> same row
        assert a == b
    finally:
        conn.close()


def test_connect_enables_wal_and_foreign_keys(tmp_path):
    conn = db.connect(tmp_path / "wal.db")
    try:
        assert conn.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"
        assert conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1
    finally:
        conn.close()


# -- config ----------------------------------------------------------------
def test_default_home_honours_env(monkeypatch, tmp_path):
    monkeypatch.setenv("PHOTO_ATLAS_HOME", str(tmp_path / "custom"))
    assert default_home() == tmp_path / "custom"
    monkeypatch.delenv("PHOTO_ATLAS_HOME", raising=False)
    assert default_home().name == ".photo_atlas"


def test_config_derived_paths(tmp_path):
    cfg = AtlasConfig(home=tmp_path / "lib")
    assert cfg.db_path == tmp_path / "lib" / "atlas.db"
    assert cfg.thumbs_dir.name == "thumbs"
    assert cfg.previews_dir.name == "previews"
    assert cfg.models_dir.name == "models"


# -- metadata pure helpers -------------------------------------------------
def test_ratio_handles_scalars_tuples_and_garbage():
    assert metadata._ratio(2.5) == 2.5
    assert metadata._ratio((10, 4)) == 2.5
    assert metadata._ratio(object()) == 0.0


def test_dms_to_decimal_sign_and_bad_input():
    north = metadata._dms_to_decimal((41, 54, 0), "N")
    south = metadata._dms_to_decimal((41, 54, 0), "S")
    assert north > 0 and south == -north
    assert metadata._dms_to_decimal((1,), "N") is None  # too few components


def test_parse_exif_datetime_variants():
    assert metadata._parse_exif_datetime("2015:08:09 11:22:33").startswith("2015-08-09")
    assert metadata._parse_exif_datetime("not a date") is None
    assert metadata._parse_exif_datetime(None) is None


def test_cached_resized_downscales_and_is_reused(tmp_path):
    from PIL import Image

    src = tmp_path / "big.jpg"
    Image.new("RGB", (3000, 2000), (20, 40, 60)).save(src, "JPEG")
    out = metadata.cached_resized(tmp_path / "cache", src, "deadbeef", 800)
    with Image.open(out) as img:
        assert max(img.size) == 800
    # Content-addressed: a second call returns the same cached file, not a rebuild.
    again = metadata.cached_resized(tmp_path / "cache", src, "deadbeef", 800)
    assert again == out and out.exists()


def test_mtime_fallback_when_no_exif(tmp_path):
    from PIL import Image

    src = tmp_path / "plain.png"
    Image.new("RGB", (16, 16), (1, 2, 3)).save(src)
    meta = metadata.extract_meta(src)
    assert meta.taken_source == "mtime" and meta.taken_at


# -- models resolution -----------------------------------------------------
def test_models_env_override_missing_file(monkeypatch, tmp_path):
    monkeypatch.setenv("PHOTO_ATLAS_YUNET", str(tmp_path / "absent.onnx"))
    with pytest.raises(FileNotFoundError):
        models._resolve("y.onnx", models.YUNET_URL, tmp_path, "PHOTO_ATLAS_YUNET", download=True)


def test_models_env_override_existing_file(monkeypatch, tmp_path):
    real = tmp_path / "local.onnx"
    real.write_bytes(b"weights")
    monkeypatch.setenv("PHOTO_ATLAS_SFACE", str(real))
    got = models._resolve("s.onnx", models.SFACE_URL, tmp_path, "PHOTO_ATLAS_SFACE", download=True)
    assert got == real


def test_models_download_disabled_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        models._resolve("m.onnx", models.YUNET_URL, tmp_path, "NOPE_ENV", download=False)


class _FakeHeaders:
    def __init__(self, d):
        self._d = d

    def get(self, key, default=None):
        return self._d.get(key, default)


def test_models_rejects_truncated_download(monkeypatch, tmp_path):
    def fake(url, filename):  # writes a tiny file (below the sanity floor)
        from pathlib import Path

        Path(filename).write_bytes(b"<html>error</html>")
        return filename, _FakeHeaders({"Content-Length": "18"})

    monkeypatch.setattr(models.urllib.request, "urlretrieve", fake)
    with pytest.raises(RuntimeError, match="incomplete"):
        models._resolve("m.onnx", models.YUNET_URL, tmp_path, "NOPE_ENV", download=True)
    assert not (tmp_path / "m.onnx").exists()  # nothing cached
    assert not (tmp_path / "m.onnx.part").exists()  # partial cleaned up


def test_models_rejects_content_length_mismatch(monkeypatch, tmp_path):
    def fake(url, filename):
        from pathlib import Path

        Path(filename).write_bytes(b"x" * 60_000)
        return filename, _FakeHeaders({"Content-Length": "99999"})  # lies about size

    monkeypatch.setattr(models.urllib.request, "urlretrieve", fake)
    with pytest.raises(RuntimeError, match="incomplete"):
        models._resolve("m.onnx", models.YUNET_URL, tmp_path, "NOPE_ENV", download=True)


def test_models_accepts_complete_download(monkeypatch, tmp_path):
    payload = b"x" * 60_000

    def fake(url, filename):
        from pathlib import Path

        Path(filename).write_bytes(payload)
        return filename, _FakeHeaders({"Content-Length": str(len(payload))})

    monkeypatch.setattr(models.urllib.request, "urlretrieve", fake)
    got = models._resolve("m.onnx", models.YUNET_URL, tmp_path, "NOPE_ENV", download=True)
    assert got.exists() and got.read_bytes() == payload
