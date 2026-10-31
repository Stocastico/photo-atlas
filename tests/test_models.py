"""Model-management helpers in ``photo_atlas.models`` (no downloads)."""

from __future__ import annotations

from photo_atlas import models


def test_yunet_default_is_latest_zoo_revision():
    # Drop-in bump to the newest OpenCV Zoo YuNet (same FaceDetectorYN API).
    assert models.YUNET_NAME == "face_detection_yunet_2026may.onnx"
    assert models.YUNET_NAME in models.YUNET_URL


def test_arcface_default_recognition_model():
    # Face recognition default is ArcFace R100 (glint360k, 512-d).
    assert "arcface" in models.ARCFACE_NAME.lower()
    assert models.ARCFACE_NAME in models.ARCFACE_URL or "recognition" in models.ARCFACE_URL


def test_ensure_arcface_env_override(tmp_path, monkeypatch):
    local = tmp_path / "my_arcface.onnx"
    local.write_bytes(b"x")
    monkeypatch.setenv("PHOTO_ATLAS_ARCFACE", str(local))
    assert models.ensure_arcface(tmp_path, download=False) == local


def test_scene_defaults_are_siglip2():
    # The shipped default scene/semantic stack is SigLIP 2 (base patch16-256).
    assert "siglip2" in models.SCENE_NAME
    assert "siglip2" in models.SCENE_URL
    assert "siglip2" in models.SCENE_TEXT_URL
    assert models.SCENE_INPUT_SIZE == 256


def test_ensure_scene_input_size_default(monkeypatch):
    monkeypatch.delenv("PHOTO_ATLAS_SCENE_INPUT_SIZE", raising=False)
    assert models.ensure_scene_input_size() == 256


def test_ensure_scene_input_size_env_override(monkeypatch):
    # A higher-resolution SigLIP 2 variant (384/512) is selectable without code.
    monkeypatch.setenv("PHOTO_ATLAS_SCENE_INPUT_SIZE", "384")
    assert models.ensure_scene_input_size() == 384


# -- download integrity -----------------------------------------------------
import hashlib  # noqa: E402
import re  # noqa: E402

import pytest  # noqa: E402


def _fake_download(monkeypatch, payload: bytes):
    def retrieve(url, dest):
        from pathlib import Path

        Path(dest).write_bytes(payload)
        return str(dest), {"Content-Length": str(len(payload))}

    monkeypatch.setattr(models.urllib.request, "urlretrieve", retrieve)


def test_download_is_verified_against_the_pinned_sha256(tmp_path, monkeypatch):
    payload = b"x" * 60_000
    monkeypatch.setitem(models.MODEL_SHA256, "fake.onnx", hashlib.sha256(payload).hexdigest())
    _fake_download(monkeypatch, payload)
    got = models._resolve("fake.onnx", "http://x/fake", tmp_path, "PHOTO_ATLAS_NOPE", True)
    assert got.read_bytes() == payload


def test_checksum_mismatch_discards_the_download(tmp_path, monkeypatch):
    monkeypatch.setitem(models.MODEL_SHA256, "fake.onnx", "0" * 64)
    _fake_download(monkeypatch, b"y" * 60_000)
    with pytest.raises(RuntimeError, match="checksum"):
        models._resolve("fake.onnx", "http://x/fake", tmp_path, "PHOTO_ATLAS_NOPE", True)
    assert not (tmp_path / "fake.onnx").exists()
    assert not list(tmp_path.glob("*.part"))  # no half-trusted leftovers


def test_unpinned_models_still_download(tmp_path, monkeypatch):
    monkeypatch.delitem(models.MODEL_SHA256, "fake.onnx", raising=False)
    _fake_download(monkeypatch, b"z" * 60_000)
    assert models._resolve("fake.onnx", "http://x/f", tmp_path, "PHOTO_ATLAS_NOPE", True).exists()


def test_huggingface_models_are_pinned_to_a_commit_and_checksum():
    hf = {
        models.ARCFACE_NAME: models.ARCFACE_URL,
        models.SCENE_NAME: models.SCENE_URL,
        models.SCENE_TEXT_NAME: models.SCENE_TEXT_URL,
        models.SCENE_TOKENIZER_NAME: models.SCENE_TOKENIZER_URL,
    }
    for name, url in hf.items():
        assert re.search(r"/resolve/[0-9a-f]{40}/", url), f"{name} must pin a commit, not 'main'"
        assert re.fullmatch(r"[0-9a-f]{64}", models.MODEL_SHA256.get(name, "")), name


def test_documented_env_var_names_are_the_ones_the_code_reads():
    """Error messages / README once told users to set PHOTO_ATLAS_TEXT_MODEL, which
    nothing reads (the real name is PHOTO_ATLAS_SCENE_TEXT_MODEL)."""

    from pathlib import Path

    root = Path(__file__).parent.parent
    files = [root / "README.md", *(root / "src" / "photo_atlas").glob("*.py")]
    for f in files:
        text = f.read_text(encoding="utf-8")
        assert not re.search(r"PHOTO_ATLAS_TEXT_MODEL\b", text), f"{f.name} cites a dead env var"
