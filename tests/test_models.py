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
