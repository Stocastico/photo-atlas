"""Model-management helpers in ``photo_atlas.models`` (no downloads)."""

from __future__ import annotations

from photo_atlas import models


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
