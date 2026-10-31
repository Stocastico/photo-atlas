"""Tunable face-detection/recognition settings (env overrides + min face size)."""

from __future__ import annotations

import numpy as np

from photo_atlas import faces, indexer
from photo_atlas.config import AtlasConfig
from tests.test_faces_backends import _Backend, _img, _row


def test_defaults(monkeypatch):
    for name in ("FACE_THRESHOLD", "CLUSTER_EPS", "REVIEW_CONFIDENCE", "FACE_SCORE",
                 "DETECT_MAX_SIDE", "MIN_FACE_PX"):
        monkeypatch.delenv(f"PHOTO_ATLAS_{name}", raising=False)
    cfg = AtlasConfig()
    assert (cfg.face_match_threshold, cfg.cluster_eps, cfg.review_confidence) == (0.5, 0.5, 0.6)
    assert cfg.face_detect_options() == {
        "score_threshold": 0.85, "detect_max_side": 1280, "min_face_px": 32,
    }


def test_env_overrides(monkeypatch):
    monkeypatch.setenv("PHOTO_ATLAS_FACE_THRESHOLD", "0.42")
    monkeypatch.setenv("PHOTO_ATLAS_CLUSTER_EPS", "0.4")
    monkeypatch.setenv("PHOTO_ATLAS_REVIEW_CONFIDENCE", "0.5")
    monkeypatch.setenv("PHOTO_ATLAS_FACE_SCORE", "0.7")
    monkeypatch.setenv("PHOTO_ATLAS_DETECT_MAX_SIDE", "1920")
    monkeypatch.setenv("PHOTO_ATLAS_MIN_FACE_PX", "20")
    cfg = AtlasConfig()
    assert cfg.face_match_threshold == 0.42 and cfg.cluster_eps == 0.4
    assert cfg.review_confidence == 0.5
    assert cfg.face_detect_options() == {
        "score_threshold": 0.7, "detect_max_side": 1920, "min_face_px": 20,
    }


def test_garbage_env_values_fall_back_to_defaults(monkeypatch):
    monkeypatch.setenv("PHOTO_ATLAS_FACE_THRESHOLD", "abc")
    monkeypatch.setenv("PHOTO_ATLAS_MIN_FACE_PX", "")
    cfg = AtlasConfig()
    assert cfg.face_match_threshold == 0.5 and cfg.min_face_px == 32


def test_small_faces_are_dropped_by_the_detector():
    rows = np.array([_row(w=20, h=40), _row(x=100, w=60, h=60)])
    backend = _Backend(rows)
    backend._min_face_px = 32
    obs = backend.detect("x.jpg", image=_img())
    assert [o.bbox[2:] for o in obs] == [(60, 60)]  # the 20px-wide face is gone

    backend = _Backend(rows)  # no minimum configured -> everything kept
    assert len(backend.detect("x.jpg", image=_img())) == 2


def test_min_size_is_measured_at_full_resolution():
    # 400x200 image, detect_max_side=100 -> scale 0.25: an 8px box on the downscaled
    # frame is a 32px face at full resolution and must be kept.
    backend = _Backend(np.array([_row(x=5, y=6, w=8, h=8)]), max_side=100)
    backend._min_face_px = 32
    assert len(backend.detect("x.jpg", image=_img(200, 400))) == 1


def test_get_backend_forwards_detection_options(monkeypatch):
    seen = {}

    class Fake:
        def __init__(self, model_dir=None, **kw):
            seen.update(kw)

    monkeypatch.setattr(faces, "YuNetArcFaceBackend", Fake)
    faces.get_backend("arcface", score_threshold=0.7, detect_max_side=2000, min_face_px=10)
    assert seen == {"score_threshold": 0.7, "detect_max_side": 2000, "min_face_px": 10}


def test_indexer_builds_backend_from_config(monkeypatch, tmp_path):
    seen = {}
    monkeypatch.setattr(
        indexer, "get_backend", lambda name, **kw: seen.update(kw) or None
    )
    monkeypatch.setenv("PHOTO_ATLAS_MIN_FACE_PX", "50")
    cfg = AtlasConfig(home=tmp_path / "lib").ensure_dirs()
    (tmp_path / "p").mkdir()
    indexer.index_path(cfg, tmp_path / "p", backend_name="arcface", geocode=False, workers=1)
    assert seen["min_face_px"] == 50 and seen["score_threshold"] == 0.85
