"""Face backends exercised with fake detectors / ONNX sessions (no models)."""

from __future__ import annotations

import numpy as np
import pytest

from photo_atlas import faces


class _FakeDetector:
    def __init__(self, rows):
        self._rows = rows
        self.input_size = None

    def setInputSize(self, size):  # noqa: N802 - cv2 API name
        self.input_size = size

    def detect(self, img):
        return 1, self._rows


def _row(x=10, y=20, w=30, h=40, score=0.9):
    lm = np.arange(10, dtype=np.float32) + 50
    return np.array([x, y, w, h, *lm, score], dtype=np.float32)


class _Backend(faces._YuNetBackend):
    """Skip the model-loading ``__init__``; inject a fake detector + embedder."""

    def __init__(self, rows, max_side=1280):
        self._detect_max_side = max_side
        self._detector = _FakeDetector(rows)
        self.embedded: list[np.ndarray] = []

    def _embed(self, image, full_row):
        self.embedded.append(full_row.copy())
        return np.array([3.0, 4.0], dtype=np.float32)


def _img(h=100, w=200):
    return np.zeros((h, w, 3), dtype=np.uint8)


def test_detect_returns_normalised_embedding_and_bbox():
    backend = _Backend(np.array([_row()]))
    (obs,) = backend.detect("ignored.jpg", image=_img())
    assert obs.bbox == (10, 20, 30, 40)
    assert obs.confidence == pytest.approx(0.9)
    assert np.linalg.norm(obs.embedding) == pytest.approx(1.0)
    assert backend._detector.input_size == (200, 100)


def test_detect_maps_boxes_back_from_downscaled_frame():
    # 400x200 image, detect_max_side=100 → scale 0.25; boxes must be mapped x4.
    backend = _Backend(np.array([_row(x=5, y=6, w=7, h=8)]), max_side=100)
    (obs,) = backend.detect("ignored.jpg", image=_img(200, 400))
    assert backend._detector.input_size == (100, 50)
    assert obs.bbox == (20, 24, 28, 32)
    assert backend.embedded[0][-1] == pytest.approx(0.9)  # score is not rescaled


def test_detect_no_faces_and_unreadable_image(monkeypatch):
    assert _Backend(None).detect("x.jpg", image=_img()) == []
    monkeypatch.setattr(faces, "_read_bgr", lambda p: None)
    assert _Backend(np.array([_row()])).detect("missing.jpg") == []


def test_arcface_embed_runs_session_on_aligned_blob():
    backend = faces.YuNetArcFaceBackend.__new__(faces.YuNetArcFaceBackend)
    seen = {}

    class Session:
        def run(self, outputs, feed):
            seen["shape"] = feed["in"].shape
            return [np.ones((1, 512), dtype=np.float32)]

    backend._session, backend._input, backend._output = Session(), "in", "out"
    # Plausible landmarks inside a 200x200 frame so the affine fit is well-posed.
    lm = np.array([70, 80, 130, 80, 100, 110, 80, 140, 120, 140], dtype=np.float32)
    full_row = np.concatenate([[50, 50, 100, 100], lm, [0.99]]).astype(np.float32)
    vec = backend._embed(np.zeros((200, 200, 3), dtype=np.uint8), full_row)
    assert vec.shape == (512,)
    assert seen["shape"] == (1, 3, 112, 112)


def test_sface_embed_delegates_to_recognizer():
    backend = faces.YuNetSFaceBackend.__new__(faces.YuNetSFaceBackend)

    class Rec:
        def alignCrop(self, image, row):  # noqa: N802
            return "aligned"

        def feature(self, aligned):
            assert aligned == "aligned"
            return np.ones((1, 128), dtype=np.float32)

    backend._recognizer = Rec()
    assert backend._embed(_img(), _row()).shape == (128,)


def test_base_hooks_are_abstract():
    backend = _Backend(None)
    with pytest.raises(NotImplementedError):
        faces._YuNetBackend._setup(backend, None, download=False)
    with pytest.raises(NotImplementedError):
        faces._YuNetBackend._embed(backend, _img(), _row())


def test_init_requires_opencv_dnn_face_module(monkeypatch):
    import cv2

    monkeypatch.delattr(cv2, "FaceDetectorYN")
    with pytest.raises(RuntimeError, match="FaceDetectorYN"):
        faces.YuNetArcFaceBackend()


def test_sface_setup_requires_recognizer_module(monkeypatch):
    import cv2

    backend = faces.YuNetSFaceBackend.__new__(faces.YuNetSFaceBackend)
    backend._cv2 = cv2
    monkeypatch.delattr(cv2, "FaceRecognizerSF")
    with pytest.raises(RuntimeError, match="FaceRecognizerSF"):
        backend._setup(None, download=False)
