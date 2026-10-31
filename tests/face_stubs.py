"""Shared fakes for the face-backend tests (no models, no cv2 detector).

Imported by bare module name (``from face_stubs import ...``) like ``scene_stub``:
``tests/`` is not a package, so ``from tests.x import ...`` fails on CI.
"""

from __future__ import annotations

import numpy as np

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
