"""Lightweight scene tagging.

A photo library benefits from a coarse "what is this a picture of" tag even
before any model is trained. :class:`SceneTagger` derives such a tag from cheap
colour / brightness statistics plus the number of faces detected in the image.

Categories: ``people``, ``landscape``, ``food``, ``document``, ``other``.

The heuristic is deliberately simple and dependency-free (Pillow + numpy). For
higher accuracy a trained classifier or a CLIP zero-shot model can be plugged in
later; the indexer only depends on the
``tag(path, face_count) -> (label, scores)`` contract.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

SCENE_LABELS = ["people", "landscape", "food", "document", "other"]


def _softmax(scores: dict[str, float]) -> dict[str, float]:
    keys = list(scores)
    arr = np.array([scores[k] for k in keys], dtype=np.float64)
    arr = arr - arr.max()
    exp = np.exp(arr)
    norm = exp / exp.sum()
    return {k: float(v) for k, v in zip(keys, norm)}


class SceneTagger:
    def tag(self, path: Path, face_count: int = 0) -> tuple[str, dict[str, float]]:
        with Image.open(path) as img:
            return self.tag_image(img, face_count)

    def tag_image(
        self, img: Image.Image, face_count: int = 0
    ) -> tuple[str, dict[str, float]]:
        """Tag an already-open image (the indexer's decode-once path)."""

        small = img.convert("RGB").resize((64, 64))
        arr = np.asarray(small, dtype=np.float32) / 255.0
        r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]

        brightness = float(arr.mean())
        mx = arr.max(axis=2)
        mn = arr.min(axis=2)
        saturation = float(np.where(mx > 0, (mx - mn) / np.maximum(mx, 1e-6), 0).mean())

        top = arr[:21]  # upper third ~ sky
        sky = float(((top[..., 2] > top[..., 0]) & (top.mean(axis=2) > 0.45)).mean())
        green = float(((g > r) & (g > b)).mean())
        warm = float(((r > 0.45) & (r > b) & (g > b) & (saturation > 0.2)).mean())
        near_white = float((arr.mean(axis=2) > 0.8).mean())
        low_sat = 1.0 - saturation

        raw = {
            "people": 0.2 + 2.5 * min(face_count, 3),
            "landscape": 0.6 + 2.0 * sky + 1.5 * green,
            "food": 0.4 + 3.0 * warm,
            "document": 0.2 + 2.5 * near_white * low_sat,
            "other": 0.9,
        }
        # A bright, evenly lit, low-saturation frame is most likely a scan/doc.
        if near_white > 0.5 and saturation < 0.15:
            raw["document"] += 1.5
        # Faces dominate: a portrait is "people" even outdoors.
        if face_count >= 1:
            raw["people"] += 1.0

        scores = _softmax(raw)
        label = max(scores, key=scores.get)
        return label, scores
