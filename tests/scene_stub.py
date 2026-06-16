"""A deterministic, picklable scene tagger for the offline test suite.

Production scene tagging is now SigLIP-only (the heuristic was removed), so the
tests can't fall back to a model-free tagger by default. This stub stands in for
the real tagger wherever a test indexes photos, keeping the suite offline (no
model download). It lives in its own importable top-level module — rather than a
conftest-local class — so it survives pickling to the spawn worker processes used
by the parallel-indexing path.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image


class StubTagger:
    """Deterministic offline tagger: ``people`` when a face is present, else ``other``.

    The tiny bit of face awareness keeps the scene facet non-trivial for the
    downstream search/facet tests (the demo library has photos both with and
    without synthetic faces) without any model.
    """

    def _result(self, face_count: int) -> tuple[str, dict[str, float]]:
        label = "people" if face_count >= 1 else "other"
        return label, {label: 1.0}

    def tag(self, path: Path, face_count: int = 0) -> tuple[str, dict[str, float]]:
        return self._result(face_count)

    def tag_image(
        self, img: Image.Image, face_count: int = 0
    ) -> tuple[str, dict[str, float]]:
        return self._result(face_count)

    def tag_embedding(
        self, embedding: object, face_count: int = 0
    ) -> tuple[str, dict[str, float]]:
        return self._result(face_count)
