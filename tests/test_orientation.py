"""EXIF-orientation handling in the indexing pipeline.

Regression test for the bug where thumbnails/previews were upright (they apply
``ImageOps.exif_transpose``) but face detection and face crops used the *raw*,
un-transposed image — so face crops from portrait-orientation phone photos (which
carry an EXIF orientation flag) rendered rotated 90°/180°. The pipeline must now
work off a single upright image for every derived artefact.
"""

from __future__ import annotations

import io

import numpy as np
from PIL import Image
from scene_stub import StubTagger

from photo_atlas import indexer
from photo_atlas.config import AtlasConfig
from photo_atlas.faces import FaceObservation

_ORIENTATION_TAG = 274  # EXIF "Orientation"


class _RecordingBackend:
    """A face backend that records the array it was handed and returns one face."""

    def __init__(self, bbox):
        self.bbox = bbox
        self.received_shape = None

    def detect(self, image_path, image=None):
        self.received_shape = None if image is None else image.shape
        return [
            FaceObservation(
                bbox=self.bbox,
                embedding=np.zeros(128, dtype=np.float32),
                confidence=1.0,
            )
        ]


def _landscape_with_orientation(path, orientation=6):
    """A 100×40 landscape JPEG flagged to *display* as 40×100 portrait."""

    img = Image.new("RGB", (100, 40), (10, 20, 30))
    exif = img.getexif()
    exif[_ORIENTATION_TAG] = orientation
    img.save(path, "JPEG", exif=exif)


def test_pipeline_uses_upright_image_for_faces_and_dimensions(tmp_path):
    src = tmp_path / "portrait.jpg"
    _landscape_with_orientation(src, orientation=6)

    config = AtlasConfig(home=tmp_path / "lib").ensure_dirs()
    # A face that only fits inside the *upright* (40×100) frame, not the raw 100×40.
    backend = _RecordingBackend(bbox=(2, 10, 30, 70))

    prepared = indexer._prepare_photo(
        config, src, backend=backend, tagger=StubTagger(),
        enrollment=None, sha1="0" * 40,
    )

    # Dimensions reflect the upright (display) orientation, not the raw pixels.
    assert (prepared.width, prepared.height) == (40, 100)

    # The detector saw the upright image (H=100, W=40), not the raw 100×40.
    assert backend.received_shape == (100, 40, 3)

    # The face crop is cut from the upright image, so it has the requested size.
    crop = Image.open(io.BytesIO(prepared.faces[0].crop_jpeg))
    assert crop.size == (30, 70)


def test_image_without_orientation_is_unchanged(tmp_path):
    src = tmp_path / "plain.jpg"
    Image.new("RGB", (80, 60), (10, 20, 30)).save(src, "JPEG")

    config = AtlasConfig(home=tmp_path / "lib").ensure_dirs()
    backend = _RecordingBackend(bbox=(0, 0, 20, 20))
    prepared = indexer._prepare_photo(
        config, src, backend=backend, tagger=StubTagger(),
        enrollment=None, sha1="1" * 40,
    )
    assert (prepared.width, prepared.height) == (80, 60)
    assert backend.received_shape == (60, 80, 3)
