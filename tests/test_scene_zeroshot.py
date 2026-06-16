"""Tests for the zero-shot (SigLIP) scene tagger.

The pure scoring logic, the bundled label matrix and the tagger-selection /
fallback behaviour are covered without any model download. A full
vision-encoder round trip runs only when ``onnxruntime`` is installed and
``PHOTO_ATLAS_SCENE_MODEL`` points at a local SigLIP vision ONNX, so the suite
stays green and offline by default.
"""

from __future__ import annotations

import os

import numpy as np
import pytest
from PIL import Image

from photo_atlas import classify
from photo_atlas.classify import (
    SCENE_LABELS,
    ZeroShotSceneTagger,
    classify_embedding,
    get_tagger,
    scene_labels_path,
)
from photo_atlas.config import AtlasConfig

# Four orthonormal label prototypes in a 5-d space; dim 4 is "slack" the probe
# can spend on no label, letting us dial each label's cosine similarity exactly.
LABELS = ["people", "landscape", "food", "document"]
MATRIX = np.eye(4, 5, dtype=np.float32)


def _probe(sims: dict[str, float]) -> np.ndarray:
    """A unit vector whose dot with each label prototype is the requested sim."""

    vec = np.zeros(5, dtype=np.float32)
    for lab, s in sims.items():
        vec[LABELS.index(lab)] = s
    used = float(np.dot(vec, vec))
    vec[4] = np.sqrt(max(0.0, 1.0 - used))  # spend the remainder on the slack axis
    return vec


def test_classify_embedding_picks_nearest_label():
    label, scores = classify_embedding(
        _probe({"food": 0.9}), MATRIX, LABELS, temperature=50.0, other_bias=-0.02
    )
    assert label == "food"
    assert scores["food"] == max(scores.values())


def test_classify_embedding_scores_are_a_full_distribution():
    _, scores = classify_embedding(
        _probe({"people": 0.5}), MATRIX, LABELS, temperature=50.0, other_bias=-0.02
    )
    assert set(scores) == set(SCENE_LABELS)
    assert sum(scores.values()) == pytest.approx(1.0, abs=1e-5)
    assert all(0.0 <= v <= 1.0 for v in scores.values())


def test_classify_embedding_falls_back_to_other_when_flat():
    # Every label weakly + equally matched; a high other_bias must win.
    label, _ = classify_embedding(
        _probe({"people": 0.3, "landscape": 0.3, "food": 0.3, "document": 0.3}),
        MATRIX, LABELS, temperature=50.0, other_bias=0.4,
    )
    assert label == "other"


def test_face_count_nudges_people_over_other():
    probe = _probe({"people": 0.43})
    # Just below the rejection bias: with no faces this reads as "other".
    label0, scores0 = classify_embedding(
        probe, MATRIX, LABELS, temperature=80.0, other_bias=0.45, face_count=0
    )
    # A detected face adds the people bonus (0.43 + 0.04 > 0.45) and flips it.
    label1, scores1 = classify_embedding(
        probe, MATRIX, LABELS, temperature=80.0, other_bias=0.45, face_count=2
    )
    assert label0 == "other"
    assert label1 == "people"
    assert scores1["people"] > scores0["people"]


def test_bundled_label_matrix_is_valid():
    path = scene_labels_path()
    assert path.exists(), "ship the bundled scene_labels.npz"
    data = np.load(path, allow_pickle=True)
    labels = [str(x) for x in data["labels"]]
    matrix = np.asarray(data["matrix"], dtype=np.float32)
    # Labels are the concrete (non-"other") scene labels.
    assert set(labels) == set(SCENE_LABELS) - {"other"}
    assert matrix.shape == (len(labels), int(data["embed_dim"]))
    # Each prototype is L2-normalised, so a raw matrix @ vec is cosine similarity.
    norms = np.linalg.norm(matrix, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-4)


def test_get_tagger_is_zeroshot_and_passes_through_the_injected_encoder():
    # The heuristic tagger is gone: get_tagger always returns the SigLIP zero-shot
    # one. Injecting a stub encoder keeps it offline (no model download) and proves
    # get_tagger forwards the encoder rather than loading its own.
    _, matrix = _bundled_matrix()
    encoder = _StubEncoder(matrix[0])
    tagger = get_tagger(AtlasConfig(), encoder=encoder)
    assert isinstance(tagger, ZeroShotSceneTagger)
    assert tagger.encoder is encoder


def test_preprocess_shape_and_normalisation():
    img = Image.new("RGB", (320, 240), (255, 255, 255))
    blob = classify._preprocess(img)
    assert blob.shape == (1, 3, 224, 224)
    assert blob.dtype == np.float32
    # White pixels -> (1.0 - 0.5) / 0.5 == 1.0 after rescale + normalise.
    assert np.allclose(blob, 1.0, atol=1e-5)


# -- tagger instance methods, driven by a stub encoder (no model download) -----
class _StubEncoder:
    """A vision encoder that returns a fixed embedding (one label prototype)."""

    def __init__(self, vector):
        self._vec = vector

    def embed_image(self, _img):
        return self._vec


def _bundled_matrix():
    data = np.load(scene_labels_path(), allow_pickle=True)
    return [str(x) for x in data["labels"]], np.asarray(data["matrix"], dtype=np.float32)


def test_zeroshot_tagger_instance_paths_with_stub_encoder(tmp_path):
    labels, matrix = _bundled_matrix()
    target = labels.index("food") if "food" in labels else 0
    tagger = ZeroShotSceneTagger(encoder=_StubEncoder(matrix[target]))

    # tag_embedding: a probe equal to the prototype must pick that label.
    label, scores = tagger.tag_embedding(matrix[target])
    assert label == labels[target]
    assert set(scores) == set(SCENE_LABELS)

    # tag_image runs through the stub encoder.
    assert tagger.tag_image(Image.new("RGB", (32, 32), (1, 2, 3)))[0] == labels[target]

    # tag(path) opens the file then delegates to tag_image.
    p = tmp_path / "x.jpg"
    Image.new("RGB", (16, 16), (4, 5, 6)).save(p)
    assert tagger.tag(p)[0] == labels[target]


def test_zeroshot_from_config_accepts_injected_encoder():
    _, matrix = _bundled_matrix()
    tagger = ZeroShotSceneTagger.from_config(AtlasConfig(), encoder=_StubEncoder(matrix[0]))
    assert isinstance(tagger, ZeroShotSceneTagger)


def test_zeroshot_requires_model_or_encoder():
    with pytest.raises(ValueError):
        ZeroShotSceneTagger()


# -- optional live round trip (needs onnxruntime + a local SigLIP vision ONNX) --
_MODEL = os.environ.get("PHOTO_ATLAS_SCENE_MODEL")


@pytest.mark.skipif(
    not _MODEL or not os.path.exists(_MODEL),
    reason="set PHOTO_ATLAS_SCENE_MODEL to a SigLIP vision ONNX to run the live test",
)
def test_zeroshot_tagger_live_round_trip():
    pytest.importorskip("onnxruntime")
    tagger = ZeroShotSceneTagger(_MODEL, scene_labels_path())
    label, scores = tagger.tag_image(Image.new("RGB", (256, 256), (255, 255, 255)))
    assert label in SCENE_LABELS
    assert set(scores) == set(SCENE_LABELS)
    assert sum(scores.values()) == pytest.approx(1.0, abs=1e-5)
