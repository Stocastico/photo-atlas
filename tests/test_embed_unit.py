"""Pure (model-free) helpers in ``photo_atlas.embed``.

The ONNX encoders need the optional ``scene`` extra + a model download, but the
preprocessing and normalisation are plain numpy and must be exact (they have to
match the bundled label matrix), so they're tested directly here.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

from photo_atlas import embed


def test_l2_normalize_unit_norm():
    out = embed.l2_normalize(np.array([3.0, 4.0], dtype=np.float32))
    assert np.allclose(out, [0.6, 0.8])
    assert np.isclose(np.linalg.norm(out), 1.0)


def test_l2_normalize_zero_vector_is_passthrough():
    z = np.zeros(4, dtype=np.float32)
    out = embed.l2_normalize(z)
    assert np.allclose(out, 0.0)  # no divide-by-zero blow-up


def test_l2_normalize_flattens_input():
    out = embed.l2_normalize(np.ones((2, 2), dtype=np.float32))
    assert out.shape == (4,)
    assert np.isclose(np.linalg.norm(out), 1.0)


def test_preprocess_image_shape_and_normalisation():
    blob = embed.preprocess_image(Image.new("RGB", (320, 240), (255, 255, 255)))
    assert blob.shape == (1, 3, 224, 224) and blob.dtype == np.float32
    # White -> (1.0 - 0.5) / 0.5 == 1.0; black would be -1.0.
    assert np.allclose(blob, 1.0, atol=1e-5)
    black = embed.preprocess_image(Image.new("RGB", (10, 10), (0, 0, 0)))
    assert np.allclose(black, -1.0, atol=1e-5)


def test_preprocess_image_converts_grayscale():
    # A non-RGB mode must be coerced to 3 channels (no crash, right shape).
    blob = embed.preprocess_image(Image.new("L", (16, 16), 128))
    assert blob.shape == (1, 3, 224, 224)


def test_preprocess_image_honours_size_override():
    # A SigLIP 2 swap (e.g. patch16-256) changes the input resolution; the size
    # is threaded through so the same preprocessing serves a different model.
    blob = embed.preprocess_image(Image.new("RGB", (64, 64)), size=256)
    assert blob.shape == (1, 3, 256, 256)


def test_input_size_from_shape_reads_static_spatial_dim():
    # NCHW with a concrete spatial dim → use it (auto-detect the model's resolution).
    assert embed._input_size_from_shape([1, 3, 256, 256]) == 256
    assert embed._input_size_from_shape([1, 3, 384, 384]) == 384


def test_input_size_from_shape_falls_back_on_dynamic_or_bad():
    # Dynamic ('width') / missing / non-positive dims fall back to the default 224.
    assert embed._input_size_from_shape(["batch", 3, "height", "width"]) == 224
    assert embed._input_size_from_shape([1, 3, 0, 0]) == 224
    assert embed._input_size_from_shape([]) == 224
    assert embed._input_size_from_shape(None) == 224
