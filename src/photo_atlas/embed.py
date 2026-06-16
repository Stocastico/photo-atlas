"""SigLIP image/text encoders for semantic search.

Photo Atlas's zero-shot scene tagger already runs a small **SigLIP** vision
encoder (see :mod:`photo_atlas.classify`); semantic search reuses the *same*
joint image/text space so a free-text query can be ranked against every photo's
image embedding by cosine similarity.

Two thin ONNX wrappers live here:

* :class:`SigLipImageEncoder` -- the vision tower, shared with the scene tagger.
  Turns an open image into a unit-norm embedding (``pooler_output``).
* :class:`SigLipTextEncoder` -- the text tower + tokenizer, used only at *query*
  time to embed the user's search phrase. Unlike the bundled scene-label matrix
  (pre-baked offline), an arbitrary query can't be precomputed, so this is the
  one place the runtime needs a tokenizer + text encoder. Both download on demand
  like the other models and need the ``scene`` extra (``onnxruntime`` +
  ``tokenizers``).

The image and text encoders must come from the *same* SigLIP model so their
embeddings are comparable; the defaults in :mod:`photo_atlas.models` keep them in
sync.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from PIL import Image

if TYPE_CHECKING:
    from .config import AtlasConfig

#: SigLIP image preprocessing: resize to 224², rescale to [0, 1] and normalise
#: with mean/std 0.5 (matches the model's ``preprocessor_config.json``).
_IMAGE_SIZE = 224
_NORM_MEAN = 0.5
_NORM_STD = 0.5
#: SigLIP pads/truncates text to a fixed 64-token window.
_TEXT_PAD_LEN = 64


def l2_normalize(vec: np.ndarray) -> np.ndarray:
    """Return ``vec`` scaled to unit L2 norm (a zero vector is returned as-is)."""

    vec = np.asarray(vec, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(vec))
    if norm < 1e-8:
        return vec
    return vec / norm


def preprocess_image(img: Image.Image) -> np.ndarray:
    """Turn an open image into a SigLIP ``(1, 3, 224, 224)`` float32 blob."""

    small = img.convert("RGB").resize((_IMAGE_SIZE, _IMAGE_SIZE), Image.Resampling.BICUBIC)
    arr = np.asarray(small, dtype=np.float32) / 255.0
    arr = (arr - _NORM_MEAN) / _NORM_STD
    return np.ascontiguousarray(arr.transpose(2, 0, 1)[None])


class SigLipImageEncoder:
    """SigLIP vision encoder: an open image -> a unit-norm embedding."""

    def __init__(self, model_path: Path | str):
        import onnxruntime as ort  # noqa: PLC0415

        self._session = ort.InferenceSession(
            str(model_path), providers=["CPUExecutionProvider"]
        )
        self._input = self._session.get_inputs()[0].name

    @classmethod
    def from_config(cls, config: AtlasConfig) -> SigLipImageEncoder:
        from .models import ensure_scene_model  # noqa: PLC0415

        return cls(ensure_scene_model(config.models_dir, download=True))

    def embed_image(self, img: Image.Image) -> np.ndarray:
        """Embed an already-open image (the indexer's decode-once path)."""

        (pooled,) = self._session.run(["pooler_output"], {self._input: preprocess_image(img)})
        return l2_normalize(pooled[0])

    def embed_path(self, path: Path | str) -> np.ndarray:
        with Image.open(path) as img:
            return self.embed_image(img)


class SigLipTextEncoder:
    """SigLIP text encoder + tokenizer: a query phrase -> a unit-norm embedding.

    The embedding lives in the same space as :class:`SigLipImageEncoder`'s image
    embeddings, so ``text @ image`` is a meaningful relevance score.
    """

    def __init__(
        self, model_path: Path | str, tokenizer_path: Path | str, *, pad_len: int = _TEXT_PAD_LEN
    ):
        import onnxruntime as ort  # noqa: PLC0415
        from tokenizers import Tokenizer  # noqa: PLC0415

        tok = Tokenizer.from_file(str(tokenizer_path))
        # SigLIP pads/truncates to a fixed window with the </s> token (mirrors
        # scripts/build_scene_embeddings.py, which built the bundled label matrix).
        pad_id = tok.token_to_id("</s>")
        pad_id = 1 if pad_id is None else pad_id
        tok.enable_truncation(pad_len)
        tok.enable_padding(length=pad_len, pad_id=pad_id, pad_token="</s>", direction="right")
        self._tok = tok
        self._session = ort.InferenceSession(
            str(model_path), providers=["CPUExecutionProvider"]
        )
        self._input = self._session.get_inputs()[0].name

    @classmethod
    def from_config(cls, config: AtlasConfig) -> SigLipTextEncoder:
        from .models import ensure_scene_text_model, ensure_scene_tokenizer  # noqa: PLC0415

        return cls(
            ensure_scene_text_model(config.models_dir, download=True),
            ensure_scene_tokenizer(config.models_dir, download=True),
        )

    def embed_text(self, text: str) -> np.ndarray:
        ids = np.array([self._tok.encode(text).ids], dtype=np.int64)
        (pooled,) = self._session.run(["pooler_output"], {self._input: ids})
        return l2_normalize(pooled[0])
