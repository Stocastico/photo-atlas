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
#: SigLIP pads/truncates text to a fixed 64-token window (both v1 and v2).
_TEXT_PAD_LEN = 64
#: Pad-token candidates, most-specific first. SigLIP 2's Gemma tokenizer uses
#: ``<pad>`` (id 0); SigLIP 1's SentencePiece tokenizer uses ``</s>`` (id 1).
_PAD_TOKEN_PREFERENCE = ("<pad>", "</s>", "<eos>")


def _resolve_pad_token(tok, prefer: tuple[str, ...] = _PAD_TOKEN_PREFERENCE) -> tuple[str, int]:
    """Pick the (token, id) to pad a SigLIP text sequence with.

    SigLIP pools the *last* token's representation over a fixed-length window, so
    the trailing pad token must be the one the model was trained with. SigLIP 1
    uses ``</s>``; SigLIP 2's Gemma tokenizer uses ``<pad>`` (id 0) — pick whichever
    the tokenizer actually defines rather than hardcoding one (SIGLIP2_MIGRATION.md
    Gap 3). Falls back to ``</s>``/1 for an exotic tokenizer with none of them.
    """

    for token in prefer:
        tid = tok.token_to_id(token)
        if tid is not None:
            return token, int(tid)
    return "</s>", 1


def configure_text_tokenizer(tok, pad_len: int = _TEXT_PAD_LEN) -> int:
    """Make ``tok`` pad/truncate to SigLIP's fixed window; return that length.

    SigLIP 2's ``onnx-community`` ``tokenizer.json`` already embeds the right
    padding (``Fixed:64`` right-pad with ``<pad>``), so we respect it as-is instead
    of clobbering it with SigLIP 1's ``</s>`` assumption. A tokenizer that ships
    without a padding config (SigLIP 1) gets configured here with the resolved pad
    token. Either way truncation is clamped to the window so a long query can't
    blow past it.
    """

    existing = getattr(tok, "padding", None)
    if existing and existing.get("length"):
        pad_len = int(existing["length"])
    else:
        token, pad_id = _resolve_pad_token(tok)
        tok.enable_padding(length=pad_len, pad_id=pad_id, pad_token=token, direction="right")
    tok.enable_truncation(pad_len)
    return pad_len


def l2_normalize(vec: np.ndarray) -> np.ndarray:
    """Return ``vec`` scaled to unit L2 norm (a zero vector is returned as-is)."""

    vec = np.asarray(vec, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(vec))
    if norm < 1e-8:
        return vec
    return vec / norm


def _select_output_name(available: list[str], prefer: tuple[str, ...]) -> str:
    """Pick which ONNX output holds the pooled embedding.

    The current SigLIP export names it ``pooler_output``; other exports (incl. some
    SigLIP 2 ONNX) use ``image_embeds`` / ``text_embeds``. Try the preferred names
    in order, then fall back to a single unambiguous output. Refuse to guess among
    several unrecognised outputs (a clear failure beats silently embedding the wrong
    tensor), so swapping in such a model is a one-line ``prefer`` update, not a
    debugging session.
    """

    for name in prefer:
        if name in available:
            return name
    if len(available) == 1:
        return available[0]
    raise ValueError(
        f"could not pick the embedding output among {available}; "
        f"expected one of {prefer}. Add the new model's output name to the "
        "preference list in embed.py."
    )


def _input_size_from_shape(shape: object, default: int = _IMAGE_SIZE) -> int:
    """Derive the square input resolution from an ONNX input shape (NCHW).

    A static spatial dim (e.g. ``[1, 3, 256, 256]``) is used directly, so swapping
    in a higher-resolution model (SigLIP 2 patch16-256/384) needs no code change —
    just point ``PHOTO_ATLAS_SCENE_MODEL`` at the new ONNX. A dynamic ('width'),
    missing or non-positive dim falls back to ``default`` (224).
    """

    try:
        last = shape[-1]  # type: ignore[index]
    except (TypeError, IndexError, KeyError):
        return default
    return int(last) if isinstance(last, int) and last > 0 else default


def _resolve_image_size(explicit: int | None, shape: object, default: int = _IMAGE_SIZE) -> int:
    """Pick the vision input resolution: explicit config > static shape > default.

    An explicitly configured size always wins — SigLIP 2's vision ONNX advertises a
    fully *dynamic* shape yet only accepts its trained resolution (256), which can't
    be recovered from the model — while a model that *does* report a static spatial
    dim (base SigLIP's 224) still self-describes when no size is forced.
    """

    if explicit:
        return int(explicit)
    return _input_size_from_shape(shape, default)


def preprocess_image(img: Image.Image, size: int = _IMAGE_SIZE) -> np.ndarray:
    """Turn an open image into a SigLIP ``(1, 3, size, size)`` float32 blob.

    ``size`` defaults to 224 (the base SigLIP input) but is overridable so a
    different-resolution model can reuse the same preprocessing.
    """

    small = img.convert("RGB").resize((size, size), Image.Resampling.BICUBIC)
    arr = np.asarray(small, dtype=np.float32) / 255.0
    arr = (arr - _NORM_MEAN) / _NORM_STD
    return np.ascontiguousarray(arr.transpose(2, 0, 1)[None])


class SigLipImageEncoder:
    """SigLIP vision encoder: an open image -> a unit-norm embedding."""

    def __init__(
        self,
        model_path: Path | str,
        *,
        image_size: int | None = None,
        default_size: int = _IMAGE_SIZE,
    ):
        import onnxruntime as ort  # noqa: PLC0415

        self._session = ort.InferenceSession(
            str(model_path), providers=["CPUExecutionProvider"]
        )
        inp = self._session.get_inputs()[0]
        self._input = inp.name
        # Resolve the input resolution: an explicit/configured size wins, else a
        # static model shape (base SigLIP's 224), else ``default_size``. SigLIP 2's
        # ONNX is dynamic-shaped, so its 256 comes from the configured default.
        self._image_size = _resolve_image_size(image_size, inp.shape, default_size)
        self._output = _select_output_name(
            [o.name for o in self._session.get_outputs()], ("pooler_output", "image_embeds")
        )

    @classmethod
    def from_config(cls, config: AtlasConfig) -> SigLipImageEncoder:
        from .models import ensure_scene_input_size, ensure_scene_model  # noqa: PLC0415

        return cls(
            ensure_scene_model(config.models_dir, download=True),
            default_size=ensure_scene_input_size(),
        )

    def embed_image(self, img: Image.Image) -> np.ndarray:
        """Embed an already-open image (the indexer's decode-once path)."""

        blob = preprocess_image(img, self._image_size)
        (pooled,) = self._session.run([self._output], {self._input: blob})
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
        # Pad/truncate to SigLIP's fixed window, honouring the tokenizer's own
        # padding config when it has one (SigLIP 2's Gemma tokenizer) and falling
        # back to a resolved pad token otherwise (SigLIP 1). Mirrors
        # scripts/build_scene_embeddings.py, which builds the bundled label matrix.
        configure_text_tokenizer(tok, pad_len)
        self._tok = tok
        self._session = ort.InferenceSession(
            str(model_path), providers=["CPUExecutionProvider"]
        )
        self._input = self._session.get_inputs()[0].name
        self._output = _select_output_name(
            [o.name for o in self._session.get_outputs()], ("pooler_output", "text_embeds")
        )

    @classmethod
    def from_config(cls, config: AtlasConfig) -> SigLipTextEncoder:
        from .models import ensure_scene_text_model, ensure_scene_tokenizer  # noqa: PLC0415

        return cls(
            ensure_scene_text_model(config.models_dir, download=True),
            ensure_scene_tokenizer(config.models_dir, download=True),
        )

    def embed_text(self, text: str) -> np.ndarray:
        ids = np.array([self._tok.encode(text).ids], dtype=np.int64)
        (pooled,) = self._session.run([self._output], {self._input: ids})
        return l2_normalize(pooled[0])
