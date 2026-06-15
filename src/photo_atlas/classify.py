"""Scene tagging.

A photo library benefits from a coarse "what is this a picture of" tag. Photo
Atlas offers two taggers behind one ``tag(path, face_count) -> (label, scores)``
contract, selected by :func:`get_tagger`:

* :class:`SceneTagger` -- the dependency-free default. Derives a tag from cheap
  colour / brightness statistics plus the detected face count. Robust to run
  anywhere, but it genuinely mislabels real photos (sunsets read as ``food``,
  snow as ``document``).
* :class:`ZeroShotSceneTagger` -- an opt-in, far more accurate tagger that runs
  a small **SigLIP** vision encoder (ONNX, via ``onnxruntime``) and compares the
  image embedding against pre-computed *text* embeddings for each label. SigLIP
  is a modern CLIP successor that beats CLIP at zero-shot for its size; we run
  only its vision encoder at index time and ship the label (text) embeddings as
  a tiny bundled matrix (see ``scripts/build_scene_embeddings.py``), so there is
  no PyTorch, no text encoder and no tokenizer in the runtime -- just the ONNX
  vision model (downloaded on demand like the face models) plus NumPy.

Categories: ``people``, ``landscape``, ``food``, ``document``, ``other``.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Protocol

import numpy as np
from PIL import Image

if TYPE_CHECKING:
    from .config import AtlasConfig

SCENE_LABELS = ["people", "landscape", "food", "document", "other"]
#: ``other`` is the catch-all fallback, not a thing we can write a text prompt
#: for, so the zero-shot label matrix only covers the concrete labels.
OTHER_LABEL = "other"


def _softmax(scores: dict[str, float]) -> dict[str, float]:
    keys = list(scores)
    arr = np.array([scores[k] for k in keys], dtype=np.float64)
    arr = arr - arr.max()
    exp = np.exp(arr)
    norm = exp / exp.sum()
    return {k: float(v) for k, v in zip(keys, norm, strict=True)}


class Tagger(Protocol):
    """The contract both taggers satisfy (and that the indexer depends on)."""

    def tag(self, path: Path, face_count: int = 0) -> tuple[str, dict[str, float]]:
        ...

    def tag_image(
        self, img: Image.Image, face_count: int = 0
    ) -> tuple[str, dict[str, float]]:
        ...


class SceneTagger:
    """Heuristic colour/brightness tagger -- the dependency-free default."""

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
        label = max(scores, key=lambda k: scores[k])
        return label, scores


# -- zero-shot (SigLIP) tagger --------------------------------------------
#: SigLIP image preprocessing: resize to 224², rescale to [0, 1] and normalise
#: with mean/std 0.5 (matches the model's ``preprocessor_config.json``).
_IMAGE_SIZE = 224
_NORM_MEAN = 0.5
_NORM_STD = 0.5
#: A detected face is a strong "people" signal; nudge that label's similarity by
#: this much (in cosine space) when faces are present, without steamrolling a
#: clear food/landscape signal.
_FACE_BONUS = 0.04


def _preprocess(img: Image.Image) -> np.ndarray:
    """Turn an open image into a SigLIP ``(1, 3, 224, 224)`` float32 blob."""

    small = img.convert("RGB").resize((_IMAGE_SIZE, _IMAGE_SIZE), Image.Resampling.BICUBIC)
    arr = np.asarray(small, dtype=np.float32) / 255.0
    arr = (arr - _NORM_MEAN) / _NORM_STD
    return np.ascontiguousarray(arr.transpose(2, 0, 1)[None])


def classify_embedding(
    embedding: np.ndarray,
    matrix: np.ndarray,
    labels: list[str],
    *,
    temperature: float,
    other_bias: float,
    face_count: int = 0,
) -> tuple[str, dict[str, float]]:
    """Score an image embedding against the per-label text-embedding ``matrix``.

    ``matrix`` is ``(L, D)`` of L2-normalised label prototypes aligned with
    ``labels`` (the concrete, non-``other`` labels). Cosine similarities become
    logits via ``temperature``; ``other_bias`` is appended as the logit for the
    catch-all ``other`` label, so a frame that matches no concrete label only
    weakly (a near-uniform similarity profile) falls through to ``other`` from a
    single softmax/argmax -- no separate threshold needed. Returns
    ``(label, scores)`` with ``scores`` a distribution over all
    :data:`SCENE_LABELS`.
    """

    vec = embedding.astype(np.float32)
    norm = np.linalg.norm(vec)
    if norm > 1e-8:
        vec = vec / norm
    sims = matrix @ vec  # (L,) cosine similarities, vecs are unit norm

    if face_count >= 1 and "people" in labels:
        sims = sims.copy()
        sims[labels.index("people")] += _FACE_BONUS

    logits = np.append(sims, other_bias) * float(temperature)
    exp = np.exp(logits - logits.max())
    probs = exp / exp.sum()

    full_labels = [*labels, OTHER_LABEL]
    scores = {lab: 0.0 for lab in SCENE_LABELS}
    for lab, p in zip(full_labels, probs, strict=True):
        scores[lab] = float(p)
    label = full_labels[int(np.argmax(probs))]
    return label, scores


class ZeroShotSceneTagger:
    """SigLIP zero-shot tagger: vision-encoder ONNX + bundled label embeddings."""

    def __init__(
        self,
        model_path: Path,
        label_path: Path,
        *,
        temperature: float = 50.0,
        other_bias: float = -0.02,
    ):
        import onnxruntime as ort  # noqa: PLC0415

        self._session = ort.InferenceSession(
            str(model_path), providers=["CPUExecutionProvider"]
        )
        self._input = self._session.get_inputs()[0].name
        data = np.load(label_path, allow_pickle=True)
        self._labels = [str(x) for x in data["labels"]]
        self._matrix = np.asarray(data["matrix"], dtype=np.float32)
        self._temperature = temperature
        self._other_bias = other_bias

    @classmethod
    def from_config(cls, config: AtlasConfig) -> ZeroShotSceneTagger:
        """Build from an :class:`AtlasConfig`, fetching the model if needed."""

        from .models import ensure_scene_model  # noqa: PLC0415

        model_path = ensure_scene_model(config.models_dir, download=True)
        return cls(
            model_path,
            scene_labels_path(),
            temperature=config.scene_temperature,
            other_bias=config.scene_other_bias,
        )

    def tag(self, path: Path, face_count: int = 0) -> tuple[str, dict[str, float]]:
        with Image.open(path) as img:
            return self.tag_image(img, face_count)

    def tag_image(
        self, img: Image.Image, face_count: int = 0
    ) -> tuple[str, dict[str, float]]:
        blob = _preprocess(img)
        (pooled,) = self._session.run(["pooler_output"], {self._input: blob})
        return classify_embedding(
            pooled[0],
            self._matrix,
            self._labels,
            temperature=self._temperature,
            other_bias=self._other_bias,
            face_count=face_count,
        )


def scene_labels_path() -> Path:
    """Path to the bundled label-embedding matrix."""

    return Path(__file__).resolve().parent / "data" / "scene_labels.npz"


def get_tagger(config: AtlasConfig) -> Tagger:
    """Return the configured scene tagger.

    ``config.scene_backend`` selects: ``heuristic`` (default), ``zeroshot``
    (SigLIP; warns and falls back to the heuristic if ``onnxruntime`` or the
    model/labels are unavailable) or ``auto`` (use the zero-shot tagger when it
    loads cleanly, else fall back silently).
    """

    backend = getattr(config, "scene_backend", "heuristic")
    if backend in ("zeroshot", "auto"):
        try:
            return ZeroShotSceneTagger.from_config(config)
        except Exception as exc:  # noqa: BLE001 - any failure means fall back
            if backend == "zeroshot":
                import sys  # noqa: PLC0415

                print(
                    "warning: zero-shot scene tagging unavailable "
                    f"({type(exc).__name__}: {exc}); install the 'scene' extra "
                    "(`uv sync --extra scene`) and ensure the model downloads. "
                    "Falling back to the heuristic tagger.",
                    file=sys.stderr,
                )
    return SceneTagger()
