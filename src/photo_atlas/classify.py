"""Scene tagging.

A photo library benefits from a coarse "what is this a picture of" tag. Photo
Atlas tags scenes with a single :class:`ZeroShotSceneTagger`: it runs a small
**SigLIP** vision encoder (ONNX, via ``onnxruntime``) and compares the image
embedding against pre-computed *text* embeddings for each label. SigLIP is a
modern CLIP successor that beats CLIP at zero-shot for its size; we run only its
vision encoder at index time and ship the label (text) embeddings as a tiny
bundled matrix (see ``scripts/build_scene_embeddings.py``), so there is no
PyTorch, no text encoder and no tokenizer in the runtime — just the ONNX vision
model (downloaded on demand like the face models) plus NumPy.

The tagger satisfies a ``tag(path, face_count) -> (label, scores)`` contract that
the indexer depends on; tests drive it with hand-built vectors / stub encoders so
the default suite needs no model download.

Categories: ``people``, ``animals``, ``landscape``, ``plants``, ``food``,
``vehicle``, ``building``, ``document``, ``screenshot``, ``other``.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Protocol

import numpy as np
from PIL import Image

from .embed import preprocess_image as _preprocess  # noqa: F401 - back-compat re-export

if TYPE_CHECKING:
    from .config import AtlasConfig
    from .embed import SigLipImageEncoder

SCENE_LABELS = [
    "people",
    "animals",
    "landscape",
    "plants",
    "food",
    "vehicle",
    "building",
    "document",
    "screenshot",
    "other",
]
#: ``other`` is the catch-all fallback, not a thing we can write a text prompt
#: for, so the zero-shot label matrix only covers the concrete labels.
OTHER_LABEL = "other"


class Tagger(Protocol):
    """The contract the tagger satisfies (and that the indexer depends on)."""

    def tag(self, path: Path, face_count: int = 0) -> tuple[str, dict[str, float]]:
        ...

    def tag_image(
        self, img: Image.Image, face_count: int = 0
    ) -> tuple[str, dict[str, float]]:
        ...



# -- zero-shot (SigLIP) tagger --------------------------------------------
#: A detected face is a strong "people" signal; nudge that label's similarity by
#: this much (in cosine space) when faces are present, without steamrolling a
#: clear food/landscape signal. Image preprocessing + the vision encoder itself
#: live in :mod:`photo_atlas.embed` (shared with semantic search).
_FACE_BONUS = 0.04


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

    vec = embedding.astype(np.float32).reshape(-1)
    # Guard the space mismatch: if the image embedding's dimension doesn't match the
    # label matrix, the bundled scene_labels.npz was built for a different model and
    # wasn't rebuilt after the swap. Fail with an actionable message rather than a
    # cryptic numpy matmul error (or, worse, silently-wrong tags when dims happen to
    # line up — see SIGLIP2_MIGRATION.md Gap 4).
    if vec.shape[0] != matrix.shape[1]:
        raise ValueError(
            f"scene-label matrix dim ({matrix.shape[1]}) != image embedding dim "
            f"({vec.shape[0]}); rebuild data/scene_labels.npz for the current model "
            "with scripts/build_scene_embeddings.py."
        )
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
    """SigLIP zero-shot tagger: vision-encoder ONNX + bundled label embeddings.

    The vision encoder is a :class:`photo_atlas.embed.SigLipImageEncoder` held as
    the public ``encoder`` attribute. The indexer can therefore share one encoder
    (one ONNX session, one inference per photo) between scene tagging and semantic
    search instead of running the same vision tower twice.
    """

    def __init__(
        self,
        model_path: Path | None = None,
        label_path: Path | None = None,
        *,
        encoder: SigLipImageEncoder | None = None,
        temperature: float = 50.0,
        other_bias: float = -0.02,
    ):
        from .embed import SigLipImageEncoder  # noqa: PLC0415

        if encoder is None:
            if model_path is None:
                raise ValueError("ZeroShotSceneTagger needs a model_path or an encoder")
            encoder = SigLipImageEncoder(model_path)
        self.encoder = encoder
        if label_path is None:
            label_path = scene_labels_path()
        data = np.load(label_path, allow_pickle=True)
        self._labels = [str(x) for x in data["labels"]]
        self._matrix = np.asarray(data["matrix"], dtype=np.float32)
        # Which model/space the bundled label matrix was built for, surfaced so a
        # stale matrix (forgotten rebuild after a model swap) is diagnosable. The
        # actual enforcement is the dim check in ``classify_embedding``.
        self.label_dim = int(self._matrix.shape[1])
        self.label_model = str(data["model"]) if "model" in data else None
        self._temperature = temperature
        self._other_bias = other_bias

    @classmethod
    def from_config(
        cls, config: AtlasConfig, *, encoder: SigLipImageEncoder | None = None
    ) -> ZeroShotSceneTagger:
        """Build from an :class:`AtlasConfig`, fetching the model if needed.

        ``encoder`` lets the caller inject an already-loaded vision encoder so the
        scene tagger and semantic-search embedding share a single ONNX session.
        """

        from .embed import SigLipImageEncoder  # noqa: PLC0415

        if encoder is None:
            encoder = SigLipImageEncoder.from_config(config)
        return cls(
            label_path=scene_labels_path(),
            encoder=encoder,
            temperature=config.scene_temperature,
            other_bias=config.scene_other_bias,
        )

    def tag(self, path: Path, face_count: int = 0) -> tuple[str, dict[str, float]]:
        with Image.open(path) as img:
            return self.tag_image(img, face_count)

    def tag_image(
        self, img: Image.Image, face_count: int = 0
    ) -> tuple[str, dict[str, float]]:
        return self.tag_embedding(self.encoder.embed_image(img), face_count)

    def tag_embedding(
        self, embedding: np.ndarray, face_count: int = 0
    ) -> tuple[str, dict[str, float]]:
        """Tag from a precomputed SigLIP image embedding (no re-inference)."""

        return classify_embedding(
            embedding,
            self._matrix,
            self._labels,
            temperature=self._temperature,
            other_bias=self._other_bias,
            face_count=face_count,
        )


def scene_labels_path() -> Path:
    """Path to the bundled label-embedding matrix."""

    return Path(__file__).resolve().parent / "data" / "scene_labels.npz"


def get_tagger(
    config: AtlasConfig, *, encoder: SigLipImageEncoder | None = None
) -> Tagger:
    """Return the SigLIP zero-shot scene tagger (the only tagger).

    Its vision model downloads on demand (cached under ``~/.photo_atlas/models``,
    overridable via ``PHOTO_ATLAS_SCENE_MODEL``). ``encoder`` injects an
    already-loaded vision encoder so scene tagging and semantic-search embedding
    can share a single ONNX session — one vision inference per photo for both.
    """

    return ZeroShotSceneTagger.from_config(config, encoder=encoder)
