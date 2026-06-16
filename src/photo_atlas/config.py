"""Runtime configuration for Photo Atlas.

All long lived artefacts (the SQLite catalog, generated thumbnails and face
crops) live under a single *library directory*. By default this is
``~/.photo_atlas`` but it can be overridden through the ``PHOTO_ATLAS_HOME``
environment variable or an explicit :class:`AtlasConfig`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def default_home() -> Path:
    """Return the library directory, honouring ``PHOTO_ATLAS_HOME``."""

    env = os.environ.get("PHOTO_ATLAS_HOME")
    if env:
        return Path(env).expanduser()
    return Path.home() / ".photo_atlas"


@dataclass
class AtlasConfig:
    """Filesystem layout and tunable thresholds for a library."""

    home: Path = field(default_factory=default_home)

    #: Edge size (px) of the long side of generated thumbnails.
    thumb_size: int = 320

    #: Edge size (px) of the long side of lightbox preview derivatives. The
    #: full-resolution original is still served on demand for download.
    preview_size: int = 1600

    #: Max geotagged points returned to the map in one response. Bounds the
    #: payload on big libraries; the default comfortably covers a 15-year
    #: personal collection (the old 20k cap could clip an iPhone library).
    map_point_limit: int = 50000

    #: Face matches closer than this cosine distance are considered the same
    #: identity when auto-recognising newly indexed photos. Tuned for SFace,
    #: where same-person pairs sit near ~0.1-0.4 and different people near ~0.9.
    face_match_threshold: float = 0.5

    #: Number of nearest enrolled faces consulted when auto-recognising a new
    #: face (k-NN majority vote). More robust than a single per-person centroid
    #: when a person's appearance drifts over the years.
    recognition_k: int = 5

    #: Scene tagger: ``heuristic`` (colour/brightness, zero deps, the default),
    #: ``zeroshot`` (SigLIP ONNX zero-shot; needs the ``scene`` extra and warns +
    #: falls back if unavailable) or ``auto`` (zero-shot when it loads, else
    #: silently the heuristic).
    scene_backend: str = "heuristic"

    #: Logit temperature applied to the zero-shot cosine similarities before the
    #: softmax. Higher = sharper label probabilities.
    scene_temperature: float = 50.0

    #: Logit (in cosine space) of the catch-all ``other`` label for the zero-shot
    #: tagger. Frames whose best concrete-label similarity does not clear this are
    #: tagged ``other``. Raise it to send more borderline frames to ``other``.
    scene_other_bias: float = -0.02

    #: Max number of photos returned by a natural-language semantic search,
    #: ranked by relevance. Semantic search ranks the whole (filtered) library, so
    #: this caps the result to the most relevant matches instead of trailing off
    #: into thousands of irrelevant photos.
    semantic_top_k: int = 200

    #: DBSCAN epsilon (cosine distance) used when clustering unknown faces.
    cluster_eps: float = 0.5

    #: Minimum number of faces required to form a cluster.
    cluster_min_samples: int = 2

    @property
    def models_dir(self) -> Path:
        return self.home / "models"

    def __post_init__(self) -> None:
        self.home = Path(self.home).expanduser()

    # -- derived paths -----------------------------------------------------
    @property
    def db_path(self) -> Path:
        return self.home / "atlas.db"

    @property
    def thumbs_dir(self) -> Path:
        return self.home / "thumbs"

    @property
    def faces_dir(self) -> Path:
        return self.home / "faces"

    @property
    def previews_dir(self) -> Path:
        return self.home / "previews"

    def ensure_dirs(self) -> AtlasConfig:
        """Create the library directory tree if it does not exist yet."""

        self.home.mkdir(parents=True, exist_ok=True)
        self.thumbs_dir.mkdir(parents=True, exist_ok=True)
        self.faces_dir.mkdir(parents=True, exist_ok=True)
        self.previews_dir.mkdir(parents=True, exist_ok=True)
        return self
