"""Deep-learning model management.

Photo Atlas uses OpenCV's DNN face stack -- **YuNet** for detection and
**SFace** for 128-d recognition embeddings. The ONNX weights live in the OpenCV
Zoo; they are small enough to fetch on demand and are cached inside the library
directory (never committed to git).

Set ``PHOTO_ATLAS_YUNET`` / ``PHOTO_ATLAS_SFACE`` to use local model files and
skip downloading (handy for offline / air-gapped setups).
"""

from __future__ import annotations

import os
import urllib.request
from pathlib import Path

ZOO = "https://github.com/opencv/opencv_zoo/raw/main/models"

YUNET_NAME = "face_detection_yunet_2023mar.onnx"
SFACE_NAME = "face_recognition_sface_2021dec.onnx"

YUNET_URL = f"{ZOO}/face_detection_yunet/{YUNET_NAME}"
SFACE_URL = f"{ZOO}/face_recognition_sface/{SFACE_NAME}"

# Zero-shot scene tagging (opt-in): the SigLIP *vision* encoder, exported to
# ONNX and quantised (~95 MB) by the transformers.js project. Only the vision
# tower runs at index time; the matching text (label) embeddings are pre-baked
# into the bundled ``data/scene_labels.npz`` (see scripts/build_scene_embeddings.py).
SCENE_NAME = "siglip_base_patch16_224_vision_quantized.onnx"
SCENE_URL = (
    "https://huggingface.co/Xenova/siglip-base-patch16-224/resolve/main/"
    "onnx/vision_model_quantized.onnx"
)

# Semantic search additionally needs the matching SigLIP *text* tower + tokenizer
# to embed a free-text query at runtime (the scene-label text embeddings are
# pre-baked, but an arbitrary query can't be). Same model/space as the vision
# tower above so image and text embeddings are comparable.
SCENE_TEXT_NAME = "siglip_base_patch16_224_text_quantized.onnx"
SCENE_TEXT_URL = (
    "https://huggingface.co/Xenova/siglip-base-patch16-224/resolve/main/"
    "onnx/text_model_quantized.onnx"
)
SCENE_TOKENIZER_NAME = "siglip_base_patch16_224_tokenizer.json"
SCENE_TOKENIZER_URL = (
    "https://huggingface.co/Xenova/siglip-base-patch16-224/resolve/main/tokenizer.json"
)

# A sanity floor: the face weights are far larger (YuNet ~230 KB, SFace ~37 MB)
# and the scene vision tower ~95 MB, so anything tiny is a truncated download or
# an error page, not a model.
_MIN_MODEL_BYTES = 50_000


def _resolve(name: str, url: str, model_dir: Path, env: str, download: bool) -> Path:
    override = os.environ.get(env)
    if override:
        p = Path(override).expanduser()
        if not p.exists():
            raise FileNotFoundError(f"{env} points to a missing file: {p}")
        return p

    dest = model_dir / name
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    if not download:
        raise FileNotFoundError(f"Model {name} not found in {model_dir} and download disabled")

    model_dir.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    _, headers = urllib.request.urlretrieve(url, tmp)  # noqa: S310 - trusted Zoo URL

    # Guard against a truncated/interrupted download caching a corrupt ONNX:
    # the partial file must be non-trivial and, when the server reports a
    # Content-Length, match it exactly. On any mismatch, discard and fail.
    size = tmp.stat().st_size
    expected = headers.get("Content-Length") if headers else None
    if size < _MIN_MODEL_BYTES or (expected is not None and int(expected) != size):
        tmp.unlink(missing_ok=True)
        raise RuntimeError(
            f"Download of {name} looks incomplete ({size} bytes"
            + (f", expected {expected}" if expected is not None else "")
            + "); please retry."
        )
    tmp.replace(dest)
    return dest


def ensure_models(model_dir: Path, download: bool = True) -> tuple[Path, Path]:
    """Return ``(yunet_path, sface_path)``, downloading them if needed."""

    model_dir = Path(model_dir)
    yunet = _resolve(YUNET_NAME, YUNET_URL, model_dir, "PHOTO_ATLAS_YUNET", download)
    sface = _resolve(SFACE_NAME, SFACE_URL, model_dir, "PHOTO_ATLAS_SFACE", download)
    return yunet, sface


def ensure_scene_model(model_dir: Path, download: bool = True) -> Path:
    """Return the SigLIP vision-encoder ONNX path, downloading it if needed.

    Set ``PHOTO_ATLAS_SCENE_MODEL`` to a local file to skip the download
    (offline / air-gapped, or to swap in a different vision encoder whose label
    matrix you have rebuilt with ``scripts/build_scene_embeddings.py``).
    """

    return _resolve(SCENE_NAME, SCENE_URL, Path(model_dir), "PHOTO_ATLAS_SCENE_MODEL", download)


def ensure_scene_text_model(model_dir: Path, download: bool = True) -> Path:
    """Return the SigLIP *text*-encoder ONNX path, downloading it if needed.

    Used only by semantic search (to embed a free-text query). Override with
    ``PHOTO_ATLAS_SCENE_TEXT_MODEL`` for an offline / swapped-in model.
    """

    return _resolve(
        SCENE_TEXT_NAME, SCENE_TEXT_URL, Path(model_dir), "PHOTO_ATLAS_SCENE_TEXT_MODEL", download
    )


def ensure_scene_tokenizer(model_dir: Path, download: bool = True) -> Path:
    """Return the SigLIP tokenizer (``tokenizer.json``) path, downloading if needed.

    Override with ``PHOTO_ATLAS_SCENE_TOKENIZER`` for an offline / swapped model.
    """

    return _resolve(
        SCENE_TOKENIZER_NAME,
        SCENE_TOKENIZER_URL,
        Path(model_dir),
        "PHOTO_ATLAS_SCENE_TOKENIZER",
        download,
    )
