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
from dataclasses import dataclass
from pathlib import Path

ZOO = "https://github.com/opencv/opencv_zoo/raw/main/models"

YUNET_NAME = "face_detection_yunet_2026may.onnx"
SFACE_NAME = "face_recognition_sface_2021dec.onnx"

YUNET_URL = f"{ZOO}/face_detection_yunet/{YUNET_NAME}"
SFACE_URL = f"{ZOO}/face_recognition_sface/{SFACE_NAME}"

# Face recognition default: **ArcFace R100** (glint360k, 512-d) — the InsightFace
# antelopev2 recognition tower, mirrored as a standalone ONNX by the immich
# project. Stronger verification accuracy than SFace (sharper "same person across
# 15 years"); it pairs with YuNet detection via a 5-point similarity alignment to
# ArcFace's canonical 112² template (see faces.YuNetArcFaceBackend). The 512-d
# embedding is stored per-row (faces.dim), so no schema change is needed. Override
# with ``PHOTO_ATLAS_ARCFACE`` for an offline / swapped-in recogniser.
ARCFACE_NAME = "arcface_glint360k_r100.onnx"
# Pinned to a commit (the mirror is a third party's; ``main`` could change or vanish)
# and verified by SHA-256 below.
ARCFACE_REVISION = "51ca7e153bccebdbae224779e23c9c243e2fec85"
ARCFACE_URL = (
    f"https://huggingface.co/immich-app/antelopev2/resolve/{ARCFACE_REVISION}/recognition/model.onnx"
)

# Zero-shot scene tagging: the SigLIP 2 *vision* encoder, exported to ONNX and
# quantised (~90 MB) by the onnx-community project. Only the vision tower runs at
# index time; the matching text (label) embeddings are pre-baked into the bundled
# ``data/scene_labels.npz`` (see scripts/build_scene_embeddings.py). SigLIP 2 is a
# drop-in architecture upgrade over the original SigLIP base — same dual-encoder /
# 768-dim space, strictly better zero-shot + retrieval (see SIGLIP2_MIGRATION.md).
SCENE_HF_REPO = "onnx-community/siglip2-base-patch16-256-ONNX"
#: Commit of that repo the checksums below were taken from; bump both together.
SCENE_HF_REVISION = "d1114256522a37ffa257a0a58017348ab0058db2"
_SCENE_BASE = f"https://huggingface.co/{SCENE_HF_REPO}/resolve/{SCENE_HF_REVISION}"
SCENE_NAME = "siglip2_base_patch16_256_vision_quantized.onnx"
SCENE_URL = f"{_SCENE_BASE}/onnx/vision_model_quantized.onnx"
#: SigLIP 2's vision ONNX advertises a fully *dynamic* input shape but is trained
#: at a fixed resolution (256² for base-patch16-256), so it can't be auto-detected
#: from the model and must be configured. Override for a 384/512 variant via
#: ``PHOTO_ATLAS_SCENE_INPUT_SIZE``.
SCENE_INPUT_SIZE = 256

# Semantic search additionally needs the matching SigLIP *text* tower + tokenizer
# to embed a free-text query at runtime (the scene-label text embeddings are
# pre-baked, but an arbitrary query can't be). Same model/space as the vision
# tower above so image and text embeddings are comparable. SigLIP 2's text tower
# uses the Gemma tokenizer (~33 MB) — the encoder reads its embedded pad config.
SCENE_TEXT_NAME = "siglip2_base_patch16_256_text_quantized.onnx"
SCENE_TEXT_URL = f"{_SCENE_BASE}/onnx/text_model_quantized.onnx"
SCENE_TOKENIZER_NAME = "siglip2_base_patch16_256_tokenizer.json"
SCENE_TOKENIZER_URL = f"{_SCENE_BASE}/tokenizer.json"

#: Expected SHA-256 per downloaded file (the Hugging Face LFS object ids), checked
#: right after download so a truncated, tampered or silently-changed model is
#: rejected instead of cached. YuNet/SFace come from the OpenCV Zoo's moving
#: ``main`` branch and are not pinned here (add their digests when you can fetch
#: them); an unpinned model is only size-checked.
MODEL_SHA256: dict[str, str] = {
    ARCFACE_NAME: "4ab1d6435d639628a6f3e5008dd4f929edf4c4124b1a7169e1048f9fef534cdf",
    SCENE_NAME: "f2eb8ccfa3dc0b3761d9ea9a39554fe0f2be71b247ad7f68a80720ec88895650",
    SCENE_TEXT_NAME: "6f59b39d880c413042314b79302b74d0dd93b273caf8fbfdb1eb2df61a7fefd4",
    SCENE_TOKENIZER_NAME: "cb9140fae3ac5122c972d37adf83e1248471a38147ad76f8215c8872c6fd8322",
}

# A sanity floor: the face weights are far larger (YuNet ~230 KB, SFace ~37 MB)
# and the scene vision tower ~95 MB, so anything tiny is a truncated download or
# an error page, not a model.
_MIN_MODEL_BYTES = 50_000


_PROVIDER_ALIASES = {
    "cpu": "CPUExecutionProvider",
    "cuda": "CUDAExecutionProvider",
    "dml": "DmlExecutionProvider",
    "directml": "DmlExecutionProvider",
    "coreml": "CoreMLExecutionProvider",
}


def onnx_providers() -> list[str]:
    """Execution providers for onnxruntime sessions, best first, CPU always last.

    Auto mode picks CUDA / DirectML when the installed onnxruntime offers them and
    otherwise stays on CPU. CoreML is opt-in (it misbehaves on Intel Macs). Set
    ``PHOTO_ATLAS_PROVIDERS`` (comma list: ``cpu``, ``cuda``, ``dml``, ``coreml``) to
    force a choice; names the runtime doesn't offer are dropped. With parallel
    indexing every worker loads its own session, so on a GPU consider ``--workers``
    small (or ``PHOTO_ATLAS_PROVIDERS=cpu``).
    """

    import onnxruntime as ort  # noqa: PLC0415

    available = set(ort.get_available_providers())
    override = os.environ.get("PHOTO_ATLAS_PROVIDERS")
    if override:
        wanted = [_PROVIDER_ALIASES.get(p.strip().lower(), "") for p in override.split(",")]
    else:
        wanted = ["CUDAExecutionProvider", "DmlExecutionProvider"]
    chosen = [p for p in wanted if p and p in available and p != "CPUExecutionProvider"]
    return [*chosen, "CPUExecutionProvider"]


# -- evaluation candidates (not used by the app unless you switch to them) ----
@dataclass(frozen=True)
class RemoteFile:
    """A model file hosted on Hugging Face, pinned to a commit and a SHA-256."""

    filename: str  # local name inside the models dir (unique across presets)
    url: str
    sha256: str


@dataclass(frozen=True)
class FacePreset:
    note: str
    model: RemoteFile


@dataclass(frozen=True)
class SearchPreset:
    note: str
    vision: RemoteFile
    text: RemoteFile
    input_size: int  # square input resolution of the vision tower


#: Face recognisers that drop into the default pipeline (112² ArcFace-style, 512-d).
FACE_PRESETS: dict[str, FacePreset] = {
    "auraface": FacePreset(
        "AuraFace v1 (ResNet100, Apache-2.0): the permissive-licence alternative to the "
        "InsightFace weights; its model card reports lower CFP-FP accuracy.",
        RemoteFile(
        "auraface_v1_glintr100.onnx",
        "https://huggingface.co/fal/AuraFace-v1/resolve/af6d057c9b0ec4071d4c49c80e3539258798b609/glintr100.onnx",
        "a7933ea5330113b01c9b60351d8f4c33003f145d8470ac5f0e52ee2effe25c60",
    ),
    ),
}

#: SigLIP 2 variants that share the default tokenizer. Switching the app to one also
#: needs its label matrix rebuilt (scripts/build_scene_embeddings.py) and a re-embed.
SEARCH_PRESETS: dict[str, SearchPreset] = {
    "siglip2-base-256-fp16": SearchPreset(
        "Same model as the default but fp16 instead of int8: does quantisation cost accuracy?",
        RemoteFile(
        "siglip2_base_256_vision_fp16.onnx",
        "https://huggingface.co/onnx-community/siglip2-base-patch16-256-ONNX/resolve/d1114256522a37ffa257a0a58017348ab0058db2/onnx/vision_model_fp16.onnx",
        "fe9ad8020a6d3d98d394c9be8f07064066135fc2f87ec11692de0b677c0ac4db",
    ),
        RemoteFile(
        "siglip2_base_256_text_fp16.onnx",
        "https://huggingface.co/onnx-community/siglip2-base-patch16-256-ONNX/resolve/d1114256522a37ffa257a0a58017348ab0058db2/onnx/text_model_fp16.onnx",
        "80954edffdc689599e5d5bc6a1738380bc9e8139a18e5c8892485f248b6b4890",
    ),
        256,
    ),
    "siglip2-base-384": SearchPreset(
        "Base model at 384 px input (int8): sharper detail for ~the same weights.",
        RemoteFile(
        "siglip2_base_384_vision_int8.onnx",
        "https://huggingface.co/onnx-community/siglip2-base-patch16-384-ONNX/resolve/99f5d40afb5331b81eecee4bb02c360fbb9cdf8a/onnx/vision_model_quantized.onnx",
        "9f354d80bbb46b319f46f9e9d0e24ae523e3eb5921aac3d34979d74dbd0db6cd",
    ),
        RemoteFile(
        "siglip2_base_384_text_int8.onnx",
        "https://huggingface.co/onnx-community/siglip2-base-patch16-384-ONNX/resolve/99f5d40afb5331b81eecee4bb02c360fbb9cdf8a/onnx/text_model_quantized.onnx",
        "30a94b43b9ddc39c04ab67f80cba4b808262682b2fa1a2a8b924bf1e7bd84c61",
    ),
        384,
    ),
    "siglip2-large-384": SearchPreset(
        "Large (int8, 384 px): ~3.4x the vision weights of the default.",
        RemoteFile(
        "siglip2_large_384_vision_int8.onnx",
        "https://huggingface.co/onnx-community/siglip2-large-patch16-384-ONNX/resolve/257daa8df5100fdd0add12f4c2187aebf0beb409/onnx/vision_model_quantized.onnx",
        "563a51bb7d04797e669d705176a2275c68ea0d2a08faf8bd08e24601e02b9df3",
    ),
        RemoteFile(
        "siglip2_large_384_text_int8.onnx",
        "https://huggingface.co/onnx-community/siglip2-large-patch16-384-ONNX/resolve/257daa8df5100fdd0add12f4c2187aebf0beb409/onnx/text_model_quantized.onnx",
        "d01fa21e4b9ab51cf7cdad55d42def718895efefee3e503e274bf2445fd3b4cc",
    ),
        384,
    ),
    "siglip2-so400m-384": SearchPreset(
        "So400m (int8, 384 px): the biggest practical SigLIP 2; slowest to index.",
        RemoteFile(
        "siglip2_so400m_384_vision_int8.onnx",
        "https://huggingface.co/onnx-community/siglip2-so400m-patch14-384-ONNX/resolve/8ecf3f2bbb135bec33f4bb721c11479bf6a8a254/onnx/vision_model_quantized.onnx",
        "f078130c78e4666a51ddf2dc00203385a584dce8388782016e16c67ca7dac27d",
    ),
        RemoteFile(
        "siglip2_so400m_384_text_int8.onnx",
        "https://huggingface.co/onnx-community/siglip2-so400m-patch14-384-ONNX/resolve/8ecf3f2bbb135bec33f4bb721c11479bf6a8a254/onnx/text_model_quantized.onnx",
        "3423f71ad99e102bf3c389fb0ebafd181faec6dedef9e28ddf17ecc6852e0397",
    ),
        384,
    ),
}

for _p in (
    *(fp.model for fp in FACE_PRESETS.values()),
    *(f for sp in SEARCH_PRESETS.values() for f in (sp.vision, sp.text)),
):
    MODEL_SHA256[_p.filename] = _p.sha256


def fetch_file(remote: RemoteFile, model_dir: Path) -> Path:
    """Download (or reuse) a preset file into ``model_dir``, verifying its SHA-256."""

    return _resolve(remote.filename, remote.url, Path(model_dir), "PHOTO_ATLAS_NO_OVERRIDE", True)


def _sha256_of(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
    expected_sha = MODEL_SHA256.get(name)
    if expected_sha is not None and _sha256_of(tmp) != expected_sha:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(
            f"Download of {name} failed its checksum (SHA-256 mismatch); the file was "
            "discarded. Retry, or if the upstream file legitimately changed, update "
            "models.MODEL_SHA256."
        )
    tmp.replace(dest)
    return dest


def ensure_yunet(model_dir: Path, download: bool = True) -> Path:
    """Return the YuNet detector ONNX path, downloading it if needed."""

    return _resolve(YUNET_NAME, YUNET_URL, Path(model_dir), "PHOTO_ATLAS_YUNET", download)


def ensure_arcface(model_dir: Path, download: bool = True) -> Path:
    """Return the ArcFace R100 recognition ONNX path, downloading it if needed.

    Override with ``PHOTO_ATLAS_ARCFACE`` for an offline / swapped-in recogniser.
    """

    return _resolve(ARCFACE_NAME, ARCFACE_URL, Path(model_dir), "PHOTO_ATLAS_ARCFACE", download)


def ensure_models(model_dir: Path, download: bool = True) -> tuple[Path, Path]:
    """Return ``(yunet_path, sface_path)`` for the legacy SFace backend."""

    model_dir = Path(model_dir)
    yunet = ensure_yunet(model_dir, download)
    sface = _resolve(SFACE_NAME, SFACE_URL, model_dir, "PHOTO_ATLAS_SFACE", download)
    return yunet, sface


def ensure_arcface_models(model_dir: Path, download: bool = True) -> tuple[Path, Path]:
    """Return ``(yunet_path, arcface_path)`` for the default YuNet+ArcFace backend."""

    model_dir = Path(model_dir)
    return ensure_yunet(model_dir, download), ensure_arcface(model_dir, download)


def ensure_scene_model(model_dir: Path, download: bool = True) -> Path:
    """Return the SigLIP vision-encoder ONNX path, downloading it if needed.

    Set ``PHOTO_ATLAS_SCENE_MODEL`` to a local file to skip the download
    (offline / air-gapped, or to swap in a different vision encoder whose label
    matrix you have rebuilt with ``scripts/build_scene_embeddings.py``).
    """

    return _resolve(SCENE_NAME, SCENE_URL, Path(model_dir), "PHOTO_ATLAS_SCENE_MODEL", download)


def scene_model_id() -> str:
    """Identifier of the vision model that produces photo embeddings (its file name).

    Honours ``PHOTO_ATLAS_SCENE_MODEL`` without needing the file to exist, so it can
    be compared against the catalog's stamp before anything is downloaded.
    """

    override = os.environ.get("PHOTO_ATLAS_SCENE_MODEL")
    return Path(override).expanduser().name if override else SCENE_NAME


def ensure_scene_input_size() -> int:
    """Return the vision encoder's expected square input resolution.

    Defaults to :data:`SCENE_INPUT_SIZE` (256 for SigLIP 2 base-patch16-256) and is
    overridable via ``PHOTO_ATLAS_SCENE_INPUT_SIZE`` to match a swapped-in higher-
    resolution variant (e.g. 384/512), since SigLIP 2's ONNX advertises a dynamic
    shape that can't be probed for the trained resolution.
    """

    override = os.environ.get("PHOTO_ATLAS_SCENE_INPUT_SIZE")
    return int(override) if override else SCENE_INPUT_SIZE


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
