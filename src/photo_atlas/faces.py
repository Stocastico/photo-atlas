"""Face detection, embedding, clustering and identity matching.

The default backend (:class:`YuNetSFaceBackend`) is a modern deep-learning
pipeline built on OpenCV's DNN face stack: **YuNet** for detection and **SFace**
for 128-d recognition embeddings (cosine-comparable). The ONNX weights are
fetched on demand by :mod:`photo_atlas.models`. On real photographs SFace cleanly
separates identities -- same person ~0.05 cosine distance, different people
~0.9.

Alternative backends share the same :class:`FaceBackend` contract:

* :class:`DlibFaceBackend`     -- optional ``face_recognition`` (dlib) embeddings.
* :class:`SyntheticFaceBackend`-- detects the cartoon faces from the bundled demo
  (and powers the offline test-suite); not for real photographs.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, cast

import numpy as np


@dataclass
class FaceObservation:
    """A single detected face within an image."""

    bbox: tuple[int, int, int, int]  # x, y, w, h
    embedding: np.ndarray
    confidence: float = 1.0


class FaceBackend(Protocol):
    def detect(
        self, image_path: Path, image: np.ndarray | None = None
    ) -> list[FaceObservation]:
        ...


def pil_to_bgr(img) -> np.ndarray | None:
    """Convert an open RGB(A) Pillow image to an OpenCV-style BGR ``uint8`` array.

    Orientation is left raw (no EXIF transpose) so face bounding boxes line up
    with the indexer's face-crop geometry, which crops the un-transposed image.
    """

    rgb = np.asarray(img.convert("RGB"))
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        return None
    return np.ascontiguousarray(rgb[:, :, ::-1])  # RGB -> BGR


def _read_bgr(image_path: Path) -> np.ndarray | None:
    """Decode an image to an OpenCV-style BGR ``uint8`` array, or ``None``.

    ``cv2.imread`` is tried first (fast C path for JPEG/PNG/etc.), but OpenCV's
    bundled ``opencv-python-headless`` build can't decode HEIC/HEIF — the default
    iPhone format and ~a fifth of a typical library — so it silently returns
    ``None`` for those. We fall back to Pillow, which decodes HEIC once
    ``pillow-heif`` has registered its opener (see :mod:`photo_atlas.metadata`).
    """

    import cv2  # noqa: PLC0415

    image = cv2.imread(str(image_path))
    if image is not None:
        return image
    try:
        from PIL import Image  # noqa: PLC0415

        from . import metadata  # noqa: F401,PLC0415 - registers the HEIF opener

        with Image.open(image_path) as img:
            return pil_to_bgr(img)
    except Exception:
        return None


def _maybe_downscale(image: np.ndarray, max_side: int) -> tuple[np.ndarray, float]:
    """Shrink ``image`` so its long edge is ``max_side``; return ``(img, scale)``.

    Running YuNet on a full 12-megapixel frame is needlessly slow; detecting on a
    downscaled copy and mapping the boxes/landmarks back (multiply by ``1/scale``)
    is far cheaper with negligible accuracy loss. ``scale`` is ``detected/original``
    (``<= 1``); an image already within the cap is returned untouched at ``1.0``.
    """

    h, w = image.shape[:2]
    long_side = max(h, w)
    if long_side <= max_side:
        return image, 1.0
    import cv2  # noqa: PLC0415

    scale = max_side / float(long_side)
    resized = cv2.resize(
        image, (max(1, round(w * scale)), max(1, round(h * scale))),
        interpolation=cv2.INTER_AREA,
    )
    return resized, scale


def l2_normalize(vec: np.ndarray) -> np.ndarray:
    vec = np.asarray(vec, dtype=np.float32)
    norm = np.linalg.norm(vec)
    if norm < 1e-8:
        return vec
    return vec / norm


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    a = l2_normalize(a)
    b = l2_normalize(b)
    return float(1.0 - np.dot(a, b))


class YuNetSFaceBackend:
    """Deep face detection (YuNet) + recognition embeddings (SFace).

    Requires OpenCV (>=4.10, with the DNN face module) and the YuNet / SFace
    ONNX weights, which :func:`photo_atlas.models.ensure_models` downloads to the
    library cache on first use.
    """

    def __init__(
        self,
        model_dir: Path | None = None,
        *,
        download: bool = True,
        score_threshold: float = 0.85,
        nms_threshold: float = 0.3,
        detect_max_side: int = 1280,
    ):
        self._detect_max_side = detect_max_side
        import cv2  # noqa: PLC0415

        if not (hasattr(cv2, "FaceDetectorYN") and hasattr(cv2, "FaceRecognizerSF")):
            raise RuntimeError(
                "This OpenCV build lacks the DNN face module (FaceDetectorYN / "
                "FaceRecognizerSF). Install opencv-python>=4.10."
            )
        from .config import AtlasConfig  # noqa: PLC0415
        from .models import ensure_models  # noqa: PLC0415

        if model_dir is None:
            model_dir = AtlasConfig().home / "models"
        yunet_path, sface_path = ensure_models(Path(model_dir), download=download)

        self._cv2 = cv2
        self._detector = cv2.FaceDetectorYN.create(
            str(yunet_path), "", (320, 320),
            score_threshold=score_threshold, nms_threshold=nms_threshold,
        )
        self._recognizer = cv2.FaceRecognizerSF.create(str(sface_path), "")

    def detect(
        self, image_path: Path, image: np.ndarray | None = None
    ) -> list[FaceObservation]:
        if image is None:
            image = _read_bgr(image_path)
        if image is None:
            return []
        # Detect on a downscaled copy for speed, but align/embed on the full-res
        # frame for the best recognition quality.
        det_img, scale = _maybe_downscale(image, self._detect_max_side)
        dh, dw = det_img.shape[:2]
        self._detector.setInputSize((dw, dh))
        _, results = self._detector.detect(det_img)
        observations: list[FaceObservation] = []
        if results is None:
            return observations
        for raw_row in results:
            # cv2's stubs type each detection row as a scalar; it's really a
            # 1-D [x, y, w, h, 5×landmark, score] array.
            row = cast(np.ndarray, raw_row)
            score = float(row[-1])
            # Map the box + 5 landmarks (indices 0..13) back to full-res coords;
            # the trailing score (index 14) is left untouched.
            full_row = row.copy()
            if scale != 1.0:
                full_row[:14] = row[:14] / scale
            x, y, bw, bh = full_row[:4]
            aligned = self._recognizer.alignCrop(image, full_row)
            feature = self._recognizer.feature(aligned).flatten()
            observations.append(
                FaceObservation(
                    bbox=(int(x), int(y), int(bw), int(bh)),
                    embedding=l2_normalize(feature),
                    confidence=score,
                )
            )
        return observations


class SyntheticFaceBackend:
    """Detect the simple drawn faces produced by :mod:`photo_atlas.demo`.

    Haar cascades only fire on realistic facial texture, so they cannot detect
    cartoon faces. This backend instead finds skin-tone blobs and embeds each by
    its colour palette, giving deterministic, clusterable "identities". It is
    meant for the bundled demo and for tests -- not for real photographs.
    """

    def __init__(self, min_area: int = 1500, max_area_frac: float = 0.2):
        import cv2  # noqa: PLC0415

        self._cv2 = cv2
        self.min_area = min_area
        self.max_area_frac = max_area_frac

    def detect(
        self, image_path: Path, image: np.ndarray | None = None
    ) -> list[FaceObservation]:
        cv2 = self._cv2
        if image is None:
            image = _read_bgr(image_path)
        if image is None:
            return []
        ih, iw = image.shape[:2]
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        sat, val = hsv[..., 1], hsv[..., 2]
        # Faces use a moderate saturation; pastel scenes / documents are below,
        # vivid food / props are above, so a band cleanly isolates faces.
        mask = ((sat >= 45) & (sat <= 170) & (val > 120)).astype("uint8")
        n, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        observations: list[FaceObservation] = []
        for i in range(1, n):
            x, y, w, h, area = stats[i]
            if area < self.min_area or area > self.max_area_frac * iw * ih:
                continue
            if not (0.5 <= w / max(h, 1) <= 2.0):  # reject background rectangles
                continue
            sel = mask[y : y + h, x : x + w].astype(bool)
            if sel.sum() == 0:
                continue
            patch = rgb[y : y + h, x : x + w].astype(np.float32)
            mean_rgb = patch[sel].mean(axis=0)
            # Chroma direction (luminance removed) separates distinct hues well
            # under cosine distance, unlike near-grey absolute skin tones.
            embedding = l2_normalize(mean_rgb - mean_rgb.mean())
            observations.append(
                FaceObservation(bbox=(int(x), int(y), int(w), int(h)), embedding=embedding)
            )
        return observations


class DlibFaceBackend:  # pragma: no cover - optional heavy dependency
    """High quality 128-d embeddings via the ``face_recognition`` package."""

    def __init__(self):
        try:
            import face_recognition  # noqa: PLC0415
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "face_recognition is not installed. Install with "
                "`pip install 'photo-atlas[dlib]'`."
            ) from exc
        self._fr = face_recognition

    def detect(
        self, image_path: Path, image: np.ndarray | None = None
    ) -> list[FaceObservation]:
        fr = self._fr
        if image is not None:
            # ``image`` is BGR (OpenCV layout); face_recognition wants RGB.
            image = np.ascontiguousarray(image[:, :, ::-1])
        else:
            image = fr.load_image_file(str(image_path))
        locations = fr.face_locations(image)
        encodings = fr.face_encodings(image, locations)
        observations: list[FaceObservation] = []
        for (top, right, bottom, left), enc in zip(locations, encodings, strict=True):
            observations.append(
                FaceObservation(
                    bbox=(int(left), int(top), int(right - left), int(bottom - top)),
                    embedding=l2_normalize(np.asarray(enc, dtype=np.float32)),
                )
            )
        return observations


def get_backend(name: str = "auto", *, model_dir: Path | None = None) -> FaceBackend | None:
    """Return a face backend by name, or ``None`` if none is available.

    ``auto`` prefers the deep YuNet/SFace pipeline, then dlib.
    """

    order = {
        "auto": ["yunet", "dlib"],
        "yunet": ["yunet"],
        "dlib": ["dlib"],
        "synthetic": ["synthetic"],
        "none": [],
    }.get(name, ["yunet"])

    for choice in order:
        try:
            if choice == "yunet":
                return YuNetSFaceBackend(model_dir=model_dir)
            if choice == "dlib":
                return DlibFaceBackend()
            if choice == "synthetic":
                return SyntheticFaceBackend()
        except (RuntimeError, ModuleNotFoundError, ImportError, OSError):
            continue
    return None


# -- clustering & recognition ---------------------------------------------
def cluster_embeddings(
    embeddings: list[np.ndarray], eps: float = 0.28, min_samples: int = 2
) -> list[int]:
    """Cluster face embeddings with DBSCAN over cosine distance.

    Returns a list of cluster ids aligned with ``embeddings``; ``-1`` marks
    noise (faces that did not join any cluster).
    """

    if not embeddings:
        return []
    if len(embeddings) == 1:
        return [-1]

    from sklearn.cluster import DBSCAN  # noqa: PLC0415

    matrix = np.vstack([l2_normalize(e) for e in embeddings])
    # cosine distance = 1 - cosine similarity; vectors are unit norm.
    similarity = np.clip(matrix @ matrix.T, -1.0, 1.0)
    distance = 1.0 - similarity
    np.fill_diagonal(distance, 0.0)
    labels = DBSCAN(eps=eps, min_samples=min_samples, metric="precomputed").fit_predict(distance)
    return [int(x) for x in labels]


def best_person_match(
    embedding: np.ndarray, centroids: dict[int, np.ndarray], threshold: float
) -> tuple[int | None, float]:
    """Find the closest person centroid within ``threshold`` cosine distance."""

    best_id: int | None = None
    best_dist = threshold
    for person_id, centroid in centroids.items():
        dist = cosine_distance(embedding, centroid)
        if dist <= best_dist:
            best_dist = dist
            best_id = person_id
    confidence = max(0.0, 1.0 - best_dist) if best_id is not None else 0.0
    return best_id, confidence
