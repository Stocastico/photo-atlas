"""Face detection, embedding, clustering and identity matching.

The default backend (:class:`YuNetArcFaceBackend`) is a modern deep-learning
pipeline: **YuNet** (OpenCV DNN) for detection and **ArcFace R100** (glint360k,
512-d, via ``onnxruntime``) for recognition embeddings (cosine-comparable). YuNet's
5 landmarks drive a similarity alignment to ArcFace's canonical 112² template. The
ONNX weights are fetched on demand by :mod:`photo_atlas.models`. On real
photographs ArcFace cleanly separates identities -- same person ~0.03 cosine
distance, different people ~1.0.

Alternative backends share the same :class:`FaceBackend` contract:

* :class:`YuNetSFaceBackend`   -- legacy YuNet + SFace (128-d) recogniser
  (``faces=sface``); OpenCV-only, no ``onnxruntime`` recogniser session.
* :class:`DlibFaceBackend`     -- optional ``face_recognition`` (dlib) embeddings.
* :class:`SyntheticFaceBackend`-- detects the cartoon faces from the bundled demo
  (and powers the offline test-suite); not for real photographs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
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

    No EXIF transpose is applied here; the caller is responsible for passing an
    already-upright image. The indexer applies ``exif_transpose`` once up front so
    detection, face crops and the thumbnail all share one consistent orientation.
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


# -- ArcFace alignment + preprocessing ------------------------------------
#: The canonical ArcFace 5-point destination template at 112² (eyes, nose,
#: mouth corners), ordered image-left-eye, image-right-eye, nose, image-left
#: mouth, image-right mouth — which matches YuNet's landmark order exactly, so the
#: detector's points map straight onto it with no reordering.
_ARCFACE_TEMPLATE_112 = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float32,
)
#: ArcFace normalises pixels to roughly [-1, 1] with mean/std 127.5.
_ARCFACE_MEAN = 127.5
_ARCFACE_STD = 127.5


def arcface_template(size: int = 112) -> np.ndarray:
    """Return the ArcFace 5-point alignment template scaled to a ``size``² crop."""

    return (_ARCFACE_TEMPLATE_112 * (size / 112.0)).astype(np.float32)


def norm_crop(image: np.ndarray, landmarks: np.ndarray, size: int = 112) -> np.ndarray:
    """Align a face to ArcFace's canonical crop via a 5-point similarity transform.

    ``landmarks`` is the ``(5, 2)`` set of detected points (YuNet's order); a
    similarity transform maps them onto :func:`arcface_template`, the standard
    InsightFace ``norm_crop`` that ArcFace expects.
    """

    import cv2  # noqa: PLC0415

    src = np.asarray(landmarks, dtype=np.float32).reshape(5, 2)
    matrix, _ = cv2.estimateAffinePartial2D(src, arcface_template(size), method=cv2.LMEDS)
    return cv2.warpAffine(image, matrix, (size, size), borderValue=(0.0, 0.0, 0.0))


def arcface_preprocess(aligned_bgr: np.ndarray) -> np.ndarray:
    """Turn an aligned ``HxWx3`` BGR crop into ArcFace's ``(1, 3, H, W)`` blob.

    ArcFace expects RGB in ``[-1, 1]``; the aligned crop arrives BGR (OpenCV), so
    the channels are swapped and the pixels are scaled ``(x - 127.5) / 127.5``.
    """

    rgb = np.asarray(aligned_bgr)[:, :, ::-1].astype(np.float32)
    rgb = (rgb - _ARCFACE_MEAN) / _ARCFACE_STD
    return np.ascontiguousarray(rgb.transpose(2, 0, 1)[None])


class _YuNetBackend:
    """Shared YuNet detection; subclasses supply the recognition ``_embed``.

    Detection runs on a downscaled copy for speed, with boxes/landmarks mapped
    back to full-res coordinates so the recogniser aligns/embeds on the best
    available pixels.
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

        if not hasattr(cv2, "FaceDetectorYN"):
            raise RuntimeError(
                "This OpenCV build lacks the DNN face module (FaceDetectorYN). "
                "Install opencv-python>=4.10."
            )
        from .config import AtlasConfig  # noqa: PLC0415

        if model_dir is None:
            model_dir = AtlasConfig().home / "models"
        self._cv2 = cv2
        yunet_path = self._setup(Path(model_dir), download=download)
        self._detector = cv2.FaceDetectorYN.create(
            str(yunet_path), "", (320, 320),
            score_threshold=score_threshold, nms_threshold=nms_threshold,
        )

    def _setup(self, model_dir: Path, *, download: bool) -> Path:
        """Fetch YuNet + the recogniser; return the YuNet path. Subclass hook."""

        raise NotImplementedError

    def _embed(self, image: np.ndarray, full_row: np.ndarray) -> np.ndarray:
        """Embed the face described by ``full_row`` (full-res box + landmarks)."""

        raise NotImplementedError

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
            observations.append(
                FaceObservation(
                    bbox=(int(x), int(y), int(bw), int(bh)),
                    embedding=l2_normalize(self._embed(image, full_row)),
                    confidence=score,
                )
            )
        return observations


class YuNetArcFaceBackend(_YuNetBackend):
    """Deep face detection (YuNet) + **ArcFace R100** recognition (512-d).

    The default deep backend. Faces are aligned to ArcFace's canonical 112²
    template via a 5-point similarity transform (:func:`norm_crop`) using YuNet's
    landmarks, then embedded by the ArcFace ONNX through ``onnxruntime``. The
    weights are fetched by :func:`photo_atlas.models.ensure_arcface_models`.
    """

    def _setup(self, model_dir: Path, *, download: bool) -> Path:
        import onnxruntime as ort  # noqa: PLC0415

        from .models import ensure_arcface_models  # noqa: PLC0415

        yunet_path, arcface_path = ensure_arcface_models(model_dir, download=download)
        self._session = ort.InferenceSession(
            str(arcface_path), providers=["CPUExecutionProvider"]
        )
        self._input = self._session.get_inputs()[0].name
        self._output = self._session.get_outputs()[0].name
        return yunet_path

    def _embed(self, image: np.ndarray, full_row: np.ndarray) -> np.ndarray:
        landmarks = full_row[4:14].reshape(5, 2)
        aligned = norm_crop(image, landmarks)
        blob = arcface_preprocess(aligned)
        (feature,) = self._session.run([self._output], {self._input: blob})
        return feature.flatten()


class YuNetSFaceBackend(_YuNetBackend):
    """Deep face detection (YuNet) + recognition embeddings (SFace, 128-d).

    The legacy recogniser, kept selectable (``faces=sface``). Requires OpenCV's
    DNN face module (``FaceRecognizerSF``); weights come from
    :func:`photo_atlas.models.ensure_models`.
    """

    def _setup(self, model_dir: Path, *, download: bool) -> Path:
        from .models import ensure_models  # noqa: PLC0415

        if not hasattr(self._cv2, "FaceRecognizerSF"):
            raise RuntimeError(
                "This OpenCV build lacks FaceRecognizerSF. Install opencv-python>=4.10."
            )
        yunet_path, sface_path = ensure_models(model_dir, download=download)
        self._recognizer = self._cv2.FaceRecognizerSF.create(str(sface_path), "")
        return yunet_path

    def _embed(self, image: np.ndarray, full_row: np.ndarray) -> np.ndarray:
        aligned = self._recognizer.alignCrop(image, full_row)
        return self._recognizer.feature(aligned).flatten()


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

    ``auto`` prefers the deep YuNet + **ArcFace** pipeline, then dlib. ``yunet`` is
    an alias for that default deep backend; ``sface`` selects the legacy YuNet +
    SFace (128-d) recogniser.
    """

    order = {
        "auto": ["arcface", "dlib"],
        "arcface": ["arcface"],
        "yunet": ["arcface"],
        "sface": ["sface"],
        "dlib": ["dlib"],
        "synthetic": ["synthetic"],
        "none": [],
    }.get(name, ["arcface"])

    for choice in order:
        try:
            if choice == "arcface":
                return YuNetArcFaceBackend(model_dir=model_dir)
            if choice == "sface":
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

    import math  # noqa: PLC0415

    from sklearn.cluster import DBSCAN  # noqa: PLC0415

    matrix = np.vstack([l2_normalize(e) for e in embeddings]).astype(np.float32)
    # On L2-normalised vectors cosine distance is monotonic in Euclidean distance:
    #   ||a - b||^2 = 2 - 2*cos(a, b)  =>  ||a - b|| = sqrt(2 * cosine_distance).
    # Clustering with a tree-based Euclidean metric therefore yields the *same*
    # neighbour graph (and thus the same DBSCAN partition) as a precomputed cosine
    # matrix, but without materialising that dense n*n matrix — which is ~13 GB at
    # 40k faces and OOMs. Memory drops from O(n^2) to O(n*d).
    eps_euclid = math.sqrt(max(0.0, 2.0 * eps))
    labels = DBSCAN(
        eps=eps_euclid, min_samples=min_samples,
        metric="euclidean", algorithm="ball_tree",
    ).fit_predict(matrix)
    return [int(x) for x in labels]


@dataclass
class Enrollment:
    """Named (enrolled) faces available for k-NN recognition.

    ``embeddings`` is an ``(n, d)`` float32 matrix of L2-normalised face vectors
    and ``person_ids`` the parallel ``(n,)`` array of their owning person ids.
    Built once per index run from the catalog's already-named faces and treated
    as read-only, so it is cheap to ship to worker processes.
    """

    embeddings: np.ndarray
    person_ids: np.ndarray
    #: "Not this person" negatives from active learning: a parallel ``(m, d)``
    #: matrix + ``(m,)`` person ids. A probe near one of a person's negatives has
    #: that person's vote penalised in :func:`knn_person_match`.
    neg_embeddings: np.ndarray = field(
        default_factory=lambda: np.empty((0, 0), dtype=np.float32)
    )
    neg_person_ids: np.ndarray = field(
        default_factory=lambda: np.empty((0,), dtype=np.int64)
    )

    @property
    def is_empty(self) -> bool:
        return int(self.person_ids.size) == 0

    @classmethod
    def from_pairs(
        cls,
        pairs: list[tuple[int, np.ndarray]],
        negatives: list[tuple[int, np.ndarray]] | None = None,
    ) -> Enrollment:
        """Build an enrollment from ``(person_id, embedding)`` positive pairs.

        ``negatives`` carries the same shape of "not this person" examples.
        """

        def _stack(items: list[tuple[int, np.ndarray]]) -> tuple[np.ndarray, np.ndarray]:
            if not items:
                return (
                    np.empty((0, 0), dtype=np.float32),
                    np.empty((0,), dtype=np.int64),
                )
            ids = np.array([pid for pid, _ in items], dtype=np.int64)
            mat = np.vstack([l2_normalize(vec) for _, vec in items]).astype(np.float32)
            return mat, ids

        pos_mat, pos_ids = _stack(pairs)
        neg_mat, neg_ids = _stack(negatives or [])
        return cls(pos_mat, pos_ids, neg_mat, neg_ids)


def knn_person_match(
    embedding: np.ndarray,
    enrollment: Enrollment,
    *,
    k: int = 5,
    threshold: float = 0.5,
) -> tuple[int | None, float]:
    """Recognise a probe face by majority vote over its k nearest enrolled faces.

    More robust than a single per-person centroid when a person's look drifts over
    years (child→adult, beards, glasses): comparing against the *nearest individual*
    enrolled faces tolerates that spread instead of averaging it away (a far-apart
    pair of enrolments would pull a centroid into the empty space between them).

    Only neighbours within ``threshold`` cosine distance get a vote; the winner is
    the person with the most votes among the ``k`` nearest, ties broken by the
    smaller mean distance. Returns ``(person_id | None, confidence)`` where
    confidence is ``1 - mean distance`` to the winning person's voting neighbours.

    **Active learning:** a person's "not this person" negatives (recorded when the
    user corrects an auto-tag) cast *negative* votes here. Each person's net vote is
    its positive neighbours minus its near negatives, so a probe that looks like a
    rejected example is penalised — or vetoed when the negatives outweigh the
    positives. With no negatives the result is identical to plain k-NN.
    """

    if enrollment.is_empty:
        return None, 0.0
    probe = l2_normalize(embedding)
    # cosine distance = 1 - cosine similarity; all vectors are unit norm.
    dists = 1.0 - (enrollment.embeddings @ probe)
    within = np.where(dists <= threshold)[0]
    if within.size == 0:
        return None, 0.0

    # The k nearest neighbours that fall within the threshold.
    nn = within[np.argsort(dists[within], kind="stable")][:k]
    nn_ids = enrollment.person_ids[nn]
    nn_dists = dists[nn]

    # Negative votes: the k nearest negatives within the threshold, tallied per
    # person, are subtracted from that person's positive votes below.
    neg_votes: dict[int, int] = {}
    if enrollment.neg_person_ids.size:
        ndists = 1.0 - (enrollment.neg_embeddings @ probe)
        nwithin = np.where(ndists <= threshold)[0]
        if nwithin.size:
            nnn = nwithin[np.argsort(ndists[nwithin], kind="stable")][:k]
            for pid in enrollment.neg_person_ids[nnn]:
                neg_votes[int(pid)] = neg_votes.get(int(pid), 0) + 1

    best_id: int | None = None
    best_votes = 0
    best_mean = float("inf")
    for pid in np.unique(nn_ids):
        mask = nn_ids == pid
        votes = int(mask.sum()) - neg_votes.get(int(pid), 0)
        if votes <= 0:
            continue  # negatives cancelled this identity out
        mean_d = float(nn_dists[mask].mean())
        if votes > best_votes or (votes == best_votes and mean_d < best_mean):
            best_votes, best_mean, best_id = votes, mean_d, int(pid)

    confidence = max(0.0, 1.0 - best_mean) if best_id is not None else 0.0
    return best_id, confidence
