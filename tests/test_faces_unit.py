"""Unit tests for the backend-agnostic face maths and backend selection."""

from __future__ import annotations

import numpy as np
import pytest

from photo_atlas import faces


def test_l2_normalize_unit_and_zero():
    v = faces.l2_normalize(np.array([3.0, 4.0]))
    assert np.isclose(np.linalg.norm(v), 1.0)
    # A (near-)zero vector is returned unchanged instead of dividing by ~0.
    z = faces.l2_normalize(np.zeros(4))
    assert np.allclose(z, 0.0)


def test_cosine_distance_bounds():
    a = np.array([1.0, 0.0])
    assert np.isclose(faces.cosine_distance(a, a), 0.0)
    assert np.isclose(faces.cosine_distance(a, np.array([0.0, 1.0])), 1.0)
    assert np.isclose(faces.cosine_distance(a, np.array([-1.0, 0.0])), 2.0)


def test_cluster_embeddings_edge_cases():
    assert faces.cluster_embeddings([]) == []
    assert faces.cluster_embeddings([np.array([1.0, 0.0])]) == [-1]


def test_cluster_embeddings_groups_two_identities():
    a1 = np.array([1.0, 0.0, 0.0])
    a2 = np.array([0.98, 0.02, 0.0])
    b1 = np.array([0.0, 0.0, 1.0])
    b2 = np.array([0.0, 0.02, 0.98])
    labels = faces.cluster_embeddings([a1, a2, b1, b2], eps=0.2, min_samples=2)
    assert labels[0] == labels[1] != -1
    assert labels[2] == labels[3] != -1
    assert labels[0] != labels[2]


def test_cluster_embeddings_matches_precomputed_cosine_partition():
    """The tree-based Euclidean clustering must reproduce, exactly, the partition
    the old dense precomputed-cosine DBSCAN produced — same groups, same noise.

    This guards the O(n^2)->O(n*d) memory refactor: it changes *how* neighbours
    are found, never *which* points are neighbours (cosine distance is monotonic
    in Euclidean distance on L2-normalised vectors).
    """

    import numpy as np
    from sklearn.cluster import DBSCAN

    from photo_atlas.faces import l2_normalize

    rng = np.random.default_rng(0)
    # Three tight blobs in 16-D plus a few outliers -> a non-trivial partition.
    centers = rng.normal(size=(3, 16))
    embs = []
    for c in centers:
        for _ in range(20):
            embs.append(c + 0.02 * rng.normal(size=16))
    embs.extend(rng.normal(size=16) for _ in range(5))  # noise

    eps = 0.3
    got = faces.cluster_embeddings(embs, eps=eps, min_samples=3)

    # Reference: the previous implementation (dense precomputed cosine matrix).
    matrix = np.vstack([l2_normalize(e) for e in embs])
    distance = 1.0 - np.clip(matrix @ matrix.T, -1.0, 1.0)
    np.fill_diagonal(distance, 0.0)
    ref_labels = DBSCAN(eps=eps, min_samples=3, metric="precomputed").fit_predict(distance)
    ref = [int(x) for x in ref_labels]

    def partition(labels):
        noise = frozenset(i for i, lab in enumerate(labels) if lab < 0)
        by_label: dict[int, set] = {}
        for i, lab in enumerate(labels):
            if lab >= 0:
                by_label.setdefault(lab, set()).add(i)
        return frozenset(frozenset(s) for s in by_label.values()), noise

    assert partition(got) == partition(ref)


def _enroll(*pairs):
    return faces.Enrollment.from_pairs([(pid, np.asarray(v, dtype=np.float32)) for pid, v in pairs])


def test_enrollment_from_pairs_shapes_and_empty():
    empty = faces.Enrollment.from_pairs([])
    assert empty.is_empty and empty.person_ids.size == 0
    e = _enroll((1, [1.0, 0.0]), (2, [0.0, 1.0]))
    assert not e.is_empty
    assert e.embeddings.shape == (2, 2)
    assert list(e.person_ids) == [1, 2]
    # Vectors are stored L2-normalised.
    norms = np.linalg.norm(e.embeddings, axis=1)
    assert np.allclose(norms, 1.0)


def test_knn_empty_enrollment_returns_none():
    pid, conf = faces.knn_person_match(np.array([1.0, 0.0]), faces.Enrollment.from_pairs([]))
    assert pid is None and conf == 0.0


def test_knn_matches_nearest_within_threshold():
    e = _enroll((1, [1.0, 0.0]), (2, [0.0, 1.0]))
    pid, conf = faces.knn_person_match(np.array([0.99, 0.01]), e, k=1, threshold=0.5)
    assert pid == 1 and conf > 0.5
    # Orthogonal probe is beyond threshold for either person -> no match.
    pid2, conf2 = faces.knn_person_match(np.array([-1.0, 0.0]), e, k=1, threshold=0.5)
    assert pid2 is None and conf2 == 0.0


def test_knn_majority_vote_beats_a_single_closer_outlier():
    # Three enrolled faces near the probe: two of person 1, one (slightly closer)
    # of person 2. Majority vote picks person 1 despite person 2's nearest face.
    e = _enroll(
        (2, [1.0, 0.02, 0.0]),     # closest single neighbour
        (1, [1.0, 0.05, 0.0]),
        (1, [1.0, 0.06, 0.0]),
    )
    probe = np.array([1.0, 0.0, 0.0])
    pid, conf = faces.knn_person_match(probe, e, k=3, threshold=0.5)
    assert pid == 1 and conf > 0.0


def test_knn_is_robust_where_a_centroid_would_fail():
    # Person 1 enrolled twice at opposite ends of a wide arc (look drifted over
    # years). Their *average* (centroid) lands near the origin -> far from either
    # real face, so a centroid match would miss a probe sitting on one end. k-NN
    # to the nearest individual face still recognises it.
    a = np.array([1.0, 0.0], dtype=np.float32)
    b = np.array([0.0, 1.0], dtype=np.float32)
    centroid = (a + b) / 2.0
    probe = np.array([0.99, 0.02], dtype=np.float32)

    # The centroid is past threshold from the probe...
    assert faces.cosine_distance(probe, centroid) > 0.25
    # ...but k-NN against the enrolled faces recognises person 1.
    e = _enroll((1, a), (1, b))
    pid, _ = faces.knn_person_match(probe, e, k=2, threshold=0.25)
    assert pid == 1


def test_knn_tie_breaks_on_smaller_mean_distance():
    # One vote each within k=2; person 1's neighbour is nearer, so it wins the tie.
    e = _enroll((1, [1.0, 0.01, 0.0]), (2, [1.0, 0.20, 0.0]))
    pid, _ = faces.knn_person_match(np.array([1.0, 0.0, 0.0]), e, k=2, threshold=0.5)
    assert pid == 1


def test_get_backend_none_and_synthetic():
    assert faces.get_backend("none") is None
    backend = faces.get_backend("synthetic")
    assert isinstance(backend, faces.SyntheticFaceBackend)


def test_synthetic_backend_detects_demo_faces(tmp_path):
    from photo_atlas import demo

    paths = demo.generate(tmp_path / "p", count=12, seed=5)
    backend = faces.SyntheticFaceBackend()
    total = sum(len(backend.detect(p)) for p in paths)
    assert total > 0


def test_read_bgr_falls_back_to_pil_when_cv2_cannot_decode(tmp_path, monkeypatch):
    """When OpenCV's bundled build can't decode a file (e.g. HEIC), ``_read_bgr``
    falls back to Pillow and still returns a correct BGR array."""

    import cv2
    from PIL import Image

    arr = np.zeros((8, 12, 3), dtype=np.uint8)
    arr[:, :, 0] = 200  # strong red in RGB
    src = tmp_path / "img.png"
    Image.fromarray(arr, "RGB").save(src)

    # Simulate a format OpenCV can't decode (HEIC behaves exactly like this).
    monkeypatch.setattr(cv2, "imread", lambda *a, **k: None)

    out = faces._read_bgr(src)
    assert out is not None and out.shape == (8, 12, 3)
    # Pillow yields RGB; _read_bgr must hand OpenCV its native BGR order.
    b, g, r = out[..., 0].mean(), out[..., 1].mean(), out[..., 2].mean()
    assert r > 150 and g < 20 and b < 20


def test_maybe_downscale_scales_long_side_and_reports_factor():
    big = np.zeros((2000, 3000, 3), dtype=np.uint8)
    out, scale = faces._maybe_downscale(big, max_side=1200)
    assert max(out.shape[:2]) == 1200
    assert np.isclose(scale, 1200 / 3000)
    # An image already within the cap is returned untouched at scale 1.0.
    small = np.zeros((100, 80, 3), dtype=np.uint8)
    out2, scale2 = faces._maybe_downscale(small, max_side=1200)
    assert out2.shape == small.shape and scale2 == 1.0


def test_detect_accepts_predecoded_image(tmp_path, monkeypatch):
    """A caller that already decoded the image (the indexer's decode-once path)
    can hand the array to ``detect`` and skip the filesystem read entirely."""

    import cv2

    from photo_atlas import demo

    [png] = demo.generate(tmp_path / "p", count=1, seed=5)
    backend = faces.SyntheticFaceBackend()
    from_path = backend.detect(png)

    bgr = faces._read_bgr(png)
    monkeypatch.setattr(cv2, "imread", lambda *a, **k: None)  # no disk read allowed
    from_array = backend.detect(png, image=bgr)
    assert len(from_array) == len(from_path)


def test_synthetic_backend_detects_faces_in_heic(tmp_path):
    """Regression: HEIC (iPhone's default, ~a fifth of a real library) must yield
    faces. ``cv2.imread`` returns None for HEIC, so detection has to route the
    decode through Pillow + pillow-heif."""

    pytest.importorskip("pillow_heif")
    import cv2
    import pillow_heif
    from PIL import Image

    from photo_atlas import demo

    pillow_heif.register_heif_opener()
    paths = demo.generate(tmp_path / "p", count=12, seed=5)
    heics = []
    for i, png in enumerate(paths):
        heic = tmp_path / f"photo_{i}.heic"
        with Image.open(png) as im:
            im.convert("RGB").save(heic, format="HEIF")
        heics.append(heic)

    # Guard the premise: OpenCV genuinely can't decode these.
    assert cv2.imread(str(heics[0])) is None

    backend = faces.SyntheticFaceBackend()
    total = sum(len(backend.detect(p)) for p in heics)
    assert total > 0
