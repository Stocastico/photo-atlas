"""Unit tests for the backend-agnostic face maths and backend selection."""

from __future__ import annotations

import numpy as np

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


def test_best_person_match_respects_threshold():
    centroids = {1: np.array([1.0, 0.0]), 2: np.array([0.0, 1.0])}
    probe = np.array([0.99, 0.01])
    pid, conf = faces.best_person_match(probe, centroids, threshold=0.5)
    assert pid == 1 and conf > 0.5
    # Orthogonal probe is beyond threshold -> no match.
    pid2, conf2 = faces.best_person_match(np.array([-1.0, 0.0]), centroids, 0.5)
    assert pid2 is None and conf2 == 0.0


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
