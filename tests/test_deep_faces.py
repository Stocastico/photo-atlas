"""Integration test for the deep YuNet + ArcFace face pipeline.

This is the "does the real model actually distinguish people" test. It needs:

* OpenCV with the DNN face module (``FaceDetectorYN``) + ``onnxruntime``,
* the YuNet + ArcFace R100 ONNX weights (downloaded to a cache), and
* a couple of real face photos (fetched from the public ``face_recognition``
  examples on GitHub).

Anything missing -> the test is **skipped** (so the offline suite stays green)
rather than failing.
"""

from __future__ import annotations

import urllib.request
from pathlib import Path

import pytest

cv2 = pytest.importorskip("cv2")
pytest.importorskip("onnxruntime")

if not hasattr(cv2, "FaceDetectorYN"):
    pytest.skip("OpenCV DNN face module unavailable", allow_module_level=True)

FACE_BASE = "https://github.com/ageitgey/face_recognition/raw/master/examples"
SAMPLES = {
    "obama.jpg": f"{FACE_BASE}/obama.jpg",
    "obama-720p.jpg": f"{FACE_BASE}/obama-720p.jpg",
    "biden.jpg": f"{FACE_BASE}/biden.jpg",
}


def _fetch(url: str, dest: Path) -> None:
    if not dest.exists():
        urllib.request.urlretrieve(url, dest)  # noqa: S310 - fixed trusted URLs


@pytest.fixture(scope="module")
def deep_backend(tmp_path_factory):
    from photo_atlas import faces, models

    cache = tmp_path_factory.mktemp("deep")
    try:
        models.ensure_arcface_models(cache, download=True)
        backend = faces.YuNetArcFaceBackend(model_dir=cache)
    except Exception as exc:  # network / model issues -> skip, not fail
        pytest.skip(f"deep face models unavailable: {exc}")
    return backend, cache


@pytest.fixture(scope="module")
def sample_faces(deep_backend):
    _, cache = deep_backend
    paths = {}
    for name, url in SAMPLES.items():
        dest = cache / name
        try:
            _fetch(url, dest)
        except Exception as exc:  # pragma: no cover - network dependent
            pytest.skip(f"sample faces unavailable: {exc}")
        paths[name] = dest
    return paths


def test_yunet_detects_single_face(deep_backend, sample_faces):
    backend, _ = deep_backend
    obs = backend.detect(sample_faces["obama.jpg"])
    assert len(obs) == 1
    assert obs[0].embedding.shape[0] == 512  # ArcFace R100
    assert obs[0].confidence > 0.85


def test_arcface_separates_identities(deep_backend, sample_faces):
    from photo_atlas.faces import cosine_distance

    backend, _ = deep_backend
    obama = backend.detect(sample_faces["obama.jpg"])[0].embedding
    obama2 = backend.detect(sample_faces["obama-720p.jpg"])[0].embedding
    biden = backend.detect(sample_faces["biden.jpg"])[0].embedding

    same = cosine_distance(obama, obama2)
    diff = cosine_distance(obama, biden)
    # Same identity must be far closer than different identities.
    assert same < 0.4
    assert diff > 0.7
    assert same < diff - 0.3


def test_clustering_groups_same_person(deep_backend, sample_faces):
    from photo_atlas.faces import cluster_embeddings

    backend, _ = deep_backend
    embs = [
        backend.detect(sample_faces["obama.jpg"])[0].embedding,
        backend.detect(sample_faces["obama-720p.jpg"])[0].embedding,
        backend.detect(sample_faces["biden.jpg"])[0].embedding,
    ]
    labels = cluster_embeddings(embs, eps=0.5, min_samples=2)
    # The two Obama photos share a cluster; Biden is separate (noise or own).
    assert labels[0] == labels[1]
    assert labels[2] != labels[0]
