"""Cluster cards: most-typical sample faces and a 'looks like <person>' suggestion."""

from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient

from photo_atlas import db, indexer, library
from photo_atlas.api import create_app
from photo_atlas.config import AtlasConfig

E1, E2, E3 = (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)


def _vec(base, jitter=0.0, seed=0):
    rng = np.random.default_rng(seed)
    v = np.array(base, dtype=np.float32) + jitter * rng.normal(size=3).astype(np.float32)
    return v / np.linalg.norm(v)


def _add(conn, vec, *, person=None, confidence=0.0, cluster=None):
    photo = db.upsert_photo(conn, {"path": f"/p/{conn.total_changes}_{id(vec)}.jpg",
                                   "filename": "x.jpg"})
    db.replace_faces(conn, photo, [{
        "person_id": person, "confidence": confidence, "cluster_id": cluster,
        "embedding": db.embedding_to_blob(vec), "dim": len(vec), "crop_path": "c.jpg",
    }])
    return conn.execute("SELECT id FROM faces WHERE photo_id=?", (photo,)).fetchone()["id"]


@pytest.fixture
def conn(tmp_path):
    c = db.connect(tmp_path / "c.db")
    yield c
    c.close()


def test_samples_are_the_most_typical_faces_not_just_the_first_ids(conn):
    # Lowest ids are the *outliers*; a naive "ORDER BY id LIMIT 6" would show them.
    outliers = [_add(conn, _vec((1, 0.9, 0), 0.0, i), cluster=1) for i in range(2)]
    typical = [_add(conn, _vec(E1, 0.02, 10 + i), cluster=1) for i in range(8)]
    conn.commit()
    (cluster,) = library.list_clusters(conn)
    shown = [s["id"] for s in cluster["samples"]]
    assert cluster["size"] == 10 and len(shown) == 6
    assert not set(shown) & set(outliers) and set(shown) <= set(typical)
    assert set(cluster["samples"][0]) == {"id", "photo_id", "crop_path"}  # no blobs leaked


def test_clusters_without_embeddings_still_list_samples(conn):
    photo = db.upsert_photo(conn, {"path": "/n.jpg", "filename": "n.jpg"})
    db.replace_faces(conn, photo, [{"cluster_id": 4, "crop_path": "c.jpg"}])
    conn.commit()
    (cluster,) = library.list_clusters(conn)
    assert cluster["size"] == 1 and len(cluster["samples"]) == 1 and "suggestion" not in cluster


def _enrolled(conn):
    anna = db.get_or_create_person(conn, "Anna")
    for i in range(4):
        _add(conn, _vec(E1, 0.02, i), person=anna, confidence=1.0)
    conn.commit()
    return anna, indexer._load_enrollment(conn)


def test_a_matching_cluster_gets_a_suggestion(conn):
    anna, enrollment = _enrolled(conn)
    for i in range(6):
        _add(conn, _vec(E1, 0.03, 20 + i), cluster=1)
    for i in range(5):
        _add(conn, _vec(E3, 0.02, 40 + i), cluster=2)
    conn.commit()
    by_id = {c["cluster_id"]: c for c in library.list_clusters(conn, enrollment=enrollment)}
    s = by_id[1]["suggestion"]
    assert s["person_id"] == anna and s["name"] == "Anna"
    assert s["confidence"] > 0.8 and s["agreement"] == 1.0
    assert "suggestion" not in by_id[2]  # nobody looks like cluster 2


def test_no_enrolment_means_no_suggestions(conn):
    for i in range(3):
        _add(conn, _vec(E1, 0.03, i), cluster=1)
    conn.commit()
    (c,) = library.list_clusters(conn, enrollment=None)
    assert "suggestion" not in c


def test_a_mixed_cluster_reports_low_agreement(conn):
    anna, enrollment = _enrolled(conn)
    for i in range(3):
        _add(conn, _vec(E1, 0.02, 30 + i), cluster=1)
    for i in range(3):
        _add(conn, _vec(E3, 0.02, 50 + i), cluster=1)  # half of the cluster isn't Anna
    conn.commit()
    (c,) = library.list_clusters(conn, enrollment=enrollment, threshold=0.9)
    if "suggestion" in c:
        assert c["suggestion"]["agreement"] <= 0.67


def test_api_includes_suggestions_and_works_with_no_one_named(tmp_path):
    config = AtlasConfig(home=tmp_path / "lib").ensure_dirs()
    c = db.connect(config.db_path)
    for i in range(3):
        _add(c, _vec(E1, 0.03, i), cluster=1)
    c.commit()
    client = TestClient(create_app(config))
    clusters = client.get("/api/clusters").json()["clusters"]
    assert clusters[0]["size"] == 3 and "suggestion" not in clusters[0]

    anna = db.get_or_create_person(c, "Anna")
    for i in range(4):
        _add(c, _vec(E1, 0.02, 90 + i), person=anna, confidence=1.0)
    c.commit()
    c.close()
    sug = client.get("/api/clusters").json()["clusters"][0]["suggestion"]
    assert sug["person_id"] == anna and sug["name"] == "Anna"
