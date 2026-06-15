"""Shared fixtures: a throwaway library, an indexed demo catalog and a client."""

from __future__ import annotations

import pytest

from photo_atlas import demo, indexer
from photo_atlas.config import AtlasConfig


@pytest.fixture
def config(tmp_path):
    return AtlasConfig(home=tmp_path / "lib").ensure_dirs()


@pytest.fixture
def indexed(config, tmp_path):
    """A populated catalog built from the synthetic demo (no network)."""
    photos_dir = tmp_path / "photos"
    demo.generate(photos_dir, count=20, seed=7)
    indexer.index_path(config, photos_dir, backend_name="synthetic", geocode=True)
    indexer.cluster_library(config)
    return config


@pytest.fixture
def client(indexed):
    from fastapi.testclient import TestClient

    from photo_atlas.api import create_app

    return TestClient(create_app(indexed))


@pytest.fixture
def person_id(client):
    """Create one named person (by naming the largest face cluster) and return it."""
    clusters = client.get("/api/clusters").json()["clusters"]
    assert clusters, "demo library should produce at least one face cluster"
    return client.post(
        f"/api/clusters/{clusters[0]['cluster_id']}/assign", json={"name": "Subject"}
    ).json()["person_id"]
