"""Unified "type of picture" facet — API wiring.

Folds the people-count buckets (portrait/group) and the scene tags into one
``kind`` filter/facet. The DB + ``_where`` behaviour is covered in
``tests/test_search_db.py`` / ``tests/test_search_unit.py``; here we pin the API
round-trip: the ``kind`` query param filters ``/api/photos`` and the ``kinds``
facet's chip counts match what selecting that token returns.
"""

from __future__ import annotations

from photo_atlas import db
from photo_atlas.config import AtlasConfig


def _client(tmp_path):
    config = AtlasConfig(home=tmp_path / "lib").ensure_dirs()
    conn = db.connect(config.db_path)
    # face_count, scene_type
    db.upsert_photo(conn, {"path": "/a.jpg", "filename": "a.jpg", "face_count": 1,
                           "scene_type": "people"})       # portrait
    db.upsert_photo(conn, {"path": "/b.jpg", "filename": "b.jpg", "face_count": 1,
                           "scene_type": "food"})         # portrait + food
    db.upsert_photo(conn, {"path": "/c.jpg", "filename": "c.jpg", "face_count": 4,
                           "scene_type": "people"})       # group
    db.upsert_photo(conn, {"path": "/d.jpg", "filename": "d.jpg", "face_count": 0,
                           "scene_type": "landscape"})    # landscape
    conn.commit()
    conn.close()

    from fastapi.testclient import TestClient

    from photo_atlas.api import create_app

    return TestClient(create_app(config))


def test_api_kind_facet_and_filter_agree(tmp_path):
    client = _client(tmp_path)

    kinds = {b["value"]: b["count"] for b in client.get("/api/facets").json()["kinds"]}
    assert kinds["portrait"] == 2
    assert kinds["group"] == 1
    assert kinds["food"] == 1
    assert kinds["landscape"] == 1

    # Each chip count equals the number of photos that selecting that token returns.
    for tok, count in kinds.items():
        data = client.get(f"/api/photos?kind={tok}").json()
        assert data["total"] == count, tok

    # OR within the facet (repeated params): portraits OR landscape.
    data = client.get("/api/photos?kind=portrait&kind=landscape").json()
    assert data["total"] == 3
