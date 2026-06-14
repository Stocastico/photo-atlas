"""Offline, deterministic tests for the Photo Atlas pipeline.

These never touch the network: they build a small synthetic library with
:mod:`photo_atlas.demo` and exercise metadata, geocoding, scene tagging, the
synthetic face backend, clustering, search, person management and the API.

The deep YuNet/SFace pipeline is covered separately in
``test_deep_faces.py`` (skipped when models / sample faces can't be fetched).
"""

from __future__ import annotations

import numpy as np
import pytest

from photo_atlas import db, demo, faces, indexer, library, search
from photo_atlas.classify import SCENE_LABELS, SceneTagger
from photo_atlas.config import AtlasConfig
from photo_atlas.geocode import Geocoder
from photo_atlas.metadata import extract_meta


@pytest.fixture
def config(tmp_path):
    return AtlasConfig(home=tmp_path / "lib").ensure_dirs()


@pytest.fixture
def demo_photos(tmp_path):
    return demo.generate(tmp_path / "photos", count=18, seed=11)


# -- metadata --------------------------------------------------------------
def test_demo_exif_roundtrip(demo_photos):
    meta = extract_meta(demo_photos[0])
    assert meta.taken_source == "exif"
    assert meta.taken_at and meta.taken_at[:2] == "20"
    assert meta.lat is not None and meta.lon is not None
    assert meta.camera_model == "DemoCam 1.0"


def test_exif_datetimeoriginal_is_read(tmp_path):
    """Real cameras store capture time in DateTimeOriginal, in the Exif sub-IFD.

    Pillow's ``Image.getexif()`` only exposes the *base* IFD, so the canonical
    capture timestamp must be read from the Exif sub-IFD (0x8769) explicitly.
    """
    from PIL import Image

    path = tmp_path / "real.jpg"
    img = Image.new("RGB", (32, 32), (10, 20, 30))
    exif = Image.Exif()
    exif[0x8769] = {0x9003: "2015:08:09 11:22:33"}  # Exif sub-IFD -> DateTimeOriginal
    img.save(path, "JPEG", exif=exif)

    meta = extract_meta(path)
    assert meta.taken_source == "exif"
    assert meta.taken_at.startswith("2015-08-09")


# -- geocoding -------------------------------------------------------------
def test_geocode_nearest_city():
    place = Geocoder(prefer_external=False).lookup(41.9, 12.5)  # Rome
    assert place is not None
    assert place.city == "Rome" and place.country == "Italy"


def test_geocode_handles_missing_coords():
    assert Geocoder(prefer_external=False).lookup(None, None) is None


def test_external_geocoder_reports_country_not_region():
    """The high-resolution backend must fill ``country`` with a country, not the
    admin1 region, and keep the ISO country code."""

    class FakeRG:
        def search(self, coords, mode=1):  # mimics reverse_geocoder.search
            return [{
                "name": "Brooklyn", "admin1": "New York", "admin2": "Kings",
                "cc": "US", "lat": "40.6500", "lon": "-73.9500",
            }]

    geo = Geocoder(prefer_external=False)
    geo._rg = FakeRG()  # inject the high-resolution backend without installing it
    place = geo.lookup(40.65, -73.95)

    assert place is not None
    assert place.city == "Brooklyn"
    assert place.country_code == "US"
    assert place.admin == "New York"
    # The bug: country was set to admin1 ("New York"). It must be the country.
    assert place.country == "United States"


# -- scene tagging ---------------------------------------------------------
def test_scene_tagger_labels_are_valid(demo_photos):
    tagger = SceneTagger()
    for photo in demo_photos[:5]:
        label, scores = tagger.tag(photo, face_count=0)
        assert label in SCENE_LABELS
        assert abs(sum(scores.values()) - 1.0) < 1e-5


def test_faces_make_scene_people(tmp_path):
    [photo] = demo.generate(tmp_path / "p", count=1, seed=1)
    label, _ = SceneTagger().tag(photo, face_count=2)
    assert label == "people"


# -- synthetic face backend + clustering -----------------------------------
def test_synthetic_backend_detects_and_separates_identities(demo_photos):
    backend = faces.get_backend("synthetic")
    assert backend is not None
    embeddings = []
    for photo in demo_photos:
        embeddings.extend(o.embedding for o in backend.detect(photo))
    assert len(embeddings) >= 3
    labels = faces.cluster_embeddings(embeddings, eps=0.5, min_samples=2)
    n_clusters = len({label for label in labels if label >= 0})
    # The demo draws exactly three distinct people.
    assert n_clusters == 3


def test_cosine_distance_and_match():
    a = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    b = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    c = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    assert faces.cosine_distance(a, b) < 1e-6
    assert faces.cosine_distance(a, c) > 0.9
    pid, conf = faces.best_person_match(a, {7: b, 9: c}, threshold=0.5)
    assert pid == 7 and conf > 0.9
    pid, _ = faces.best_person_match(a, {9: c}, threshold=0.5)
    assert pid is None


# -- db --------------------------------------------------------------------
def test_upsert_photo_returns_stable_id_on_insert_and_update():
    conn = db.connect(":memory:")
    base = {"filename": "x.jpg"}
    id_a = db.upsert_photo(conn, {**base, "path": "/x/a.jpg"})
    db.upsert_photo(conn, {**base, "path": "/x/b.jpg"})  # advance the rowid

    # Re-upserting an existing path updates in place and returns the same id.
    id_a2 = db.upsert_photo(conn, {**base, "path": "/x/a.jpg", "filename": "a2.jpg"})
    assert id_a2 == id_a
    assert conn.execute(
        "SELECT filename FROM photos WHERE id=?", (id_a,)
    ).fetchone()[0] == "a2.jpg"
    assert conn.execute("SELECT COUNT(*) FROM photos").fetchone()[0] == 2


# -- indexing + search + library ------------------------------------------
@pytest.fixture
def indexed(config, tmp_path):
    photos_dir = tmp_path / "photos"
    demo.generate(photos_dir, count=24, seed=7)
    indexer.index_path(config, photos_dir, backend_name="synthetic", geocode=True)
    indexer.cluster_library(config)
    return config


def test_index_populates_catalog(indexed):
    conn = db.connect(indexed.db_path)
    total = conn.execute("SELECT COUNT(*) FROM photos").fetchone()[0]
    faces_n = conn.execute("SELECT COUNT(*) FROM faces").fetchone()[0]
    assert total == 24
    assert faces_n > 0
    # Every photo got a scene tag and most got a place from GPS.
    placed = conn.execute("SELECT COUNT(*) FROM photos WHERE place_country IS NOT NULL").fetchone()[0]
    assert placed == 24


def test_thumbnail_path_is_content_addressed(config, tmp_path):
    """Thumbnail filenames must be a deterministic function of file content, not
    of ``hash()`` (which is salted per process), so re-indexing reuses them."""
    from photo_atlas.metadata import sha1_of

    [photo] = demo.generate(tmp_path / "p", count=1, seed=5)
    sha1 = sha1_of(photo)
    expected = config.thumbs_dir / sha1[:2] / f"{sha1}.jpg"
    assert indexer.thumb_path_for(config, sha1) == expected


def test_reindex_reuses_thumbnail_path(config, tmp_path):
    photos_dir = tmp_path / "photos"
    demo.generate(photos_dir, count=2, seed=8)
    indexer.index_path(config, photos_dir, backend_name="none", geocode=False)
    conn = db.connect(config.db_path)
    before = {r["id"]: r["thumb_path"] for r in conn.execute("SELECT id, thumb_path FROM photos")}

    indexer.index_path(config, photos_dir, backend_name="none", geocode=False, recompute=True)
    after = {r["id"]: r["thumb_path"] for r in conn.execute("SELECT id, thumb_path FROM photos")}
    assert before == after


def test_reindex_is_idempotent(indexed, tmp_path):
    before = db.connect(indexed.db_path).execute("SELECT COUNT(*) FROM photos").fetchone()[0]
    indexer.index_path(indexed, tmp_path / "photos", backend_name="synthetic")
    after = db.connect(indexed.db_path).execute("SELECT COUNT(*) FROM photos").fetchone()[0]
    assert before == after


def test_search_filters(indexed):
    conn = db.connect(indexed.db_path)
    people, total = search.search_photos(conn, {"scene": "people"})
    assert total >= 1 and all(p["scene_type"] == "people" for p in people)

    f = search.facets(conn)
    assert f["total"] == 24
    assert any(s["value"] == "people" for s in f["scenes"])
    assert f["countries"]

    # Free-text search reaches beyond the filename into camera/place fields.
    by_camera, total_cam = search.search_photos(conn, {"q": "DemoCam"})
    assert total_cam == 24 and all("DemoCam" in (p["camera_model"] or "") for p in by_camera)

    country = f["countries"][0]["value"]
    by_country, total_country = search.search_photos(conn, {"q": country})
    assert total_country >= 1


def test_cluster_assignment_and_recognition(indexed):
    conn = db.connect(indexed.db_path)
    clusters = library.list_clusters(conn)
    assert clusters, "expected at least one unnamed cluster"

    pid = library.assign_cluster(conn, clusters[0]["cluster_id"], name="Alice")
    persons = library.list_persons(conn)
    alice = next(p for p in persons if p["name"] == "Alice")
    assert alice["face_count"] == clusters[0]["size"]

    # Filtering by the new person returns only their photos.
    photos, total = search.search_photos(conn, {"person_id": pid})
    assert total == alice["photo_count"]

    library.rename_person(conn, pid, "Alicia")
    assert any(p["name"] == "Alicia" for p in library.list_persons(conn))

    library.delete_person(conn, pid)
    assert not any(p["name"] == "Alicia" for p in library.list_persons(conn))
    # Faces are detached, not deleted, so they can be re-clustered later.
    orphaned = conn.execute("SELECT COUNT(*) FROM faces WHERE person_id IS NULL").fetchone()[0]
    assert orphaned > 0


def test_auto_recognition_of_new_photos(config, tmp_path):
    """A named person is auto-recognised when new photos are indexed."""

    first = tmp_path / "first"
    demo.generate(first, count=12, seed=3)
    indexer.index_path(config, first, backend_name="synthetic")
    indexer.cluster_library(config)

    conn = db.connect(config.db_path)
    clusters = library.list_clusters(conn)
    library.assign_cluster(conn, clusters[0]["cluster_id"], name="Bob")

    second = tmp_path / "second"
    demo.generate(second, count=12, seed=99)
    stats = indexer.index_path(config, second, backend_name="synthetic")
    assert stats.recognized > 0  # Bob recognised in the new batch without re-clustering


# -- API -------------------------------------------------------------------
def test_api_endpoints(indexed):
    from fastapi.testclient import TestClient

    from photo_atlas.api import create_app

    client = TestClient(create_app(indexed))

    assert client.get("/").status_code == 200
    facets = client.get("/api/facets").json()
    assert facets["total"] == 24

    photos = client.get("/api/photos?scene=people").json()
    assert photos["total"] >= 1
    photo_id = photos["photos"][0]["id"]
    assert client.get(f"/api/thumb/{photo_id}").status_code == 200
    assert client.get(f"/api/photos/{photo_id}").json()["id"] == photo_id

    clusters = client.get("/api/clusters").json()["clusters"]
    res = client.post(f"/api/clusters/{clusters[0]['cluster_id']}/assign", json={"name": "Carol"})
    assert res.json()["ok"] is True
    assert any(p["name"] == "Carol" for p in client.get("/api/persons").json()["persons"])
