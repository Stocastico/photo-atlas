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
    enrollment = faces.Enrollment.from_pairs([(7, b), (9, c)])
    pid, conf = faces.knn_person_match(a, enrollment, k=1, threshold=0.5)
    assert pid == 7 and conf > 0.9
    pid, _ = faces.knn_person_match(a, faces.Enrollment.from_pairs([(9, c)]), k=1, threshold=0.5)
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
    placed = conn.execute(
        "SELECT COUNT(*) FROM photos WHERE place_country IS NOT NULL"
    ).fetchone()[0]
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

    # Facet counts are filter-aware: constraining to one scene must not inflate
    # any facet beyond that scene's own total, and the scene facet itself stays
    # fully listed (its own dimension is excluded from the constraint).
    scene_total = next(s["count"] for s in f["scenes"] if s["value"] == "people")
    fp = search.facets(conn, {"scene": "people"})
    assert {s["value"] for s in fp["scenes"]} == {s["value"] for s in f["scenes"]}
    assert all(c["count"] <= scene_total for c in fp["countries"])
    assert sum(p["count"] for p in fp["persons"]) >= 0  # persons facet still resolves


def test_sort_options(indexed):
    conn = db.connect(indexed.db_path)

    newest, _ = search.search_photos(conn, {"sort": "newest"}, limit=500)
    oldest, _ = search.search_photos(conn, {"sort": "oldest"}, limit=500)
    # Oldest-first is the exact reverse ordering of newest-first by date.
    assert [p["id"] for p in oldest][::-1] == [p["id"] for p in newest]
    dates = [p["taken_at"] for p in newest]
    assert dates == sorted(dates, reverse=True)

    az, _ = search.search_photos(conn, {"sort": "filename"}, limit=500)
    za, _ = search.search_photos(conn, {"sort": "filename_desc"}, limit=500)
    names = [p["filename"].lower() for p in az]
    assert names == sorted(names)
    assert [p["id"] for p in za] == [p["id"] for p in az][::-1]

    recent, _ = search.search_photos(conn, {"sort": "indexed"}, limit=500)
    stamps = [p["indexed_at"] for p in recent]
    assert stamps == sorted(stamps, reverse=True)

    # An unknown sort key falls back to the default (newest) rather than erroring.
    fallback, _ = search.search_photos(conn, {"sort": "bogus"}, limit=500)
    assert [p["id"] for p in fallback] == [p["id"] for p in newest]


def test_sort_pagination_is_stable(indexed):
    """Paging must not drop or duplicate rows even when the sort key ties."""
    conn = db.connect(indexed.db_path)
    for sort in ("newest", "oldest", "filename", "filename_desc", "indexed"):
        full, total = search.search_photos(conn, {"sort": sort}, limit=500)
        page1, _ = search.search_photos(conn, {"sort": sort}, limit=10, offset=0)
        page2, _ = search.search_photos(conn, {"sort": sort}, limit=10, offset=10)
        paged_ids = [p["id"] for p in page1] + [p["id"] for p in page2]
        assert paged_ids == [p["id"] for p in full][: len(paged_ids)]
        assert len(set(paged_ids)) == len(paged_ids)  # no duplicates across pages


def test_multi_select_filters(indexed):
    conn = db.connect(indexed.db_path)
    f = search.facets(conn)
    countries = [c["value"] for c in f["countries"]]
    assert len(countries) >= 2, "demo library should span several countries"
    c1, c2 = countries[0], countries[1]

    # A scalar value and a single-element list behave identically.
    one, t1 = search.search_photos(conn, {"country": c1})
    one_list, t1l = search.search_photos(conn, {"country": [c1]})
    assert t1 == t1l and {p["id"] for p in one} == {p["id"] for p in one_list}

    # OR within a facet: two countries return the union (they are disjoint).
    a, ta = search.search_photos(conn, {"country": [c1]})
    b, tb = search.search_photos(conn, {"country": [c2]})
    both, tboth = search.search_photos(conn, {"country": [c1, c2]})
    assert tboth == ta + tb
    assert {p["id"] for p in both} == {p["id"] for p in a} | {p["id"] for p in b}
    assert all(p["place_country"] in {c1, c2} for p in both)

    # AND across facets still narrows the union.
    scenes = [s["value"] for s in f["scenes"]]
    combined, tcomb = search.search_photos(conn, {"country": [c1, c2], "scene": scenes[:1]})
    assert tcomb <= tboth
    assert all(p["scene_type"] == scenes[0] for p in combined)

    # The facet's own dimension stays fully listed under multi-select.
    fm = search.facets(conn, {"country": [c1, c2]})
    assert {c["value"] for c in fm["countries"]} == set(countries)

    # OR across people via the faces join.
    clusters = library.list_clusters(conn)
    if len(clusters) >= 2:
        p1 = library.assign_cluster(conn, clusters[0]["cluster_id"], name="MP1")
        p2 = library.assign_cluster(conn, clusters[1]["cluster_id"], name="MP2")
        a2, _ = search.search_photos(conn, {"person_id": [p1]})
        b2, _ = search.search_photos(conn, {"person_id": [p2]})
        ab, _ = search.search_photos(conn, {"person_id": [p1, p2]})
        assert {p["id"] for p in ab} == {p["id"] for p in a2} | {p["id"] for p in b2}


def test_has_faces_and_date_filters(indexed):
    conn = db.connect(indexed.db_path)
    f = search.facets(conn)

    # Facet payload carries the quick-filter count and the date bounds.
    assert f["with_faces"] >= 1
    assert f["date_min"] and f["date_max"] and f["date_min"] <= f["date_max"]

    withf, tf = search.search_photos(conn, {"has_faces": True})
    assert tf == f["with_faces"]
    assert all(p["face_count"] > 0 for p in withf)

    # A full-span range returns every dated photo (bounds are inclusive).
    dated = conn.execute("SELECT COUNT(*) FROM photos WHERE taken_at IS NOT NULL").fetchone()[0]
    _, tb = search.search_photos(conn, {"date_from": f["date_min"], "date_to": f["date_max"]})
    assert tb == dated

    # Capping the upper bound at the earliest date keeps only that day or before.
    narrow, tn = search.search_photos(conn, {"date_to": f["date_min"]})
    assert 1 <= tn <= dated
    assert all(p["taken_at"][:10] <= f["date_min"] for p in narrow)

    # has_faces AND a date range combine (never more than either alone).
    _, tcomb = search.search_photos(conn, {"has_faces": True, "date_to": f["date_min"]})
    assert tcomb <= tf and tcomb <= tn


def test_empty_library_onboarding_signal(config):
    """An empty catalog reports zero totals (the UI's onboarding trigger) and
    the page ships the onboarding + toast scaffolding."""
    from fastapi.testclient import TestClient

    from photo_atlas.api import create_app

    client = TestClient(create_app(config))
    assert client.get("/api/photos").json()["total"] == 0
    assert client.get("/api/facets").json()["total"] == 0

    html = client.get("/").text
    assert 'id="onboarding"' in html
    assert 'id="toast"' in html
    assert 'photo-atlas index' in html  # onboarding shows the CLI hint


def test_preview_endpoint_caps_size_and_caches(indexed):
    import io

    from fastapi.testclient import TestClient
    from PIL import Image

    from photo_atlas.api import create_app

    client = TestClient(create_app(indexed))
    pid = client.get("/api/photos").json()["photos"][0]["id"]

    res = client.get(f"/api/preview/{pid}")
    assert res.status_code == 200
    assert res.headers["content-type"].startswith("image/")

    img = Image.open(io.BytesIO(res.content))
    assert max(img.size) <= indexed.preview_size

    # The derivative is cached to disk content-addressed by sha1, so a second
    # request reuses the file rather than re-encoding.
    cached = list(indexed.previews_dir.rglob("*.jpg"))
    assert cached, "preview should be written to the previews cache"
    again = client.get(f"/api/preview/{pid}")
    assert again.status_code == 200

    # The full-resolution original is still reachable for download.
    assert client.get(f"/api/image/{pid}").status_code == 200
    assert client.get("/api/preview/999999").status_code == 404


def test_thumb_size_variant(indexed):
    """The thumb endpoint serves a cached 2x (retina) derivative on demand and
    leaves the default thumbnail untouched."""
    import io

    from fastapi.testclient import TestClient
    from PIL import Image

    from photo_atlas.api import create_app

    client = TestClient(create_app(indexed))
    pid = client.get("/api/photos").json()["photos"][0]["id"]

    default = client.get(f"/api/thumb/{pid}")
    assert default.status_code == 200

    variant = client.get(f"/api/thumb/{pid}", params={"size": 640})
    assert variant.status_code == 200
    img = Image.open(io.BytesIO(variant.content))
    assert max(img.size) <= 640
    # The on-demand size is cached content-addressed alongside the thumbnails.
    assert list(indexed.thumbs_dir.rglob("*_640.jpg"))
    # Out-of-range sizes are rejected by the query validation.
    assert client.get(f"/api/thumb/{pid}", params={"size": 5000}).status_code == 422


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


def test_merge_persons(indexed):
    conn = db.connect(indexed.db_path)
    clusters = library.list_clusters(conn)
    assert len(clusters) >= 2, "need two clusters to exercise a merge"

    a = library.assign_cluster(conn, clusters[0]["cluster_id"], name="Ann")
    b = library.assign_cluster(conn, clusters[1]["cluster_id"], name="Bob")

    faces_a = {r["id"] for r in conn.execute("SELECT id FROM faces WHERE person_id=?", (a,))}
    faces_b = {r["id"] for r in conn.execute("SELECT id FROM faces WHERE person_id=?", (b,))}
    assert faces_a and faces_b

    merged = library.merge_persons(conn, source_id=a, target_id=b)
    assert merged == b
    # Ann is gone; all her faces now belong to Bob.
    assert not any(p["name"] == "Ann" for p in library.list_persons(conn))
    now_b = {r["id"] for r in conn.execute("SELECT id FROM faces WHERE person_id=?", (b,))}
    assert now_b == faces_a | faces_b

    import pytest

    with pytest.raises(ValueError):
        library.merge_persons(conn, source_id=b, target_id=b)
    with pytest.raises(ValueError):
        library.merge_persons(conn, source_id=999, target_id=b)


def test_cover_face_picker(indexed):
    import pytest

    conn = db.connect(indexed.db_path)
    clusters = library.list_clusters(conn)
    pid = library.assign_cluster(conn, clusters[0]["cluster_id"], name="Cara")
    faces_of = library.list_person_faces(conn, pid)
    assert faces_of, "named person should have faces with crops"

    library.set_cover_face(conn, pid, faces_of[0]["id"])
    cover = conn.execute("SELECT cover_face_id FROM persons WHERE id=?", (pid,)).fetchone()[0]
    assert cover == faces_of[0]["id"]
    # list_persons surfaces the pinned cover rather than the first-available one.
    cara = next(p for p in library.list_persons(conn) if p["id"] == pid)
    assert cara["cover_face_id"] == faces_of[0]["id"]

    # A face that belongs to nobody (or someone else) is rejected.
    orphan = conn.execute(
        "SELECT id FROM faces WHERE person_id IS NULL LIMIT 1"
    ).fetchone()
    if orphan is not None:
        with pytest.raises(ValueError):
            library.set_cover_face(conn, pid, orphan["id"])


def test_person_management_api(indexed):
    from fastapi.testclient import TestClient

    from photo_atlas.api import create_app

    client = TestClient(create_app(indexed))
    clusters = client.get("/api/clusters").json()["clusters"]
    assert len(clusters) >= 2
    pa = client.post(
        f"/api/clusters/{clusters[0]['cluster_id']}/assign", json={"name": "Dee"}
    ).json()["person_id"]
    pb = client.post(
        f"/api/clusters/{clusters[1]['cluster_id']}/assign", json={"name": "Eve"}
    ).json()["person_id"]

    # Cover picker: list a person's faces, then pin one.
    faces = client.get(f"/api/persons/{pa}/faces").json()["faces"]
    assert faces
    assert (
        client.put(f"/api/persons/{pa}/cover", json={"face_id": faces[0]["id"]}).status_code
        == 200
    )
    # Pinning a face that isn't theirs is a 400.
    other = client.get(f"/api/persons/{pb}/faces").json()["faces"]
    assert (
        client.put(f"/api/persons/{pa}/cover", json={"face_id": other[0]["id"]}).status_code
        == 400
    )

    # Empty rename is rejected.
    assert client.patch(f"/api/persons/{pa}", json={"name": "   "}).status_code == 400

    # Merge Eve into Dee over HTTP.
    assert client.post(f"/api/persons/{pa}/merge", json={"source_id": pb}).json()["ok"] is True
    names = [p["name"] for p in client.get("/api/persons").json()["persons"]]
    assert "Dee" in names and "Eve" not in names


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

    # Multi-select round-trips as repeated query params (OR within a facet).
    countries = [c["value"] for c in facets["countries"]][:2]
    assert len(countries) == 2
    a = client.get("/api/photos", params={"country": countries[0]}).json()
    b = client.get("/api/photos", params={"country": countries[1]}).json()
    both = client.get("/api/photos", params={"country": countries}).json()
    assert both["total"] == a["total"] + b["total"]
    # Filter-aware facets accept the same repeated params.
    fac = client.get("/api/facets", params={"country": countries}).json()
    assert fac["total"] == 24

    # has_faces toggle and the inclusive date range round-trip over HTTP.
    hf = client.get("/api/photos", params={"has_faces": "true"}).json()
    assert hf["total"] == facets["with_faces"]

    # Number-of-people and known-people buckets: each facet is present and every
    # bucket's count matches what filtering by that bucket returns.
    for facet_key, param in (("people", "people"), ("known", "known")):
        assert facets[facet_key]
        for bucket in facets[facet_key]:
            got = client.get("/api/photos", params={param: bucket["value"]}).json()
            assert got["total"] == bucket["count"]
    dated = client.get(
        "/api/photos", params={"date_from": facets["date_min"], "date_to": facets["date_max"]}
    ).json()
    assert dated["total"] >= 1

    clusters = client.get("/api/clusters").json()["clusters"]
    res = client.post(f"/api/clusters/{clusters[0]['cluster_id']}/assign", json={"name": "Carol"})
    assert res.json()["ok"] is True
    assert any(p["name"] == "Carol" for p in client.get("/api/persons").json()["persons"])
