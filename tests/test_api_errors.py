"""Error paths and media endpoints of the FastAPI surface.

The happy paths live in ``test_photo_atlas.py``; this file pins the 404 / 400 /
422 branches and the media fallbacks so a regression there can't slip through.
"""

from __future__ import annotations

from pathlib import Path

from photo_atlas import db, indexer


def _first_photo_id(client):
    return client.get("/api/photos?limit=1").json()["photos"][0]["id"]


def _a_face_with_crop(conn):
    return conn.execute(
        "SELECT id, crop_path FROM faces WHERE crop_path IS NOT NULL LIMIT 1"
    ).fetchone()


# -- not-found branches ----------------------------------------------------
def test_photo_detail_404(client):
    assert client.get("/api/photos/999999").status_code == 404


def test_image_404_for_unknown_id(client):
    assert client.get("/api/image/999999").status_code == 404


def test_preview_404_for_unknown_id(client):
    assert client.get("/api/preview/999999").status_code == 404


def test_thumb_404_for_unknown_id(client):
    assert client.get("/api/thumb/999999").status_code == 404


def test_face_404_for_unknown_id(client):
    assert client.get("/api/face/999999").status_code == 404


# -- media happy paths -----------------------------------------------------
def test_image_and_preview_and_thumb_served(client):
    pid = _first_photo_id(client)
    assert client.get(f"/api/image/{pid}").status_code == 200
    assert client.get(f"/api/preview/{pid}").status_code == 200
    assert client.get(f"/api/thumb/{pid}").status_code == 200
    # The retina 2x variant is generated and cached on demand.
    r = client.get(f"/api/thumb/{pid}?size=640")
    assert r.status_code == 200


def test_thumb_size_out_of_range_is_422(client):
    pid = _first_photo_id(client)
    assert client.get(f"/api/thumb/{pid}?size=4096").status_code == 422


# -- query-param validation ------------------------------------------------
def test_negative_offset_is_422(client):
    assert client.get("/api/photos?offset=-5").status_code == 422


def test_zero_limit_is_422(client):
    assert client.get("/api/photos?limit=0").status_code == 422


def test_malformed_date_is_422(client):
    assert client.get("/api/photos?date_from=not-a-date").status_code == 422
    assert client.get("/api/facets?date_to=garbage").status_code == 422


# -- face-crop re-save (recover from a missing crop) -----------------------
def test_missing_face_crop_is_regenerated_on_demand(client, indexed):
    """A face whose crop file is gone (or never written: ``crop_path=NULL``)
    rebuilds the crop from the source photo instead of 404ing forever."""
    conn = db.connect(indexed.db_path)
    face = _a_face_with_crop(conn)
    assert face is not None, "demo library should produce at least one face crop"
    fid = face["id"]
    # Simulate the index-time write failure: drop the file and the stored path.
    Path(face["crop_path"]).unlink()
    conn.execute("UPDATE faces SET crop_path=NULL WHERE id=?", (fid,))
    conn.commit()
    conn.close()

    assert client.get(f"/api/face/{fid}").status_code == 200

    conn = db.connect(indexed.db_path)
    row = conn.execute("SELECT crop_path FROM faces WHERE id=?", (fid,)).fetchone()
    conn.close()
    assert row["crop_path"] and Path(row["crop_path"]).exists()


def test_regenerate_face_crop_none_when_source_gone(client, indexed):
    conn = db.connect(indexed.db_path)
    face = _a_face_with_crop(conn)
    fid = face["id"]
    # Point the owning photo at a path that no longer exists: no retry possible.
    conn.execute(
        "UPDATE photos SET path=? WHERE id=(SELECT photo_id FROM faces WHERE id=?)",
        ("/nonexistent/gone.jpg", fid),
    )
    conn.execute("UPDATE faces SET crop_path=NULL WHERE id=?", (fid,))
    conn.commit()
    assert indexer.regenerate_face_crop(conn, indexed, fid) is None
    assert indexer.regenerate_face_crop(conn, indexed, 999999) is None
    conn.close()

    assert client.get(f"/api/face/{fid}").status_code == 404


def test_wellformed_date_is_accepted(client):
    assert client.get("/api/photos?date_from=2020-01-01&date_to=2026-12-31").status_code == 200


# -- cross-origin write guard ----------------------------------------------
def _a_face_id(client):
    photos = client.get("/api/photos").json()["photos"]
    photo = next(p for p in photos if p["face_count"] > 0)
    return client.get(f"/api/photos/{photo['id']}").json()["faces"][0]["id"]


def test_cross_origin_write_is_forbidden(client):
    fid = _a_face_id(client)
    resp = client.post(
        f"/api/faces/{fid}/unassign", headers={"Origin": "http://evil.example"}
    )
    assert resp.status_code == 403


def test_same_origin_write_is_allowed(client):
    fid = _a_face_id(client)
    # TestClient's Host is "testserver"; a same-origin browser sends a matching Origin.
    resp = client.post(
        f"/api/faces/{fid}/unassign", headers={"Origin": "http://testserver"}
    )
    assert resp.status_code == 200


def test_write_without_origin_is_allowed(client):
    fid = _a_face_id(client)
    assert client.post(f"/api/faces/{fid}/unassign").status_code == 200


def test_cross_origin_read_is_allowed(client):
    # Only state-changing methods are guarded; GETs are unaffected.
    assert client.get("/api/photos", headers={"Origin": "http://evil.example"}).status_code == 200


# -- map -------------------------------------------------------------------
def test_map_endpoint_returns_geotagged_points(client):
    data = client.get("/api/map").json()
    assert data["points"]  # demo photos all carry GPS EXIF
    pt = data["points"][0]
    assert pt["lat"] is not None and pt["lon"] is not None and "id" in pt


def test_map_endpoint_respects_filters(client):
    # A country that doesn't exist yields no points but still a valid shape.
    data = client.get("/api/map?country=Nowhere").json()
    assert data["points"] == []


def test_map_endpoint_caps_at_configured_limit(indexed):
    from fastapi.testclient import TestClient

    from photo_atlas.api import create_app

    indexed.map_point_limit = 5  # AtlasConfig is a mutable dataclass
    client = TestClient(create_app(indexed))
    data = client.get("/api/map").json()
    assert len(data["points"]) == 5  # the 20 geotagged demo photos are capped


# -- validation / mutation error branches ----------------------------------
def test_rename_empty_name_is_400(client, person_id):
    assert client.patch(f"/api/persons/{person_id}", json={"name": "   "}).status_code == 400


def test_rename_and_delete_person(client, person_id):
    assert client.patch(f"/api/persons/{person_id}", json={"name": "Renamed"}).json()["ok"]
    assert any(p["name"] == "Renamed" for p in client.get("/api/persons").json()["persons"])
    assert client.delete(f"/api/persons/{person_id}").json()["ok"]


def test_merge_into_missing_person_is_400(client, person_id):
    # Merging a nonexistent source into the person is rejected.
    r = client.post(f"/api/persons/{person_id}/merge", json={"source_id": 999999})
    assert r.status_code == 400


def test_cover_face_not_owned_is_400(client, person_id):
    assert client.put(
        f"/api/persons/{person_id}/cover", json={"face_id": 999999}
    ).status_code == 400


def test_assign_face_without_name_or_id_is_400(client):
    pid = _first_photo_id(client)
    faces = client.get(f"/api/photos/{pid}").json()["faces"]
    if not faces:
        return
    r = client.post(f"/api/faces/{faces[0]['id']}/assign", json={})
    assert r.status_code == 400


def test_assign_and_unassign_single_face(client):
    pid = _first_photo_id(client)
    faces = client.get(f"/api/photos/{pid}").json()["faces"]
    if not faces:
        return
    fid = faces[0]["id"]
    assert client.post(f"/api/faces/{fid}/assign", json={"name": "Casey"}).json()["ok"]
    detail = client.get(f"/api/photos/{pid}").json()
    assert any(f["person_name"] == "Casey" for f in detail["faces"])
    # Send it back to unknown.
    assert client.post(f"/api/faces/{fid}/unassign").json()["ok"]
    detail2 = client.get(f"/api/photos/{pid}").json()
    assert all(f["id"] != fid or f["person_id"] is None for f in detail2["faces"])


def test_assign_cluster_without_name_is_400(client):
    clusters = client.get("/api/clusters").json()["clusters"]
    if not clusters:
        return
    r = client.post(f"/api/clusters/{clusters[0]['cluster_id']}/assign", json={})
    assert r.status_code == 400
