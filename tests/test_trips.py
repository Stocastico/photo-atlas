"""Trip auto-detection — group the library into trips from time gaps + GPS.

A pure pass over ``taken_at`` (and GPS) that splits the chronological photo
stream on multi-day breaks or far geographic jumps, dropping tiny clusters and
labelling each run by place. Surfaced over ``/api/trips``.
"""

from __future__ import annotations

from photo_atlas import db, search
from photo_atlas.config import AtlasConfig


def _insert(conn, path, taken_at, **cols):
    cols.setdefault("filename", path.rsplit("/", 1)[-1])
    return db.upsert_photo(conn, {"path": path, "taken_at": taken_at, **cols})


def _rome_week(conn, n=4):
    """A four-day Rome trip (one geotagged photo per day)."""
    for i in range(n):
        _insert(
            conn, f"/rome{i}.jpg", f"2019-06-0{i + 1}T10:00:00",
            place_label="Rome, Italy", lat=41.9, lon=12.5,
        )


def test_split_on_time_gap(tmp_path):
    conn = db.connect(tmp_path / "s.db")
    try:
        _rome_week(conn)  # June 1–4
        # …then a separate four-day trip a month later.
        for i in range(4):
            _insert(
                conn, f"/paris{i}.jpg", f"2019-07-1{i}T10:00:00",
                place_label="Paris, France", lat=48.85, lon=2.35,
            )
        trips = search.detect_trips(conn, min_photos=3)
        assert len(trips) == 2
        # Newest-first ordering.
        assert trips[0]["place"] == "Paris, France"
        assert trips[1]["place"] == "Rome, Italy"
        assert trips[1]["start"] == "2019-06-01" and trips[1]["end"] == "2019-06-04"
        assert trips[1]["count"] == 4
        # Centroid + a geotagged cover are filled in.
        assert round(trips[1]["lat"], 1) == 41.9
        assert trips[1]["cover_id"] is not None
    finally:
        conn.close()


def test_split_on_far_gps_jump_within_day_window(tmp_path):
    """Two cities the same day split even though the time gap is small."""
    conn = db.connect(tmp_path / "s.db")
    try:
        for i in range(3):
            _insert(conn, f"/a{i}.jpg", f"2020-03-01T0{i}:00:00", lat=40.0, lon=-74.0)
        # A few hours later but ~5500 km away.
        for i in range(3):
            _insert(conn, f"/b{i}.jpg", f"2020-03-01T1{i}:00:00", lat=51.5, lon=-0.12)
        trips = search.detect_trips(conn, min_photos=3, gap_km=200)
        assert len(trips) == 2
    finally:
        conn.close()


def test_no_split_for_nearby_same_window(tmp_path):
    """A small hop within the same window stays one trip."""
    conn = db.connect(tmp_path / "s.db")
    try:
        for i in range(6):
            _insert(
                conn, f"/c{i}.jpg", f"2020-03-0{i + 1}T10:00:00",
                lat=40.0 + i * 0.01, lon=-74.0,
            )
        trips = search.detect_trips(conn, min_photos=3, gap_km=200)
        assert len(trips) == 1 and trips[0]["count"] == 6
    finally:
        conn.close()


def test_min_photos_drops_stray_clusters(tmp_path):
    conn = db.connect(tmp_path / "s.db")
    try:
        _rome_week(conn)                                   # 4 photos → kept
        _insert(conn, "/stray.jpg", "2019-09-01T10:00:00")  # lone shot → dropped
        trips = search.detect_trips(conn, min_photos=4)
        assert len(trips) == 1 and trips[0]["count"] == 4
    finally:
        conn.close()


def test_undated_photos_are_ignored(tmp_path):
    conn = db.connect(tmp_path / "s.db")
    try:
        _rome_week(conn)
        _insert(conn, "/nodate.jpg", None)
        trips = search.detect_trips(conn, min_photos=3)
        assert len(trips) == 1 and trips[0]["count"] == 4  # the undated one isn't counted
    finally:
        conn.close()


def test_place_label_falls_back_to_folder_then_geo(tmp_path):
    conn = db.connect(tmp_path / "s.db")
    try:
        for i in range(3):
            _insert(conn, f"/f{i}.jpg", f"2021-01-0{i + 1}T10:00:00", folder_place="Alps 2021")
        for i in range(3):
            _insert(
                conn, f"/g{i}.jpg", f"2021-05-0{i + 1}T10:00:00",
                place_city="Oslo", place_country="Norway",
            )
        trips = sorted(search.detect_trips(conn, min_photos=3), key=lambda t: t["start"])
        assert trips[0]["place"] == "Alps 2021"
        assert trips[1]["place"] == "Oslo, Norway"
    finally:
        conn.close()


# -- API --------------------------------------------------------------------
def test_api_trips(tmp_path):
    config = AtlasConfig(home=tmp_path / "lib").ensure_dirs()
    conn = db.connect(config.db_path)
    for i in range(4):
        _insert(conn, f"/t{i}.jpg", f"2022-08-0{i + 1}T10:00:00", place_label="Lisbon")
    conn.commit()
    conn.close()
    from fastapi.testclient import TestClient

    from photo_atlas.api import create_app

    client = TestClient(create_app(config))
    data = client.get("/api/trips").json()
    assert data["count"] == 1
    assert data["trips"][0]["place"] == "Lisbon"
    assert data["trips"][0]["count"] == 4
