"""'On this day' memories — photos taken on the same month/day across years.

A pure SQL slice on the ``taken_at`` month/day, grouped by year (newest first),
surfaced over ``/api/memories`` (defaulting to the server's current date).
"""

from __future__ import annotations

import datetime

from photo_atlas import db, search
from photo_atlas.config import AtlasConfig


def _insert(conn, path, taken_at, **cols):
    cols.setdefault("filename", path.rsplit("/", 1)[-1])
    return db.upsert_photo(conn, {"path": path, "taken_at": taken_at, **cols})


def test_on_this_day_groups_by_year_newest_first(tmp_path):
    conn = db.connect(tmp_path / "s.db")
    try:
        _insert(conn, "/a.jpg", "2021-06-16T10:00:00")
        _insert(conn, "/b.jpg", "2020-06-16T09:00:00")
        _insert(conn, "/c.jpg", "2020-06-16T18:30:00")
        _insert(conn, "/d.jpg", "2019-06-15T12:00:00")  # different day
        _insert(conn, "/e.jpg", "2018-12-16T12:00:00")  # different month
        _insert(conn, "/f.jpg", None)                    # no date at all

        groups = search.on_this_day(conn, 6, 16)
        assert [g["year"] for g in groups] == ["2021", "2020"]
        assert [g["count"] for g in groups] == [1, 2]
        # Each group carries its (capped) photos, lightened of the scene_scores blob.
        assert len(groups[1]["photos"]) == 2
        assert "scene_scores" not in groups[1]["photos"][0]
    finally:
        conn.close()


def test_on_this_day_per_year_caps_photos_but_not_count(tmp_path):
    conn = db.connect(tmp_path / "s.db")
    try:
        for i in range(5):
            _insert(conn, f"/p{i}.jpg", f"2020-06-16T0{i}:00:00")
        groups = search.on_this_day(conn, 6, 16, per_year=2)
        assert len(groups) == 1
        assert groups[0]["count"] == 5          # full count
        assert len(groups[0]["photos"]) == 2    # capped sample
    finally:
        conn.close()


def test_on_this_day_empty_when_no_match(tmp_path):
    conn = db.connect(tmp_path / "s.db")
    try:
        _insert(conn, "/a.jpg", "2020-01-01T00:00:00")
        assert search.on_this_day(conn, 6, 16) == []
    finally:
        conn.close()


# -- API --------------------------------------------------------------------
def _client(tmp_path, rows):
    config = AtlasConfig(home=tmp_path / "lib").ensure_dirs()
    conn = db.connect(config.db_path)
    for path, taken in rows:
        _insert(conn, path, taken)
    conn.commit()
    conn.close()
    from fastapi.testclient import TestClient

    from photo_atlas.api import create_app

    return TestClient(create_app(config))


def test_api_memories_explicit_date(tmp_path):
    client = _client(tmp_path, [("/a.jpg", "2021-06-16"), ("/b.jpg", "2020-06-16")])
    data = client.get("/api/memories", params={"month": 6, "day": 16}).json()
    assert data["month"] == 6 and data["day"] == 16
    assert data["total"] == 2
    assert [g["year"] for g in data["groups"]] == ["2021", "2020"]


def test_api_memories_defaults_to_today(tmp_path):
    today = datetime.date.today()
    iso = f"{today.year}-{today.month:02d}-{today.day:02d}T08:00:00"
    client = _client(tmp_path, [("/today.jpg", iso)])
    data = client.get("/api/memories").json()
    assert data["month"] == today.month and data["day"] == today.day
    assert data["total"] == 1


def test_api_memories_rejects_bad_date(tmp_path):
    client = _client(tmp_path, [])
    assert client.get("/api/memories", params={"month": 13, "day": 1}).status_code == 422
    assert client.get("/api/memories", params={"month": 6, "day": 40}).status_code == 422
