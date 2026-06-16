"""Person-management edge cases: name collisions and cover-face integrity."""

from __future__ import annotations

import pytest

from photo_atlas import db, library


def test_rename_to_existing_name_raises_value_error(tmp_path):
    conn = db.connect(tmp_path / "a.db")
    db.get_or_create_person(conn, "Alice")
    bob = db.get_or_create_person(conn, "Bob")
    # Renaming Bob to "Alice" violates the UNIQUE(name) constraint; the library
    # must surface a clean ValueError (the API maps it to 409), not let a raw
    # sqlite3.IntegrityError bubble up as a 500.
    with pytest.raises(ValueError):
        library.rename_person(conn, bob, "Alice")


def test_rename_to_own_name_is_fine(tmp_path):
    conn = db.connect(tmp_path / "a.db")
    bob = db.get_or_create_person(conn, "Bob")
    library.rename_person(conn, bob, "Bob")  # no-op, must not raise


def _add_face(conn, photo_id, person_id, crop_path):
    cur = conn.execute(
        "INSERT INTO faces (photo_id, person_id, crop_path) VALUES (?, ?, ?)",
        (photo_id, person_id, crop_path),
    )
    conn.commit()
    return cur.lastrowid


def test_list_persons_heals_dangling_cover_face(tmp_path):
    conn = db.connect(tmp_path / "a.db")
    pid = db.get_or_create_person(conn, "Alice")
    photo = conn.execute(
        "INSERT INTO photos (path, filename) VALUES ('/x.jpg', 'x.jpg')"
    ).lastrowid
    f1 = _add_face(conn, photo, pid, "/c1.jpg")
    f2 = _add_face(conn, photo, pid, "/c2.jpg")

    library.set_cover_face(conn, pid, f1)
    # The pinned cover face is removed (e.g. its photo was deleted/reindexed).
    conn.execute("DELETE FROM faces WHERE id=?", (f1,))
    conn.commit()

    cover = library.list_persons(conn)[0]["cover_face_id"]
    # Must fall back to a still-present crop, not serve the now-404 pinned id.
    assert cover == f2


def _a_face_id(client):
    photos = client.get("/api/photos").json()["photos"]
    photo = next(p for p in photos if p["face_count"] > 0)
    faces = client.get(f"/api/photos/{photo['id']}").json()["faces"]
    return faces[0]["id"]


def test_rename_collision_returns_409(client, person_id):
    # person_id already created the person "Subject"; make a second person.
    other = client.post(
        f"/api/faces/{_a_face_id(client)}/assign", json={"name": "Other"}
    ).json()["person_id"]
    resp = client.patch(f"/api/persons/{other}", json={"name": "Subject"})
    assert resp.status_code == 409
