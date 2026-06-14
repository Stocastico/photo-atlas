"""Translate filter dictionaries into SQL queries over the catalog.

Supported filters (all optional, combined with AND):

``person_id``  only photos containing this person.
``scene``      scene tag (people/landscape/food/document/other).
``country``    place country (from GPS).
``city``       place city (from GPS).
``place``      trip/region label mined from the folder name.
``year``       capture year (int or str).
``date_from``  / ``date_to`` -- ISO date bounds on ``taken_at``.
``camera``     camera model substring.
``has_faces``  ``True`` -> at least one face.
``q``          free-text substring matched across filename, city, country,
               place label, folder/trip and camera make/model.
"""

from __future__ import annotations

import sqlite3
from typing import Any


def _where(filters: dict[str, Any]) -> tuple[str, list[Any], str]:
    clauses: list[str] = []
    params: list[Any] = []
    join = ""

    person_id = filters.get("person_id")
    if person_id:
        join = "JOIN faces f ON f.photo_id = p.id AND f.person_id = ?"
        params.append(int(person_id))

    if filters.get("scene"):
        clauses.append("p.scene_type = ?")
        params.append(filters["scene"])
    if filters.get("country"):
        clauses.append("p.place_country = ?")
        params.append(filters["country"])
    if filters.get("city"):
        clauses.append("p.place_city = ?")
        params.append(filters["city"])
    if filters.get("place"):
        clauses.append("p.folder_place = ?")
        params.append(filters["place"])
    if filters.get("year"):
        clauses.append("substr(p.taken_at, 1, 4) = ?")
        params.append(str(filters["year"]))
    if filters.get("date_from"):
        clauses.append("p.taken_at >= ?")
        params.append(filters["date_from"])
    if filters.get("date_to"):
        clauses.append("p.taken_at <= ?")
        params.append(filters["date_to"])
    if filters.get("camera"):
        clauses.append("p.camera_model LIKE ?")
        params.append(f"%{filters['camera']}%")
    if filters.get("has_faces"):
        clauses.append("p.face_count > 0")
    if filters.get("q"):
        like = f"%{filters['q']}%"
        clauses.append(
            "(p.filename LIKE ? OR p.place_city LIKE ? OR p.place_country LIKE ? "
            "OR p.place_label LIKE ? OR p.folder_place LIKE ? "
            "OR p.camera_make LIKE ? OR p.camera_model LIKE ?)"
        )
        params.extend([like] * 7)

    where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
    return where, params, join


def search_photos(
    conn: sqlite3.Connection, filters: dict[str, Any], limit: int = 60, offset: int = 0
) -> tuple[list[dict], int]:
    where, params, join = _where(filters)
    base = f"FROM photos p {join}{where}"

    total = conn.execute(f"SELECT COUNT(DISTINCT p.id) {base}", params).fetchone()[0]

    order = "p.taken_at DESC" if filters.get("sort") != "oldest" else "p.taken_at ASC"
    rows = conn.execute(
        f"SELECT DISTINCT p.* {base} ORDER BY {order} LIMIT ? OFFSET ?",
        [*params, int(limit), int(offset)],
    ).fetchall()
    return [dict(r) for r in rows], int(total)


def photo_detail(conn: sqlite3.Connection, photo_id: int) -> dict | None:
    row = conn.execute("SELECT * FROM photos WHERE id=?", (photo_id,)).fetchone()
    if row is None:
        return None
    photo = dict(row)
    faces = conn.execute(
        "SELECT f.id, f.person_id, f.cluster_id, f.bbox_x, f.bbox_y, f.bbox_w, "
        "f.bbox_h, f.confidence, f.crop_path, pr.name AS person_name "
        "FROM faces f LEFT JOIN persons pr ON pr.id = f.person_id "
        "WHERE f.photo_id=? ORDER BY f.id",
        (photo_id,),
    ).fetchall()
    photo["faces"] = [dict(f) for f in faces]
    return photo


def facets(conn: sqlite3.Connection, filters: dict[str, Any] | None = None) -> dict:
    """Aggregate counts used to build the filter sidebar.

    Counts are *filter-aware*: each facet reflects the other active filters but
    not its own dimension, so the number next to a chip is how many photos you
    would see by adding that value to the current selection (the classic
    faceted-search behaviour). ``total`` stays the unfiltered library size.
    """

    filters = filters or {}

    def facet(column: str, own_key: str, *, order_by_value: bool = False) -> list[dict]:
        # Exclude this facet's own filter so all of its options stay visible.
        sub = {k: v for k, v in filters.items() if k != own_key}
        where, params, join = _where(sub)
        order = f"{column} DESC" if order_by_value else "c DESC, v"
        sql = (
            f"SELECT {column} AS v, COUNT(DISTINCT p.id) AS c "
            f"FROM photos p {join}{where} GROUP BY v ORDER BY {order}"
        )
        return [
            {"value": r["v"], "count": r["c"]}
            for r in conn.execute(sql, params).fetchall()
            if r["v"] is not None
        ]

    def person_facet() -> list[dict]:
        sub = {k: v for k, v in filters.items() if k != "person_id"}
        where, params, _join = _where(sub)
        sql = (
            "SELECT pr.id AS id, pr.name AS name, COUNT(DISTINCT p.id) AS c "
            "FROM persons pr JOIN faces f ON f.person_id = pr.id "
            f"JOIN photos p ON p.id = f.photo_id {where} "
            "GROUP BY pr.id ORDER BY c DESC, pr.name"
        )
        return [
            {"id": r["id"], "name": r["name"], "count": r["c"]}
            for r in conn.execute(sql, params).fetchall()
        ]

    total = conn.execute("SELECT COUNT(*) FROM photos").fetchone()[0]
    return {
        "total": int(total),
        "scenes": facet("p.scene_type", "scene"),
        "countries": facet("p.place_country", "country"),
        "cities": facet("p.place_city", "city"),
        "places": facet("p.folder_place", "place"),
        "years": facet("substr(p.taken_at,1,4)", "year", order_by_value=True),
        "cameras": facet("p.camera_model", "camera"),
        "persons": person_facet(),
    }
