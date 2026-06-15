"""Translate filter dictionaries into SQL queries over the catalog.

Supported filters (all optional, combined with AND across keys). The facet
filters (``person_id``, ``scene``, ``country``, ``city``, ``place``, ``year``,
``camera``) each accept a single value *or* a list of values; a list matches
any of them (OR within the facet, AND across facets).

``person_id``  only photos containing this person (or any of these people).
``scene``      scene tag (people/landscape/food/document/other).
``country``    place country (from GPS).
``city``       place city (from GPS).
``place``      trip/region label mined from the folder name.
``year``       capture year (int or str).
``date_from``  / ``date_to`` -- ISO date bounds on ``taken_at``.
``camera``     exact ``camera_model`` (the value emitted by the camera facet).
``has_faces``  ``True`` -> at least one face.
``q``          free-text substring matched across filename, city, country,
               place label, folder/trip and camera make/model.
``sort``       result ordering: ``newest`` (default), ``oldest``, ``filename``,
               ``filename_desc`` or ``indexed`` (most recently indexed first).
"""

from __future__ import annotations

import sqlite3
from typing import Any

from . import db

# Columns returned for grid/list rows: every photo column except the
# ``scene_scores`` JSON blob, which only the single-photo detail view uses.
# Shipping it for 60 photos/page bloats the payload (and the browser decodes and
# discards it), so the list query selects an explicit set instead of ``p.*``.
_LIST_COLUMNS = ", ".join(
    ["p.id", *(f"p.{c}" for c in db.PHOTO_COLUMNS if c != "scene_scores")]
)


def _as_list(value: Any) -> list[Any]:
    """Normalise a scalar / list / None filter value to a list of non-empty items."""
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return [v for v in value if v is not None and v != ""]
    if value == "":
        return []
    return [value]


def _like_escape(term: str) -> str:
    r"""Escape LIKE wildcards so a user typing ``%`` or ``_`` matches literally.

    Pairs with an ``ESCAPE '\'`` clause on the LIKE; the backslash itself is
    escaped first so it can act as the escape character.
    """
    return term.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


# People-count buckets: token -> SQL predicate on ``p.face_count``. Shared by
# the ``people`` filter and its facet so chip counts and results always agree.
# "1" is a portrait; "2-4"/"5+" are groups of people. Predicates are literal
# (no bound params), so they're safe to splice straight into the WHERE.
PEOPLE_BUCKETS: list[tuple[str, str]] = [
    ("0", "p.face_count = 0"),
    ("1", "p.face_count = 1"),
    ("2-4", "p.face_count BETWEEN 2 AND 4"),
    ("5+", "p.face_count >= 5"),
]
_PEOPLE_PREDICATE = dict(PEOPLE_BUCKETS)

# A single CASE mapping face_count -> bucket token, for the facet's GROUP BY.
_PEOPLE_CASE = (
    "CASE WHEN p.face_count = 0 THEN '0' "
    "WHEN p.face_count = 1 THEN '1' "
    "WHEN p.face_count BETWEEN 2 AND 4 THEN '2-4' ELSE '5+' END"
)


def _where(filters: dict[str, Any]) -> tuple[str, list[Any]]:
    clauses: list[str] = []
    params: list[Any] = []

    persons = _as_list(filters.get("person_id"))
    if persons:
        placeholders = ", ".join(["?"] * len(persons))
        # An EXISTS subquery (rather than a JOIN) keeps the result one row per
        # photo, so callers never need a `DISTINCT` to undo a fan-out. The `ef`
        # alias is local to the subquery and won't collide with any outer `f`.
        clauses.append(
            f"EXISTS (SELECT 1 FROM faces ef WHERE ef.photo_id = p.id "
            f"AND ef.person_id IN ({placeholders}))"
        )
        params.extend(int(p) for p in persons)

    def add_in(column: str, key: str, cast=lambda v: v) -> None:
        values = _as_list(filters.get(key))
        if not values:
            return
        placeholders = ", ".join(["?"] * len(values))
        clauses.append(f"{column} IN ({placeholders})")
        params.extend(cast(v) for v in values)

    add_in("p.scene_type", "scene")
    add_in("p.place_country", "country")
    add_in("p.place_city", "city")
    add_in("p.folder_place", "place")
    add_in("substr(p.taken_at, 1, 4)", "year", cast=str)
    # Camera is matched exactly (the sidebar facet emits whole ``camera_model``
    # values), so the chip's count matches the result count even when one model
    # name is a substring of another. Free-text substring search is via ``q``.
    add_in("p.camera_model", "camera")

    if filters.get("date_from"):
        clauses.append("substr(p.taken_at, 1, 10) >= ?")
        params.append(filters["date_from"])
    if filters.get("date_to"):
        clauses.append("substr(p.taken_at, 1, 10) <= ?")
        params.append(filters["date_to"])
    if filters.get("has_faces"):
        clauses.append("p.face_count > 0")
    # Number-of-people buckets (OR within the facet); unknown tokens are ignored.
    people = [
        _PEOPLE_PREDICATE[b]
        for b in _as_list(filters.get("people"))
        if b in _PEOPLE_PREDICATE
    ]
    if people:
        clauses.append("(" + " OR ".join(people) + ")")
    if filters.get("q"):
        like = f"%{_like_escape(str(filters['q']))}%"
        clauses.append(
            "(p.filename LIKE ? ESCAPE '\\' OR p.place_city LIKE ? ESCAPE '\\' "
            "OR p.place_country LIKE ? ESCAPE '\\' OR p.place_label LIKE ? ESCAPE '\\' "
            "OR p.folder_place LIKE ? ESCAPE '\\' OR p.camera_make LIKE ? ESCAPE '\\' "
            "OR p.camera_model LIKE ? ESCAPE '\\')"
        )
        params.extend([like] * 7)

    where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
    return where, params


# Sort keys exposed by the API/UI -> ORDER BY clause. Each ends with an ``id``
# tiebreaker so paging through ``LIMIT/OFFSET`` is stable when the primary key
# ties (e.g. many photos sharing a folder-synthesised date).
SORTS: dict[str, str] = {
    "newest": "p.taken_at DESC, p.id DESC",
    "oldest": "p.taken_at ASC, p.id ASC",
    "filename": "p.filename COLLATE NOCASE ASC, p.id ASC",
    "filename_desc": "p.filename COLLATE NOCASE DESC, p.id DESC",
    "indexed": "p.indexed_at DESC, p.id DESC",
}


def _order_by(sort: Any) -> str:
    return SORTS.get(sort or "newest", SORTS["newest"])


def search_photos(
    conn: sqlite3.Connection,
    filters: dict[str, Any],
    limit: int = 60,
    offset: int = 0,
    *,
    count: bool = True,
) -> tuple[list[dict], int]:
    """Return ``(rows, total)`` for one page of results.

    With the person filter now expressed as an ``EXISTS`` subquery, the query is
    one row per photo, so neither ``DISTINCT`` (which would otherwise hash every
    wide row, JSON column included) nor ``COUNT(DISTINCT)`` is needed. ``count``
    can be ``False`` for infinite-scroll pages after the first — ``total`` is
    page-invariant, so re-counting the whole filtered set each scroll step is
    wasted work; the sentinel ``-1`` signals "unchanged".
    """

    where, params = _where(filters)
    base = f"FROM photos p{where}"

    total = -1
    if count:
        total = conn.execute(f"SELECT COUNT(*) {base}", params).fetchone()[0]

    order = _order_by(filters.get("sort"))
    rows = conn.execute(
        f"SELECT {_LIST_COLUMNS} {base} ORDER BY {order} LIMIT ? OFFSET ?",
        [*params, int(limit), int(offset)],
    ).fetchall()
    return [dict(r) for r in rows], int(total)


def map_points(
    conn: sqlite3.Connection, filters: dict[str, Any], limit: int = 20000
) -> list[dict]:
    """Geotagged photos matching ``filters`` as ``{id, lat, lon, year}`` points.

    Only rows with both coordinates are returned (the map can't place the rest).
    ``limit`` bounds the payload for very large libraries.
    """

    where, params = _where(filters)
    glue = " AND " if where else " WHERE "
    sql = (
        "SELECT p.id, p.lat, p.lon, substr(p.taken_at, 1, 4) AS year "
        f"FROM photos p{where}{glue}p.lat IS NOT NULL AND p.lon IS NOT NULL "
        "LIMIT ?"
    )
    rows = conn.execute(sql, [*params, int(limit)]).fetchall()
    return [dict(r) for r in rows]


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
        where, params = _where(sub)
        order = f"{column} DESC" if order_by_value else "c DESC, v"
        # One row per photo (person filter is an EXISTS, no join fan-out), so a
        # plain COUNT(*) per group equals the old COUNT(DISTINCT p.id).
        sql = (
            f"SELECT {column} AS v, COUNT(*) AS c "
            f"FROM photos p{where} GROUP BY v ORDER BY {order}"
        )
        return [
            {"value": r["v"], "count": r["c"]}
            for r in conn.execute(sql, params).fetchall()
            if r["v"] is not None
        ]

    def person_facet() -> list[dict]:
        sub = {k: v for k, v in filters.items() if k != "person_id"}
        where, params = _where(sub)
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

    def with_faces_count() -> int:
        # Photos containing at least one face, under the other active filters.
        sub = {k: v for k, v in filters.items() if k != "has_faces"}
        where, params = _where(sub)
        glue = " AND " if where else " WHERE "
        sql = f"SELECT COUNT(*) FROM photos p{where}{glue}p.face_count > 0"
        return int(conn.execute(sql, params).fetchone()[0])

    def people_facet() -> list[dict]:
        # Number-of-people buckets (portrait = 1, group = 2+), in canonical order,
        # filter-aware against the other dimensions but not the people bucket itself.
        sub = {k: v for k, v in filters.items() if k != "people"}
        where, params = _where(sub)
        sql = f"SELECT {_PEOPLE_CASE} AS v, COUNT(*) AS c FROM photos p{where} GROUP BY v"
        counts = {r["v"]: r["c"] for r in conn.execute(sql, params).fetchall()}
        return [
            {"value": tok, "count": counts[tok]}
            for tok, _ in PEOPLE_BUCKETS
            if counts.get(tok)
        ]

    drow = conn.execute(
        "SELECT MIN(substr(taken_at,1,10)), MAX(substr(taken_at,1,10)) "
        "FROM photos WHERE taken_at IS NOT NULL"
    ).fetchone()

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
        "people": people_facet(),
        "with_faces": with_faces_count(),
        "date_min": drow[0],
        "date_max": drow[1],
    }
