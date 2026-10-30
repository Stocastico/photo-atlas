"""Catalog search: filtered listing, facets, map points and photo detail.

The filter-dict -> SQL translation lives in :mod:`photo_atlas.filters`, the
embedding-based ranking in :mod:`photo_atlas.semantic` and the derived groupings
(memories, trips, bursts) in :mod:`photo_atlas.groups`. They are re-exported here so
``photo_atlas.search`` stays the single public entry point.
"""

from __future__ import annotations

import sqlite3
from typing import Any

from .classify import SCENE_LABELS as _SCENE_LABELS  # noqa: F401
from .filters import (  # noqa: F401
    _KIND_PREDICATE,
    _KNOWN_CASE,
    _KNOWN_COL,
    _KNOWN_PREDICATE,
    _LIST_COLUMNS,
    _PEOPLE_CASE,
    _PEOPLE_PREDICATE,
    KNOWN_BUCKETS,
    PEOPLE_BUCKETS,
    PICTURE_TYPES,
    SORTS,
    _as_list,
    _like_escape,
    _order_by,
    _where,
)
from .groups import (  # noqa: F401
    _build_trip,
    _cover_score,
    _dominant_place,
    _haversine_km,
    _parse_taken_at,
    detect_trips,
    find_burst_groups,
    on_this_day,
    phash_distance,
)
from .semantic import (  # noqa: F401
    FaceIndex,
    RegionIndex,
    SemanticIndex,
    _normalize,
    _rows_for_ranked,
    grounded_search,
    semantic_search,
    similar_faces,
    similar_photos,
)


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


def taken_source_counts(conn: sqlite3.Connection) -> dict[str, int]:
    """How many photos got their capture date from each source.

    Keys are the ``taken_source`` values (``exif`` / ``filename`` / ``folder`` /
    ``mtime`` / ``video``); the counts let a ``stats`` run report how much the
    filename/folder mining recovered over the bare filesystem-mtime fallback.
    """

    rows = conn.execute(
        "SELECT COALESCE(taken_source, 'mtime') AS src, COUNT(*) AS n "
        "FROM photos GROUP BY src"
    ).fetchall()
    return {r["src"]: r["n"] for r in rows}


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

    def favorites_count() -> int:
        # Starred photos under the other active filters (powers the ★ chip).
        sub = {k: v for k, v in filters.items() if k != "favorite"}
        where, params = _where(sub)
        glue = " AND " if where else " WHERE "
        sql = f"SELECT COUNT(*) FROM photos p{where}{glue}p.favorite = 1"
        return int(conn.execute(sql, params).fetchone()[0])

    def hidden_count() -> int:
        # Hidden photos under the other active filters (powers the 🙈 chip). Forces
        # ``hidden=True`` so the default exclusion doesn't zero it out.
        sub = {k: v for k, v in filters.items() if k != "hidden"}
        sub["hidden"] = True
        where, params = _where(sub)
        return int(conn.execute(f"SELECT COUNT(*) FROM photos p{where}", params).fetchone()[0])

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

    def known_facet() -> list[dict]:
        # Buckets by how many faces in a photo are assigned to a named person.
        sub = {k: v for k, v in filters.items() if k != "known"}
        where, params = _where(sub)
        sql = f"SELECT {_KNOWN_CASE} AS v, COUNT(*) AS c FROM photos p{where} GROUP BY v"
        counts = {r["v"]: r["c"] for r in conn.execute(sql, params).fetchall()}
        return [
            {"value": tok, "count": counts[tok]}
            for tok, _ in KNOWN_BUCKETS
            if counts.get(tok)
        ]

    def kinds_facet() -> list[dict]:
        # Unified "type of picture": portrait/group (face_count) + the scene tags.
        # The tokens overlap (a photo can be both a portrait and a "food" shot), so
        # a single GROUP BY can't bucket them — count each token independently with
        # one conditional-aggregate scan, filter-aware against the other dimensions.
        sub = {k: v for k, v in filters.items() if k != "kind"}
        where, params = _where(sub)
        parts = ", ".join(
            f"SUM(CASE WHEN {pred} THEN 1 ELSE 0 END)" for _, pred in PICTURE_TYPES
        )
        row = conn.execute(f"SELECT {parts} FROM photos p{where}", params).fetchone()
        return [
            {"value": tok, "count": int(c)}
            for (tok, _), c in zip(PICTURE_TYPES, row, strict=True)
            if c
        ]

    def date_bounds() -> tuple:
        # The date slider's bounds are a facet too: filter-aware against every
        # other dimension, but not its own (date_from/date_to), so the handles
        # span exactly the dates reachable under the current selection.
        sub = {k: v for k, v in filters.items() if k not in ("date_from", "date_to")}
        where, params = _where(sub)
        glue = " AND " if where else " WHERE "
        sql = (
            "SELECT MIN(substr(p.taken_at,1,10)), MAX(substr(p.taken_at,1,10)) "
            f"FROM photos p{where}{glue}p.taken_at IS NOT NULL"
        )
        return conn.execute(sql, params).fetchone()

    drow = date_bounds()

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
        "known": known_facet(),
        "kinds": kinds_facet(),
        "with_faces": with_faces_count(),
        "favorites": favorites_count(),
        "hidden": hidden_count(),
        "date_min": drow[0],
        "date_max": drow[1],
    }
