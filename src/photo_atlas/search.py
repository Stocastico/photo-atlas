"""Translate filter dictionaries into SQL queries over the catalog.

Supported filters (all optional, combined with AND across keys). The facet
filters (``person_id``, ``scene``, ``country``, ``city``, ``place``, ``year``,
``camera``) each accept a single value *or* a list of values; a list matches
any of them (OR within the facet, AND across facets).

``person_id``  only photos containing this person (or any of these people).
``scene``      scene tag (see ``classify.SCENE_LABELS``).
``country``    place country (from GPS).
``city``       place city (from GPS).
``place``      trip/region label mined from the folder name.
``year``       capture year (int or str).
``date_from``  / ``date_to`` -- ISO date bounds on ``taken_at``.
``camera``     exact ``camera_model`` (the value emitted by the camera facet).
``has_faces``  ``True`` -> at least one face.
``favorite``   ``True`` -> only starred photos.
``q``          free-text substring matched across filename, city, country,
               place label, folder/trip and camera make/model.
``sort``       result ordering: ``newest`` (default), ``oldest``, ``filename``,
               ``filename_desc`` or ``indexed`` (most recently indexed first).
"""

from __future__ import annotations

import sqlite3
from typing import Any

import numpy as np

from . import db

# Columns returned for grid/list rows: every photo column except the
# ``scene_scores`` JSON blob, which only the single-photo detail view uses.
# Shipping it for 60 photos/page bloats the payload (and the browser decodes and
# discards it), so the list query selects an explicit set instead of ``p.*``.
# ``favorite`` is appended explicitly: it's deliberately not in ``PHOTO_COLUMNS``
# (so a re-index can't reset a user's star), but the grid needs it per card to
# render the star state.
_LIST_COLUMNS = ", ".join(
    ["p.id", *(f"p.{c}" for c in db.PHOTO_COLUMNS if c != "scene_scores"), "p.favorite"]
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

# Number of *known* (named) people in a photo: how many of its faces are assigned
# to a person. Read straight from the trigger-maintained ``named_face_count``
# column (denormalised in ``db``), so this is a plain indexed read rather than a
# per-row correlated subquery. Buckets: nobody identified / one / two-or-more.
_KNOWN_COL = "p.named_face_count"
KNOWN_BUCKETS: list[tuple[str, str]] = [
    ("0", f"{_KNOWN_COL} = 0"),
    ("1", f"{_KNOWN_COL} = 1"),
    ("2+", f"{_KNOWN_COL} >= 2"),
]
_KNOWN_PREDICATE = dict(KNOWN_BUCKETS)
_KNOWN_CASE = (
    f"CASE WHEN {_KNOWN_COL} = 0 THEN '0' "
    f"WHEN {_KNOWN_COL} = 1 THEN '1' ELSE '2+' END"
)


def _where(filters: dict[str, Any]) -> tuple[str, list[Any]]:
    clauses: list[str] = []
    params: list[Any] = []

    persons = _as_list(filters.get("person_id"))
    if persons:
        # ``person_mode='all'`` requires every selected person to be present
        # (one AND-ed EXISTS each); the default 'any' matches any of them (a
        # single EXISTS over an IN). EXISTS keeps the result one row per photo,
        # so callers never need a DISTINCT. The ``ef`` alias is local to each
        # subquery and won't collide with any outer ``f``.
        if filters.get("person_mode") == "all":
            for pid in persons:
                clauses.append(
                    "EXISTS (SELECT 1 FROM faces ef "
                    "WHERE ef.photo_id = p.id AND ef.person_id = ?)"
                )
                params.append(int(pid))
        else:
            placeholders = ", ".join(["?"] * len(persons))
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
    if filters.get("favorite"):
        clauses.append("p.favorite = 1")
    # Number-of-people buckets (OR within the facet); unknown tokens are ignored.
    people = [
        _PEOPLE_PREDICATE[b]
        for b in _as_list(filters.get("people"))
        if b in _PEOPLE_PREDICATE
    ]
    if people:
        clauses.append("(" + " OR ".join(people) + ")")
    # Number-of-known-(named)-people buckets (OR within the facet).
    known = [
        _KNOWN_PREDICATE[b]
        for b in _as_list(filters.get("known"))
        if b in _KNOWN_PREDICATE
    ]
    if known:
        clauses.append("(" + " OR ".join(known) + ")")
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


# -- natural-language semantic search -------------------------------------
class SemanticIndex:
    """In-memory matrix of photo image embeddings for cosine ranking.

    Loaded once from the catalog and cached by the web layer (see
    ``api._get_semantic_index``); ranking a free-text query against it is a single
    matrix-vector product. ``matrix`` is ``(N, D)`` of L2-normalised embeddings and
    ``ids`` the parallel ``(N,)`` photo ids.
    """

    def __init__(self, ids: np.ndarray, matrix: np.ndarray):
        self.ids = ids
        self.matrix = matrix

    @classmethod
    def load(cls, conn: sqlite3.Connection) -> SemanticIndex:
        rows = conn.execute(
            "SELECT id, embedding FROM photos WHERE embedding IS NOT NULL ORDER BY id"
        ).fetchall()
        ids = np.array([int(r["id"]) for r in rows], dtype=np.int64)
        vectors = [db.blob_to_embedding(r["embedding"]) for r in rows]
        if not vectors:
            return cls(ids, np.empty((0, 0), dtype=np.float32))
        matrix = np.vstack([_normalize(v) for v in vectors]).astype(np.float32)
        return cls(ids, matrix)

    @property
    def size(self) -> int:
        return int(self.ids.shape[0])

    def vector_for(self, photo_id: int) -> np.ndarray | None:
        """Return the (L2-normalised) embedding row for ``photo_id``, or ``None``.

        ``None`` when the matrix is empty or the photo has no stored embedding —
        powers "more like this" by using a photo's own vector as the query.
        """

        if self.size == 0:
            return None
        matches = np.nonzero(self.ids == int(photo_id))[0]
        if matches.size == 0:
            return None
        return self.matrix[int(matches[0])]

    def rank(
        self,
        query_vec: np.ndarray,
        allowed_ids: set[int] | None = None,
        *,
        top_k: int | None = None,
    ) -> list[tuple[int, float]]:
        """Return ``(photo_id, score)`` sorted by descending cosine similarity.

        ``allowed_ids`` (when given) restricts the result to that set — the photos
        that also pass the structured filters — so semantic ranking ANDs cleanly
        with people/place/date filters. ``top_k`` caps the number returned.
        """

        if self.size == 0:
            return []
        scores = self.matrix @ _normalize(query_vec)
        order = np.argsort(-scores, kind="stable")
        out: list[tuple[int, float]] = []
        for idx in order:
            pid = int(self.ids[idx])
            if allowed_ids is not None and pid not in allowed_ids:
                continue
            out.append((pid, float(scores[idx])))
            if top_k is not None and len(out) >= top_k:
                break
        return out


def _normalize(vec: np.ndarray | None) -> np.ndarray:
    arr = np.asarray(vec, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(arr))
    return arr / norm if norm > 1e-8 else arr


def semantic_search(
    conn: sqlite3.Connection,
    filters: dict[str, Any],
    query_vec: np.ndarray,
    index: SemanticIndex,
    *,
    top_k: int = 200,
    limit: int = 60,
    offset: int = 0,
) -> tuple[list[dict], int]:
    """Rank embedded photos by relevance to ``query_vec``, ANDed with ``filters``.

    Returns ``(rows, total)`` where ``rows`` is one page (in ranked order, each with
    a ``score``) and ``total`` is the number of matches up to ``top_k``. The
    structured filters (people, place, date, …) are applied first as a hard mask;
    the visual query then orders what remains.
    """

    sub = {k: v for k, v in filters.items() if k != "text"}
    where, params = _where(sub)
    glue = " AND " if where else " WHERE "
    allowed = {
        int(r[0])
        for r in conn.execute(
            f"SELECT p.id FROM photos p{where}{glue}p.embedding IS NOT NULL", params
        ).fetchall()
    }
    ranked = index.rank(query_vec, allowed_ids=allowed, top_k=top_k)
    total = len(ranked)
    return _rows_for_ranked(conn, ranked[offset : offset + limit]), total


def _rows_for_ranked(conn: sqlite3.Connection, page: list[tuple[int, float]]) -> list[dict]:
    """Fetch the grid columns for a ranked ``(photo_id, score)`` page.

    Preserves the ranked order and attaches each row's ``score`` (the heavy
    ``scene_scores`` blob is excluded via ``_LIST_COLUMNS``). Shared by
    ``semantic_search`` and ``similar_photos``.
    """

    if not page:
        return []
    ids = [pid for pid, _ in page]
    placeholders = ", ".join(["?"] * len(ids))
    by_id = {
        r["id"]: dict(r)
        for r in conn.execute(
            f"SELECT {_LIST_COLUMNS} FROM photos p WHERE p.id IN ({placeholders})", ids
        ).fetchall()
    }
    rows: list[dict] = []
    for pid, score in page:
        row = by_id.get(pid)
        if row is not None:
            row["score"] = score
            rows.append(row)
    return rows


def similar_photos(
    conn: sqlite3.Connection,
    photo_id: int,
    index: SemanticIndex,
    *,
    top_k: int = 200,
    limit: int = 60,
    offset: int = 0,
) -> tuple[list[dict], int]:
    """Rank the library by visual similarity to ``photo_id`` ("more like this").

    Uses the target photo's own SigLIP image embedding as the query vector and
    cosine-ranks every *other* embedded photo. Returns ``(rows, total)`` where the
    target is always excluded and ``total`` is the number of neighbours up to
    ``top_k``. If the target has no embedding, returns ``([], 0)``.
    """

    query_vec = index.vector_for(photo_id)
    if query_vec is None:
        return [], 0
    # Rank one extra candidate so dropping the target (cosine 1.0 with itself, so
    # always rank 0) still leaves a full top_k of genuine neighbours.
    ranked = [
        (pid, score)
        for pid, score in index.rank(query_vec, top_k=top_k + 1)
        if pid != photo_id
    ][:top_k]
    total = len(ranked)
    return _rows_for_ranked(conn, ranked[offset : offset + limit]), total


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

    def favorites_count() -> int:
        # Starred photos under the other active filters (powers the ★ chip).
        sub = {k: v for k, v in filters.items() if k != "favorite"}
        where, params = _where(sub)
        glue = " AND " if where else " WHERE "
        sql = f"SELECT COUNT(*) FROM photos p{where}{glue}p.favorite = 1"
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
        "with_faces": with_faces_count(),
        "favorites": favorites_count(),
        "date_min": drow[0],
        "date_max": drow[1],
    }
