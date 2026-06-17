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
``hidden``     ``True`` -> only hidden photos; ``False`` -> exclude hidden (the
               browsing default, set by the API); absent -> no clause.
``q``          free-text substring matched across filename, city, country,
               place label, folder/trip and camera make/model.
``sort``       result ordering: ``newest`` (default), ``oldest``, ``filename``,
               ``filename_desc`` or ``indexed`` (most recently indexed first).
"""

from __future__ import annotations

import datetime as _dt
import math as _math
import sqlite3
from collections import Counter
from typing import Any

import numpy as np

from . import db
from .classify import SCENE_LABELS as _SCENE_LABELS

# Columns returned for grid/list rows: every photo column except the
# ``scene_scores`` JSON blob, which only the single-photo detail view uses.
# Shipping it for 60 photos/page bloats the payload (and the browser decodes and
# discards it), so the list query selects an explicit set instead of ``p.*``.
# ``favorite`` is appended explicitly: it's deliberately not in ``PHOTO_COLUMNS``
# (so a re-index can't reset a user's star), but the grid needs it per card to
# render the star state.
_LIST_COLUMNS = ", ".join(
    [
        "p.id",
        *(f"p.{c}" for c in db.PHOTO_COLUMNS if c != "scene_scores"),
        "p.favorite",
        "p.is_video",
        "p.hidden",
    ]
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

# Unified "type of picture" facet: token -> SQL predicate. Folds the people-count
# buckets and the scene tags into one OR-within facet (the UI's single "Type of
# picture" section), so the sidebar isn't split between "Number of people" and
# "Scene". ``portrait``/``group`` come from ``face_count`` (a portrait is one
# detected face, a group is two or more); the rest mirror ``classify.SCENE_LABELS``
# minus ``people``, which is folded into portrait/group rather than shown twice.
# Predicates are literal (no bound params), so they splice straight into the WHERE
# like the other bucket facets; the scene labels are our own controlled constants.
PICTURE_TYPES: list[tuple[str, str]] = [
    ("portrait", "p.face_count = 1"),
    ("group", "p.face_count >= 2"),
    *(
        (label, f"p.scene_type = '{label}'")
        for label in _SCENE_LABELS
        if label != "people"
    ),
]
_KIND_PREDICATE = dict(PICTURE_TYPES)


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
    # ``hidden`` is tri-state: absent → no clause (the contract that ``_where({})``
    # is empty); ``True`` → only hidden (the review chip); ``False`` → exclude
    # hidden (the browsing default, set by the API). User-hidden photos are thus
    # invisible everywhere the API passes ``hidden=False`` unless explicitly asked.
    if "hidden" in filters:
        clauses.append("p.hidden = 1" if filters["hidden"] else "p.hidden = 0")
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
    # Unified "type of picture" buckets (OR within the facet); unknown tokens are
    # ignored. Tokens overlap by design (a portrait can also be a "food" shot), so
    # OR-ing them matches any selected type.
    kinds = [
        _KIND_PREDICATE[t]
        for t in _as_list(filters.get("kind"))
        if t in _KIND_PREDICATE
    ]
    if kinds:
        clauses.append("(" + " OR ".join(kinds) + ")")
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


class FaceIndex:
    """In-memory matrix of face (SFace) embeddings for cosine ranking.

    Parallel to :class:`SemanticIndex` but over the ``faces`` table: ``matrix`` is
    ``(N, D)`` of L2-normalised face embeddings, with ``face_ids`` and the owning
    ``photo_ids`` as parallel ``(N,)`` arrays. Powers "more like this person" —
    ranking a chosen face against every other face — without any model download
    (it reuses the embeddings detection already stored). Cached by the web layer.
    """

    def __init__(self, face_ids: np.ndarray, photo_ids: np.ndarray, matrix: np.ndarray):
        self.face_ids = face_ids
        self.photo_ids = photo_ids
        self.matrix = matrix

    @classmethod
    def load(cls, conn: sqlite3.Connection) -> FaceIndex:
        rows = conn.execute(
            "SELECT id, photo_id, embedding FROM faces "
            "WHERE embedding IS NOT NULL ORDER BY id"
        ).fetchall()
        face_ids = np.array([int(r["id"]) for r in rows], dtype=np.int64)
        photo_ids = np.array([int(r["photo_id"]) for r in rows], dtype=np.int64)
        vectors = [db.blob_to_embedding(r["embedding"]) for r in rows]
        if not vectors:
            return cls(face_ids, photo_ids, np.empty((0, 0), dtype=np.float32))
        matrix = np.vstack([_normalize(v) for v in vectors]).astype(np.float32)
        return cls(face_ids, photo_ids, matrix)

    @property
    def size(self) -> int:
        return int(self.face_ids.shape[0])

    def _row_for(self, face_id: int) -> int | None:
        if self.size == 0:
            return None
        matches = np.nonzero(self.face_ids == int(face_id))[0]
        return int(matches[0]) if matches.size else None

    def vector_for(self, face_id: int) -> np.ndarray | None:
        row = self._row_for(face_id)
        return None if row is None else self.matrix[row]

    def photo_for(self, face_id: int) -> int | None:
        row = self._row_for(face_id)
        return None if row is None else int(self.photo_ids[row])


def similar_faces(
    conn: sqlite3.Connection,
    face_id: int,
    index: FaceIndex,
    *,
    top_k: int = 200,
    limit: int = 60,
    offset: int = 0,
) -> tuple[list[dict], int]:
    """Rank photos by SFace similarity to ``face_id`` ("more like this person").

    Cosine-ranks every other face against the chosen one and collapses the result
    to photos, keeping each photo's best (nearest) face score. The source face's
    own photo is always excluded (a photo is never "similar" to itself, even when
    it holds another near-identical face). Returns ``(rows, total)`` with the target
    excluded and ``total`` capped at ``top_k`` distinct photos; ``([], 0)`` when the
    face has no stored embedding.
    """

    query_vec = index.vector_for(face_id)
    if query_vec is None:
        return [], 0
    src_photo = index.photo_for(face_id)
    scores = index.matrix @ _normalize(query_vec)
    order = np.argsort(-scores, kind="stable")
    # First time a photo is seen (in descending-score order) is its best score, so
    # the first ``top_k`` distinct photos are exactly the top_k by best face.
    best_per_photo: dict[int, float] = {}
    for idx in order:
        pid = int(index.photo_ids[idx])
        if pid == src_photo or pid in best_per_photo:
            continue
        if int(index.face_ids[idx]) == int(face_id):
            continue
        best_per_photo[pid] = float(scores[idx])
        if len(best_per_photo) >= top_k:
            break
    ranked = list(best_per_photo.items())
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


def on_this_day(
    conn: sqlite3.Connection, month: int, day: int, *, per_year: int = 24
) -> list[dict]:
    """Photos taken on ``month``/``day`` in past years ("on this day").

    Returns one group per year (newest first), each ``{year, count, photos}`` where
    ``count`` is the full number of matches that year and ``photos`` is a capped
    sample (``per_year``, newest first) carrying the grid columns (minus the heavy
    ``scene_scores`` blob). The match is on the ``taken_at`` month/day prefix, so a
    time component doesn't matter and undated photos are skipped.
    """

    mm, dd = f"{int(month):02d}", f"{int(day):02d}"
    rows = conn.execute(
        f"SELECT {_LIST_COLUMNS}, substr(p.taken_at, 1, 4) AS year FROM photos p "
        "WHERE substr(p.taken_at, 6, 2) = ? AND substr(p.taken_at, 9, 2) = ? "
        "ORDER BY p.taken_at DESC, p.id DESC",
        (mm, dd),
    ).fetchall()

    groups: dict[str, dict] = {}
    order: list[str] = []
    for r in rows:
        row = dict(r)
        year = row.pop("year")
        grp = groups.get(year)
        if grp is None:
            grp = groups[year] = {"year": year, "count": 0, "photos": []}
            order.append(year)
        grp["count"] += 1
        if len(grp["photos"]) < per_year:
            grp["photos"].append(row)
    return [groups[y] for y in order]


def _parse_taken_at(value: str | None) -> _dt.datetime | None:
    if not value:
        return None
    try:
        return _dt.datetime.fromisoformat(value)
    except ValueError:  # pragma: no cover - defensive against odd stored values
        return None


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance between two WGS84 points, in kilometres."""

    radius = 6371.0
    p1, p2 = _math.radians(lat1), _math.radians(lat2)
    dphi = _math.radians(lat2 - lat1)
    dlam = _math.radians(lon2 - lon1)
    a = _math.sin(dphi / 2) ** 2 + _math.cos(p1) * _math.cos(p2) * _math.sin(dlam / 2) ** 2
    return 2 * radius * _math.asin(_math.sqrt(a))


def _dominant_place(rows: list[dict]) -> str | None:
    """The most common place label across a trip's photos (best available source)."""

    for key in ("place_label", "folder_place"):
        counts = Counter(r[key] for r in rows if r.get(key))
        if counts:
            return counts.most_common(1)[0][0]
    counts = Counter(
        ", ".join(filter(None, (r.get("place_city"), r.get("place_country"))))
        for r in rows
        if r.get("place_city") or r.get("place_country")
    )
    return counts.most_common(1)[0][0] if counts else None


def _build_trip(rows: list[dict], per_trip: int) -> dict:
    geos = [
        (r["lat"], r["lon"])
        for r in rows
        if r.get("lat") is not None and r.get("lon") is not None
    ]
    lat = lon = None
    cover_id = rows[0]["id"]
    if geos:
        lat = sum(a for a, _ in geos) / len(geos)
        lon = sum(b for _, b in geos) / len(geos)
        for r in rows:  # prefer a geotagged cover so the trip pins on the map
            if r.get("lat") is not None and r.get("lon") is not None:
                cover_id = r["id"]
                break
    return {
        "start": rows[0]["taken_at"][:10],
        "end": rows[-1]["taken_at"][:10],
        "count": len(rows),
        "place": _dominant_place(rows),
        "lat": lat,
        "lon": lon,
        "cover_id": cover_id,
        "photos": rows[:per_trip],
    }


def detect_trips(
    conn: sqlite3.Connection,
    *,
    gap_days: float = 2.0,
    gap_km: float = 200.0,
    min_photos: int = 4,
    per_trip: int = 12,
) -> list[dict]:
    """Group the library into trips from capture-time gaps + GPS proximity.

    Walks every dated photo in chronological order and starts a new trip whenever
    there's a break longer than ``gap_days`` *or* (within that window) a geographic
    jump farther than ``gap_km`` between consecutive geotagged shots. Clusters
    smaller than ``min_photos`` are dropped. Each trip carries its date span, photo
    ``count``, a ``place`` label (most common place/folder), a centroid + cover for
    the map, and a capped ``photos`` sample. Returned newest-first.
    """

    rows = conn.execute(
        f"SELECT {_LIST_COLUMNS} FROM photos p "
        "WHERE p.taken_at IS NOT NULL ORDER BY p.taken_at ASC, p.id ASC"
    ).fetchall()

    trips: list[dict] = []
    current: list[dict] = []
    prev_dt: _dt.datetime | None = None
    prev_ll: tuple[float, float] | None = None

    def flush() -> None:
        if len(current) >= min_photos:
            trips.append(_build_trip(current, per_trip))
        current.clear()

    for r in rows:
        row = dict(r)
        dt = _parse_taken_at(row.get("taken_at"))
        ll = (
            (row["lat"], row["lon"])
            if row.get("lat") is not None and row.get("lon") is not None
            else None
        )
        if current and dt is not None and prev_dt is not None:
            gap = (dt - prev_dt).total_seconds() / 86400.0
            jumped = (
                ll is not None
                and prev_ll is not None
                and _haversine_km(*prev_ll, *ll) > gap_km
            )
            if gap > gap_days or (gap > 0 and jumped):
                flush()
        current.append(row)
        if dt is not None:
            prev_dt = dt
        if ll is not None:
            prev_ll = ll
    flush()

    trips.sort(key=lambda t: (t["end"], t["start"]), reverse=True)
    return trips


def phash_distance(a: str | None, b: str | None) -> int:
    """Hamming distance between two dHash hex strings.

    A missing hash on either side is treated as maximally distant (64 bits for the
    default 64-bit dHash), so an un-hashed photo never groups with anything.
    """

    if not a or not b:
        return 64
    return int(bin(int(a, 16) ^ int(b, 16)).count("1"))


def _cover_score(row: dict) -> tuple[int, int, int]:
    """Sort key for picking the "best of N" cover shot of a burst.

    Prefer a favorite, then the highest resolution (most pixels), then the
    earliest frame (lowest id) as a stable tiebreak. Used as a descending sort, so
    the winner sorts first.
    """

    pixels = (row.get("width") or 0) * (row.get("height") or 0)
    return (1 if row.get("favorite") else 0, int(pixels), -int(row["id"]))


def find_burst_groups(
    conn: sqlite3.Connection,
    *,
    max_distance: int = 10,
    max_gap_seconds: float = 10.0,
    min_group: int = 2,
) -> list[dict]:
    """Group near-identical shots taken close together in time (bursts / dupes).

    Two signals define a group: perceptual similarity (dHash Hamming distance
    ``<= max_distance``) *and* temporal proximity (captured within
    ``max_gap_seconds`` of a neighbour). A union-find over the time-sorted stream
    — comparing each photo only against the still-recent ones in a sliding window —
    forms connected components, so a burst survives the odd blurry/off frame and
    the scan stays ``O(N · window)`` rather than ``O(N²)``. Components smaller than
    ``min_group`` are dropped.

    Hidden and undated photos are excluded (videos carry no phash, so they fall out
    naturally). Each returned group is ``{count, cover_id, photos}`` where ``photos``
    carries the full grid columns sorted best-first (the "best of N" cover leads,
    so the UI can offer to keep it and hide/delete the rest). Returned newest-first.
    """

    from collections import deque

    rows = conn.execute(
        f"SELECT {_LIST_COLUMNS}, p.phash AS phash FROM photos p "
        "WHERE p.phash IS NOT NULL AND p.taken_at IS NOT NULL AND p.hidden = 0 "
        "ORDER BY p.taken_at ASC, p.id ASC"
    ).fetchall()

    photos: dict[int, dict] = {}
    phashes: dict[int, str] = {}
    parent: dict[int, int] = {}

    def find(x: int) -> int:
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:  # path compression
            parent[x], x = root, parent[x]
        return root

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    window: deque[tuple[int, _dt.datetime]] = deque()
    for raw in rows:
        row = dict(raw)
        phash = row.pop("phash")
        pid = int(row["id"])
        dt = _parse_taken_at(row.get("taken_at"))
        photos[pid] = row
        parent[pid] = pid
        if dt is None:  # malformed timestamp: keep as a singleton, never groups
            continue
        while window and (dt - window[0][1]).total_seconds() > max_gap_seconds:
            window.popleft()
        for prev_id, _prev_dt in window:
            if phash_distance(phash, phashes[prev_id]) <= max_distance:
                union(pid, prev_id)
        phashes[pid] = phash
        window.append((pid, dt))

    components: dict[int, list[int]] = {}
    for pid in parent:
        components.setdefault(find(pid), []).append(pid)

    groups: list[dict] = []
    for member_ids in components.values():
        if len(member_ids) < min_group:
            continue
        members = sorted(
            (photos[i] for i in member_ids), key=_cover_score, reverse=True
        )
        groups.append(
            {
                "count": len(members),
                "cover_id": members[0]["id"],
                "photos": members,
            }
        )
    # Newest-first by the group's latest capture time (its first chronological
    # member is the lowest id; use the max taken_at across members).
    groups.sort(key=lambda g: max(p["taken_at"] for p in g["photos"]), reverse=True)
    return groups


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
