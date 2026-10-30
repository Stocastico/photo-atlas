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

from typing import Any

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
