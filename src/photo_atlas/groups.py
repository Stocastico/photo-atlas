"""Derived groupings: on-this-day memories, trips and near-duplicate bursts."""

from __future__ import annotations

import datetime as _dt
import math as _math
import sqlite3
from collections import Counter, deque

from .filters import _LIST_COLUMNS


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
