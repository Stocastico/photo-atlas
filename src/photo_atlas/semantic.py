"""Semantic / similarity search: cached embedding indexes and cosine ranking."""

from __future__ import annotations

import sqlite3
from typing import Any

import numpy as np

from . import db
from .filters import _LIST_COLUMNS, _where


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
        if allowed_ids is None:
            cand = np.arange(self.size)
        else:
            cand = np.flatnonzero(np.isin(self.ids, np.fromiter(allowed_ids, dtype=np.int64)))
        sub = scores[cand]
        if top_k is not None and 0 < top_k < sub.shape[0]:
            # Keep everything >= the k-th best score (so ties at the cut aren't
            # dropped arbitrarily), then stable-sort only that small set.
            threshold = np.partition(sub, -top_k)[-top_k]
            keep = np.flatnonzero(sub >= threshold)
            cand, sub = cand[keep], sub[keep]
        order = np.argsort(-sub, kind="stable")
        if top_k is not None:
            order = order[:top_k]
        return [(int(self.ids[cand[i]]), float(sub[i])) for i in order]


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


class RegionIndex:
    """In-memory matrix of per-face *region* embeddings for grounded ranking.

    Where :class:`SemanticIndex` ranks a whole photo against a text query, this
    ranks the **region around a named person's face** — so "Stefano eating food"
    scores Stefano's region rather than the whole frame (per-person grounding).
    Loaded only for faces that both carry a region embedding *and* are assigned to
    a person (an unnamed face can't be addressed by name). ``matrix`` is ``(N, D)``
    of L2-normalised region embeddings, with parallel ``face_ids``, ``photo_ids``
    and ``person_ids`` ``(N,)`` arrays. Cached by the web layer like the others.
    """

    def __init__(
        self,
        face_ids: np.ndarray,
        photo_ids: np.ndarray,
        person_ids: np.ndarray,
        matrix: np.ndarray,
    ):
        self.face_ids = face_ids
        self.photo_ids = photo_ids
        self.person_ids = person_ids
        self.matrix = matrix

    @classmethod
    def load(cls, conn: sqlite3.Connection) -> RegionIndex:
        rows = conn.execute(
            "SELECT id, photo_id, person_id, region_embedding FROM faces "
            "WHERE region_embedding IS NOT NULL AND person_id IS NOT NULL ORDER BY id"
        ).fetchall()
        face_ids = np.array([int(r["id"]) for r in rows], dtype=np.int64)
        photo_ids = np.array([int(r["photo_id"]) for r in rows], dtype=np.int64)
        person_ids = np.array([int(r["person_id"]) for r in rows], dtype=np.int64)
        vectors = [db.blob_to_embedding(r["region_embedding"]) for r in rows]
        if not vectors:
            return cls(face_ids, photo_ids, person_ids, np.empty((0, 0), dtype=np.float32))
        matrix = np.vstack([_normalize(v) for v in vectors]).astype(np.float32)
        return cls(face_ids, photo_ids, person_ids, matrix)

    @property
    def size(self) -> int:
        return int(self.face_ids.shape[0])

    def has_any(self, person_ids: set[int] | list[int]) -> bool:
        """True when at least one loaded region belongs to one of ``person_ids``.

        Lets the caller decide whether grounding is even possible for a query
        before reaching for the (heavier) ranking path; falls back to whole-image
        semantic search when it isn't.
        """

        if self.size == 0:
            return False
        targets = {int(p) for p in person_ids}
        return any(int(pid) in targets for pid in self.person_ids)

    def rank_photos(
        self,
        query_vec: np.ndarray,
        *,
        person_ids: set[int] | list[int] | None = None,
        allowed_ids: set[int] | None = None,
        top_k: int | None = None,
    ) -> list[tuple[int, float]]:
        """Rank photos by their best matching region for one of ``person_ids``.

        Each photo's score is the maximum cosine similarity between ``query_vec``
        and any region of a target person in that photo (so a photo wins on its
        *most* relevant person-region). ``allowed_ids`` masks to the photos that
        also pass the structured filters; ``top_k`` caps the count. Returns
        ``(photo_id, score)`` sorted by descending score.
        """

        if self.size == 0:
            return []
        targets = None if person_ids is None else {int(p) for p in person_ids}
        scores = self.matrix @ _normalize(query_vec)
        order = np.argsort(-scores, kind="stable")
        best: dict[int, float] = {}
        for idx in order:
            if targets is not None and int(self.person_ids[idx]) not in targets:
                continue
            pid = int(self.photo_ids[idx])
            if allowed_ids is not None and pid not in allowed_ids:
                continue
            if pid in best:  # first time seen (descending order) is its best region
                continue
            best[pid] = float(scores[idx])
            if top_k is not None and len(best) >= top_k:
                break
        return list(best.items())


def grounded_search(
    conn: sqlite3.Connection,
    filters: dict[str, Any],
    query_vec: np.ndarray,
    index: RegionIndex,
    person_ids: list[int],
    *,
    top_k: int = 200,
    limit: int = 60,
    offset: int = 0,
) -> tuple[list[dict], int]:
    """Rank a named person's photos by how well *their* region matches ``query_vec``.

    The structured ``filters`` (which already include the person filter from the
    planner) form a hard mask; within that set each photo is scored by its best
    region embedding for one of ``person_ids``. Returns ``(rows, total)`` in ranked
    order, each row carrying its ``score``. ``([], 0)`` when nothing qualifies.
    """

    sub = {k: v for k, v in filters.items() if k != "text"}
    where, params = _where(sub)
    allowed = {
        int(r[0]) for r in conn.execute(f"SELECT p.id FROM photos p{where}", params).fetchall()
    }
    ranked = index.rank_photos(
        query_vec, person_ids=person_ids, allowed_ids=allowed, top_k=top_k
    )
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
