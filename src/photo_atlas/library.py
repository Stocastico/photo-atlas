"""Person & cluster management operations.

These helpers back both the HTTP API and the CLI: naming a cluster, renaming or
merging people, and listing the unnamed face clusters that still need a label.
"""

from __future__ import annotations

import sqlite3
from typing import cast

import numpy as np

from . import db
from .faces import Enrollment, knn_person_match, l2_normalize


def list_persons(conn: sqlite3.Connection) -> list[dict]:
    # The cover face falls back to the person's first crop when none is pinned —
    # or when the pinned face no longer exists / lost its crop (its photo was
    # deleted or re-indexed), which would otherwise serve a 404 avatar. The first
    # COALESCE arm validates the pin (still present, still this person's, has a
    # crop); the second is the fallback. Both are correlated subqueries so the
    # whole thing stays one aggregate query (no N+1 on big libraries).
    rows = conn.execute(
        "SELECT pr.id, pr.name, "
        "       COALESCE("
        "         (SELECT f0.id FROM faces f0 "
        "          WHERE f0.id = pr.cover_face_id AND f0.person_id = pr.id "
        "            AND f0.crop_path IS NOT NULL), "
        "         (SELECT f2.id FROM faces f2 "
        "          WHERE f2.person_id = pr.id AND f2.crop_path IS NOT NULL "
        "          ORDER BY COALESCE(f2.quality, 0.6) DESC, f2.id LIMIT 1)"
        "       ) AS cover_face_id, "
        "       COUNT(f.id) AS face_count, "
        "       COUNT(DISTINCT f.photo_id) AS photo_count "
        "FROM persons pr LEFT JOIN faces f ON f.person_id = pr.id "
        "GROUP BY pr.id ORDER BY pr.name COLLATE NOCASE"
    ).fetchall()
    return [dict(r) for r in rows]


def rename_person(conn: sqlite3.Connection, person_id: int, name: str) -> None:
    # ``persons.name`` is UNIQUE; renaming onto an existing name must surface a
    # clean error (the API maps it to 409) instead of a raw IntegrityError 500.
    try:
        conn.execute("UPDATE persons SET name=? WHERE id=?", (name.strip(), person_id))
        conn.commit()
    except sqlite3.IntegrityError as exc:
        conn.rollback()
        raise ValueError(f"a person named {name.strip()!r} already exists") from exc


def delete_person(conn: sqlite3.Connection, person_id: int) -> None:
    # Detach faces (keep them for re-clustering) then drop the person.
    conn.execute("UPDATE faces SET person_id=NULL WHERE person_id=?", (person_id,))
    conn.execute("DELETE FROM persons WHERE id=?", (person_id,))
    conn.commit()


def set_cover_face(conn: sqlite3.Connection, person_id: int, face_id: int) -> None:
    """Pin ``face_id`` as the person's avatar (it must belong to them)."""

    row = conn.execute("SELECT person_id FROM faces WHERE id=?", (face_id,)).fetchone()
    if row is None:
        raise ValueError("face not found")
    if row["person_id"] != person_id:
        raise ValueError("face does not belong to this person")
    conn.execute("UPDATE persons SET cover_face_id=? WHERE id=?", (face_id, person_id))
    conn.commit()


def list_person_faces(conn: sqlite3.Connection, person_id: int, limit: int = 200) -> list[dict]:
    """Faces assigned to a person (for the avatar/cover picker), oldest first."""

    rows = conn.execute(
        "SELECT id, photo_id, crop_path FROM faces "
        "WHERE person_id=? AND crop_path IS NOT NULL ORDER BY id LIMIT ?",
        (person_id, limit),
    ).fetchall()
    return [dict(r) for r in rows]


def merge_persons(conn: sqlite3.Connection, source_id: int, target_id: int) -> int:
    """Fold ``source_id``'s faces into ``target_id`` and drop the source person.

    Used when two named clusters turn out to be the same individual. Returns the
    surviving ``target_id``.
    """

    if source_id == target_id:
        raise ValueError("Cannot merge a person into themselves")
    for pid in (source_id, target_id):
        if conn.execute("SELECT 1 FROM persons WHERE id=?", (pid,)).fetchone() is None:
            raise ValueError(f"person {pid} not found")
    conn.execute("UPDATE faces SET person_id=? WHERE person_id=?", (target_id, source_id))
    # The source's pinned cover face now belongs to the target; drop the row.
    conn.execute("DELETE FROM persons WHERE id=?", (source_id,))
    conn.commit()
    return target_id


def assign_cluster(
    conn: sqlite3.Connection,
    cluster_id: int,
    *,
    name: str | None = None,
    person_id: int | None = None,
) -> int:
    """Assign every face in ``cluster_id`` to a person (human-confirmed) and clear the cluster."""

    if person_id is None:
        if not name:
            raise ValueError("Provide either a person name or an existing person_id")
        person_id = db.get_or_create_person(conn, name)
    # Naming a cluster is a human label: its faces become ground truth (confidence 1.0),
    # which is what recognition enrolment, calibration and --recompute carry-over rely
    # on. Any earlier "not this person" correction against *this* person is stale now.
    conn.execute(
        "DELETE FROM face_negatives WHERE person_id=? AND face_id IN "
        "(SELECT id FROM faces WHERE cluster_id=? AND person_id IS NULL)",
        (person_id, cluster_id),
    )
    conn.execute(
        "UPDATE faces SET person_id=?, cluster_id=NULL, confidence=1.0 "
        "WHERE cluster_id=? AND person_id IS NULL",
        (person_id, cluster_id),
    )
    conn.commit()
    return person_id


def unassign_cluster(
    conn: sqlite3.Connection, cluster_id: int, face_ids: list[int], person_id: int
) -> int:
    """Undo :func:`assign_cluster` for exactly ``face_ids``: back into the group, unnamed.

    Only faces still assigned to ``person_id`` are touched. A person left with no faces
    at all (typically one the undone naming just created) is removed. Returns the
    number of faces restored.
    """

    clean = [int(i) for i in face_ids]
    if not clean:
        return 0
    marks = ", ".join("?" * len(clean))
    cur = conn.execute(
        f"UPDATE faces SET person_id=NULL, cluster_id=?, confidence=0.0 "
        f"WHERE id IN ({marks}) AND person_id=?",
        [cluster_id, *clean, person_id],
    )
    restored = cur.rowcount
    left = conn.execute("SELECT COUNT(*) FROM faces WHERE person_id=?", (person_id,)).fetchone()[0]
    if left == 0:
        conn.execute("DELETE FROM persons WHERE id=?", (person_id,))
    conn.commit()
    return int(restored)


def assign_face(
    conn: sqlite3.Connection,
    face_id: int,
    *,
    name: str | None = None,
    person_id: int | None = None,
) -> int:
    """Assign a face to a person (a human label, so confidence becomes 1.0).

    Active learning: reassigning a face that was tagged as *someone else* records a
    "not that person" negative, and any stale negative against the new person is
    cleared (the user just confirmed it *is* them).
    """

    if person_id is None:
        if not name:
            raise ValueError("Provide either a person name or an existing person_id")
        person_id = db.get_or_create_person(conn, name)
    prev = conn.execute("SELECT person_id FROM faces WHERE id=?", (face_id,)).fetchone()
    if prev is not None and prev["person_id"] not in (None, person_id):
        db.add_face_negative(conn, face_id, int(prev["person_id"]))
    db.remove_face_negative(conn, face_id, person_id)
    conn.execute(
        "UPDATE faces SET person_id=?, cluster_id=NULL, confidence=1.0 WHERE id=?",
        (person_id, face_id),
    )
    conn.commit()
    return person_id


def unassign_face(conn: sqlite3.Connection, face_id: int) -> None:
    """Send a face back to unknown, recording a negative for the person it had."""

    prev = conn.execute("SELECT person_id FROM faces WHERE id=?", (face_id,)).fetchone()
    if prev is not None and prev["person_id"] is not None:
        db.add_face_negative(conn, face_id, int(prev["person_id"]))
    conn.execute("UPDATE faces SET person_id=NULL WHERE id=?", (face_id,))
    conn.commit()


def low_confidence_faces(
    conn: sqlite3.Connection, *, max_confidence: float, limit: int = 60
) -> list[dict]:
    """Auto-recognised faces below ``max_confidence``, for the review queue.

    Only machine guesses (``confidence`` strictly between 0 and the threshold) with
    a crop are returned, weakest first; a human assignment sets confidence to 1.0
    and so drops out once reviewed.
    """

    rows = conn.execute(
        "SELECT f.id, f.photo_id, f.person_id, f.confidence, f.crop_path, "
        "p.name AS person_name "
        "FROM faces f JOIN persons p ON p.id = f.person_id "
        "WHERE f.person_id IS NOT NULL AND f.crop_path IS NOT NULL "
        "AND f.confidence > 0 AND f.confidence < ? "
        "ORDER BY f.confidence ASC, f.id ASC LIMIT ?",
        (max_confidence, limit),
    ).fetchall()
    return [dict(r) for r in rows]


def list_clusters(
    conn: sqlite3.Connection,
    limit: int = 50,
    *,
    enrollment: Enrollment | None = None,
    k: int = 5,
    threshold: float = 0.5,
) -> list[dict]:
    """Unnamed face clusters, largest first, ready to be named in one click.

    ``samples`` are the six **most typical, best-quality** faces (close to the centroid and
    legible), not simply the first ids, so the card shows who the cluster really is rather than
    its outliers. With an ``enrollment`` of confirmed people, a cluster whose centroid
    matches one gets a ``suggestion`` (``person_id``, ``name``, ``confidence`` and
    ``agreement`` = share of up to 12 typical members that match the same person).
    """

    groups = conn.execute(
        "SELECT cluster_id, COUNT(*) AS size FROM faces "
        "WHERE person_id IS NULL AND cluster_id IS NOT NULL "
        "GROUP BY cluster_id ORDER BY size DESC LIMIT ?",
        (limit,),
    ).fetchall()
    use_enrollment = enrollment is not None and not enrollment.is_empty
    names = (
        {int(r["id"]): r["name"] for r in conn.execute("SELECT id, name FROM persons")}
        if use_enrollment
        else {}
    )
    clusters = []
    for g in groups:
        rows = conn.execute(
            "SELECT id, photo_id, crop_path, embedding, quality FROM faces "
            "WHERE cluster_id=? AND person_id IS NULL ORDER BY id",
            (g["cluster_id"],),
        ).fetchall()
        embs = [db.blob_to_embedding(r["embedding"]) for r in rows]
        usable = [i for i, e in enumerate(embs) if e is not None]
        order = list(range(len(rows)))
        centroid: np.ndarray | None = None
        if usable and len({cast(np.ndarray, embs[i]).shape[0] for i in usable}) == 1:
            mat = np.vstack([l2_normalize(cast(np.ndarray, embs[i])) for i in usable])
            centroid = l2_normalize(mat.mean(axis=0))
            # Most typical *and* most legible: closeness to the centroid, discounted by
            # quality (an unscored face counts as middling).
            quality = np.array([
                0.6 if rows[i]["quality"] is None else rows[i]["quality"] for i in usable
            ])
            score = (mat @ centroid) * (0.5 + 0.5 * quality)
            ranked = [usable[j] for j in np.argsort(-score, kind="stable")]
            order = ranked + [i for i in range(len(rows)) if i not in set(usable)]
        entry: dict = {
            "cluster_id": int(g["cluster_id"]),
            "size": int(g["size"]),
            "samples": [
                {"id": rows[i]["id"], "photo_id": rows[i]["photo_id"],
                 "crop_path": rows[i]["crop_path"]}
                for i in order[:6]
            ],
        }
        if use_enrollment and centroid is not None and enrollment is not None:
            if centroid.shape[0] == enrollment.embeddings.shape[1]:
                pid, conf = knn_person_match(centroid, enrollment, k=k, threshold=threshold)
                if pid is not None:
                    votes = [
                        knn_person_match(
                            cast(np.ndarray, embs[i]), enrollment, k=k, threshold=threshold
                        )[0]
                        for i in order[:12] if embs[i] is not None
                    ]
                    entry["suggestion"] = {
                        "person_id": pid,
                        "name": names.get(pid),
                        "confidence": round(conf, 3),
                        "agreement": round(sum(v == pid for v in votes) / len(votes), 2)
                        if votes else 0.0,
                    }
        clusters.append(entry)
    return clusters
