"""Person & cluster management operations.

These helpers back both the HTTP API and the CLI: naming a cluster, renaming or
merging people, and listing the unnamed face clusters that still need a label.
"""

from __future__ import annotations

import sqlite3

from . import db


def list_persons(conn: sqlite3.Connection) -> list[dict]:
    rows = conn.execute(
        "SELECT pr.id, pr.name, pr.cover_face_id, COUNT(f.id) AS face_count, "
        "       COUNT(DISTINCT f.photo_id) AS photo_count "
        "FROM persons pr LEFT JOIN faces f ON f.person_id = pr.id "
        "GROUP BY pr.id ORDER BY pr.name COLLATE NOCASE"
    ).fetchall()
    result = []
    for r in rows:
        person = dict(r)
        cover = person.get("cover_face_id")
        if not cover:
            top = conn.execute(
                "SELECT id FROM faces WHERE person_id=? AND crop_path IS NOT NULL LIMIT 1",
                (person["id"],),
            ).fetchone()
            person["cover_face_id"] = top["id"] if top else None
        result.append(person)
    return result


def rename_person(conn: sqlite3.Connection, person_id: int, name: str) -> None:
    conn.execute("UPDATE persons SET name=? WHERE id=?", (name.strip(), person_id))
    conn.commit()


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
    """Assign every face in ``cluster_id`` to a person and clear the cluster."""

    if person_id is None:
        if not name:
            raise ValueError("Provide either a person name or an existing person_id")
        person_id = db.get_or_create_person(conn, name)
    conn.execute(
        "UPDATE faces SET person_id=?, cluster_id=NULL WHERE cluster_id=? AND person_id IS NULL",
        (person_id, cluster_id),
    )
    conn.commit()
    return person_id


def assign_face(
    conn: sqlite3.Connection,
    face_id: int,
    *,
    name: str | None = None,
    person_id: int | None = None,
) -> int:
    if person_id is None:
        if not name:
            raise ValueError("Provide either a person name or an existing person_id")
        person_id = db.get_or_create_person(conn, name)
    conn.execute(
        "UPDATE faces SET person_id=?, cluster_id=NULL WHERE id=?", (person_id, face_id)
    )
    conn.commit()
    return person_id


def unassign_face(conn: sqlite3.Connection, face_id: int) -> None:
    conn.execute("UPDATE faces SET person_id=NULL WHERE id=?", (face_id,))
    conn.commit()


def list_clusters(conn: sqlite3.Connection, limit: int = 50) -> list[dict]:
    """Unnamed face clusters, largest first, with sample face ids."""

    groups = conn.execute(
        "SELECT cluster_id, COUNT(*) AS size FROM faces "
        "WHERE person_id IS NULL AND cluster_id IS NOT NULL "
        "GROUP BY cluster_id ORDER BY size DESC LIMIT ?",
        (limit,),
    ).fetchall()
    clusters = []
    for g in groups:
        samples = conn.execute(
            "SELECT id, photo_id, crop_path FROM faces "
            "WHERE cluster_id=? AND person_id IS NULL ORDER BY id LIMIT 6",
            (g["cluster_id"],),
        ).fetchall()
        clusters.append(
            {
                "cluster_id": int(g["cluster_id"]),
                "size": int(g["size"]),
                "samples": [dict(s) for s in samples],
            }
        )
    return clusters
