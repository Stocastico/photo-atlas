"""Catalog housekeeping: orphan sweep, prune, trash, hard delete, export, clustering.

Split out of :mod:`photo_atlas.indexer` (which re-exports these names)."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import cast

import numpy as np

from . import db
from . import faces as faces_module
from .config import AtlasConfig
from .faces import (
    cap_confidence_for_quality,
    cluster_embeddings,
    face_quality,
    knn_person_match,
)
from .fsutil import unlink_quietly


def sweep_orphan_derivatives(config: AtlasConfig) -> int:
    """Delete derivative files no catalog row references; return how many were removed.

    Thumbnails and preview/retina variants are content-addressed by the source
    photo's SHA-1 (``{sha1}.jpg`` / ``{sha1}_{size}.jpg``), and face crops live
    under ``faces_dir/<photo_id>/``. A crash mid-index, or a source photo whose
    bytes (and thus SHA-1) changed, can strand such files with no owning row. This
    reclaims them: any thumb/preview whose SHA-1 isn't in the catalog, any leftover
    ``.part`` temp from an interrupted write, and any face-crop dir for a photo id
    that no longer exists. Referenced files are always kept.
    """

    import shutil

    conn = db.connect(config.db_path, ensure_schema=False)
    try:
        live_sha1 = {
            r[0] for r in conn.execute("SELECT sha1 FROM photos WHERE sha1 IS NOT NULL")
        }
        live_ids = {str(r[0]) for r in conn.execute("SELECT id FROM photos")}
    finally:
        conn.close()

    removed = 0
    # Content-addressed thumbnails, preview/retina variants and video posters.
    for cache_dir in (config.thumbs_dir, config.previews_dir, config.posters_dir):
        if not cache_dir.exists():
            continue
        for f in cache_dir.rglob("*"):
            if not f.is_file():
                continue
            # A leftover atomic-write temp is an orphan regardless of name.
            if f.name.endswith(".part"):
                removed += unlink_quietly(f)
                continue
            # Filenames are ``{sha1}.jpg`` or ``{sha1}_{size}.jpg``; pull the sha1.
            sha1 = f.name.split(".", 1)[0].split("_", 1)[0]
            if sha1 not in live_sha1:
                removed += unlink_quietly(f)
    # Face-crop directories keyed by photo id.
    if config.faces_dir.exists():
        for sub in config.faces_dir.iterdir():
            if sub.is_dir() and sub.name not in live_ids:
                shutil.rmtree(sub, ignore_errors=True)
                removed += 1
    return removed


def prune_library(config: AtlasConfig) -> dict[str, int]:
    """Reconcile the catalog with the filesystem (rows *and* derivative files).

    Indexing only ever adds or updates rows, so moved/deleted photos linger as
    dead entries that 404 in the UI. ``prune`` removes the photo row for each
    missing source file (its faces cascade away) plus that row's thumbnail and
    face crops, then sweeps any remaining orphaned derivative files
    (:func:`sweep_orphan_derivatives`) — e.g. preview/retina variants of the
    removed photos, or files stranded by an interrupted index. Returns
    ``{removed, kept, orphans}``.
    """

    import shutil

    conn = db.connect(config.db_path)
    removed = kept = 0
    try:
        rows = conn.execute("SELECT id, path, thumb_path FROM photos").fetchall()
        for row in rows:
            if Path(row["path"]).exists():
                kept += 1
                continue
            if row["thumb_path"]:
                unlink_quietly(Path(row["thumb_path"]))
            crop_dir = config.faces_dir / str(row["id"])
            if crop_dir.exists():
                shutil.rmtree(crop_dir, ignore_errors=True)
            conn.execute("DELETE FROM photos WHERE id=?", (row["id"],))
            removed += 1
        conn.commit()
    finally:
        conn.close()
    # Now that dead rows are gone, reclaim any derivative files they (or a crash)
    # left without an owning row.
    orphans = sweep_orphan_derivatives(config)
    return {"removed": removed, "kept": kept, "orphans": orphans}



def _move_to_trash(config: AtlasConfig, photo_id: int, src: Path) -> None:
    """Move ``src`` into the library trash and log where it came from."""

    import shutil

    config.trash_dir.mkdir(parents=True, exist_ok=True)
    dest = config.trash_dir / f"{photo_id}_{src.name}"
    shutil.move(str(src), dest)
    with (config.trash_dir / "manifest.jsonl").open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({"id": photo_id, "original": str(src), "trashed": dest.name}) + "\n")


def empty_trash(config: AtlasConfig) -> int:
    """Permanently delete everything in the trash; returns the files removed."""

    if not config.trash_dir.exists():
        return 0
    removed = 0
    for entry in config.trash_dir.iterdir():
        if entry.name != "manifest.jsonl" and entry.is_file() and unlink_quietly(entry):
            removed += 1
    unlink_quietly(config.trash_dir / "manifest.jsonl")
    return removed


def restore_trash(config: AtlasConfig, ids: list[int] | None = None) -> dict[str, int]:
    """Move trashed originals back to where they came from.

    Restores every manifest entry, or only those whose photo id is in ``ids``. A file
    is never overwritten: if something now lives at the original path it counts as a
    ``conflict`` and stays in the trash (and in the manifest). An entry whose trashed
    file has vanished is ``missing`` and is dropped. Restored files are plain files
    again — run ``photo-atlas index`` to catalogue them. Returns
    ``{restored, missing, conflicts}``.
    """

    import shutil

    manifest = config.trash_dir / "manifest.jsonl"
    result = {"restored": 0, "missing": 0, "conflicts": 0}
    if not manifest.exists():
        return result
    wanted = None if ids is None else {int(i) for i in ids}
    kept: list[dict] = []
    for line in manifest.read_text(encoding="utf-8").splitlines():
        try:
            entry = json.loads(line)
        except ValueError:
            continue  # a corrupt line can't be restored; drop it
        if wanted is not None and entry.get("id") not in wanted:
            kept.append(entry)
            continue
        trashed = config.trash_dir / entry["trashed"]
        original = Path(entry["original"])
        if not trashed.exists():
            result["missing"] += 1
        elif original.exists():
            result["conflicts"] += 1
            kept.append(entry)
        else:
            original.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(trashed), original)
            result["restored"] += 1
    if kept:
        manifest.write_text("".join(json.dumps(e) + "\n" for e in kept), encoding="utf-8")
    else:
        manifest.unlink()
    return result


def delete_photos(config: AtlasConfig, ids: list[int]) -> dict[str, int]:
    """Delete photos from the catalog and move their source files to the trash.

    For each id this moves the source file into ``config.trash_dir`` (recoverable
    until ``empty-trash``; ``manifest.jsonl`` records the original path), removes the
    photo's thumbnail and face-crop directory, then the catalog row (faces cascade); a
    final sweep reclaims any remaining content-addressed derivatives (preview /
    retina variants, video posters) the deleted rows owned. Returns
    ``{rows, files, orphans}`` — rows removed from the catalog, source files
    unlinked from disk, and orphaned derivative files swept.
    """

    import shutil

    clean = [int(i) for i in ids]
    conn = db.connect(config.db_path)
    files_removed = 0
    try:
        if clean:
            placeholders = ", ".join(["?"] * len(clean))
            rows = conn.execute(
                f"SELECT id, path, thumb_path FROM photos WHERE id IN ({placeholders})",
                clean,
            ).fetchall()
            for row in rows:
                src = Path(row["path"]) if row["path"] else None
                if src is not None and src.exists():
                    try:
                        _move_to_trash(config, row["id"], src)
                        files_removed += 1
                    except OSError as exc:  # surface, don't crash the whole batch
                        print(f"warning: could not trash {src}: {exc}", file=sys.stderr)
                if row["thumb_path"]:
                    unlink_quietly(Path(row["thumb_path"]))
                crop_dir = config.faces_dir / str(row["id"])
                if crop_dir.exists():
                    shutil.rmtree(crop_dir, ignore_errors=True)
        removed = db.delete_photos(conn, clean)
    finally:
        conn.close()
    orphans = sweep_orphan_derivatives(config)
    return {"rows": removed, "files": files_removed, "orphans": orphans}


def export_photos(config: AtlasConfig, ids: list[int], dest: Path | str) -> dict[str, int]:
    """Copy the chosen photos' original files into ``dest``; never moves them.

    A multi-select bulk action: pull a filtered/selected set out of the library as
    plain files. Originals are copied (with metadata, ``copy2``), so the catalog is
    untouched. A row whose source file is gone — or an unknown id — counts toward
    ``missing``. Basename collisions (two photos sharing a filename from different
    folders) are disambiguated with an id suffix rather than clobbering. Returns
    ``{requested, copied, missing}``.
    """

    import shutil

    dest = Path(dest).expanduser()
    clean = [int(i) for i in ids]
    copied = missing = 0
    if clean:
        dest.mkdir(parents=True, exist_ok=True)
        conn = db.connect(config.db_path, ensure_schema=False)
        try:
            placeholders = ", ".join(["?"] * len(clean))
            found = {
                int(r["id"]): r
                for r in conn.execute(
                    f"SELECT id, path, filename FROM photos WHERE id IN ({placeholders})",
                    clean,
                ).fetchall()
            }
        finally:
            conn.close()
        for i in clean:
            row = found.get(i)
            if row is None:
                missing += 1
                continue
            src = Path(row["path"]) if row["path"] else None
            if src is None or not src.exists():
                missing += 1
                continue
            target = dest / (row["filename"] or src.name)
            if target.exists():  # don't clobber a same-named file from another folder
                target = dest / f"{src.stem}_{i}{src.suffix}"
            try:
                shutil.copy2(src, target)
                copied += 1
            except OSError as exc:
                print(f"warning: could not export {src}: {exc}", file=sys.stderr)
                missing += 1
    return {"requested": len(clean), "copied": copied, "missing": missing}


def cluster_library(config: AtlasConfig) -> dict[str, int]:
    """Cluster all unnamed faces so groups can be labelled in one go."""

    conn = db.connect(config.db_path)
    try:
        rows = conn.execute(
            "SELECT id, embedding FROM faces WHERE person_id IS NULL AND embedding IS NOT NULL "
            "AND (quality IS NULL OR quality >= ?)",
            (config.min_face_quality,),
        ).fetchall()
        ids = [int(r["id"]) for r in rows]
        # The WHERE clause guarantees non-NULL embeddings, so each decodes to an
        # array (never None); cast to keep the list type aligned with ``ids``.
        embeddings = [cast(np.ndarray, db.blob_to_embedding(r["embedding"])) for r in rows]
        labels = cluster_embeddings(
            embeddings, eps=config.cluster_eps, min_samples=config.cluster_min_samples
        )

        # Reset previous clustering for unnamed faces, then assign new labels.
        conn.execute("UPDATE faces SET cluster_id=NULL WHERE person_id IS NULL")
        n_clusters = 0
        seen: set[int] = set()
        for face_id, label in zip(ids, labels, strict=True):
            if label < 0:
                continue
            conn.execute("UPDATE faces SET cluster_id=? WHERE id=?", (label, face_id))
            if label not in seen:
                seen.add(label)
                n_clusters += 1
        conn.commit()
        return {"faces": len(ids), "clusters": n_clusters}
    finally:
        conn.close()


def calibrate_faces(config: AtlasConfig) -> dict | None:
    """Suggest recognition thresholds from the faces a human has named.

    Uses only faces assigned by hand (``confidence >= 1.0``) so auto-tags can't bias
    the result, and only the dominant embedding dimension (so a library mixing
    recogniser backends isn't compared across spaces). Returns the
    :func:`photo_atlas.faces.calibrate_threshold` stats plus ``faces`` / ``people``
    counts, or ``None`` when there isn't enough named data yet.
    """

    from collections import Counter  # noqa: PLC0415

    conn = db.connect(config.db_path)
    try:
        rows = conn.execute(
            "SELECT person_id, embedding, dim FROM faces "
            "WHERE person_id IS NOT NULL AND confidence >= 1.0 AND embedding IS NOT NULL"
        ).fetchall()
    finally:
        conn.close()
    if not rows:
        return None
    dim = Counter(r["dim"] for r in rows).most_common(1)[0][0]
    rows = [r for r in rows if r["dim"] == dim]
    embeddings = np.vstack(
        [cast(np.ndarray, db.blob_to_embedding(r["embedding"])) for r in rows]
    )
    person_ids = np.array([r["person_id"] for r in rows])
    result = faces_module.calibrate_threshold(embeddings, person_ids)
    if result is None:
        return None
    return {**result, "faces": len(rows), "people": int(np.unique(person_ids).size)}


def recognize_faces(config: AtlasConfig, *, dry_run: bool = False) -> dict:
    """Name the faces already in the catalog using the people you've confirmed.

    Recognition used to happen only while indexing, so naming someone never reached the
    faces indexed before. This re-runs the app's k-NN match (negative-aware, same
    threshold) from the **stored** embeddings — seconds, no photo is touched. Only
    faces a human hasn't confirmed (confidence < 1) are considered, and they are
    re-evaluated against the current enrolment: unnamed faces that now match get tagged,
    wrong guesses move to the right person, and guesses that no longer hold are cleared.
    Faces in another embedding space (different dimension) are skipped. Idempotent.

    Returns counts: ``tagged`` / ``retagged`` / ``cleared`` / ``unchanged`` /
    ``skipped_dim`` / ``need_review`` (resulting guesses below ``review_confidence``,
    shown under *Review guesses*), ``enrolled`` (confirmed faces learned from) and
    ``people`` (distinct people newly assigned).
    """

    from .indexer import _load_enrollment  # noqa: PLC0415 - indexer imports this module

    result: dict = {
        "enrolled": 0, "examined": 0, "tagged": 0, "retagged": 0, "cleared": 0,
        "unchanged": 0, "skipped_dim": 0, "need_review": 0, "people": 0,
        "threshold": config.face_match_threshold,
        "review_confidence": config.review_confidence, "dry_run": dry_run,
    }
    conn = db.connect(config.db_path)
    try:
        enrollment = _load_enrollment(conn, config.min_face_quality)
        result["enrolled"] = int(enrollment.person_ids.size)
        if enrollment.is_empty:
            return result
        dim = int(enrollment.embeddings.shape[1])
        rows = conn.execute(
            "SELECT id, person_id, confidence, embedding, quality FROM faces "
            "WHERE embedding IS NOT NULL AND (confidence IS NULL OR confidence < 1.0) "
            "AND (quality IS NULL OR quality >= ?)",
            (config.min_face_quality,),
        ).fetchall()
        updates: list[tuple[int | None, float, int]] = []
        newly: set[int] = set()
        for r in rows:
            vec = db.blob_to_embedding(r["embedding"])
            if vec is None or vec.shape[0] != dim:
                result["skipped_dim"] += 1
                continue
            result["examined"] += 1
            pid, conf = knn_person_match(
                vec, enrollment, k=config.recognition_k, threshold=config.face_match_threshold
            )
            old, old_conf = r["person_id"], r["confidence"] or 0.0
            conf = cap_confidence_for_quality(
                conf, r["quality"],
                review_confidence=config.review_confidence,
                review_quality=config.review_quality,
            )
            if pid is None:
                if old is None:
                    result["unchanged"] += 1
                else:
                    result["cleared"] += 1
                    updates.append((None, 0.0, r["id"]))
                continue
            if old is None:
                result["tagged"] += 1
                newly.add(pid)
            elif old != pid:
                result["retagged"] += 1
                newly.add(pid)
            else:
                result["unchanged"] += 1
            if conf < config.review_confidence:
                result["need_review"] += 1
            if old != pid or abs(old_conf - conf) > 1e-6:
                updates.append((pid, conf, r["id"]))
        result["people"] = len(newly)
        if not dry_run:
            for i in range(0, len(updates), 5000):
                conn.executemany(
                    "UPDATE faces SET person_id=?, confidence=?, "
                    "cluster_id = CASE WHEN ? IS NULL THEN cluster_id ELSE NULL END WHERE id=?",
                    [(p, c, p, fid) for p, c, fid in updates[i : i + 5000]],
                )
            conn.commit()
        return result
    finally:
        conn.close()


def backfill_face_quality(config: AtlasConfig, *, progress=None) -> int:
    """Score faces indexed before quality existed, from their saved crops.

    A crop has no landmarks, so pose is left neutral; size comes from the stored box and
    sharpness from the crop itself. Faces whose crop file is missing stay unscored (they
    are never excluded or demoted). Returns the number of faces scored.
    """

    import cv2  # noqa: PLC0415

    conn = db.connect(config.db_path)
    scored = 0
    try:
        rows = conn.execute(
            "SELECT id, crop_path, bbox_w, bbox_h FROM faces "
            "WHERE quality IS NULL AND crop_path IS NOT NULL"
        ).fetchall()
        for i, r in enumerate(rows):
            crop = cv2.imread(str(r["crop_path"]))
            if crop is None:
                continue
            h, w = crop.shape[:2]
            side = min(r["bbox_w"] or w, r["bbox_h"] or h)
            conn.execute(
                "UPDATE faces SET quality=? WHERE id=?",
                (face_quality(crop, (0, 0, w, h), None, side=float(side)), r["id"]),
            )
            scored += 1
            if scored % 500 == 0:
                conn.commit()
            if progress is not None:
                progress(i + 1, len(rows))
        conn.commit()
    finally:
        conn.close()
    return scored
