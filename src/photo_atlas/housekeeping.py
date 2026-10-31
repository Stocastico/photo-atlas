"""Catalog housekeeping: orphan sweep, prune, trash, hard delete, export, clustering.

Split out of :mod:`photo_atlas.indexer` (which re-exports these names)."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import cast

import numpy as np

from . import db
from .config import AtlasConfig
from .faces import (
    cluster_embeddings,
)


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
                f.unlink(missing_ok=True)
                removed += 1
                continue
            # Filenames are ``{sha1}.jpg`` or ``{sha1}_{size}.jpg``; pull the sha1.
            sha1 = f.name.split(".", 1)[0].split("_", 1)[0]
            if sha1 not in live_sha1:
                f.unlink(missing_ok=True)
                removed += 1
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
                Path(row["thumb_path"]).unlink(missing_ok=True)
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
        if entry.name != "manifest.jsonl" and entry.is_file():
            entry.unlink()
            removed += 1
    (config.trash_dir / "manifest.jsonl").unlink(missing_ok=True)
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
                    Path(row["thumb_path"]).unlink(missing_ok=True)
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
            "SELECT id, embedding FROM faces WHERE person_id IS NULL AND embedding IS NOT NULL"
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
