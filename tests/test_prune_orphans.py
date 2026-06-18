"""Crash-safe indexing: sweep orphaned derivative files + auto-prune.

Re-running ``index`` already resumes (it skips already-indexed paths), so the new
piece is reclaiming derivative files — thumbnails, preview/retina variants and
face crops — that no catalog row references any more (left behind by a crash
mid-index, or when a source photo's bytes/sha1 changed). ``prune`` now sweeps
them, and ``index --prune`` runs the whole reconciliation in one step.
"""

from __future__ import annotations

from photo_atlas import cli, db, demo, indexer
from photo_atlas.config import AtlasConfig


def _seed_photo(conn, path, sha1):
    return db.upsert_photo(conn, {"path": path, "filename": path.rsplit("/", 1)[-1], "sha1": sha1})


def test_sweep_orphan_derivatives_removes_unreferenced_only(tmp_path):
    cfg = AtlasConfig(home=tmp_path / "lib").ensure_dirs()
    conn = db.connect(cfg.db_path)
    live_sha1 = "a" * 40
    pid = _seed_photo(conn, "/live.jpg", live_sha1)
    conn.commit()
    conn.close()

    # Referenced derivatives for the live photo (content-addressed by sha1).
    live_thumb = indexer.thumb_path_for(cfg, live_sha1)
    live_thumb.parent.mkdir(parents=True, exist_ok=True)
    live_thumb.write_bytes(b"thumb")
    live_retina = cfg.thumbs_dir / live_sha1[:2] / f"{live_sha1}_640.jpg"
    live_retina.write_bytes(b"retina")
    live_preview = cfg.previews_dir / live_sha1[:2] / f"{live_sha1}_1600.jpg"
    live_preview.parent.mkdir(parents=True, exist_ok=True)
    live_preview.write_bytes(b"preview")
    live_crop_dir = cfg.faces_dir / str(pid)
    live_crop_dir.mkdir(parents=True, exist_ok=True)
    (live_crop_dir / "face_0.jpg").write_bytes(b"crop")

    # Orphans: a different sha1's thumb + preview, a leftover .part temp, and a
    # face-crop dir for a photo id that doesn't exist.
    dead_sha1 = "b" * 40
    dead_thumb = cfg.thumbs_dir / dead_sha1[:2] / f"{dead_sha1}.jpg"
    dead_thumb.parent.mkdir(parents=True, exist_ok=True)
    dead_thumb.write_bytes(b"x")
    dead_preview = cfg.previews_dir / dead_sha1[:2] / f"{dead_sha1}_1600.jpg"
    dead_preview.parent.mkdir(parents=True, exist_ok=True)
    dead_preview.write_bytes(b"x")
    stale_part = cfg.thumbs_dir / live_sha1[:2] / f"{live_sha1}_640.jpg.999.part"
    stale_part.write_bytes(b"half")
    dead_crop_dir = cfg.faces_dir / "9999"
    dead_crop_dir.mkdir(parents=True, exist_ok=True)
    (dead_crop_dir / "face_0.jpg").write_bytes(b"x")

    removed = indexer.sweep_orphan_derivatives(cfg)

    # The four orphans (dead thumb, dead preview, .part temp, dead crop dir) go.
    assert removed == 4
    assert not dead_thumb.exists()
    assert not dead_preview.exists()
    assert not stale_part.exists()
    assert not dead_crop_dir.exists()
    # Everything the catalog still references survives.
    assert live_thumb.exists() and live_retina.exists() and live_preview.exists()
    assert (live_crop_dir / "face_0.jpg").exists()


def test_sweep_is_idempotent(tmp_path):
    cfg = AtlasConfig(home=tmp_path / "lib").ensure_dirs()
    db.connect(cfg.db_path).close()
    orphan = cfg.thumbs_dir / "cc" / f"{'c' * 40}.jpg"
    orphan.parent.mkdir(parents=True, exist_ok=True)
    orphan.write_bytes(b"x")
    assert indexer.sweep_orphan_derivatives(cfg) == 1
    assert indexer.sweep_orphan_derivatives(cfg) == 0  # nothing left to do


def test_prune_library_reports_orphans(tmp_path):
    cfg = AtlasConfig(home=tmp_path / "lib").ensure_dirs()
    db.connect(cfg.db_path).close()
    orphan = cfg.previews_dir / "dd" / f"{'d' * 40}_1600.jpg"
    orphan.parent.mkdir(parents=True, exist_ok=True)
    orphan.write_bytes(b"x")

    result = indexer.prune_library(cfg)
    assert result["orphans"] == 1
    assert not orphan.exists()
    # The existing row-reconciliation keys are still present.
    assert "removed" in result and "kept" in result


def test_index_prune_flag_reconciles(tmp_path, capsys):
    photos = tmp_path / "pics"
    demo.generate(photos, count=3, seed=4)
    home = tmp_path / "lib"
    cfg = AtlasConfig(home=home).ensure_dirs()
    # Pre-seed an orphan derivative that --prune should sweep.
    orphan = cfg.thumbs_dir / "ee" / f"{'e' * 40}.jpg"
    orphan.parent.mkdir(parents=True, exist_ok=True)
    orphan.write_bytes(b"x")

    rc = cli.main(
        ["--home", str(home), "index", str(photos), "--faces", "none", "--prune", "--workers", "1"]
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "Pruned" in out  # reconciliation ran as part of index
    assert not orphan.exists()
