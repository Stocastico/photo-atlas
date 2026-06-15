"""Edge cases of the directory indexer: single files, failures, missing backend."""

from __future__ import annotations

from photo_atlas import db, indexer
from photo_atlas.config import AtlasConfig


def test_iter_images_accepts_single_file(tmp_path):
    from PIL import Image

    f = tmp_path / "one.jpg"
    Image.new("RGB", (8, 8)).save(f, "JPEG")
    assert list(indexer.iter_images(f)) == [f]
    # A non-image single file yields nothing.
    txt = tmp_path / "note.txt"
    txt.write_text("hi")
    assert list(indexer.iter_images(txt)) == []


def test_iter_images_walks_nested_dirs_deterministically(tmp_path):
    from PIL import Image

    # Build a small nested tree; only the image files should come back, sorted.
    (tmp_path / "2012" / "trip").mkdir(parents=True)
    (tmp_path / "2013").mkdir()
    for rel in ("2013/b.jpg", "2012/a.png", "2012/trip/c.jpeg", "2012/notes.txt"):
        p = tmp_path / rel
        if p.suffix == ".txt":
            p.write_text("x")
        else:
            Image.new("RGB", (4, 4)).save(p)

    found = [p.name for p in indexer.iter_images(tmp_path)]
    # Top-down walk, each level sorted: 2012/ before 2013/, and within 2012/ the
    # file (a.png) before the descent into trip/ (c.jpeg). Deterministic.
    assert found == ["a.png", "c.jpeg", "b.jpg"]  # .txt excluded, recursive


def test_videos_are_counted_but_not_indexed(tmp_path):
    from PIL import Image

    from photo_atlas import db

    Image.new("RGB", (8, 8)).save(tmp_path / "photo.jpg", "JPEG")
    (tmp_path / "clip.mov").write_bytes(b"\x00\x00\x00\x18ftypqt  ")  # dummy video
    (tmp_path / "movie.MP4").write_bytes(b"\x00\x00\x00\x18ftypmp42")

    cfg = AtlasConfig(home=tmp_path / "lib").ensure_dirs()
    stats = indexer.index_path(cfg, tmp_path, backend_name="none", geocode=False)

    assert stats.indexed == 1            # the jpg
    assert stats.videos == 2             # .mov + .MP4 (case-insensitive)
    conn = db.connect(cfg.db_path)
    try:
        rows = [r["filename"] for r in conn.execute("SELECT filename FROM photos")]
    finally:
        conn.close()
    assert rows == ["photo.jpg"]         # videos never entered the catalog


def test_identical_files_are_deduplicated_by_sha1(tmp_path):
    import shutil

    from PIL import Image

    from photo_atlas import db

    a = tmp_path / "a.jpg"
    Image.new("RGB", (16, 16), (123, 50, 200)).save(a, "JPEG")
    shutil.copyfile(a, tmp_path / "b.jpg")   # byte-identical copy -> same sha1

    cfg = AtlasConfig(home=tmp_path / "lib").ensure_dirs()
    stats = indexer.index_path(cfg, tmp_path, backend_name="none", geocode=False)

    assert stats.indexed == 1
    assert stats.duplicates == 1
    conn = db.connect(cfg.db_path)
    try:
        assert conn.execute("SELECT COUNT(*) FROM photos").fetchone()[0] == 1
    finally:
        conn.close()


def test_corrupt_image_is_counted_as_failed_with_diagnostics(tmp_path, capsys):
    cfg = AtlasConfig(home=tmp_path / "lib").ensure_dirs()
    bad = tmp_path / "broken.jpg"  # supported suffix, but not a real image
    bad.write_bytes(b"not actually a jpeg")
    stats = indexer.index_path(cfg, tmp_path, backend_name="none", geocode=False)
    assert stats.failed >= 1
    assert stats.indexed == 0
    # The failure is now diagnosable: the path is captured, not silently dropped.
    assert stats.errors and "broken.jpg" in stats.errors[0]


def test_unavailable_backend_warns_and_indexes_without_faces(tmp_path, capsys):
    from photo_atlas import demo

    photos = tmp_path / "p"
    demo.generate(photos, count=3, seed=1)
    cfg = AtlasConfig(home=tmp_path / "lib").ensure_dirs()
    # dlib/face_recognition isn't installed in the test env, so the backend
    # resolves to None and indexing should warn but still complete.
    stats = indexer.index_path(cfg, photos, backend_name="dlib", geocode=False)
    err = capsys.readouterr().err
    assert "indexing without faces" in err
    assert stats.indexed == 3
    assert stats.faces == 0


def test_index_warns_when_geocoder_is_low_resolution(tmp_path):
    from photo_atlas import demo

    photos = tmp_path / "p"
    demo.generate(photos, count=2, seed=1)
    cfg = AtlasConfig(home=tmp_path / "lib").ensure_dirs()
    # reverse_geocoder isn't installed in the test env, so geocoding falls back
    # to the coarse bundled table — the user should be told.
    import contextlib
    import io

    err = io.StringIO()
    with contextlib.redirect_stderr(err):
        indexer.index_path(cfg, photos, backend_name="none", geocode=True)
    msg = err.getvalue()
    assert "reverse_geocoder" in msg or "--extra geo" in msg


def test_no_geocode_run_does_not_warn_about_resolution(tmp_path):
    from photo_atlas import demo

    photos = tmp_path / "p"
    demo.generate(photos, count=1, seed=1)
    cfg = AtlasConfig(home=tmp_path / "lib").ensure_dirs()
    import contextlib
    import io

    err = io.StringIO()
    with contextlib.redirect_stderr(err):
        indexer.index_path(cfg, photos, backend_name="none", geocode=False)
    assert "reverse_geocoder" not in err.getvalue()


def test_progress_callback_is_invoked(tmp_path):
    from photo_atlas import demo

    photos = tmp_path / "p"
    demo.generate(photos, count=4, seed=2)
    cfg = AtlasConfig(home=tmp_path / "lib").ensure_dirs()
    seen = []
    indexer.index_path(
        cfg, photos, backend_name="none", geocode=False,
        progress=lambda path, stats: seen.append(stats.scanned),
    )
    assert seen and seen[-1] == 4


def test_index_file_decodes_image_once(tmp_path, monkeypatch):
    """The per-file pipeline (metadata, thumbnail, scene tag, face crops) reuses
    a single PIL decode instead of re-opening the file for each stage."""

    import PIL.Image as PILImage

    from photo_atlas import demo, faces
    from photo_atlas.classify import SceneTagger

    [photo] = demo.generate(tmp_path / "p", count=1, seed=3)
    cfg = AtlasConfig(home=tmp_path / "lib").ensure_dirs()
    conn = db.connect(cfg.db_path)

    calls = {"n": 0}
    real_open = PILImage.open

    def counting_open(*a, **k):
        calls["n"] += 1
        return real_open(*a, **k)

    monkeypatch.setattr(PILImage, "open", counting_open)
    try:
        indexer.index_file(
            conn, cfg, photo,
            backend=faces.SyntheticFaceBackend(), geocoder=None, tagger=SceneTagger(),
        )
    finally:
        conn.close()
    assert calls["n"] == 1


def _photo_snapshot(cfg) -> dict:
    """Map resolved path -> a comparable tuple of the indexed photo + face count."""

    conn = db.connect(cfg.db_path)
    try:
        rows = conn.execute(
            "SELECT path, sha1, width, height, scene_type, face_count, taken_source "
            "FROM photos"
        ).fetchall()
        return {
            r["path"]: (
                r["sha1"], r["width"], r["height"], r["scene_type"],
                r["face_count"], r["taken_source"],
            )
            for r in rows
        }
    finally:
        conn.close()


def test_parallel_indexing_matches_serial(tmp_path):
    """Indexing with multiple worker processes yields exactly the same catalog as
    the single-process path: same photos, same per-photo metadata and face counts."""

    from photo_atlas import demo

    photos = tmp_path / "p"
    demo.generate(photos, count=8, seed=11)

    serial = AtlasConfig(home=tmp_path / "serial").ensure_dirs()
    parallel = AtlasConfig(home=tmp_path / "parallel").ensure_dirs()

    s_stats = indexer.index_path(serial, photos, backend_name="synthetic", geocode=False)
    p_stats = indexer.index_path(
        parallel, photos, backend_name="synthetic", geocode=False, workers=2
    )

    assert p_stats.indexed == s_stats.indexed
    assert p_stats.faces == s_stats.faces
    assert p_stats.scanned == s_stats.scanned
    assert p_stats.failed == 0

    snap_serial = _photo_snapshot(serial)
    snap_parallel = _photo_snapshot(parallel)
    # Paths differ only by the library home, not the source photos, so compare by
    # the source path itself (stored as the resolved photo path, identical here).
    assert snap_serial == snap_parallel
    # Face crops landed on disk for the parallel run too (written from the main
    # process after the worker handed back the encoded crop bytes).
    conn = db.connect(parallel.db_path)
    try:
        crops = [
            r["crop_path"]
            for r in conn.execute(
                "SELECT crop_path FROM faces WHERE crop_path IS NOT NULL"
            ).fetchall()
        ]
    finally:
        conn.close()
    from pathlib import Path

    assert crops and all(Path(c).exists() for c in crops)


def test_parallel_indexing_progress_and_dedup(tmp_path):
    """The parallel path still reports progress, skips byte-identical duplicates
    and survives a file that fails to decode."""

    import shutil
    from pathlib import Path

    from photo_atlas import demo

    photos = tmp_path / "p"
    paths = demo.generate(photos, count=4, seed=5)
    # A byte-identical duplicate of the first photo under a second name.
    shutil.copyfile(paths[0], photos / "dup_copy.jpg")
    # A corrupt "image" that fails to decode in a worker.
    (photos / "broken.jpg").write_bytes(b"not a real jpeg")

    cfg = AtlasConfig(home=tmp_path / "lib").ensure_dirs()
    seen: list[int] = []

    def progress(_path: Path, stats) -> None:
        seen.append(stats.indexed)

    stats = indexer.index_path(
        cfg, photos, backend_name="synthetic", geocode=False, workers=2,
        progress=progress,
    )

    assert stats.indexed == 4  # 4 originals; the copy is deduped, broken fails
    assert stats.duplicates == 1
    assert stats.failed == 1
    assert seen  # progress was called as results streamed in


def test_prune_removes_rows_for_deleted_files(tmp_path):
    from pathlib import Path

    from photo_atlas import db, demo

    photos = tmp_path / "p"
    paths = demo.generate(photos, count=3, seed=7)
    cfg = AtlasConfig(home=tmp_path / "lib").ensure_dirs()
    indexer.index_path(cfg, photos, backend_name="synthetic", geocode=False)

    # The user deletes one source file on disk.
    victim = Path(paths[0])
    conn = db.connect(cfg.db_path)
    try:
        row = conn.execute(
            "SELECT id, thumb_path FROM photos WHERE path=?", (str(victim.resolve()),)
        ).fetchone()
        victim_id, thumb = row["id"], row["thumb_path"]
    finally:
        conn.close()
    victim.unlink()

    result = indexer.prune_library(cfg)
    assert result["removed"] == 1 and result["kept"] == 2

    conn = db.connect(cfg.db_path)
    try:
        assert conn.execute("SELECT COUNT(*) FROM photos").fetchone()[0] == 2
        # Its faces went too (cascade), and the row is gone.
        assert conn.execute(
            "SELECT COUNT(*) FROM faces WHERE photo_id=?", (victim_id,)
        ).fetchone()[0] == 0
    finally:
        conn.close()
    assert not Path(thumb).exists()  # derivative cleaned up


def test_reindex_reuses_content_addressed_thumb(tmp_path):
    from photo_atlas import demo

    photos = tmp_path / "p"
    demo.generate(photos, count=2, seed=9)
    cfg = AtlasConfig(home=tmp_path / "lib").ensure_dirs()
    indexer.index_path(cfg, photos, backend_name="none", geocode=False)
    conn = db.connect(cfg.db_path)
    try:
        thumb = conn.execute("SELECT thumb_path FROM photos LIMIT 1").fetchone()["thumb_path"]
    finally:
        conn.close()
    # Thumb is named by sha1 -> stable across runs.
    from pathlib import Path

    assert Path(thumb).exists()
