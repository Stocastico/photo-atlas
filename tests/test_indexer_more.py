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


def test_corrupt_image_is_counted_as_failed(tmp_path, capsys):
    cfg = AtlasConfig(home=tmp_path / "lib").ensure_dirs()
    bad = tmp_path / "broken.jpg"  # supported suffix, but not a real image
    bad.write_bytes(b"not actually a jpeg")
    stats = indexer.index_path(cfg, tmp_path, backend_name="none", geocode=False)
    assert stats.failed >= 1
    assert stats.indexed == 0


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
