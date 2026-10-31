"""Re-tagging scenes in place, without a full re-index."""

from __future__ import annotations

from photo_atlas import cli, db, demo, indexer
from photo_atlas.config import AtlasConfig


class _FixedTagger:
    """A stand-in tagger that always returns the same label."""

    def tag_image(self, img, face_count=0):
        return "food", {"food": 1.0, "people": 0.0}

    def tag(self, path, face_count=0):
        return "food", {"food": 1.0, "people": 0.0}


def test_retag_scenes_rewrites_only_scene(indexed):
    conn = db.connect(indexed.db_path)
    before = {
        r["id"]: (r["scene_type"], r["face_count"], r["thumb_path"])
        for r in conn.execute("SELECT id, scene_type, face_count, thumb_path FROM photos")
    }
    conn.close()
    assert before, "demo library should have photos"
    # Not everything was 'food' to begin with (else the test proves nothing).
    assert {v[0] for v in before.values()} != {"food"}

    n = indexer.retag_scenes(indexed, tagger=_FixedTagger())
    assert n == len(before)

    conn = db.connect(indexed.db_path)
    after = {
        r["id"]: (r["scene_type"], r["face_count"], r["thumb_path"])
        for r in conn.execute("SELECT id, scene_type, face_count, thumb_path FROM photos")
    }
    conn.close()
    # Every scene became 'food'; face_count and thumbnails are untouched.
    assert {v[0] for v in after.values()} == {"food"}
    for pid, (_, fc, thumb) in after.items():
        assert (fc, thumb) == (before[pid][1], before[pid][2])


def test_retag_scenes_warns_on_unreadable_file(tmp_path, capsys):
    """A corrupt/unreadable source is skipped (not counted) but reported on stderr
    instead of failing silently."""
    from photo_atlas import db as _db
    from photo_atlas.config import AtlasConfig

    cfg = AtlasConfig(home=tmp_path / "lib").ensure_dirs()
    conn = _db.connect(cfg.db_path)
    # A row whose "image" is actually garbage bytes -> Image.open/load raises.
    bad = tmp_path / "broken.jpg"
    bad.write_bytes(b"not an image")
    _db.upsert_photo(conn, {"path": str(bad), "filename": "broken.jpg", "face_count": 0})
    conn.commit()
    conn.close()

    n = indexer.retag_scenes(cfg, tagger=_FixedTagger())
    assert n == 0  # the unreadable photo was skipped
    assert "could not retag" in capsys.readouterr().err


def test_retag_scenes_cli_runs(tmp_path, capsys):
    photos = tmp_path / "pics"
    demo.generate(photos, count=5, seed=4)
    home = tmp_path / "lib"
    cli.main(["--home", str(home), "index", str(photos), "--faces", "none", "--workers", "1"])
    capsys.readouterr()
    rc = cli.main(["--home", str(home), "retag-scenes"])
    assert rc == 0
    assert "retagged" in capsys.readouterr().out



class _EmbeddingTagger:
    """Tags from a stored embedding; blows up if asked to decode an image."""

    def __init__(self):
        self.from_embedding = []

    def tag_embedding(self, embedding, face_count=0):
        self.from_embedding.append(tuple(embedding))
        return "food", {"food": 1.0}

    def tag_image(self, img, face_count=0):
        raise AssertionError("must not decode when an embedding is stored")

    def tag(self, path, face_count=0):
        raise AssertionError("must not decode when an embedding is stored")


def test_retag_uses_stored_embeddings_without_decoding(tmp_path):
    import numpy as np

    config = AtlasConfig(home=tmp_path).ensure_dirs()
    conn = db.connect(config.db_path)
    # Source files don't even exist: the stored embedding is all that's needed.
    for i in range(3):
        pid = db.upsert_photo(conn, {"path": f"/gone/{i}.jpg", "filename": f"{i}.jpg",
                                     "scene_type": "other", "face_count": 0})
        db.set_photo_embedding(conn, pid, np.array([1.0, 0.0, float(i)], dtype=np.float32))
    conn.commit()
    conn.close()

    tagger = _EmbeddingTagger()
    assert indexer.retag_scenes(config, tagger=tagger) == 3
    assert len(tagger.from_embedding) == 3
    conn = db.connect(config.db_path)
    assert {r[0] for r in conn.execute("SELECT scene_type FROM photos")} == {"food"}
    conn.close()


def test_retag_falls_back_to_decoding_when_no_embedding(indexed):
    """Photos without a stored embedding (or a tagger lacking tag_embedding) still work."""

    n = indexer.retag_scenes(indexed, tagger=_FixedTagger())  # no tag_embedding at all
    assert n > 0
