"""Re-tagging scenes in place, without a full re-index."""

from __future__ import annotations

from photo_atlas import cli, db, demo, indexer


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


def test_retag_scenes_cli_runs(tmp_path, capsys):
    photos = tmp_path / "pics"
    demo.generate(photos, count=5, seed=4)
    home = tmp_path / "lib"
    cli.main(["--home", str(home), "index", str(photos), "--faces", "none"])
    capsys.readouterr()
    rc = cli.main(["--home", str(home), "retag-scenes", "--scene", "heuristic"])
    assert rc == 0
    assert "retagged" in capsys.readouterr().out

