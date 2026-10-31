"""Deriving face thresholds from the user's own named faces."""

from __future__ import annotations

import numpy as np

from photo_atlas import cli, db, faces, housekeeping
from photo_atlas.config import AtlasConfig


def _people(n_people=6, per_person=30, noise=0.25, dim=64, seed=0):
    rng = np.random.default_rng(seed)
    centers = rng.normal(size=(n_people, dim))
    embs, pids = [], []
    for p, c in enumerate(centers):
        for _ in range(per_person):
            embs.append(c + noise * rng.normal(size=dim))
            pids.append(p + 1)
    return np.array(embs, dtype=np.float32), np.array(pids)


def test_threshold_separates_genuine_from_impostor_pairs():
    embs, pids = _people()
    r = faces.calibrate_threshold(embs, pids)
    assert r is not None
    assert r["genuine_p50"] < r["impostor_p1"]  # the classes are separable here
    assert r["genuine_p50"] <= r["threshold"] <= r["impostor_p50"]
    assert r["recall"] > 0.9 and r["false_accept"] <= 0.002
    assert r["genuine_pairs"] > 0 and r["impostor_pairs"] > 0
    assert 0.2 <= r["threshold"] <= 0.7  # clamped to a sane range
    assert 1 - r["threshold"] <= r["review_confidence"] <= 0.9


def test_calibration_is_deterministic():
    embs, pids = _people()
    assert faces.calibrate_threshold(embs, pids) == faces.calibrate_threshold(embs, pids)


def test_noisier_faces_give_a_looser_threshold():
    tight = faces.calibrate_threshold(*_people(noise=0.15))
    loose = faces.calibrate_threshold(*_people(noise=0.6))
    assert loose["genuine_p50"] > tight["genuine_p50"]


def test_not_enough_data_returns_none():
    embs, pids = _people(n_people=1)
    assert faces.calibrate_threshold(embs, pids) is None  # no impostor pairs
    embs, pids = _people(n_people=3, per_person=1)
    assert faces.calibrate_threshold(embs, pids) is None  # no genuine pairs


def _seed_catalog(config, embs, pids, confidence=1.0):
    conn = db.connect(config.db_path)
    names = {int(p): db.get_or_create_person(conn, f"P{p}") for p in set(pids.tolist())}
    for e, p in zip(embs, pids, strict=True):
        photo = db.upsert_photo(conn, {"path": f"/x/{len(e)}_{np.random.rand()}.jpg",
                                       "filename": "x.jpg"})
        db.replace_faces(conn, photo, [{
            "person_id": names[int(p)], "embedding": db.embedding_to_blob(e),
            "dim": len(e), "confidence": confidence,
        }])
    conn.commit()
    conn.close()


def test_calibrate_uses_only_human_confirmed_faces(tmp_path):
    config = AtlasConfig(home=tmp_path).ensure_dirs()
    embs, pids = _people(n_people=3, per_person=10)
    _seed_catalog(config, embs, pids, confidence=0.7)  # auto-tags only
    assert housekeeping.calibrate_faces(config) is None


def test_cli_prints_suggested_environment(tmp_path, capsys):
    config = AtlasConfig(home=tmp_path).ensure_dirs()
    _seed_catalog(config, *_people(n_people=4, per_person=15))
    assert cli.main(["--home", str(tmp_path), "calibrate-faces"]) == 0
    out = capsys.readouterr().out
    assert "PHOTO_ATLAS_FACE_THRESHOLD=" in out and "PHOTO_ATLAS_CLUSTER_EPS=" in out
    assert "PHOTO_ATLAS_REVIEW_CONFIDENCE=" in out


def test_cli_explains_when_there_is_not_enough_data(tmp_path, capsys):
    AtlasConfig(home=tmp_path).ensure_dirs()
    assert cli.main(["--home", str(tmp_path), "calibrate-faces"]) == 1
    assert "name at least two people" in capsys.readouterr().out.lower()
