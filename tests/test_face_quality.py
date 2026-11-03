"""Face quality: sharpness, size and pose, stored per face and used for display/review."""

from __future__ import annotations

import cv2
import numpy as np
import pytest
from face_stubs import _Backend

from photo_atlas import cli, db, faces, housekeeping, library
from photo_atlas.config import AtlasConfig

# YuNet landmark order: right eye, left eye, nose, right mouth corner, left mouth corner.
FRONTAL = np.array([[70, 80], [130, 80], [100, 110], [80, 140], [120, 140]], dtype=np.float32)
PROFILE = np.array([[70, 80], [130, 80], [136, 110], [80, 140], [120, 140]], dtype=np.float32)


def _textured(size=240, seed=0):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 255, size=(size, size, 3), dtype=np.uint8)


def test_sharp_beats_blurred():
    sharp = _textured()
    blurred = cv2.GaussianBlur(sharp, (31, 31), 12)
    bbox = (40, 40, 160, 160)
    gap = faces.face_quality(sharp, bbox, FRONTAL) - faces.face_quality(blurred, bbox, FRONTAL)
    assert gap > 0.2


def test_bigger_faces_score_higher():
    img = _textured()
    small = faces.face_quality(img, (50, 50, 20, 20), FRONTAL)
    big = faces.face_quality(img, (20, 20, 180, 180), FRONTAL)
    assert big > small


def test_profile_scores_lower_than_frontal():
    img = _textured()
    bbox = (40, 40, 160, 160)
    assert faces.face_quality(img, bbox, FRONTAL) > faces.face_quality(img, bbox, PROFILE)


def test_range_and_edge_cases():
    img = _textured()
    for bbox in [(0, 0, 240, 240), (200, 200, 100, 100), (-30, -30, 80, 80)]:
        q = faces.face_quality(img, bbox, FRONTAL)
        assert 0.0 <= q <= 1.0
    assert faces.face_quality(img, (500, 500, 50, 50), FRONTAL) == 0.0  # entirely outside
    assert faces.face_quality(img, (10, 10, 0, 0), FRONTAL) == 0.0


def test_missing_landmarks_do_not_penalise_pose():
    img = _textured()
    bbox = (40, 40, 160, 160)
    with_landmarks = faces.face_quality(img, bbox, FRONTAL)
    assert faces.face_quality(img, bbox, None) == pytest.approx(with_landmarks)


def test_detect_attaches_quality():
    backend = _Backend(np.array([[10, 20, 60, 60, *FRONTAL.ravel(), 0.95]], dtype=np.float32))
    (obs,) = backend.detect("x.jpg", image=_textured(200))
    assert obs.quality is not None and 0.0 <= obs.quality <= 1.0


# -- storage -----------------------------------------------------------------
def test_quality_is_stored_with_new_faces(indexed):
    conn = db.connect(indexed.db_path)
    try:
        qs = [r["quality"] for r in conn.execute("SELECT quality FROM faces")]
    finally:
        conn.close()
    assert qs and all(q is not None and 0.0 <= q <= 1.0 for q in qs)


def test_migration_v3_adds_the_column(tmp_path):
    path = tmp_path / "old.db"
    conn = db.connect(path)
    conn.execute("ALTER TABLE faces DROP COLUMN quality")
    conn.execute("PRAGMA user_version = 2")
    conn.commit()
    conn.close()
    conn = db.connect(path)
    try:
        assert "quality" in {r["name"] for r in conn.execute("PRAGMA table_info(faces)")}
        assert conn.execute("PRAGMA user_version").fetchone()[0] == db.SCHEMA_VERSION >= 3
    finally:
        conn.close()


def _face(conn, *, quality=None, vec=(1.0, 0.0, 0.0), person=None, confidence=0.0, cluster=None,
          crop=None):
    photo = db.upsert_photo(conn, {"path": f"/q/{conn.total_changes}_{np.random.rand()}.jpg",
                                   "filename": "x.jpg"})
    db.replace_faces(conn, photo, [{
        "person_id": person, "confidence": confidence, "cluster_id": cluster, "quality": quality,
        "embedding": db.embedding_to_blob(np.array(vec, dtype=np.float32)), "dim": len(vec),
        "crop_path": crop, "bbox_x": 0, "bbox_y": 0, "bbox_w": 120, "bbox_h": 120,
    }])
    return conn.execute("SELECT id FROM faces WHERE photo_id=?", (photo,)).fetchone()["id"]


# -- backfill ----------------------------------------------------------------
def test_backfill_scores_old_faces_from_their_crops(tmp_path):
    config = AtlasConfig(home=tmp_path / "lib").ensure_dirs()
    sharp, blurry = tmp_path / "sharp.jpg", tmp_path / "blurry.jpg"
    img = _textured(120)
    cv2.imwrite(str(sharp), img)
    cv2.imwrite(str(blurry), cv2.GaussianBlur(img, (31, 31), 12))
    conn = db.connect(config.db_path)
    a = _face(conn, crop=str(sharp))
    b = _face(conn, crop=str(blurry))
    c = _face(conn, crop=str(tmp_path / "missing.jpg"))
    conn.commit()
    conn.close()

    assert housekeeping.backfill_face_quality(config) == 2  # the missing crop can't be scored
    conn = db.connect(config.db_path)
    try:
        q = {r["id"]: r["quality"] for r in conn.execute("SELECT id, quality FROM faces")}
    finally:
        conn.close()
    assert q[a] > q[b] and q[c] is None
    assert housekeeping.backfill_face_quality(config) == 0  # only fills what's missing


def test_cli_face_quality(tmp_path, capsys):
    config = AtlasConfig(home=tmp_path / "lib").ensure_dirs()
    crop = tmp_path / "c.jpg"
    cv2.imwrite(str(crop), _textured(120))
    conn = db.connect(config.db_path)
    _face(conn, crop=str(crop))
    conn.commit()
    conn.close()
    assert cli.main(["--home", str(config.home), "face-quality"]) == 0
    assert "Scored 1" in capsys.readouterr().out


# -- display -----------------------------------------------------------------
def test_cluster_samples_prefer_good_quality_among_equally_typical_faces(tmp_path):
    conn = db.connect(tmp_path / "c.db")
    try:
        blurry = [_face(conn, quality=0.1, cluster=1, crop="c") for _ in range(6)]
        sharp = [_face(conn, quality=0.95, cluster=1, crop="c") for _ in range(6)]
        conn.commit()
        (cluster,) = library.list_clusters(conn)
        shown = {s["id"] for s in cluster["samples"]}
        assert shown == set(sharp) and not shown & set(blurry)
    finally:
        conn.close()


def test_default_person_cover_is_the_best_quality_face(tmp_path):
    conn = db.connect(tmp_path / "c.db")
    try:
        anna = db.get_or_create_person(conn, "Anna")
        _face(conn, quality=0.2, person=anna, confidence=1.0, crop="a")
        best = _face(conn, quality=0.9, person=anna, confidence=1.0, crop="b")
        _face(conn, quality=None, person=anna, confidence=1.0, crop="c")
        conn.commit()
        (p,) = library.list_persons(conn)
        assert p["cover_face_id"] == best
    finally:
        conn.close()


# -- review routing & gating -------------------------------------------------
E1 = (1.0, 0.0, 0.0)


def _lib(tmp_path):
    config = AtlasConfig(home=tmp_path / "lib").ensure_dirs()
    conn = db.connect(config.db_path)
    anna = db.get_or_create_person(conn, "Anna")
    for i in range(3):
        _face(conn, person=anna, confidence=1.0, quality=0.9,
              vec=tuple(np.array(E1) + 0.01 * i))
    conn.commit()
    return config, conn, anna


def test_weak_quality_guesses_are_routed_to_review(tmp_path):
    config, conn, anna = _lib(tmp_path)
    try:
        good = _face(conn, quality=0.9, vec=(1.0, 0.01, 0.0))
        poor = _face(conn, quality=0.1, vec=(1.0, 0.01, 0.0))
        conn.commit()
        housekeeping.recognize_faces(config)
        c = {r["id"]: r["confidence"] for r in conn.execute("SELECT id, confidence FROM faces")}
        assert c[good] >= config.review_confidence      # trusted: not in the review queue
        assert 0 < c[poor] < config.review_confidence   # same face, bad crop: a human checks it
    finally:
        conn.close()


def test_min_face_quality_gates_enrolment_and_recognition(tmp_path, monkeypatch):
    config, conn, anna = _lib(tmp_path)
    try:
        bob = db.get_or_create_person(conn, "Bob")
        _face(conn, person=bob, confidence=1.0, quality=0.1, vec=(0.0, 1.0, 0.0))  # blurry example
        probe_good = _face(conn, quality=0.9, vec=(0.0, 1.0, 0.0))
        poor_unknown = _face(conn, quality=0.1, vec=(1.0, 0.01, 0.0))
        legacy_unknown = _face(conn, quality=None, vec=(1.0, 0.01, 0.0))  # not scored yet
        conn.commit()
        monkeypatch.setenv("PHOTO_ATLAS_MIN_FACE_QUALITY", "0.5")
        housekeeping.recognize_faces(AtlasConfig(home=config.home))  # re-reads the env
        r = {x["id"]: x["person_id"] for x in conn.execute("SELECT id, person_id FROM faces")}
        assert r[probe_good] is None      # Bob's only example was too blurry to learn from
        assert r[poor_unknown] is None    # too poor to name automatically
        assert r[legacy_unknown] == anna  # no score yet: not penalised
    finally:
        conn.close()


def test_gating_is_off_by_default(monkeypatch):
    monkeypatch.delenv("PHOTO_ATLAS_MIN_FACE_QUALITY", raising=False)
    cfg = AtlasConfig()
    assert cfg.min_face_quality == 0.0 and 0.0 < cfg.review_quality < 1.0


def test_unscored_faces_cluster_but_poor_ones_are_left_out_when_gated(tmp_path, monkeypatch):
    config = AtlasConfig(home=tmp_path / "lib").ensure_dirs()
    conn = db.connect(config.db_path)
    try:
        good = [_face(conn, quality=0.9, vec=(1.0, 0.01 * i, 0.0)) for i in range(3)]
        poor = [_face(conn, quality=0.1, vec=(1.0, 0.01 * i, 0.0)) for i in range(3)]
        conn.commit()
        monkeypatch.setenv("PHOTO_ATLAS_MIN_FACE_QUALITY", "0.5")
        housekeeping.cluster_library(AtlasConfig(home=config.home))
        rows = {r["id"]: r["cluster_id"] for r in conn.execute("SELECT id, cluster_id FROM faces")}
        assert all(rows[i] is not None for i in good)
        assert all(rows[i] is None for i in poor)
    finally:
        conn.close()
