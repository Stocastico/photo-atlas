"""Propagating the names you've given to the faces already in the catalog."""

from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient

from photo_atlas import cli, db, housekeeping
from photo_atlas.api import create_app
from photo_atlas.config import AtlasConfig

E1, E2, E3 = (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)


def _vec(base, jitter=0.0, seed=0):
    rng = np.random.default_rng(seed)
    v = np.array(base, dtype=np.float32) + jitter * rng.normal(size=3).astype(np.float32)
    return v / np.linalg.norm(v)


def _add(conn, vec, *, person=None, confidence=0.0, cluster=None, dim=None):
    photo = db.upsert_photo(conn, {"path": f"/p/{conn.total_changes}_{id(vec)}.jpg",
                                   "filename": "x.jpg"})
    db.replace_faces(conn, photo, [{
        "person_id": person, "confidence": confidence, "cluster_id": cluster,
        "embedding": db.embedding_to_blob(vec), "dim": dim or len(vec),
    }])
    return conn.execute("SELECT id FROM faces WHERE photo_id=?", (photo,)).fetchone()["id"]


def _state(conn, face_id):
    r = conn.execute("SELECT person_id, confidence, cluster_id FROM faces WHERE id=?",
                     (face_id,)).fetchone()
    return r["person_id"], r["confidence"], r["cluster_id"]


@pytest.fixture
def lib(tmp_path):
    config = AtlasConfig(home=tmp_path / "lib").ensure_dirs()
    conn = db.connect(config.db_path)
    anna, bob = db.get_or_create_person(conn, "Anna"), db.get_or_create_person(conn, "Bob")
    for i in range(3):
        _add(conn, _vec(E1, 0.02, i), person=anna, confidence=1.0)
        _add(conn, _vec(E2, 0.02, i), person=bob, confidence=1.0)
    conn.commit()
    yield config, conn, anna, bob
    conn.close()


def test_unnamed_faces_get_the_name_of_the_person_they_match(lib):
    config, conn, anna, bob = lib
    near_anna = [_add(conn, _vec(E1, 0.03, 10 + i), cluster=7) for i in range(2)]
    near_bob = _add(conn, _vec(E2, 0.03, 20))
    stranger = _add(conn, _vec(E3, 0.02, 30), cluster=8)
    conn.commit()

    result = housekeeping.recognize_faces(config)

    assert result["tagged"] == 3 and result["unchanged"] == 1 and result["cleared"] == 0
    for f in near_anna:
        person, conf, cluster = _state(conn, f)
        assert person == anna and 0.5 <= conf < 1.0 and cluster is None
    assert _state(conn, near_bob)[0] == bob
    assert _state(conn, stranger) == (None, 0.0, 8)  # no one close enough: untouched
    # The denormalised "known people" counter follows (trigger-maintained).
    n = conn.execute("SELECT SUM(named_face_count) FROM photos").fetchone()[0]
    assert n == 6 + 3


def test_it_is_idempotent(lib):
    config, conn, *_ = lib
    _add(conn, _vec(E1, 0.03, 10))
    conn.commit()
    assert housekeeping.recognize_faces(config)["tagged"] == 1
    again = housekeeping.recognize_faces(config)
    assert again["tagged"] == 0 and again["retagged"] == 0 and again["cleared"] == 0


def test_human_labels_are_never_overridden(lib):
    config, conn, anna, bob = lib
    mine = _add(conn, _vec(E1, 0.02, 5), person=bob, confidence=1.0)  # looks like Anna, I said Bob
    conn.commit()
    housekeeping.recognize_faces(config)
    assert _state(conn, mine)[:2] == (bob, 1.0)


def test_earlier_guesses_are_reevaluated(lib):
    config, conn, anna, bob = lib
    wrong = _add(conn, _vec(E1, 0.03, 40), person=bob, confidence=0.62)   # mis-tagged
    orphan = _add(conn, _vec(E3, 0.02, 41), person=anna, confidence=0.55)  # no longer matches
    conn.commit()
    result = housekeeping.recognize_faces(config)
    assert result["retagged"] == 1 and result["cleared"] == 1
    assert _state(conn, wrong)[0] == anna
    assert _state(conn, orphan)[:2] == (None, 0.0)


def test_corrections_you_made_are_respected(lib):
    config, conn, anna, bob = lib
    # One more Anna example is not enough to overcome a recorded "this is NOT Anna":
    # build a tiny enrolment where a single negative cancels the single positive.
    conn.execute("DELETE FROM faces WHERE person_id=?", (anna,))
    pos = _add(conn, _vec(E1, 0.0, 0), person=anna, confidence=1.0)
    twin = _add(conn, _vec(E1, 0.01, 1))
    probe = _add(conn, _vec(E1, 0.02, 2))
    db.add_face_negative(conn, twin, anna)  # "this near-identical face is not Anna"
    conn.commit()
    housekeeping.recognize_faces(config)
    assert _state(conn, probe)[0] != anna and pos


def test_dry_run_reports_without_writing(lib):
    config, conn, *_ = lib
    f = _add(conn, _vec(E1, 0.03, 10))
    conn.commit()
    result = housekeeping.recognize_faces(config, dry_run=True)
    assert result["tagged"] == 1 and result["dry_run"] is True
    assert _state(conn, f)[0] is None


def test_no_confirmed_faces_means_nothing_to_learn_from(tmp_path):
    config = AtlasConfig(home=tmp_path / "e").ensure_dirs()
    conn = db.connect(config.db_path)
    _add(conn, _vec(E1))
    conn.commit()
    conn.close()
    result = housekeeping.recognize_faces(config)
    assert result["tagged"] == 0 and result["enrolled"] == 0


def test_faces_from_another_embedding_space_are_skipped(lib):
    config, conn, *_ = lib
    other = _add(conn, np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32))  # 4-d, not 3-d
    conn.commit()
    result = housekeeping.recognize_faces(config)
    assert result["skipped_dim"] == 1 and _state(conn, other)[0] is None


def test_review_count_reflects_weak_guesses(lib):
    config, conn, *_ = lib
    _add(conn, _vec(E1, 0.35, 99))  # noisy: matches, but only just
    conn.commit()
    r = housekeeping.recognize_faces(config)
    assert r["tagged"] == 1 and r["need_review"] in (0, 1)
    assert "threshold" in r


def test_api_and_cli(lib, capsys):
    config, conn, anna, bob = lib
    _add(conn, _vec(E1, 0.03, 10))
    _add(conn, _vec(E2, 0.03, 11))
    conn.commit()
    client = TestClient(create_app(config))
    res = client.post("/api/faces/recognize")
    assert res.status_code == 200 and res.json()["tagged"] == 2

    _add(conn, _vec(E1, 0.03, 12))
    conn.commit()
    assert cli.main(["--home", str(config.home), "recognize", "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "Would tag 1" in out
    assert cli.main(["--home", str(config.home), "recognize"]) == 0
    assert "Tagged 1" in capsys.readouterr().out
