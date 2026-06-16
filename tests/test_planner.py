"""Hybrid query decomposition (planner) — pure-logic + API integration.

The planner peels known person names and "alone / with others" phrases out of a
natural-language query into structured filters, leaving the residual for SigLIP.
Pure planning is tested without any model; the API path uses a stub text encoder.
"""

from __future__ import annotations

import numpy as np

from photo_atlas import db, embed
from photo_atlas.config import AtlasConfig
from photo_atlas.planner import plan_query

PERSONS = [
    {"id": 1, "name": "Stefano"},
    {"id": 2, "name": "Anna"},
    {"id": 3, "name": "Anna Maria"},
]


def test_peels_one_name_leaving_visual_residual():
    plan = plan_query("Stefano eating food", PERSONS)
    assert plan.person_ids == [1]
    assert plan.person_names == ["Stefano"]
    assert plan.person_mode is None
    assert plan.people == []
    assert plan.text == "eating food"


def test_strips_preamble_and_connectives():
    plan = plan_query("a photo of Anna at the beach", PERSONS)
    assert plan.person_ids == [2]
    assert plan.text == "beach"


def test_two_names_use_and_mode_in_query_order():
    plan = plan_query("Stefano and Anna at a wedding", PERSONS)
    assert plan.person_ids == [1, 2]
    assert plan.person_mode == "all"
    assert plan.text == "wedding"


def test_longest_name_wins_over_prefix():
    plan = plan_query("Anna Maria smiling", PERSONS)
    # "Anna Maria" matches, not the shorter "Anna".
    assert plan.person_ids == [3]
    assert plan.person_names == ["Anna Maria"]
    assert plan.text == "smiling"


def test_group_phrase_maps_to_people_buckets():
    plan = plan_query("Stefano with other people", PERSONS)
    assert plan.person_ids == [1]
    assert plan.people == ["2-4", "5+"]
    assert plan.text == ""
    assert plan.is_structured_only is True


def test_alone_phrase_maps_to_portrait_bucket():
    plan = plan_query("Stefano alone", PERSONS)
    assert plan.people == ["1"]
    assert plan.text == ""
    assert plan.is_structured_only is True


def test_case_insensitive_name_match():
    plan = plan_query("STEFANO on a boat", PERSONS)
    assert plan.person_ids == [1]
    assert plan.text == "boat"


def test_pure_visual_query_has_no_structured_legs():
    plan = plan_query("red sports car at sunset", PERSONS)
    assert plan.person_ids == [] and plan.people == []
    assert plan.text == "red sports car at sunset"
    assert plan.is_structured_only is False


def test_substring_is_not_a_false_name_match():
    # "Annabelle" must not match the person "Anna" (word-boundary).
    plan = plan_query("Annabelle doll", PERSONS)
    assert plan.person_ids == []
    assert plan.text == "Annabelle doll"


# -- API integration --------------------------------------------------------
class _StubTextEncoder:
    def __init__(self, vec):
        self._vec = np.asarray(vec, dtype=np.float32)

    def embed_text(self, _text):
        return self._vec


def _unit(*vals):
    v = np.array(vals, dtype=np.float32)
    return v / np.linalg.norm(v)


def _build_library(tmp_path):
    """A tiny catalog: Stefano in two photos (one alone, one in a group)."""

    config = AtlasConfig(home=tmp_path / "lib").ensure_dirs()
    conn = db.connect(config.db_path)
    pid = db.get_or_create_person(conn, "Stefano")

    def photo(path, vec, faces):
        photo_id = db.upsert_photo(conn, {"path": path, "filename": path.rsplit("/")[-1],
                                          "face_count": faces})
        db.set_photo_embedding(conn, photo_id, vec)
        return photo_id

    # solo: 1 face (Stefano), embedding "beachy". group: 3 faces, embedding "beachy".
    solo = photo("/solo.jpg", _unit(1, 0, 0), 1)
    group = photo("/group.jpg", _unit(0.9, 0.1, 0), 3)
    other = photo("/other.jpg", _unit(0, 1, 0), 1)  # someone else, not Stefano
    conn.execute("INSERT INTO faces (photo_id, person_id) VALUES (?, ?)", (solo, pid))
    conn.execute("INSERT INTO faces (photo_id, person_id) VALUES (?, ?)", (group, pid))
    conn.execute("INSERT INTO faces (photo_id, person_id) VALUES (?, ?)", (group, None))
    conn.execute("INSERT INTO faces (photo_id, person_id) VALUES (?, ?)", (group, None))
    conn.execute("INSERT INTO faces (photo_id, person_id) VALUES (?, ?)", (other, None))
    conn.commit()
    conn.close()
    return config, {"solo": solo, "group": group, "other": other}


def _client(config):
    from fastapi.testclient import TestClient

    from photo_atlas.api import create_app

    return TestClient(create_app(config))


def test_api_hybrid_person_plus_visual(tmp_path, monkeypatch):
    config, ids = _build_library(tmp_path)
    monkeypatch.setattr(
        embed.SigLipTextEncoder, "from_config",
        classmethod(lambda cls, c: _StubTextEncoder(_unit(1, 0, 0))),
    )
    data = _client(config).get("/api/photos", params={"text": "Stefano at the beach"}).json()
    # Only Stefano's two photos qualify (the "other" person's photo is filtered out),
    # ranked by the visual residual; the plan is echoed back for the UI.
    assert {p["id"] for p in data["photos"]} == {ids["solo"], ids["group"]}
    assert data["plan"]["persons"] == ["Stefano"]
    assert data["plan"]["text"] == "beach"


def test_api_hybrid_structured_only_needs_no_encoder(tmp_path, monkeypatch):
    config, ids = _build_library(tmp_path)
    # No residual visual text -> the text encoder is never consulted. Make it blow
    # up to prove it isn't called for "Stefano alone".
    def _boom(cls, c):
        raise AssertionError("text encoder must not be built for a structured-only query")

    monkeypatch.setattr(embed.SigLipTextEncoder, "from_config", classmethod(_boom))
    data = _client(config).get("/api/photos", params={"text": "Stefano alone"}).json()
    # "alone" -> exactly one face; only the solo Stefano photo matches.
    assert [p["id"] for p in data["photos"]] == [ids["solo"]]
    assert data["plan"]["people"] == ["1"]
    assert data["plan"]["text"] == ""
