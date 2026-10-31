"""Embeddings remember which model produced them, and refuse to be mixed."""

from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from photo_atlas import cli, db, embed, faces, indexer, models
from photo_atlas.api import create_app
from photo_atlas.config import AtlasConfig


# -- db.claim_model ---------------------------------------------------------
def test_claim_model_sets_then_accepts_same_then_rejects_different(tmp_path):
    conn = db.connect(tmp_path / "s.db")
    try:
        db.claim_model(conn, "embedding_model", "a.onnx")
        assert db.get_meta(conn, "embedding_model") == "a.onnx"
        db.claim_model(conn, "embedding_model", "a.onnx")  # same: fine
        with pytest.raises(db.ModelMismatchError, match="a.onnx.*b.onnx"):
            db.claim_model(conn, "embedding_model", "b.onnx")
        assert db.get_meta(conn, "embedding_model") == "a.onnx"  # unchanged
        db.claim_model(conn, "embedding_model", "b.onnx", replace=True)
        assert db.get_meta(conn, "embedding_model") == "b.onnx"
    finally:
        conn.close()


def test_scene_model_id_follows_the_env_override(monkeypatch):
    monkeypatch.delenv("PHOTO_ATLAS_SCENE_MODEL", raising=False)
    assert models.scene_model_id() == models.SCENE_NAME
    monkeypatch.setenv("PHOTO_ATLAS_SCENE_MODEL", "/somewhere/other_vision.onnx")
    assert models.scene_model_id() == "other_vision.onnx"  # no file needed to compute it


# -- photo embeddings -------------------------------------------------------
class _Enc:
    def embed_image(self, _img):
        return np.array([1.0, 0.0], dtype=np.float32)


def _library_with_photos(tmp_path):
    config = AtlasConfig(home=tmp_path / "lib").ensure_dirs()
    conn = db.connect(config.db_path)
    for i in range(2):
        p = tmp_path / f"i{i}.jpg"
        Image.new("RGB", (8, 8), (i * 40, 0, 0)).save(p)
        db.upsert_photo(conn, {"path": str(p), "filename": p.name})
    conn.commit()
    conn.close()
    return config


def test_embed_library_stamps_and_blocks_a_model_swap(tmp_path, monkeypatch):
    monkeypatch.delenv("PHOTO_ATLAS_SCENE_MODEL", raising=False)
    config = _library_with_photos(tmp_path)
    assert indexer.embed_library(config, image_encoder=_Enc()) == 2
    conn = db.connect(config.db_path)
    assert db.get_meta(conn, "embedding_model") == models.SCENE_NAME
    conn.close()

    monkeypatch.setenv("PHOTO_ATLAS_SCENE_MODEL", "/m/new_vision.onnx")
    with pytest.raises(db.ModelMismatchError, match="--recompute"):
        indexer.embed_library(config, image_encoder=_Enc())
    # A library-wide recompute rebuilds every vector, so it may re-stamp.
    assert indexer.embed_library(config, image_encoder=_Enc(), recompute=True) == 2
    conn = db.connect(config.db_path)
    assert db.get_meta(conn, "embedding_model") == "new_vision.onnx"
    conn.close()


def test_accept_model_change_allows_a_partial_run(tmp_path, monkeypatch):
    monkeypatch.delenv("PHOTO_ATLAS_SCENE_MODEL", raising=False)
    config = _library_with_photos(tmp_path)
    indexer.embed_library(config, image_encoder=_Enc())
    monkeypatch.setenv("PHOTO_ATLAS_SCENE_MODEL", "/m/new_vision.onnx")
    indexer.embed_library(config, image_encoder=_Enc(), accept_model_change=True)
    conn = db.connect(config.db_path)
    assert db.get_meta(conn, "embedding_model") == "new_vision.onnx"
    conn.close()


# -- face embeddings --------------------------------------------------------
class _OtherBackend(faces.SyntheticFaceBackend):
    model_id = "arcface_r50.onnx"


def test_synthetic_backend_has_a_model_id():
    assert faces.SyntheticFaceBackend().model_id == "synthetic"


def test_face_backend_swap_is_refused_unless_accepted(tmp_path, monkeypatch):
    config = AtlasConfig(home=tmp_path / "lib").ensure_dirs()
    photos = tmp_path / "p"
    photos.mkdir()
    Image.new("RGB", (32, 32), (9, 9, 9)).save(photos / "a.jpg")
    indexer.index_path(config, photos, backend_name="synthetic", geocode=False, workers=1)
    conn = db.connect(config.db_path)
    assert db.get_meta(conn, "face_model") == "synthetic"
    conn.close()

    monkeypatch.setattr(indexer, "get_backend", lambda name, **kw: _OtherBackend())
    with pytest.raises(db.ModelMismatchError, match="--accept-model-change"):
        indexer.index_path(config, photos, backend_name="x", geocode=False, workers=1)
    indexer.index_path(
        config, photos, backend_name="x", geocode=False, workers=1, accept_model_change=True
    )
    conn = db.connect(config.db_path)
    assert db.get_meta(conn, "face_model") == "arcface_r50.onnx"
    conn.close()


def test_cli_reports_a_model_mismatch_cleanly(tmp_path, monkeypatch, capsys):
    home = tmp_path / "lib"
    photos = tmp_path / "p"
    photos.mkdir()
    Image.new("RGB", (32, 32), (9, 9, 9)).save(photos / "a.jpg")
    args = ["--home", str(home), "index", str(photos), "--no-geocode", "--workers", "1"]
    assert cli.main([*args, "--faces", "synthetic"]) == 0
    monkeypatch.setattr(indexer, "get_backend", lambda name, **kw: _OtherBackend())
    capsys.readouterr()
    assert cli.main([*args, "--faces", "arcface"]) == 2
    assert "--accept-model-change" in capsys.readouterr().err
    assert cli.main([*args, "--faces", "arcface", "--accept-model-change"]) == 0


# -- API: stale embeddings -------------------------------------------------
def _api_with_stamp(tmp_path, stamp):
    config = AtlasConfig(home=tmp_path / "lib").ensure_dirs()
    conn = db.connect(config.db_path)
    pid = db.upsert_photo(conn, {"path": "/a.jpg", "filename": "a.jpg"})
    db.set_photo_embedding(conn, pid, np.array([1.0, 0.0, 0.0], dtype=np.float32))
    if stamp:
        db.claim_model(conn, "embedding_model", stamp)
    conn.commit()
    conn.close()
    return TestClient(create_app(config))


def test_api_flags_embeddings_from_another_model(tmp_path, monkeypatch):
    monkeypatch.delenv("PHOTO_ATLAS_SCENE_MODEL", raising=False)
    client = _api_with_stamp(tmp_path, "old_model.onnx")
    caps = client.get("/api/capabilities").json()
    assert caps["semantic"] is False and caps["stale_embeddings"] is True
    res = client.get("/api/photos", params={"text": "a beach"})
    assert res.status_code == 409 and "embed --recompute" in res.json()["detail"]
    # "More like this" only compares stored vectors (all from the same old model).
    assert client.get("/api/photos/1/similar").status_code == 200


def test_api_is_happy_when_the_stamp_matches_or_is_absent(tmp_path, monkeypatch):
    monkeypatch.delenv("PHOTO_ATLAS_SCENE_MODEL", raising=False)
    monkeypatch.setattr(
        embed.SigLipTextEncoder, "from_config",
        classmethod(lambda cls, c: type("T", (), {"embed_text": lambda s, t: np.array(
            [1.0, 0.0, 0.0], dtype=np.float32)})()),
    )
    for stamp in (models.SCENE_NAME, None):  # None = catalog from before stamping
        client = _api_with_stamp(tmp_path / str(stamp), stamp)
        assert client.get("/api/photos", params={"text": "a beach"}).status_code == 200
        assert client.get("/api/capabilities").json().get("stale_embeddings") is False
