"""Regression guards for the 'SigLIP-only scene tagger' refactor.

These lock in the contract after removing the heuristic tagger: the heuristic and
its config knob are gone, ``get_tagger`` is always the zero-shot SigLIP one, the
``--scene`` CLI flags no longer exist, the embed path reuses the tagger's single
vision encoder, and an end-to-end index still writes scene tags (offline, via the
injected stub tagger). They complement the existing scene/indexer suites.
"""

from __future__ import annotations

import numpy as np
import pytest
from scene_stub import StubTagger

from photo_atlas import classify, db, demo, indexer
from photo_atlas.classify import ZeroShotSceneTagger, get_tagger
from photo_atlas.config import AtlasConfig


# -- removal is real --------------------------------------------------------
def test_heuristic_tagger_class_is_gone():
    assert not hasattr(classify, "SceneTagger")
    # The internal softmax helper went with it.
    assert not hasattr(classify, "_softmax")


def test_config_has_no_scene_backend_knob():
    assert not hasattr(AtlasConfig(), "scene_backend")


def test_get_tagger_is_always_zeroshot():
    # Inject a stub encoder so this stays offline; the point is the *type*.
    class _Enc:
        def embed_image(self, _img):
            return np.zeros(4, dtype=np.float32)

    tagger = get_tagger(AtlasConfig(), encoder=_Enc())
    assert isinstance(tagger, ZeroShotSceneTagger)


# -- CLI no longer accepts --scene -----------------------------------------
def test_cli_index_rejects_removed_scene_flag(tmp_path):
    from photo_atlas import cli

    photos = tmp_path / "p"
    photos.mkdir()
    # argparse exits non-zero on an unknown option.
    with pytest.raises(SystemExit):
        cli.main(["--home", str(tmp_path / "lib"), "index", str(photos), "--scene", "zeroshot"])


def test_cli_retag_rejects_removed_scene_flag(tmp_path):
    from photo_atlas import cli

    with pytest.raises(SystemExit):
        cli.main(["--home", str(tmp_path / "lib"), "retag-scenes", "--scene", "heuristic"])


# -- embed reuses the tagger's single vision encoder -----------------------
def test_build_encoders_reuses_zeroshot_encoder_for_embeddings(monkeypatch):
    """When embeddings are requested, the image encoder is the very same object the
    zero-shot tagger holds — one ONNX session / one inference per photo, not two."""

    matrix = np.load(classify.scene_labels_path(), allow_pickle=True)["matrix"]

    class _Enc:
        def embed_image(self, _img):
            return np.asarray(matrix[0], dtype=np.float32)

    tagger = ZeroShotSceneTagger(encoder=_Enc())
    monkeypatch.setattr(indexer, "get_tagger", lambda config, **kw: tagger)

    image_encoder, built = indexer._build_encoders(AtlasConfig(), embed=True)
    assert built is tagger
    assert image_encoder is tagger.encoder  # shared, not a second encoder

    # With embed off, no encoder is loaded at all.
    enc_off, _ = indexer._build_encoders(AtlasConfig(), embed=False)
    assert enc_off is None


# -- end-to-end: indexing still tags scenes (offline, injected tagger) ------
def test_index_writes_scene_tags_with_injected_tagger(tmp_path):
    photos = tmp_path / "p"
    demo.generate(photos, count=4, seed=2)
    cfg = AtlasConfig(home=tmp_path / "lib").ensure_dirs()

    stats = indexer.index_path(
        cfg, photos, backend_name="none", geocode=False, tagger=StubTagger()
    )
    assert stats.indexed == 4

    conn = db.connect(cfg.db_path)
    try:
        scenes = [r[0] for r in conn.execute("SELECT scene_type FROM photos")]
    finally:
        conn.close()
    # Every indexed photo got a scene tag; with no face backend the stub tags 'other'.
    assert len(scenes) == 4
    assert all(s == "other" for s in scenes)


def test_indexed_fixture_has_scene_tags_from_stub(indexed):
    # The shared `indexed` fixture runs the full pipeline (via the autouse stub
    # tagger). Regression: scene_type is always populated and within the stub's
    # vocabulary (face photos -> 'people', the rest -> 'other').
    conn = db.connect(indexed.db_path)
    try:
        scenes = {r[0] for r in conn.execute("SELECT scene_type FROM photos")}
    finally:
        conn.close()
    assert scenes and scenes <= {"people", "other"}
