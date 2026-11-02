"""Eval presets (downloadable candidates) and candidate-spec parsing."""

from __future__ import annotations

import re

import pytest

from photo_atlas import evaluate, models


def test_every_preset_is_pinned_and_checksummed():
    files = []
    for p in models.FACE_PRESETS.values():
        files.append(p.model)
    for p in models.SEARCH_PRESETS.values():
        files += [p.vision, p.text]
    assert files
    names = [f.filename for f in files]
    assert len(names) == len(set(names)), "local file names must be unique"
    for f in files:
        assert re.search(r"/resolve/[0-9a-f]{40}/", f.url), f"{f.filename}: pin a commit"
        assert re.fullmatch(r"[0-9a-f]{64}", f.sha256), f.filename
        assert models.MODEL_SHA256[f.filename] == f.sha256  # _resolve will verify it


def test_search_presets_reuse_the_pinned_default_tokenizer():
    # The SigLIP 2 repos ship an identical tokenizer; presets rely on that.
    assert models.MODEL_SHA256[models.SCENE_TOKENIZER_NAME]
    for p in models.SEARCH_PRESETS.values():
        assert p.input_size in (256, 384, 512)


def test_fetch_preset_downloads_through_resolve(tmp_path, monkeypatch):
    seen = []

    def fake_resolve(name, url, model_dir, env, download):
        seen.append((name, url))
        return tmp_path / name

    monkeypatch.setattr(models, "_resolve", fake_resolve)
    p = models.FACE_PRESETS["auraface"]
    assert models.fetch_file(p.model, tmp_path) == tmp_path / p.model.filename
    assert seen == [(p.model.filename, p.model.url)]


# -- candidate specs ---------------------------------------------------------
def test_parse_face_candidate():
    assert evaluate.parse_face_candidate("r50=/m/r50.onnx") == ("r50", "/m/r50.onnx")
    assert evaluate.parse_face_candidate("/m/glintr100.onnx") == ("glintr100", "/m/glintr100.onnx")
    with pytest.raises(ValueError, match="NAME=PATH"):
        evaluate.parse_face_candidate("")


def test_parse_search_candidate():
    c = evaluate.parse_search_candidate("large=/m/v.onnx,/m/t.onnx,/m/tok.json,384")
    assert c == ("large", "/m/v.onnx", "/m/t.onnx", "/m/tok.json", 384)
    c = evaluate.parse_search_candidate("x=/v.onnx,/t.onnx")
    assert c == ("x", "/v.onnx", "/t.onnx", None, None)
    c = evaluate.parse_search_candidate("x=/v.onnx,/t.onnx,384")  # size without tokenizer
    assert c == ("x", "/v.onnx", "/t.onnx", None, 384)
    with pytest.raises(ValueError, match="VISION,TEXT"):
        evaluate.parse_search_candidate("x=/only-one.onnx")


# -- candidate builders (fakes for the heavy classes) -----------------------
class _FakeEmbedder:
    def __init__(self, path):
        self.path = path


def test_build_face_candidates_default_specs_and_presets(tmp_path, monkeypatch):
    from photo_atlas.config import AtlasConfig

    cfg = AtlasConfig(home=tmp_path)
    default = tmp_path / "arc_default.onnx"
    default.write_bytes(b"x" * 2_000_000)
    custom = tmp_path / "r50.onnx"
    custom.write_bytes(b"y" * 1_000_000)
    pre = tmp_path / "auraface_v1_glintr100.onnx"
    pre.write_bytes(b"z" * 3_000_000)
    monkeypatch.setattr(models, "ensure_arcface", lambda d, download=True: default)
    fetched = []
    monkeypatch.setattr(models, "fetch_file", lambda f, d: fetched.append(f.filename) or pre)

    out = evaluate.build_face_candidates(
        cfg, specs=[f"r50={custom}"], presets=["auraface"], include_default=True,
        embedder_factory=_FakeEmbedder,
    )
    assert [c.name for c in out] == ["default", "r50", "auraface"]
    assert [c.size_mb for c in out] == [2.0, 1.0, 3.0]
    assert out[1].embedder.path == custom and fetched == ["auraface_v1_glintr100.onnx"]

    no_default = evaluate.build_face_candidates(
        cfg, specs=[f"r50={custom}"], presets=[], include_default=False,
        embedder_factory=_FakeEmbedder,
    )
    assert [c.name for c in no_default] == ["r50"]


def test_build_face_candidates_rejects_unknown_preset_and_missing_files(tmp_path):
    from photo_atlas.config import AtlasConfig

    cfg = AtlasConfig(home=tmp_path)
    with pytest.raises(ValueError, match="unknown face preset 'nope'.*auraface"):
        evaluate.build_face_candidates(cfg, [], ["nope"], include_default=False,
                                       embedder_factory=_FakeEmbedder)
    with pytest.raises(FileNotFoundError, match="missing.onnx"):
        evaluate.build_face_candidates(cfg, [f"m={tmp_path / 'missing.onnx'}"], [],
                                       include_default=False, embedder_factory=_FakeEmbedder)


def test_build_search_candidates_default_spec_and_preset(tmp_path, monkeypatch):
    from photo_atlas.config import AtlasConfig

    cfg = AtlasConfig(home=tmp_path)
    files = {}
    for n, mb in (("dv", 1), ("dt", 2), ("tok", 0), ("cv", 4), ("ct", 8), ("pv", 16), ("pt", 32)):
        f = tmp_path / f"{n}.bin"
        f.write_bytes(b"0" * (mb * 1_000_000))
        files[n] = f
    monkeypatch.setattr(models, "ensure_scene_model", lambda d, download=True: files["dv"])
    monkeypatch.setattr(models, "ensure_scene_text_model", lambda d, download=True: files["dt"])
    monkeypatch.setattr(models, "ensure_scene_tokenizer", lambda d, download=True: files["tok"])
    monkeypatch.setattr(
        models, "fetch_file",
        lambda f, d: files["pv"] if "vision" in f.filename else files["pt"],
    )
    made = []

    class Img:
        def __init__(self, path, image_size=None, default_size=256):
            made.append(("img", path.name, image_size, default_size))

    class Txt:
        def __init__(self, path, tokenizer):
            made.append(("txt", path.name, tokenizer.name))

    out = evaluate.build_search_candidates(
        cfg, specs=[f"custom={files['cv']},{files['ct']},384"], presets=["siglip2-large-384"],
        include_default=True, image_cls=Img, text_cls=Txt,
    )
    assert [c.name for c in out] == ["default", "custom", "siglip2-large-384"]
    assert [c.size_mb for c in out] == [3.0, 12.0, 48.0]  # vision + text weights
    assert ("img", "cv.bin", 384, 256) in made and ("txt", "ct.bin", "tok.bin") in made
    assert ("img", "pv.bin", 384, 256) in made  # preset resolution comes from the preset
    with pytest.raises(ValueError, match="unknown search preset"):
        evaluate.build_search_candidates(cfg, [], ["x"], include_default=False,
                                         image_cls=Img, text_cls=Txt)
