"""SigLIP encoder classes driven by fake ONNX sessions / tokenizers (offline)."""

from __future__ import annotations

import numpy as np
from PIL import Image

from photo_atlas import embed, models
from photo_atlas.config import AtlasConfig


class _Node:
    def __init__(self, name, shape=None):
        self.name, self.shape = name, shape


class _FakeSession:
    def __init__(self, path, providers=None, *, out="pooler_output", shape=(1, 3, 224, 224)):
        self.path, self.providers = path, providers
        self._out, self._shape = out, shape
        self.feeds: list[dict] = []

    def get_inputs(self):
        return [_Node("pixel_values", list(self._shape))]

    def get_outputs(self):
        return [_Node("last_hidden_state"), _Node(self._out)]

    def run(self, outputs, feed):
        self.feeds.append(feed)
        assert outputs == [self._out]
        return [np.array([[3.0, 4.0, 0.0]], dtype=np.float32)]


def _patch_ort(monkeypatch, **kwargs):
    import onnxruntime

    created: list[_FakeSession] = []

    def factory(path, providers=None):
        s = _FakeSession(path, providers, **kwargs)
        created.append(s)
        return s

    monkeypatch.setattr(onnxruntime, "InferenceSession", factory)
    return created


def test_image_encoder_embeds_image_and_path(monkeypatch, tmp_path):
    created = _patch_ort(monkeypatch)
    enc = embed.SigLipImageEncoder("model.onnx")
    assert created[0].providers == ["CPUExecutionProvider"]
    vec = enc.embed_image(Image.new("RGB", (40, 30), "red"))
    assert np.linalg.norm(vec) == np.float32(1.0) or abs(np.linalg.norm(vec) - 1) < 1e-6
    assert created[0].feeds[0]["pixel_values"].shape == (1, 3, 224, 224)

    p = tmp_path / "x.png"
    Image.new("RGB", (8, 8)).save(p)
    assert enc.embed_path(p).shape == (3,)


def test_image_encoder_dynamic_shape_uses_configured_size(monkeypatch):
    created = _patch_ort(monkeypatch, shape=(1, 3, "h", "w"))
    enc = embed.SigLipImageEncoder("m.onnx", default_size=256)
    enc.embed_image(Image.new("RGB", (10, 10)))
    assert created[0].feeds[0]["pixel_values"].shape == (1, 3, 256, 256)


def test_image_encoder_from_config_downloads_via_models(monkeypatch, tmp_path):
    _patch_ort(monkeypatch)
    seen = {}

    def fake_model(models_dir, download):
        seen["dir"], seen["download"] = models_dir, download
        return tmp_path / "vision.onnx"

    monkeypatch.setattr(models, "ensure_scene_model", fake_model)
    monkeypatch.setattr(models, "ensure_scene_input_size", lambda: 224)
    cfg = AtlasConfig(home=tmp_path)
    embed.SigLipImageEncoder.from_config(cfg)
    assert seen == {"dir": cfg.models_dir, "download": True}


class _Enc:
    def __init__(self, ids):
        self.ids = ids


class _FakeTok:
    def __init__(self):
        self.encoded: list[str] = []

    def encode(self, text):
        self.encoded.append(text)
        return _Enc([1, 2, 3])


def test_text_encoder_embeds_and_from_config(monkeypatch, tmp_path):
    import tokenizers

    tok = _FakeTok()
    monkeypatch.setattr(tokenizers.Tokenizer, "from_file", staticmethod(lambda p: tok))
    monkeypatch.setattr(embed, "configure_text_tokenizer", lambda t, n: None)
    created = _patch_ort(monkeypatch, out="text_embeds")

    enc = embed.SigLipTextEncoder("text.onnx", "tok.json")
    vec = enc.embed_text("a dog on a beach")
    assert tok.encoded == ["a dog on a beach"]
    assert abs(np.linalg.norm(vec) - 1) < 1e-6
    assert created[0].feeds[0]["pixel_values"].dtype == np.int64

    monkeypatch.setattr(models, "ensure_scene_text_model", lambda d, download: tmp_path / "t.onnx")
    monkeypatch.setattr(models, "ensure_scene_tokenizer", lambda d, download: tmp_path / "t.json")
    assert isinstance(
        embed.SigLipTextEncoder.from_config(AtlasConfig(home=tmp_path)), embed.SigLipTextEncoder
    )
