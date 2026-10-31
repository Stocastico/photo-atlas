"""Execution-provider selection for onnxruntime sessions."""

from __future__ import annotations

import onnxruntime
import pytest

from photo_atlas import embed, faces, models


@pytest.fixture
def available(monkeypatch):
    def _set(*names):
        monkeypatch.setattr(onnxruntime, "get_available_providers", lambda: list(names))

    monkeypatch.delenv("PHOTO_ATLAS_PROVIDERS", raising=False)
    return _set


def test_cpu_only_machine_uses_cpu(available):
    available("CPUExecutionProvider")
    assert models.onnx_providers() == ["CPUExecutionProvider"]


def test_gpu_providers_preferred_then_cpu_fallback(available):
    available("CUDAExecutionProvider", "DmlExecutionProvider", "CPUExecutionProvider")
    assert models.onnx_providers() == [
        "CUDAExecutionProvider", "DmlExecutionProvider", "CPUExecutionProvider",
    ]


def test_coreml_is_opt_in_only(available, monkeypatch):
    available("CoreMLExecutionProvider", "CPUExecutionProvider")
    assert models.onnx_providers() == ["CPUExecutionProvider"]  # deferred by default
    monkeypatch.setenv("PHOTO_ATLAS_PROVIDERS", "coreml")
    assert models.onnx_providers() == ["CoreMLExecutionProvider", "CPUExecutionProvider"]


def test_env_override_aliases_and_unavailable_are_dropped(available, monkeypatch):
    available("CUDAExecutionProvider", "CPUExecutionProvider")
    monkeypatch.setenv("PHOTO_ATLAS_PROVIDERS", "cpu")
    assert models.onnx_providers() == ["CPUExecutionProvider"]
    monkeypatch.setenv("PHOTO_ATLAS_PROVIDERS", "dml, cuda")  # dml isn't installed
    assert models.onnx_providers() == ["CUDAExecutionProvider", "CPUExecutionProvider"]
    monkeypatch.setenv("PHOTO_ATLAS_PROVIDERS", "nonsense")
    assert models.onnx_providers() == ["CPUExecutionProvider"]


def test_sessions_are_created_with_the_selected_providers(available, monkeypatch, tmp_path):
    available("CUDAExecutionProvider", "CPUExecutionProvider")
    seen = []

    class FakeSession:
        def __init__(self, path, providers=None):
            seen.append(providers)

        def get_inputs(self):
            return [type("N", (), {"name": "x", "shape": [1, 3, 224, 224]})()]

        def get_outputs(self):
            return [type("N", (), {"name": "pooler_output"})()]

    monkeypatch.setattr(onnxruntime, "InferenceSession", FakeSession)
    embed.SigLipImageEncoder("m.onnx")
    expected = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    assert seen == [expected]
    assert "providers=onnx_providers()" in open(faces.__file__).read()
