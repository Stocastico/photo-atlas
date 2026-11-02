"""evaluate_faces end-to-end with a fake detector and fake recognisers (no models)."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from photo_atlas import db, evaluate
from photo_atlas.config import AtlasConfig

BOX = (10, 10, 40, 40)
PERSON_COLORS = [(250, 10, 10), (10, 250, 10), (10, 10, 250), (250, 250, 10)]


class FakeDetector:
    """Always finds one face where the library says it is."""

    def detect_rows(self, image):
        x, y, w, h = BOX
        return [np.array([x, y, w, h, *([0] * 10), 0.99], dtype=np.float32)]


class IdentityEmbedder:
    """Sees the identity (via the face's colour) plus a little noise."""

    model_id = "good.onnx"

    def __init__(self):
        self.calls = 0
        self._rng = np.random.default_rng(0)

    def embed(self, image, row):
        self.calls += 1
        x, y, w, h = (int(v) for v in row[:4])
        colour = image[y + h // 2, x + w // 2].astype(np.float32)  # BGR
        base = np.concatenate([colour, np.zeros(13, dtype=np.float32)])
        base = base / np.linalg.norm(base)
        return base + 0.05 * self._rng.normal(size=16).astype(np.float32)


class RandomEmbedder:
    model_id = "bad.onnx"

    def __init__(self):
        self._rng = np.random.default_rng(1)

    def embed(self, image, row):
        return self._rng.normal(size=16).astype(np.float32)


def _library(tmp_path, per_person=8, confidence=1.0, people=4):
    config = AtlasConfig(home=tmp_path / "lib").ensure_dirs()
    conn = db.connect(config.db_path)
    for p in range(people):
        pid = db.get_or_create_person(conn, f"P{p}")
        for i in range(per_person):
            path = tmp_path / f"p{p}_{i}.png"
            Image.new("RGB", (100, 100), PERSON_COLORS[p]).save(path)
            photo = db.upsert_photo(conn, {"path": str(path), "filename": path.name})
            db.replace_faces(conn, photo, [{
                "person_id": pid, "bbox_x": BOX[0], "bbox_y": BOX[1], "bbox_w": BOX[2],
                "bbox_h": BOX[3], "confidence": confidence,
            }])
    conn.commit()
    conn.close()
    return config


def test_good_recogniser_beats_a_random_one(tmp_path):
    config = _library(tmp_path)
    result = evaluate.evaluate_faces(
        config,
        [evaluate.FaceCandidate("good", IdentityEmbedder()),
         evaluate.FaceCandidate("bad", RandomEmbedder(), size_mb=12.5)],
        detector=FakeDetector(),
    )
    assert result["faces"] == 32 and result["people"] == 4 and result["unmatched"] == 0
    good, bad = result["candidates"]
    assert (good["name"], bad["name"]) == ("good", "bad")
    assert good["verification"]["auc"] > 0.99 and good["identification"]["top1_accuracy"] == 1.0
    assert 0.35 < bad["verification"]["auc"] < 0.65
    assert bad["identification"]["top1_accuracy"] < 0.6
    assert bad["size_mb"] == 12.5 and good["dim"] == 16 and good["ms_per_face"] >= 0


def test_only_human_confirmed_faces_are_used(tmp_path):
    config = _library(tmp_path, confidence=0.7)  # auto-tags only
    with pytest.raises(evaluate.EvaluationError, match="name at least two people"):
        evaluate.evaluate_faces(
            config, [evaluate.FaceCandidate("g", IdentityEmbedder())], detector=FakeDetector()
        )


def test_caps_keep_the_run_bounded_and_deterministic(tmp_path):
    config = _library(tmp_path, per_person=10)
    emb1, emb2 = IdentityEmbedder(), IdentityEmbedder()
    r1 = evaluate.evaluate_faces(
        config, [evaluate.FaceCandidate("g", emb1)], detector=FakeDetector(), max_per_person=3
    )
    r2 = evaluate.evaluate_faces(
        config, [evaluate.FaceCandidate("g", emb2)], detector=FakeDetector(), max_per_person=3
    )
    assert r1["faces"] == 12 and emb1.calls == 12  # 4 people x 3
    assert r1["candidates"][0]["verification"] == r2["candidates"][0]["verification"]
    total_cap = evaluate.evaluate_faces(
        config, [evaluate.FaceCandidate("g", IdentityEmbedder())],
        detector=FakeDetector(), max_faces=8,
    )
    assert total_cap["faces"] <= 8 and total_cap["people"] == 4


def test_faces_the_detector_misses_are_counted_not_fatal(tmp_path):
    config = _library(tmp_path)

    class Blind(FakeDetector):
        def __init__(self):
            self.n = 0

        def detect_rows(self, image):
            self.n += 1
            return [] if self.n % 4 == 0 else super().detect_rows(image)

    result = evaluate.evaluate_faces(
        config, [evaluate.FaceCandidate("g", IdentityEmbedder())], detector=Blind()
    )
    assert result["unmatched"] == 8 and result["faces"] == 24


def test_missing_photo_files_are_skipped(tmp_path):
    config = _library(tmp_path)
    for f in list(tmp_path.glob("p0_*.png")):
        f.unlink()  # person 0's originals are gone
    result = evaluate.evaluate_faces(
        config, [evaluate.FaceCandidate("g", IdentityEmbedder())], detector=FakeDetector()
    )
    assert result["people"] == 3 and result["unmatched"] == 8


def test_format_faces_report_is_readable(tmp_path):
    config = _library(tmp_path)
    result = evaluate.evaluate_faces(
        config, [evaluate.FaceCandidate("good", IdentityEmbedder())], detector=FakeDetector()
    )
    text = evaluate.format_faces_report(result)
    assert "good" in text and "AUC" in text and "TAR@FAR" in text and "32 faces" in text


def test_a_broken_candidate_is_reported_without_sinking_the_others(tmp_path):
    config = _library(tmp_path)

    class Broken:
        def embed(self, image, row):
            raise RuntimeError("bad input shape")

    result = evaluate.evaluate_faces(
        config,
        [
            evaluate.FaceCandidate("broken", Broken()),
            evaluate.FaceCandidate("ok", IdentityEmbedder()),
        ],
        detector=FakeDetector(),
    )
    broken, ok = result["candidates"]
    assert "bad input shape" in broken["error"] and "verification" not in broken
    assert ok["verification"]["auc"] > 0.99
    assert "FAILED" in evaluate.format_faces_report(result)
