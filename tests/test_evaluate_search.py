"""evaluate_search end-to-end with fake image/text encoders (no models)."""

from __future__ import annotations

import json

import numpy as np
import pytest
from PIL import Image

from photo_atlas import db, evaluate
from photo_atlas.config import AtlasConfig

COLORS = {"red": (250, 10, 10), "green": (10, 250, 10), "blue": (10, 10, 250)}
AXES = {"red": 0, "green": 1, "blue": 2}


class GoodImage:
    def embed_image(self, img):
        r, g, b = (float(v) for v in np.asarray(img.convert("RGB"))[5, 5])
        v = np.array([r, g, b] + [0.0] * 5, dtype=np.float32)
        return v / np.linalg.norm(v)


class GoodText:
    def embed_text(self, text):
        v = np.zeros(8, dtype=np.float32)
        for word, axis in AXES.items():
            if word in text.lower():
                v[axis] = 1.0
        return v


class RandomEncoders:
    def __init__(self):
        self._rng = np.random.default_rng(7)

    def embed_image(self, img):
        return self._rng.normal(size=8).astype(np.float32)

    def embed_text(self, text):
        return self._rng.normal(size=8).astype(np.float32)


def _library(tmp_path, per_color=6, with_bursts=False):
    config = AtlasConfig(home=tmp_path / "lib").ensure_dirs()
    photos = tmp_path / "photos"
    photos.mkdir()
    conn = db.connect(config.db_path)
    names = {c: [] for c in COLORS}
    n = 0
    for color, rgb in COLORS.items():
        for i in range(per_color):
            f = photos / f"{color}_{i}.png"
            Image.new("RGB", (16, 16), rgb).save(f)
            pid = db.upsert_photo(conn, {
                "path": str(f), "filename": f.name,
                "taken_at": f"2024-01-{(n % 28) + 1:02d}T10:00:00",
            })
            if with_bursts and i < 3:  # first three of each colour form a burst
                conn.execute(
                    "UPDATE photos SET taken_at=?, phash=? WHERE id=?",
                    (f"2024-06-0{AXES[color] + 1}T10:00:0{i}", "0f0f0f0f0f0f0f0f", pid),
                )
            names[color].append(f.name)
            n += 1
    conn.commit()
    conn.close()
    return config, names


def _queries(tmp_path, names):
    f = tmp_path / "q.jsonl"
    f.write_text("\n".join(
        json.dumps({"query": f"a {c} thing", "relevant": names[c]}) for c in COLORS
    ), encoding="utf-8")
    return f


def _candidates():
    rnd = RandomEncoders()
    return [
        evaluate.SearchCandidate("good", GoodImage(), GoodText(), size_mb=400.0),
        evaluate.SearchCandidate("bad", rnd, rnd),
    ]


def test_text_queries_rank_the_right_model_first(tmp_path):
    config, names = _library(tmp_path)
    result = evaluate.evaluate_search(
        config, _candidates(), queries_path=_queries(tmp_path, names), bursts=False
    )
    assert result["pool"] == 18 and result["queries"] == 3
    good, bad = result["candidates"]
    assert good["text"]["hit@1"] == 1.0 and good["text"]["mrr"] == 1.0
    assert good["text"]["ndcg@10"] == pytest.approx(1.0)
    assert bad["text"]["ndcg@10"] < good["text"]["ndcg@10"]
    assert bad["text"]["mrr"] <= good["text"]["mrr"]
    assert good["size_mb"] == 400.0 and good["dim"] == 8 and good["images_per_sec"] > 0
    assert good["bursts"] is None


def test_bursts_give_a_label_free_image_to_image_check(tmp_path):
    config, _ = _library(tmp_path, with_bursts=True)
    result = evaluate.evaluate_search(config, _candidates(), queries_path=None, bursts=True)
    assert result["burst_cases"] == 3
    good, bad = result["candidates"]
    assert good["text"] is None
    assert good["bursts"]["recall@5"] == 1.0
    assert bad["bursts"]["recall@5"] < good["bursts"]["recall@5"]


def test_filenames_and_full_paths_both_resolve_and_unknown_ones_are_reported(tmp_path):
    config, names = _library(tmp_path)
    full = str(tmp_path / "photos" / names["red"][0])
    qf = tmp_path / "q.jsonl"
    qf.write_text(json.dumps({"query": "red", "relevant": [full, names["red"][1], "nope.jpg"]}))
    result = evaluate.evaluate_search(config, _candidates()[:1], queries_path=qf, bursts=False)
    assert result["queries"] == 1
    assert result["unresolved"] == ["nope.jpg"]


def test_ambiguous_filenames_are_not_guessed(tmp_path):
    config, names = _library(tmp_path)
    other = tmp_path / "other"
    other.mkdir()
    dup = other / names["red"][0]
    Image.new("RGB", (16, 16), COLORS["red"]).save(dup)
    conn = db.connect(config.db_path)
    db.upsert_photo(conn, {"path": str(dup), "filename": dup.name})
    conn.commit()
    conn.close()
    qf = tmp_path / "q.jsonl"
    qf.write_text(json.dumps({"query": "red", "relevant": [names["red"][0], names["red"][1]]}))
    result = evaluate.evaluate_search(config, _candidates()[:1], queries_path=qf, bursts=False)
    assert result["unresolved"] == [names["red"][0] + " (ambiguous)"]


def test_pool_is_capped_but_always_contains_the_relevant_photos(tmp_path):
    config, names = _library(tmp_path, per_color=10)
    qf = tmp_path / "q.jsonl"
    qf.write_text(json.dumps({"query": "green", "relevant": names["green"][:2]}))
    result = evaluate.evaluate_search(
        config, _candidates()[:1], queries_path=qf, bursts=False, max_photos=5
    )
    assert result["pool"] == 5
    assert result["candidates"][0]["text"]["hit@1"] == 1.0


def test_nothing_to_evaluate_is_an_actionable_error(tmp_path):
    config, _ = _library(tmp_path)
    with pytest.raises(evaluate.EvaluationError, match="eval-init-queries|queries"):
        evaluate.evaluate_search(config, _candidates()[:1], queries_path=None, bursts=True)


def test_report_lists_models_and_metrics(tmp_path):
    config, names = _library(tmp_path)
    result = evaluate.evaluate_search(
        config, _candidates(), queries_path=_queries(tmp_path, names), bursts=False
    )
    text = evaluate.format_search_report(result)
    assert "good" in text and "bad" in text and "MRR" in text and "img/s" in text
