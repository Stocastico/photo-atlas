"""Pure metric functions of the model-evaluation harness."""

from __future__ import annotations

import json

import numpy as np
import pytest

from photo_atlas import evaluate


# -- verification -----------------------------------------------------------
def test_verification_perfect_separation():
    m = evaluate.verification_metrics(np.array([0.1, 0.2, 0.3]), np.array([0.8, 0.9, 1.0]))
    assert m["auc"] == 1.0 and m["eer"] == 0.0
    assert m["tar_at_far_1pct"] == 1.0 and m["tar_at_far_0p1pct"] == 1.0
    assert m["d_prime"] > 5


def test_verification_complete_overlap_is_chance():
    rng = np.random.default_rng(0)
    both = rng.uniform(0, 1, 4000)
    m = evaluate.verification_metrics(both[:2000], both[2000:])
    assert 0.46 < m["auc"] < 0.54 and 0.4 < m["eer"] < 0.6
    assert abs(m["d_prime"]) < 0.15


def test_verification_auc_counts_ties_as_half():
    m = evaluate.verification_metrics(np.array([0.5]), np.array([0.5]))
    assert m["auc"] == 0.5


def test_verification_tar_at_far_uses_the_impostor_tail():
    imp = np.linspace(0.5, 1.5, 1001)       # 1st percentile of impostors = 0.51
    gen = np.array([0.4, 0.45, 0.5, 0.55])  # only the first three are below it
    m = evaluate.verification_metrics(gen, imp)
    assert m["tar_at_far_1pct"] == pytest.approx(0.75)
    assert m["thr_at_far_1pct"] == pytest.approx(0.51, abs=1e-6)


# -- identification ---------------------------------------------------------
def _clusters(n_people=4, per=6, noise=0.05, dim=16, seed=1):
    rng = np.random.default_rng(seed)
    centers = rng.normal(size=(n_people, dim))
    emb = np.vstack([c + noise * rng.normal(size=(per, dim)) for c in centers])
    labels = np.repeat(np.arange(n_people), per)
    return emb.astype(np.float32), labels


def test_identification_perfect_on_separated_identities():
    emb, labels = _clusters()
    m = evaluate.identification_metrics(emb, labels, threshold=0.5, k=3)
    assert m["probes"] == 24
    assert m["top1_accuracy"] == 1.0
    assert m["auto_correct"] == 1.0 and m["auto_wrong"] == 0.0 and m["auto_unknown"] == 0.0
    assert m["auto_precision"] == 1.0


def test_identification_tight_threshold_leaves_probes_unknown():
    emb, labels = _clusters()
    m = evaluate.identification_metrics(emb, labels, threshold=0.0001, k=3)
    assert m["auto_unknown"] == 1.0 and m["auto_correct"] == 0.0
    assert m["auto_precision"] is None  # nothing was tagged


def test_identification_ignores_singletons_and_counts_wrong_tags():
    emb, labels = _clusters(n_people=2, per=4)
    emb = np.vstack([emb, emb[0] + 0.001])          # a lone face of a 3rd identity...
    labels = np.append(labels, 9)                    # ...that looks exactly like person 0
    m = evaluate.identification_metrics(emb, labels, threshold=0.5, k=1)
    assert m["probes"] == 8  # the singleton is not a probe
    assert m["auto_correct"] < 1.0  # person 0's nearest neighbour is now the impostor twin
    assert m["auto_wrong"] > 0.0


# -- retrieval --------------------------------------------------------------
def test_rank_metrics():
    rankings = [["a", "b", "c", "d"], ["x", "y", "z", "w"]]
    relevant = [{"b"}, {"w", "q"}]
    m = evaluate.rank_metrics(rankings, relevant, ks=(1, 2, 4))
    assert m["queries"] == 2
    assert m["hit@1"] == 0.0 and m["hit@2"] == 0.5 and m["hit@4"] == 1.0
    assert m["recall@4"] == pytest.approx((1.0 + 0.5) / 2)
    assert m["mrr"] == pytest.approx((1 / 2 + 1 / 4) / 2)
    # nDCG@10 (binary gains): q1 -> 1/log2(3); q2 -> (1/log2(5)) / (1 + 1/log2(3)).
    ndcg1 = 1 / np.log2(3)
    ndcg2 = (1 / np.log2(5)) / (1 + 1 / np.log2(3))
    assert m["ndcg@10"] == pytest.approx((ndcg1 + ndcg2) / 2)


def test_rank_metrics_perfect_and_empty():
    perfect = evaluate.rank_metrics([["a", "b"]], [{"a", "b"}], ks=(1,))
    assert perfect["mrr"] == 1.0 and perfect["ndcg@10"] == pytest.approx(1.0)
    assert evaluate.rank_metrics([], [])["queries"] == 0


def test_rank_pool_orders_by_cosine_and_can_exclude_self():
    pool = np.array([[1, 0], [0.9, 0.1], [0, 1]], dtype=np.float32)
    top = evaluate.rank_pool(np.array([[1.0, 0.0]], dtype=np.float32), pool, top=3)
    assert top[0].tolist() == [0, 1, 2]
    excl = evaluate.rank_pool(pool[:1], pool, top=2, exclude=[0])
    assert excl[0].tolist() == [1, 2]


# -- queries file -----------------------------------------------------------
def test_load_queries_skips_blank_and_comment_lines(tmp_path):
    f = tmp_path / "q.jsonl"
    f.write_text(
        "# my queries\n\n"
        + json.dumps({"query": "tramonto al mare", "relevant": ["a.jpg", "b.jpg"]}) + "\n"
        + json.dumps({"query": "x", "relevant": []}) + "\n",
        encoding="utf-8",
    )
    qs = evaluate.load_queries(f)
    assert [q.text for q in qs] == ["tramonto al mare"]  # a query with no answers is useless
    assert qs[0].relevant == ["a.jpg", "b.jpg"]


def test_load_queries_reports_bad_lines_with_their_number(tmp_path):
    f = tmp_path / "q.jsonl"
    f.write_text('{"query": "ok", "relevant": ["a"]}\nnot json\n', encoding="utf-8")
    with pytest.raises(ValueError, match="line 2"):
        evaluate.load_queries(f)
    f.write_text('{"relevant": ["a"]}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="line 1"):
        evaluate.load_queries(f)


def test_query_template_round_trips(tmp_path):
    f = tmp_path / "t.jsonl"
    evaluate.write_query_template(f)
    assert evaluate.load_queries(f)  # the template itself is valid
