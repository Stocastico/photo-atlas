"""Model evaluation harness: compare candidate models on *your own* library.

Two questions are worth measuring before swapping a default model:

* **Faces** — does a different recogniser separate the people you have named better?
  (:func:`evaluate_faces`: verification AUC/EER/TAR@FAR and nearest-neighbour
  identification on faces you assigned by hand.)
* **Search** — does a different image-text encoder retrieve better for the way you
  search? (:func:`evaluate_search`: Hit/Recall/MRR/nDCG on your own query file, plus a
  label-free image→image check on near-duplicate bursts.)

The metric functions here are pure NumPy so they can be tested exactly; the
orchestrators take injectable detectors/encoders so the whole flow is testable
offline. See ``MODELS.md`` and ``photo-atlas eval-faces`` / ``eval-search``.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


# -- face verification / identification metrics ---------------------------
def verification_metrics(genuine: np.ndarray, impostor: np.ndarray) -> dict:
    """How well distances separate same-person from different-person pairs.

    ``genuine`` / ``impostor`` are cosine *distances* (smaller = more alike).
    ``auc`` is P(a same-person pair is closer than a different-person pair) with ties
    counting half; ``eer`` the error rate where false-accept equals false-reject;
    ``tar_at_far_*`` the share of same-person pairs still accepted when only 1% /
    0.1% of different-person pairs are; ``d_prime`` a threshold-free separation score.
    """

    gen = np.sort(np.asarray(genuine, dtype=np.float64))
    imp = np.sort(np.asarray(impostor, dtype=np.float64))
    left = np.searchsorted(imp, gen, side="left")
    right = np.searchsorted(imp, gen, side="right")
    auc = float(np.mean(((imp.size - right) + 0.5 * (right - left)) / imp.size))

    def tar_at(far: float) -> tuple[float, float]:
        thr = float(np.percentile(imp, far * 100))
        return float(np.mean(gen <= thr)), thr

    tar1, thr1 = tar_at(0.01)
    tar01, thr01 = tar_at(0.001)

    candidates = np.unique(np.quantile(np.concatenate([gen, imp]), np.linspace(0, 1, 2001)))
    far = np.searchsorted(imp, candidates, side="right") / imp.size
    frr = 1.0 - np.searchsorted(gen, candidates, side="right") / gen.size
    at = int(np.argmin(np.abs(far - frr)))
    pooled = np.sqrt(0.5 * (gen.var() + imp.var()))
    return {
        "auc": auc,
        "eer": float((far[at] + frr[at]) / 2),
        "tar_at_far_1pct": tar1,
        "thr_at_far_1pct": thr1,
        "tar_at_far_0p1pct": tar01,
        "thr_at_far_0p1pct": thr01,
        "d_prime": float((imp.mean() - gen.mean()) / pooled) if pooled > 0 else float("inf"),
        "genuine_pairs": int(gen.size),
        "impostor_pairs": int(imp.size),
    }


def identification_metrics(
    embeddings: np.ndarray, labels: Sequence, *, threshold: float, k: int = 5
) -> dict:
    """Leave-one-out identification mirroring how the app auto-tags a new face.

    Each face whose person has at least one other face is a probe. ``top1_accuracy`` is
    whether its single nearest neighbour is the same person. ``auto_*`` apply the app's
    rule: the ``k`` nearest neighbours within ``threshold`` vote; no neighbour within
    range means the face stays *unknown*. ``auto_precision`` is correct / (correct +
    wrong) among the faces that did get a name (``None`` if none did).
    """

    emb = np.asarray(embeddings, dtype=np.float32)
    emb = emb / np.maximum(np.linalg.norm(emb, axis=1, keepdims=True), 1e-8)
    lab = np.asarray(labels)
    dist = 1.0 - emb @ emb.T
    np.fill_diagonal(dist, np.inf)
    counts = Counter(lab.tolist())
    probes = [i for i in range(len(lab)) if counts[lab[i].item()] >= 2]
    nn_ok = correct = wrong = unknown = 0
    for i in probes:
        order = np.argsort(dist[i], kind="stable")[:k]
        nn_ok += int(lab[order[0]] == lab[i])
        within = [j for j in order if dist[i][j] <= threshold]
        if not within:
            unknown += 1
            continue
        votes: dict = {}
        for j in within:
            n, best = votes.get(lab[j].item(), (0, np.inf))
            votes[lab[j].item()] = (n + 1, min(best, dist[i][j]))
        winner = max(votes.items(), key=lambda kv: (kv[1][0], -kv[1][1]))[0]
        if winner == lab[i].item():
            correct += 1
        else:
            wrong += 1
    n = len(probes)
    tagged = correct + wrong
    return {
        "probes": n,
        "top1_accuracy": nn_ok / n if n else 0.0,
        "auto_correct": correct / n if n else 0.0,
        "auto_wrong": wrong / n if n else 0.0,
        "auto_unknown": unknown / n if n else 0.0,
        "auto_precision": correct / tagged if tagged else None,
    }


# -- retrieval metrics -----------------------------------------------------
def rank_metrics(
    rankings: Sequence[Sequence], relevant: Sequence[set], ks: Sequence[int] = (1, 5, 10)
) -> dict:
    """Retrieval quality over ranked id lists (best first) vs relevant-id sets.

    ``hit@k`` = a relevant item is in the top k; ``recall@k`` = share of the relevant
    items found in the top k; ``mrr`` = mean reciprocal rank of the first relevant
    item; ``ndcg@10`` = binary-gain nDCG over the top 10. All averaged over queries.
    """

    out: dict = {"queries": len(rankings)}
    if not rankings:
        return out
    hits: dict[int, list[float]] = {k: [] for k in ks}
    recalls: dict[int, list[float]] = {k: [] for k in ks}
    rr: list[float] = []
    ndcg: list[float] = []
    for ranked, rel in zip(rankings, relevant, strict=True):
        ranked = list(ranked)
        first = next((i for i, x in enumerate(ranked) if x in rel), None)
        rr.append(0.0 if first is None else 1.0 / (first + 1))
        for k in ks:
            found = sum(1 for x in ranked[:k] if x in rel)
            hits[k].append(1.0 if found else 0.0)
            recalls[k].append(found / len(rel) if rel else 0.0)
        dcg = sum(1.0 / np.log2(i + 2) for i, x in enumerate(ranked[:10]) if x in rel)
        ideal = sum(1.0 / np.log2(i + 2) for i in range(min(len(rel), 10)))
        ndcg.append(dcg / ideal if ideal else 0.0)
    for k in ks:
        out[f"hit@{k}"] = float(np.mean(hits[k]))
        out[f"recall@{k}"] = float(np.mean(recalls[k]))
    out["mrr"] = float(np.mean(rr))
    out["ndcg@10"] = float(np.mean(ndcg))
    return out


def rank_pool(
    queries: np.ndarray, pool: np.ndarray, *, top: int, exclude: Sequence[int | None] | None = None
) -> list[np.ndarray]:
    """Indices of the ``top`` pool rows most cosine-similar to each query vector.

    ``exclude[i]`` (a pool index) is removed from query ``i``'s ranking — used so an
    image never retrieves itself in the image→image check.
    """

    q = np.asarray(queries, dtype=np.float32)
    p = np.asarray(pool, dtype=np.float32)
    q = q / np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-8)
    p = p / np.maximum(np.linalg.norm(p, axis=1, keepdims=True), 1e-8)
    sims = q @ p.T
    result = []
    for i in range(sims.shape[0]):
        row = sims[i]
        order = np.argsort(-row, kind="stable")
        if exclude is not None and exclude[i] is not None:
            order = order[order != exclude[i]]
        result.append(order[:top])
    return result


# -- query file -------------------------------------------------------------
@dataclass
class Query:
    text: str
    relevant: list[str] = field(default_factory=list)


def load_queries(path: Path | str) -> list[Query]:
    """Read a JSONL file of ``{"query": "...", "relevant": ["path-or-filename", ...]}``.

    Blank lines and ``#`` comments are skipped; a query with no ``relevant`` photos is
    dropped (nothing to score). Errors name the offending line number.
    """

    queries: list[Query] = []
    for n, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        try:
            obj = json.loads(line)
            text = str(obj["query"]).strip()
            relevant = [str(r) for r in obj.get("relevant", [])]
        except (ValueError, KeyError, TypeError, AttributeError) as exc:
            raise ValueError(
                f'{path}: line {n}: expected {{"query", "relevant"}} ({exc})'
            ) from exc
        if text and relevant:
            queries.append(Query(text, relevant))
    return queries


QUERY_TEMPLATE = (
    "# One JSON object per line. 'relevant' lists photos (full path or just the file name)\n"
    "# that a good search for the query should return. 15-30 queries in the way YOU\n"
    "# search (Italian is fine) are enough to compare models.\n"
    '{"query": "tramonto sul mare", "relevant": ["IMG_1234.jpg", "IMG_1240.jpg"]}\n'
    '{"query": "cena in famiglia", "relevant": ["/photos/2024/natale/IMG_9001.jpg"]}\n'
)


def write_query_template(path: Path | str) -> None:
    Path(path).write_text(QUERY_TEMPLATE, encoding="utf-8")
