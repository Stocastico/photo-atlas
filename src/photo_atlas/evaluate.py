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
from typing import Any, Protocol

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
    embeddings: np.ndarray, labels: Sequence[Any] | np.ndarray, *, threshold: float, k: int = 5
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


# -- face evaluation --------------------------------------------------------
class EvaluationError(ValueError):
    """The library doesn't have what an evaluation needs (message says what to do)."""


class FaceEmbedder(Protocol):
    def embed(self, image: np.ndarray, full_row: np.ndarray) -> np.ndarray: ...


@dataclass
class FaceCandidate:
    """A recogniser to evaluate: anything with ``embed(image_bgr, detection_row)``."""

    name: str
    embedder: FaceEmbedder
    size_mb: float | None = None


def load_upright_bgr(path: Path | str) -> np.ndarray | None:
    """Decode a photo the way the indexer does (EXIF-upright), as BGR; ``None`` on failure."""

    from PIL import Image, ImageOps  # noqa: PLC0415

    from . import metadata  # noqa: F401, PLC0415 - registers the HEIC opener
    from .faces import pil_to_bgr  # noqa: PLC0415

    try:
        with Image.open(path) as raw:
            raw.load()
            return pil_to_bgr(ImageOps.exif_transpose(raw) or raw)
    except Exception:
        return None


def _iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix = max(0.0, min(ax + aw, bx + bw) - max(ax, bx))
    iy = max(0.0, min(ay + ah, by + bh) - max(ay, by))
    inter = ix * iy
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def _load_labelled_faces(conn, *, max_faces: int, max_per_person: int, seed: int) -> list[dict]:
    """Faces a human named (confidence 1.0), capped per person and overall, reproducibly."""

    rows = conn.execute(
        "SELECT f.id AS face_id, f.person_id, f.bbox_x, f.bbox_y, f.bbox_w, f.bbox_h, p.path "
        "FROM faces f JOIN photos p ON p.id = f.photo_id "
        "WHERE f.person_id IS NOT NULL AND f.confidence >= 1.0 "
        "AND f.bbox_w > 0 AND f.bbox_h > 0 ORDER BY f.id"
    ).fetchall()
    by_person: dict[int, list[dict]] = {}
    for r in rows:
        by_person.setdefault(r["person_id"], []).append(dict(r))
    by_person = {p: fs for p, fs in by_person.items() if len(fs) >= 2}
    rng = np.random.default_rng(seed)
    cap = max_per_person
    if sum(min(len(fs), cap) for fs in by_person.values()) > max_faces and by_person:
        cap = max(2, max_faces // len(by_person))
    chosen: list[dict] = []
    for _, fs in sorted(by_person.items()):
        picked = rng.permutation(len(fs))[:cap]
        chosen.extend(fs[i] for i in sorted(picked))
        if len(chosen) >= max_faces:
            break
    return chosen[:max_faces]


def evaluate_faces(
    config,
    candidates: Sequence[FaceCandidate],
    *,
    detector,
    max_faces: int = 2000,
    max_per_person: int = 40,
    k: int = 5,
    seed: int = 0,
    loader=load_upright_bgr,
) -> dict:
    """Compare face recognisers on the people you have named by hand.

    The ground truth is every face you assigned or confirmed yourself (confidence 1.0).
    Each source photo is decoded and run through ``detector.detect_rows`` **once**; the
    detection matching each stored face (IoU ≥ 0.4) is then aligned and embedded by every
    candidate, so all of them see exactly the same faces. Per candidate the result holds
    ``verification`` (see :func:`verification_metrics`), ``identification`` (at the
    threshold where 0.1% of different-person pairs would be accepted — an equal
    false-accept budget for everyone) and speed. Raises :class:`EvaluationError` when
    fewer than two people have two named faces each.
    """

    from . import db, faces  # noqa: PLC0415

    conn = db.connect(config.db_path)
    try:
        labelled = _load_labelled_faces(
            conn, max_faces=max_faces, max_per_person=max_per_person, seed=seed
        )
    finally:
        conn.close()

    by_path: dict[str, list[dict]] = {}
    for f in labelled:
        by_path.setdefault(f["path"], []).append(f)

    kept: list[dict] = []
    vectors: list[list[np.ndarray]] = [[] for _ in candidates]
    seconds = [0.0] * len(candidates)
    errors: dict[int, str] = {}
    import time  # noqa: PLC0415

    for path, fs in by_path.items():
        image = loader(path) if Path(path).exists() else None
        rows = detector.detect_rows(image) if image is not None else []
        for f in fs:
            box = (f["bbox_x"], f["bbox_y"], f["bbox_w"], f["bbox_h"])
            scored = [(_iou(box, tuple(r[:4])), r) for r in rows]
            best = max(scored, key=lambda t: t[0], default=(0.0, None))
            if best[1] is None or best[0] < 0.4:
                continue
            assert image is not None  # a row only exists if the image decoded
            embedded: list[np.ndarray | None] = []
            for ci, cand in enumerate(candidates):
                if ci in errors:
                    embedded.append(None)
                    continue
                t0 = time.perf_counter()
                try:
                    vec = np.asarray(cand.embedder.embed(image, best[1]), dtype=np.float32)
                except Exception as exc:  # a broken candidate must not sink the others
                    errors[ci] = f"{type(exc).__name__}: {exc}"
                    embedded.append(None)
                    continue
                seconds[ci] += time.perf_counter() - t0
                embedded.append(vec.reshape(-1))
            kept.append(f)
            for ci, got in enumerate(embedded):
                if got is not None:
                    vectors[ci].append(got)

    # Drop people left with fewer than two matched faces.
    counts = Counter(f["person_id"] for f in kept)
    keep_idx = [i for i, f in enumerate(kept) if counts[f["person_id"]] >= 2]
    person_ids = np.array([kept[i]["person_id"] for i in keep_idx])
    if len(set(person_ids.tolist())) < 2:
        raise EvaluationError(
            "Not enough named faces to evaluate: name at least two people in the web UI, "
            "each with 2+ faces you assigned or confirmed yourself (and whose photos are "
            "still on disk)."
        )

    results = []
    for ci, cand in enumerate(candidates):
        entry: dict = {"name": cand.name, "size_mb": cand.size_mb}
        if ci in errors:
            entry["error"] = errors[ci]
            results.append(entry)
            continue
        emb = np.vstack([vectors[ci][i] for i in keep_idx])
        pairs = faces.sample_pair_distances(emb, person_ids, seed=seed)
        if pairs is None:
            entry["error"] = "no usable pairs"
            results.append(entry)
            continue
        ver = verification_metrics(*pairs)
        entry.update(
            dim=int(emb.shape[1]),
            ms_per_face=round(1000 * seconds[ci] / max(1, len(vectors[ci])), 2),
            verification=ver,
            identification=identification_metrics(
                emb, person_ids, threshold=ver["thr_at_far_0p1pct"], k=k
            ),
        )
        results.append(entry)

    return {
        "faces": len(keep_idx),
        "people": len(set(person_ids.tolist())),
        "unmatched": len(labelled) - len(keep_idx),
        "candidates": results,
    }


def format_faces_report(result: dict) -> str:
    """A fixed-width table of :func:`evaluate_faces` output."""

    lines = [
        f"Face recognisers on {result['faces']} faces of {result['people']} people "
        f"({result['unmatched']} named faces could not be re-detected and were skipped).",
        "Identification uses each model's own threshold at 0.1% false-accept; "
        "higher is better except EER, wrong, unknown and ms/face.",
        "",
        f"{'model':<22}{'MB':>7}{'dim':>6}{'ms/face':>9}{'AUC':>8}{'EER':>8}"
        f"{'TAR@FAR1%':>11}{'TAR@FAR.1%':>12}{'top1':>7}{'right':>7}{'wrong':>7}{'unk':>6}",
    ]
    for c in result["candidates"]:
        if "error" in c:
            lines.append(f"{c['name']:<22} FAILED: {c['error']}")
            continue
        v, i = c["verification"], c["identification"]
        size = f"{c['size_mb']:.0f}" if c.get("size_mb") else "-"
        lines.append(
            f"{c['name']:<22}{size:>7}{c['dim']:>6}{c['ms_per_face']:>9.1f}"
            f"{v['auc']:>8.4f}{v['eer']:>8.4f}{v['tar_at_far_1pct']:>11.3f}"
            f"{v['tar_at_far_0p1pct']:>12.3f}{i['top1_accuracy']:>7.3f}{i['auto_correct']:>7.3f}"
            f"{i['auto_wrong']:>7.3f}{i['auto_unknown']:>6.3f}"
        )
    return "\n".join(lines)
