"""Build the bundled zero-shot scene-label embeddings.

The runtime scene tagger (:class:`photo_atlas.classify.ZeroShotSceneTagger`) only
runs a vision encoder and compares the image embedding against a small, fixed
matrix of *text* embeddings — one per scene label. Those text embeddings never
change between runs, so we compute them **once, offline** with this script and
ship the result as a tiny ``.npz`` next to the package. That keeps the runtime
free of a text encoder, a tokenizer, and any per-image text work.

The output matrix lives in the same joint image/text space as the vision
encoder that runs at index time, so the two must come from the *same* model
(default: ``Xenova/siglip-base-patch16-224``). To switch models — e.g. to a
MobileCLIP2 or SigLIP 2 export — point ``--model`` at the new repo and rebuild;
nothing else in the runtime changes.

Usage (needs ``onnxruntime`` + ``tokenizers``; the encoders/tokenizer are pulled
from the Hugging Face hub on demand and cached under ``--cache``)::

    python scripts/build_scene_embeddings.py

Each label is described by an *ensemble* of natural-language prompts; their
embeddings are averaged and re-normalised, which is the standard trick for
sturdier zero-shot prototypes than any single prompt.
"""

from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path

import numpy as np

# The concrete, text-describable labels. ``other`` is intentionally absent: it
# is the runtime's low-confidence fallback, not a thing we can write a prompt
# for. Keep this in sync with the non-"other" half of
# ``photo_atlas.classify.SCENE_LABELS``.
LABEL_PROMPTS: dict[str, list[str]] = {
    "people": [
        "a photo of a person",
        "a portrait of a person",
        "a photo of a group of people",
        "a selfie of people",
        "a photo of a family",
    ],
    "landscape": [
        "a landscape photograph",
        "a photo of scenery",
        "a photo of nature outdoors",
        "a photo of mountains and sky",
        "a photo of a beach",
    ],
    "food": [
        "a photo of food",
        "a photo of a meal on a plate",
        "a close-up photo of a dish",
        "a photo of a dessert",
    ],
    "document": [
        "a scan of a document",
        "a screenshot of text",
        "a photo of a page of text",
        "a photo of a receipt",
    ],
}

DEFAULT_MODEL = "Xenova/siglip-base-patch16-224"
HF = "https://huggingface.co/{model}/resolve/main/{path}"


def _download(model: str, path: str, cache: Path) -> Path:
    dest = cache / model.replace("/", "__") / path.replace("/", "__")
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        url = HF.format(model=model, path=path)
        print(f"  fetch {url}")
        urllib.request.urlretrieve(url, dest)  # noqa: S310 - trusted HF hub URL
    return dest


def build(model: str, cache: Path, out: Path, *, text_file: str, pad_len: int) -> None:
    import onnxruntime as ort  # noqa: PLC0415
    from tokenizers import Tokenizer  # noqa: PLC0415

    tok_path = _download(model, "tokenizer.json", cache)
    text_path = _download(model, f"onnx/{text_file}", cache)

    tok = Tokenizer.from_file(str(tok_path))
    # SigLIP pads/truncates to a fixed 64-token window with the </s> token.
    pad_id = tok.token_to_id("</s>")
    pad_id = 1 if pad_id is None else pad_id
    tok.enable_truncation(pad_len)
    tok.enable_padding(length=pad_len, pad_id=pad_id, pad_token="</s>", direction="right")

    sess = ort.InferenceSession(str(text_path), providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name

    labels = sorted(LABEL_PROMPTS)
    rows = []
    for label in labels:
        prompts = LABEL_PROMPTS[label]
        ids = np.array([tok.encode(p).ids for p in prompts], dtype=np.int64)
        (pooled,) = sess.run(["pooler_output"], {in_name: ids})
        vecs = pooled / np.linalg.norm(pooled, axis=1, keepdims=True)
        proto = vecs.mean(axis=0)
        proto = proto / np.linalg.norm(proto)
        rows.append(proto.astype(np.float32))
        print(f"  {label:<10} <- {len(prompts)} prompts")

    matrix = np.vstack(rows)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out,
        labels=np.array(labels),
        matrix=matrix,
        model=np.array(model),
        embed_dim=np.array(matrix.shape[1]),
    )
    print(f"wrote {out}  (labels={labels}, matrix={matrix.shape})")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--text-file", default="text_model_quantized.onnx")
    ap.add_argument("--pad-len", type=int, default=64)
    ap.add_argument("--cache", type=Path, default=Path("/tmp/photo_atlas_scene_build"))
    ap.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "src/photo_atlas/data/scene_labels.npz",
    )
    args = ap.parse_args()
    build(args.model, args.cache, args.out, text_file=args.text_file, pad_len=args.pad_len)


if __name__ == "__main__":
    main()
