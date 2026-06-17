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
    "animals": [
        "a photo of an animal",
        "a photo of a pet",
        "a photo of a dog",
        "a photo of a cat",
        "a photo of a bird",
    ],
    "landscape": [
        "a landscape photograph",
        "a photo of scenery",
        "a photo of nature outdoors",
        "a photo of mountains and sky",
        "a photo of a beach",
    ],
    "plants": [
        "a photo of a flower",
        "a close-up photo of a plant",
        "a photo of a garden",
        "a macro photo of a blossom",
    ],
    "food": [
        "a photo of food",
        "a photo of a meal on a plate",
        "a close-up photo of a dish",
        "a photo of a dessert",
    ],
    "vehicle": [
        "a photo of a car",
        "a photo of a vehicle",
        "a photo of a truck",
        "a photo of a motorcycle",
    ],
    "building": [
        "a photo of a building",
        "a photo of architecture",
        "a photo of a city street",
        "a photo of a house",
    ],
    "document": [
        "a scan of a document",
        "a photo of a page of text",
        "a photo of a receipt",
        "a photo of a book page",
    ],
    "screenshot": [
        "a screenshot of an app",
        "a screenshot of a phone screen",
        "a screenshot of a website",
        "a screenshot of a user interface",
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

    from photo_atlas.embed import _select_output_name, configure_text_tokenizer  # noqa: PLC0415

    tok_path = _download(model, "tokenizer.json", cache)
    text_path = _download(model, f"onnx/{text_file}", cache)

    tok = Tokenizer.from_file(str(tok_path))
    # Pad/truncate to SigLIP's fixed window the same way the runtime text encoder
    # does — honouring SigLIP 2's embedded Gemma <pad> config, configuring SigLIP
    # 1's </s> ourselves — so the bundled label matrix lives in the same space as
    # query embeddings. ``--pad-len`` is only a fallback for a config-less tokenizer.
    configure_text_tokenizer(tok, pad_len)

    sess = ort.InferenceSession(str(text_path), providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name
    out_name = _select_output_name(
        [o.name for o in sess.get_outputs()], ("pooler_output", "text_embeds")
    )

    labels = sorted(LABEL_PROMPTS)
    rows = []
    for label in labels:
        prompts = LABEL_PROMPTS[label]
        ids = np.array([tok.encode(p).ids for p in prompts], dtype=np.int64)
        (pooled,) = sess.run([out_name], {in_name: ids})
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
