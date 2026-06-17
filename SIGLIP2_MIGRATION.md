# Migration scope — SigLIP base → SigLIP 2 (scene tagging + semantic search)

Implementation-ready scope for the top TODO follow-up: swap the shared SigLIP
vision/text tower (`classify.py` + `embed.py`) for **SigLIP 2**. SigLIP 2 is a
drop-in *architecture* (same dual-encoder, same `onnxruntime`/NumPy/Pillow
pipeline, no PyTorch) with strictly better zero-shot and retrieval quality, so it
upgrades scene tags **and** natural-language search from a single swap. See
[`MODELS.md`](MODELS.md) §3 for the comparison that picked it.

This document is the *how*: the exact files, the gaps that make it more than "change
a URL", the runbook, the eval, and rollback. No swap is committed here — a candidate
must be A/B'd on a real library first (the suite stays offline). **One enabling
refactor is included in this branch** (vision input-size auto-detect; see Gap 1).

## Candidate

| Variant | Input | Embed dim | ONNX source | Notes |
| --- | --- | --- | --- | --- |
| **siglip2-base-patch16-256** (recommended) | 256² | 768 | `onnx-community/siglip2-base-patch16-256` (quantised) | Same dim as today (768) → **no re-embed needed for dim**, only a space change. Modest extra cost from 224→256. |
| siglip2-base-patch16-384 | 384² | 768 | `onnx-community/...-384` | Higher quality, ~2× the vision cost per photo. |
| siglip2-large-patch16-256 | 256² | 1024 | `onnx-community/...large...` | Best quality; **dim 768→1024 ⇒ full re-embed + scene-matrix rebuild**; ~2–3× size/cost. |

Start with **base-patch16-256**: same embedding dimension as the current model, so
the only required data migration is rebuilding the label matrix (cheap) and
re-embedding (because the *space* changed even though the dim didn't).

## Already supported (no work)

- **Per-tower env overrides** — `PHOTO_ATLAS_SCENE_MODEL` (vision),
  `PHOTO_ATLAS_SCENE_TEXT_MODEL` (text), `PHOTO_ATLAS_SCENE_TOKENIZER`
  (`models._resolve`). An A/B is a config change, not a redeploy.
- **Offline-safe label matrix** — `scripts/build_scene_embeddings.py --model <repo>`
  rebuilds `src/photo_atlas/data/scene_labels.npz` in the new space; the runtime
  needs no text encoder for scene tagging.
- **Dim-aware storage** — `photos.embed_dim` is stored per row; `faces.dim` exists
  for the parallel recognition story. A dim change is representable.
- **Offline test suite** — encoders are stubbed (`tests/scene_stub.StubTagger`,
  hand-built vectors); a model swap never touches CI.

## Gaps (the actual work)

### Gap 1 — Vision input size was hardcoded → **fixed in this branch**
`embed.preprocess_image` resized to a literal 224². SigLIP 2 uses 256²/384². Now
`SigLipImageEncoder` reads the model's input shape and threads the resolution into
`preprocess_image(img, size=…)` (`embed._input_size_from_shape`, default 224 for
dynamic/unknown shapes). The vision-side swap is now genuinely config-only.
*Covered by `tests/test_embed_unit.py`.*

### Gap 2 — Output tensor name was hardcoded `"pooler_output"` → **fixed in this branch**
Both encoders called `session.run(["pooler_output"], …)`. Other exports (incl. some
`onnx-community` SigLIP 2 ONNX) name the pooled output differently (e.g.
`image_embeds` / `text_embeds`). Now `embed._select_output_name` resolves the output
from a preference list (`pooler_output` then `image_embeds`/`text_embeds`), falls back
to a sole output, and raises a clear error rather than guessing among several
unrecognised outputs; both encoders store the resolved name once in `__init__`.
Adding a new model's output name is a one-line `prefer`-tuple edit. *Covered by
`tests/test_embed_unit.py`.*

### Gap 3 — Text tokenizer assumptions (SentencePiece `</s>`, pad 64)
`embed.SigLipTextEncoder` and `build_scene_embeddings.py` both hardcode SigLIP 1's
SentencePiece padding: `pad_id = token_to_id("</s>")`, `pad_len = 64`, right pad.
**SigLIP 2's multilingual text tower uses the Gemma tokenizer**, where that pad
token/scheme differs. **Action:** before swapping the text side, verify the chosen
export's `tokenizer.json` — confirm the pad token id and the model's expected
sequence length, and parameterise `_TEXT_PAD_LEN` / pad token (read from the
tokenizer's `padding` config rather than assuming `</s>`). If the base (English)
SigLIP 2 export keeps a SigLIP-style tokenizer, this may be a no-op — **verify, don't
assume.** This only affects *query-time* text + the offline matrix build, not stored
data.

### Gap 4 — Silent scene-label / image-embedding space mismatch → **partly fixed in this branch**
The `.npz` already records `model` and `embed_dim`, but `classify.py` ignored them.
If someone swaps the vision model and forgets to rebuild `scene_labels.npz`:
- **dim changed** → now caught: `classify_embedding` checks the image-embedding dim
  against the label-matrix dim and raises an **actionable** error ("rebuild
  data/scene_labels.npz … with scripts/build_scene_embeddings.py") instead of an
  opaque numpy matmul error. `ZeroShotSceneTagger` also exposes `label_model` /
  `label_dim` (read from the `.npz`) so the matrix's provenance is diagnosable.
- **dim unchanged (e.g. base-256, still 768)** → **still a residual risk**: the dim
  check can't distinguish two same-dim spaces. Mitigation: the runbook always rebuilds
  the matrix, and `label_model` records which model the bundled matrix was built for.
  A fuller guard (persist the active model id alongside embeddings and compare) is a
  possible follow-up if same-dim swaps become common.

*Covered by `tests/test_scene_zeroshot.py`.*

### Gap 5 — Re-embed migration + cache invalidation
Semantic search compares within one space, so after the swap **every stored image
embedding must be recomputed**: `photo-atlas embed --recompute`. Two caveats:
- The new index runs (`index --embed`) and `retag-scenes` automatically use the new
  model; only the *already-stored* embeddings are stale.
- `api._embed_signature` keys the cached `SemanticIndex` on `(count, max_id)`. An
  in-place `--recompute` changes neither, so a **running** server won't reload the
  matrix — restart `serve` after re-embedding (or extend the signature to include a
  cheap content hash / a stored model tag). Document the restart; consider the
  signature improvement if live model swaps become common.

### Gap 6 — Normalisation constants (likely a no-op, verify)
`_NORM_MEAN = _NORM_STD = 0.5` matches SigLIP's `preprocessor_config.json`. SigLIP 2
keeps 0.5/0.5 mean/std as far as the published configs show, so this is expected to
be unchanged — but confirm against the chosen export's preprocessor config and, if it
ever diverges, thread mean/std through `preprocess_image` the same way Gap 1 threads
size.

## Runbook (once a candidate passes eval)

```bash
# 1. Point the three model env vars at the SigLIP 2 ONNX files (or bump the
#    *_NAME/*_URL constants in models.py to make it the new default).
export PHOTO_ATLAS_SCENE_MODEL=…/siglip2-base-patch16-256/vision_model_quantized.onnx
export PHOTO_ATLAS_SCENE_TEXT_MODEL=…/text_model_quantized.onnx
export PHOTO_ATLAS_SCENE_TOKENIZER=…/tokenizer.json

# 2. Rebuild the bundled scene-label matrix in the new space (commit the .npz).
python scripts/build_scene_embeddings.py \
    --model onnx-community/siglip2-base-patch16-256 \
    --text-file text_model_quantized.onnx --pad-len <verified>

# 3. Refresh stored data in the new space.
photo-atlas embed --recompute      # re-embed all images (semantic search)
photo-atlas retag-scenes           # re-tag scenes (reuses stored face_count)

# 4. Restart `serve` (Gap 5 cache note).
```

If making it the default (step 1 via `models.py`): bump `SCENE_NAME/_URL`,
`SCENE_TEXT_NAME/_URL`, `SCENE_TOKENIZER_NAME/_URL`, ship the rebuilt `.npz`, and note
in the README that an existing library needs `embed --recompute` + `retag-scenes`.

## A/B evaluation (gate before committing the swap)

Reuse [`MODELS.md` §"How to evaluate"](MODELS.md#how-to-evaluate-before-swapping):
- **Scene tags:** per-class correct + well-separated; out-of-vocabulary → `other`.
- **Semantic search:** a handful of held-out NL queries with known target photos;
  compare the target's rank old-vs-new (point the env vars at each model in turn).
- **Cost:** photos/sec at index time and model size — the vision tower runs once per
  photo over the whole library; 256² is ~(256/224)² ≈ 1.3× the 224² compute, 384² ≈
  2.9×.
Keep both models env-selectable so the A/B is a config flip; CI stays offline.

## Rollback

Unset the three env vars (or revert the `models.py` constants), restore the previous
`scene_labels.npz` from git, and `embed --recompute` + `retag-scenes` back into the
old space. Because everything is env/file driven, rollback is symmetric with the swap.

## Effort estimate

| Piece | Effort | Risk |
| --- | --- | --- |
| Gap 1 (input size) | **done** | none |
| Gap 2 (output name) | **done** | low |
| Gap 3 (tokenizer verify/param) | ~½–1 day | **medium** (Gemma tokenizer) |
| Gap 4 (dim mismatch guard) | **done** (same-dim case residual) | low |
| Gap 5 (re-embed + cache note) | runbook only (+~½ day if improving the signature) | low |
| Eval harness + A/B run | ~1 day | medium (needs a real labelled slice) |

Net: roughly **2–3 focused days** plus the eval, with the tokenizer the only genuine
unknown. No architectural change; all behind the existing env overrides.
