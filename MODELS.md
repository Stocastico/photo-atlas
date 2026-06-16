# Model survey — upgrade options for the deep-learning stack

Investigation deliverable for the TODO item *"Investigate newer / better models
everywhere a deep-learning net is used."* This is a **written comparison to inform
a future swap**, not a code change. It deliberately makes no swap yet — each
candidate should be validated on a real slice of the target library first (see
[How to evaluate before swapping](#how-to-evaluate-before-swapping)).

## Hard constraints (carried from CLAUDE.md / the request)

- **ONNX Runtime only at inference time. No PyTorch dependency.** Any candidate
  must have (or allow) a clean ONNX export that runs under `onnxruntime` with just
  NumPy/Pillow/OpenCV pre-processing.
- **On-demand download + env override.** Every weight is fetched lazily to
  `~/.photo_atlas/models` and overridable via a `PHOTO_ATLAS_*` env var
  (`models._resolve`). A swap is "change the URL/name + env, rebuild any derived
  artefact," never a code rewrite — the architecture is already model-agnostic.
- **Offline test suite stays green.** Pure logic is tested with hand-built vectors
  / stub encoders; live round-trips are env-gated and skipped by default. A model
  swap must not change that.

## Current stack

| Job | Module | Model today | Size | Notes |
| --- | --- | --- | --- | --- |
| Face **detection** | `faces.py` / `models.py` | YuNet `2023mar` (OpenCV Zoo) | ~230 KB | Fast CNN detector, 5 landmarks; run on a ≤1280px copy. |
| Face **recognition** (embedding) | `faces.py` | SFace `2021dec` (OpenCV Zoo) | ~37 MB | 128-d embedding; cosine + k-NN vote enrol/recognise. |
| **Scene** tag + **semantic** search | `classify.py` / `embed.py` | SigLIP base patch16-224, vision + text (Xenova ONNX, quantised) | ~95 MB vision (+ text + tokenizer) | Zero-shot scene labels (pre-baked text matrix) and NL search share the vision tower. |

All three are solid, conservative 2021–2023 choices. None is *bad*; the question is
whether a current model meaningfully improves accuracy at acceptable size/speed and
with a no-PyTorch ONNX path.

---

## 1. Face detection — YuNet → ?

**Why consider a change.** YuNet is excellent for its size but is a small model; on
a 15-year mixed library (group shots, tiny/blurry/profile faces, old scans) recall
on hard faces is the usual weak point, which then starves recognition.

| Candidate | ONNX / no-PyTorch | Pros | Cons / risk |
| --- | --- | --- | --- |
| **YuNet (latest Zoo revision)** | ✅ native ONNX in OpenCV Zoo | Drop-in: same pre/post-processing, same `cv2.FaceDetectorYN` API; just a newer `.onnx`. Zero integration cost. | Marginal gains; still a small model. **Lowest-risk first step.** |
| **SCRFD** (InsightFace, e.g. `scrfd_10g_bnkps`) | ✅ ONNX ships in InsightFace model zoo (no PyTorch to run) | Strong recall/precision at small/medium sizes; well-proven on WIDER FACE; 5 landmarks for alignment. | New pre/post-processing (anchors, distance2bbox); not a `FaceDetectorYN` drop-in — needs a small decode layer in `faces.py`. |
| **RetinaFace (R50)** | ✅ ONNX available | Very high recall, robust landmarks. | Heavier/slower; overkill for an index pass over tens of thousands of photos. |
| **YOLO-face variants** | ⚠️ ONNX yes, but ecosystems often pull torch/ultralytics | Good speed/recall. | Licensing (some are AGPL/Ultralytics) and dependency-creep risk; vet carefully. |

**Recommendation.** (1) Bump to the **newest YuNet Zoo revision** first — near-zero
risk, possibly free accuracy. (2) If hard-face recall is still the bottleneck on a
real eval, adopt **SCRFD** (e.g. `scrfd_10g`) as a `PHOTO_ATLAS_YUNET`-style
alternate backend — it's the best accuracy/size/ONNX trade-up and pairs naturally
with an ArcFace recogniser below. Keep YuNet as the lightweight default.

## 2. Face recognition / embedding — SFace → ?

**Why consider a change.** SFace (128-d) is good but older and lower-dimensional
than the current state of the art. The project's k-NN-over-enrolments design already
mitigates appearance drift, but a stronger embedding raises the ceiling on
"same person across 15 years" (child→adult, beards, glasses).

| Candidate | ONNX / no-PyTorch | Pros | Cons / risk |
| --- | --- | --- | --- |
| **ArcFace (InsightFace `buffalo_l` / glint360k R100), 512-d** | ✅ ONNX ships in InsightFace zoo, runs on onnxruntime | The de-facto strong baseline; 512-d, excellent verification accuracy; same SCRFD+ArcFace family for a coherent detect+recog pair. | Larger (~166 MB R100; `buffalo_s` is smaller). **Changing `dim` 128→512 means re-embedding all faces** and re-clustering. Needs ArcFace-standard alignment (5-pt similarity transform). |
| **AdaFace (IR-100, MS1MV2/WebFace)** | ✅ exportable to ONNX | Better on **low-quality / blurry** faces than vanilla ArcFace — directly relevant to old/scanned photos. | Export is a manual step (no canonical zoo ONNX as turnkey as InsightFace); validate the export carefully. |
| **SFace (current)** | ✅ | Tiny, integrated, fast. | Older; 128-d ceiling. |

**Recommendation.** Prototype **ArcFace R100 (buffalo_l)** behind a
`PHOTO_ATLAS_SFACE`-style override and compare verification accuracy on the user's
own labelled faces. If many photos are low-quality, also try **AdaFace**. Treat this
as the **highest-value upgrade** (recognition quality is the product's backbone), but
also the **most invasive**: it changes the embedding dimension and alignment, so it
needs a re-embed/re-cluster migration and a dim-aware schema (the `faces.dim` column
already exists, so this is tractable).

## 3. Scene tagging + semantic search — SigLIP base → ?

**Why consider a change.** This is where the field moved most since the base SigLIP
checkpoint. Newer encoders give better zero-shot scene separation **and** better NL
retrieval *from the same swap*, since both jobs share the vision/text space.

| Candidate | ONNX / no-PyTorch | Pros | Cons / risk |
| --- | --- | --- | --- |
| **SigLIP 2** (base/large patch16-256/384) | ✅ ONNX via transformers.js (Xenova/onnx-community), quantised | Direct successor to the current model; better zero-shot + retrieval; **same architecture/pipeline** as today, so the lowest-friction quality bump. Bigger inputs (256/384) cost a little speed. | Larger than base-224; pick the size/patch that fits the index-time budget. |
| **MobileCLIP2** (Apple) | ✅ ONNX exportable | Excellent **accuracy-per-FLOP**; great if index-time speed on CPU matters most. | Newer ecosystem; confirm a clean quantised ONNX + tokenizer. |
| **jina-clip-v2 / EVA-CLIP / MetaCLIP2** | ⚠️ ONNX varies | Strong retrieval; some multilingual (jina). | Export maturity and licensing vary; heavier; verify no-PyTorch ONNX path before committing. |

**Mechanics of this swap (already supported).** Point `PHOTO_ATLAS_SCENE_MODEL`
(and `PHOTO_ATLAS_SCENE_TEXT_MODEL` + `PHOTO_ATLAS_SCENE_TOKENIZER`) at the new ONNX
files, then **rebuild the bundled label matrix** with
`scripts/build_scene_embeddings.py --model <hf-repo>` so the scene-label text
embeddings live in the new space. If `embed_dim` changes, **re-run `photo-atlas
embed`** to refresh stored image embeddings (semantic search compares within one
space). `pre/post`-processing constants (image size, normalisation) must match the
new model — keep them in `embed.preprocess_image`.

**Recommendation.** **SigLIP 2 (base or large, patch16-256)** is the clear,
low-friction win: same pipeline, strictly better quality for both scene tags and
search. Prefer it as the first scene/semantic upgrade. Evaluate **MobileCLIP2** only
if CPU index-time speed becomes the binding constraint.

---

## Priorities (suggested order)

1. **SigLIP → SigLIP 2** — biggest quality-per-effort win, no architectural change
   (swap URLs, rebuild `scene_labels.npz`, re-embed). Improves scene tags *and*
   search together.
2. **YuNet → latest Zoo YuNet** — trivial, possibly-free detection gain.
3. **SFace → ArcFace R100 (or AdaFace for low-quality)** — highest ceiling on
   recognition, but the most invasive (dim/alignment change ⇒ re-embed + re-cluster
   migration). Do it deliberately, behind the existing env override, with a real
   accuracy comparison first.
4. **YuNet → SCRFD** — only if hard-face *recall* is shown to be the bottleneck.

## How to evaluate before swapping

- **Recognition:** assemble a small labelled set from the user's own named people
  (the catalog already has them) and compare verification accuracy / cluster purity
  old-vs-new on identical crops. Watch the `dim` change.
- **Scene tagging:** reuse the ground-truth-style check already used for the current
  zero-shot tagger (correct + well-separated per class; out-of-vocabulary → `other`).
- **Semantic search:** a handful of held-out NL queries with known target photos;
  compare rank of the target old-vs-new.
- **Cost:** measure index-time throughput (photos/sec) and model size — an index
  pass runs the vision tower once per photo over the whole library.
- Keep every candidate **env-overridable** so an A/B is a config change, and keep the
  default suite offline (stub encoders), never downloading a model in CI.
