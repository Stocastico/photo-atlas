# Photo Atlas

Navigate years of photos by **who's in them**, **what they show**, **where**
and **when** — with a modern deep-learning face pipeline, offline reverse
geocoding, automatic scene tagging and a clean web UI.

Point it at a folder (or 15 years of folders), let it build a catalog, then
browse and filter your library and put names to faces. New imports are
recognised automatically once a person has been named.

![pipeline](https://img.shields.io/badge/faces-YuNet%20%2B%20ArcFace-4c8dff) ![offline](https://img.shields.io/badge/geocoding-offline-2ea043)

---

## Features

- **Deep face recognition** — OpenCV's [YuNet](https://github.com/opencv/opencv_zoo)
  detector + [ArcFace R100](https://github.com/deepinsight/insightface) 512-d
  embeddings (glint360k). On real photos same-person pairs sit at ~0.03 cosine
  distance and different people at ~1.0, so identities separate cleanly. (The
  legacy 128-d SFace recogniser stays selectable with `--faces sface`.)
- **Name people once** — unrecognised faces are grouped into clusters; name a
  cluster and every matching photo becomes filterable. Future imports are
  auto-recognised against the people you've named.
- **Filter by anything** — person, scene type (`people` / `animals` /
  `landscape` / `plants` / `food` / `vehicle` / `building` / `document` /
  `screenshot` / `other`), country, city, place (trip/folder), year, camera, or
  filename — combined.
- **Natural-language semantic search** — describe a photo in words ("kids on the
  beach at sunset", "my red car") and rank the library by visual similarity — no
  tags, no folders. Each photo's [SigLIP 2](https://huggingface.co/onnx-community/siglip2-base-patch16-256-ONNX)
  image embedding is stored at index time and a free-text query is embedded into
  the same space at search time; the ✨ **Smart** toggle in the search bar switches
  it on, and it ANDs with every other filter.
- **Near-duplicate & burst grouping** — a perceptual hash (dHash) is stored for
  every photo at index time, and the **Duplicates** tab groups near-identical
  shots taken moments apart (camera bursts, re-saved copies). Each set keeps a
  best-of-N cover (favorite → highest resolution → earliest) and lets you **hide**
  the rest (reversible) or **delete** them from disk (irreversible) in one click.
- **Find more like this** — from the lightbox, page **visually similar** photos
  (cosine over the stored SigLIP image embeddings) or, per face, **other shots of
  the same person** (cosine over the ArcFace face embeddings) — the latter works even
  for an *unnamed* face the person filter can't reach. Neither needs a model download.
- **Memories, Trips & Favorites** — an "On this day" **Memories** tab, auto-detected
  **Trips** (split on capture-time gaps + GPS jumps), and a ★ **Favorites** star with
  its own quick filter. Plus **Smart albums** — save any filter set by name and reload
  it later.
- **Multi-select & bulk actions** — a Select mode turns the grid into a multi-select;
  apply **favorite / hide** to a whole set, **export** the originals to a folder, or
  (from the Duplicates tab) delete redundant shots.
- **Offline reverse geocoding** — GPS EXIF → city + country using a bundled
  dataset (no network). Install `reverse_geocoder` for ~150k-city resolution.
- **Rich metadata** — capture date (EXIF, with file-mtime fallback), camera,
  dimensions, GPS, thumbnails.
- **Folder-name mining** — for libraries organised like `2012/2012_05_Sardegna`,
  the year, month and trip/place label are recovered from the folder names and
  used to fill in dates EXIF lacks (synthesised as `YYYY-MM-01`,
  `taken_source='folder'`) and to add a filterable **Place** facet. EXIF always
  wins; only dated folders (with a 4-digit year) are trusted.
- **Web UI** — a virtualised gallery with lazy thumbnails and a detail lightbox
  (zoom/pan, slideshow, EXIF panel), **Memories / Trips / Duplicates / Map** tabs, an
  interactive map of your geotagged photos, a People page and a "Name faces" workflow.
  No build step (vanilla JS); Leaflet is vendored locally.
- **Everything local** — a single SQLite catalog plus thumbnails/face crops
  under `~/.photo_atlas`. Your photos never leave your machine. (The map fetches
  basemap tiles from OpenStreetMap, so that view needs network — but only tile
  coordinates are requested; no photo or location data is uploaded.)

## Install

Photo Atlas uses [uv](https://docs.astral.sh/uv/) for environment and
dependency management:

```bash
uv sync                     # core app + test tooling into .venv (dev group)
uv sync --extra geo         # + high-resolution offline reverse geocoding
uv sync --extra heic        # + HEIC/HEIF decoding (default iPhone format)
uv sync --extra dlib        # + face_recognition (dlib) backend (needs CMake/C++)
uv run photo-atlas --help   # run the CLI inside the managed environment
uv run pytest               # run the test suite
```

`uv` resolves a compatible Python (3.11+) automatically and records exact
versions in `uv.lock`. Plain `pip install -e .` still works for the runtime
package if you prefer to manage your own environment.

The YuNet + ArcFace R100 ONNX weights (~260 MB) are downloaded on first use to
`~/.photo_atlas/models`. For offline/air-gapped setups, point
`PHOTO_ATLAS_YUNET` / `PHOTO_ATLAS_ARCFACE` at local model files (or
`--faces sface` for the lighter ~38 MB YuNet + SFace pair, overridable via
`PHOTO_ATLAS_SFACE`).

### Scene tagging

Scenes are tagged **zero-shot** by a small **SigLIP 2** vision encoder (a modern
CLIP successor that beats CLIP at zero-shot for its size) — no training, no
PyTorch. It runs on every index automatically:

```bash
photo-atlas index ~/Pictures
```

It tags the classes `people`, `animals`, `landscape`, `plants`, `food`,
`vehicle`, `building`, `document`, `screenshot` (plus the catch-all `other`),
which show up as options in the scene filter.

Tuning the tagger doesn't need a full re-index — scene tags are independent of
faces/thumbnails, so re-tag in place:

```bash
photo-atlas retag-scenes   # recompute just the scene column
```

Only SigLIP 2's *vision* tower runs at index time (a ~90 MB quantised ONNX at
256² resolution, downloaded on first use via [ONNX Runtime](https://onnxruntime.ai/)
— **no PyTorch**). The per-label *text* embeddings are pre-computed once and
shipped as a tiny bundled matrix (`data/scene_labels.npz`), so there is no text
encoder or tokenizer at runtime — just a dot product. To swap in a different /
newer encoder (e.g. a larger SigLIP 2 or MobileCLIP2), rebuild that matrix with
`scripts/build_scene_embeddings.py --model <hf-repo>` and point
`PHOTO_ATLAS_SCENE_MODEL` at the matching vision ONNX (and
`PHOTO_ATLAS_SCENE_INPUT_SIZE` if its resolution differs from 256).

### Semantic search

The same SigLIP space powers **natural-language search**: store each photo's
image embedding, then embed a free-text query and rank by cosine similarity.

```bash
photo-atlas index ~/Pictures --embed     # store embeddings as you index
# already indexed? backfill without re-detecting faces (decodes once):
photo-atlas embed
```

Then in `serve`, the search bar grows a ✨ **Smart** toggle: type "kids on the
beach at sunset" or "my red car" and the grid re-ranks by visual relevance.
Semantic queries combine with every other filter (person, place, date, scene…)
— the structured filters narrow the set, the query orders what's left — and are
capped at the most relevant `config.semantic_top_k` (default 200) matches.

**Hybrid person + visual queries.** SigLIP has no idea *who* "Stefano" is —
identity is the face pipeline's job — so a query like "Stefano eating food" is
**decomposed** server-side (`planner.py`) rather than fed whole to the model:
known person names are peeled into a person filter (multiple names → the People
AND-mode), count phrases like "alone" / "with other people" map to the
number-of-people buckets, and only the residual ("eating food") goes to SigLIP.
The legs are AND-ed, and the UI shows how it split your words. So "Stefano alone"
is answered with zero ML at query time (pure filters), while "Anna at the beach"
filters to Anna and ranks by the beach. The visual score is whole-image, so it
means a photo *containing* Stefano that *looks like* the residual — not that
Stefano is the one eating.

Unlike scene tagging (whose label text is pre-baked), an arbitrary query can't be
precomputed, so semantic search additionally downloads SigLIP's *text* tower and
tokenizer on first use (still **no PyTorch**). Image and text embeddings come from
the same model, so they're directly comparable. Storing one ~768-d float32 vector
per photo adds ~3 KB/photo to the catalog.

## Try it in 30 seconds

```bash
photo-atlas demo      # paint & index a synthetic, geotagged library
photo-atlas serve     # open http://127.0.0.1:8000
```

The demo creates cartoon photos with EXIF dates spread across 2010–2024, GPS
near real cities, and three recurring "people" so you can exercise filtering,
clustering and naming without any real photos.

## Use it on your photos

```bash
photo-atlas index ~/Pictures        # walk the tree, extract metadata + faces
photo-atlas cluster                 # group the unnamed faces
photo-atlas serve                   # browse, filter, and name people
photo-atlas embed                   # backfill SigLIP embeddings for semantic search
photo-atlas dedup                   # backfill perceptual hashes for the Duplicates tab
photo-atlas retag-scenes            # recompute scene tags in place (no re-index)
photo-atlas stats                   # quick catalog summary
photo-atlas prune                   # reconcile: drop dead rows + sweep orphaned derivatives
photo-atlas export-labels           # write person names to portable XMP sidecars
```

`index` is incremental and crash-safe — already-known photos are skipped (use
`--recompute` to force), so an interrupted run resumes cleanly on the next invocation.
Add `--prune` to reconcile in one step: drop rows for deleted/moved files and sweep
orphaned thumbnail/preview/crop files (otherwise `photo-atlas prune` does the same on
demand). Choose the face backend with
`--faces {auto,arcface,yunet,sface,dlib,synthetic,none}`
(default `auto` → YuNet + ArcFace R100; `sface` for the legacy 128-d recogniser).
Indexing fans out over worker processes
(`--workers N`, default = CPU count; `--workers 1` for serial): each file is
decoded once and face detection runs on a downscaled copy, while the single main
process performs all database writes. Byte-identical duplicates (same photo in
two folders) are detected by SHA-1 and skipped; *near*-duplicates (bursts,
re-saved copies) are grouped in the **Duplicates** tab via a perceptual hash
computed at index time (run `photo-atlas dedup` to backfill it on a library
indexed before this feature). Video files are catalogued when
`ffmpeg`/`ffprobe` are on `PATH` — a poster frame plus the capture date/GPS are
extracted so clips are browsable (and playable) alongside photos; without ffmpeg
they're counted but skipped.

> **HEIC needs the `heic` extra.** iPhone HEIC photos (often a fifth of a
> library) only decode — for thumbnails *and* face detection — once
> `pillow-heif` is installed (`uv sync --extra heic`).

> **City labels need the `geo` extra.** Without `reverse_geocoder`
> (`uv sync --extra geo`) GPS is matched against a tiny bundled table and
> `index` warns that city/country labels will be coarse.

`export-labels` writes a `<photo>.xmp` sidecar next to each photo that has named
people (or use `--dest DIR`), recording the names as `dc:subject` + `People|Name`
keywords so the naming work survives a catalog loss and is read by digiKam,
Lightroom and Bridge. Originals are never modified.

## How it works

```
indexer ─┬─ metadata.py    EXIF date / camera / GPS  + thumbnails
         ├─ folder_meta.py year / month / place mined from folder names
         ├─ geocode.py     GPS → city, country (offline nearest-city)
         ├─ faces.py      YuNet detect → ArcFace embed → DBSCAN cluster
         ├─ classify.py   scene tag (SigLIP 2 zero-shot)
         ├─ embed.py      SigLIP 2 image/text embeddings for semantic search
         └─ db.py         SQLite catalog (photos / persons / faces)

api.py (FastAPI)  →  web/  (gallery · filters · people · name-faces · ✨ smart search)
   ├─ search.py    filter→SQL + SigLIP relevance ranking (SemanticIndex)
   └─ planner.py   decompose NL queries → person/people filters + visual text
```

- **Recognition.** When new photos are indexed, each detected face is matched by
  **k-nearest-neighbour vote** against every already-named face: the `recognition_k`
  closest enrolled faces within `face_match_threshold` cosine distance vote, and
  the majority person wins. Matching the nearest individual examples (rather than a
  single averaged "centroid") is more robust when a person's look drifts over years.
- **Clustering.** Unnamed faces are grouped with DBSCAN over cosine distance
  (`cluster_eps`, `cluster_min_samples`) so you can name a whole group at once.
- **Tunables** live in `photo_atlas.config.AtlasConfig`.

## API

`photo-atlas serve` exposes a small JSON API (see `src/photo_atlas/api.py`):

| Method & path | Purpose |
| --- | --- |
| `GET /api/facets` | counts for the filter sidebar |
| `GET /api/capabilities` | feature flags for the UI (e.g. `{"semantic": true}`) |
| `GET /api/photos?person_id=&scene=&country=&city=&place=&year=&camera=&q=` | filtered list |
| `GET /api/photos?text=...` | natural-language semantic search (ranked; ANDs with the filters) |
| `GET /api/map?...` | geotagged `{id, lat, lon, year}` points for the map (same filters) |
| `GET /api/photos/{id}` | photo detail + faces |
| `GET /api/photos/{id}/similar`, `GET /api/faces/{id}/similar` | "more like this" (visual) · "more of this person" (SFace) |
| `GET /api/memories?month=&day=`, `GET /api/trips`, `GET /api/duplicates` | On-this-day · auto-detected trips · near-duplicate/burst groups |
| `GET /api/exif/{id}` | on-demand capture settings for the lightbox info panel |
| `GET /api/image\|preview\|thumb/{id}`, `GET /api/face/{id}` | media (preview = bounded lightbox derivative; `thumb?size=` for retina) |
| `PUT /api/photos/{id}/favorite`, `POST /api/photos/bulk` | star a photo · bulk favorite/hide a selection |
| `POST /api/photos/export`, `POST /api/photos/delete` | copy a selection's originals to a folder · hard-delete (rows + files) |
| `GET/POST /api/albums`, `DELETE /api/albums/{id}` | smart albums (saved searches) |
| `GET /api/persons`, `PATCH/DELETE /api/persons/{id}` | manage people |
| `GET /api/persons/{id}/faces`, `PUT .../cover`, `POST .../merge` | cover picker · merge two people |
| `GET /api/clusters`, `POST /api/clusters/{id}/assign` | name a face group |
| `POST /api/faces/{id}/assign\|unassign`, `GET /api/faces/review` | name/unname a face · low-confidence review queue |

State-changing endpoints (POST/PUT/PATCH/DELETE) are guarded against cross-origin
requests; see `_same_origin_writes` in `api.py`.

## Tests

```bash
uv run pytest                            # offline suite (deterministic, no network)
uv run pytest tests/test_deep_faces.py   # deep YuNet/ArcFace test on real faces*
```

The offline suite is deterministic, runs on every push/PR via GitHub Actions
(Python 3.11–3.12) and enforces **≥80 % coverage** (`--cov-fail-under=80`); it
currently sits near 98 %. A coverage summary prints after each run.

\* downloads the models + a few public sample faces; **skips** (never fails)
when offline. It asserts the deep model puts the same person far closer than two
different people, and that clustering groups repeat photos of one identity.

## Notes

- Built and verified against `opencv-python 4.11–4.13`, which ships the DNN face
  module (`FaceDetectorYN`; `FaceRecognizerSF` for the legacy `sface` backend).
  The code targets `opencv>=4.10` and is forward compatible. ArcFace recognition
  runs on ONNX Runtime, so it needs no OpenCV recognition module.
- **OpenCV 5 (as of June 2026):** still not released on PyPI — 4.13.0 (Dec 2025)
  is the latest published wheel and the 5.0 milestone is feature-complete but
  unreleased. When 5.0 lands it keeps the same `FaceDetectorYN`/`FaceRecognizerSF`
  API, so no code change is expected; the YuNet zoo weights are unchanged.
  Note: `opencv-python-headless` and `onnxruntime` both dropped macOS-13 Intel
  wheels (at 4.13 / 1.24 respectively) — see the uv constraints in `pyproject.toml`.
- **Recognition backbone.** The default is [ArcFace R100](https://github.com/deepinsight/insightface)
  (glint360k, 512-d) via the InsightFace antelopev2 ONNX, aligned with YuNet's
  landmarks to ArcFace's 112² template. It separates identities better than the
  legacy 128-d SFace (~0.03 vs ~0.05 same-person; ~1.0 vs ~0.9 different) and runs
  on ONNX Runtime with no dlib/CMake build. SFace stays available via
  `--faces sface`. A still-stronger detector (SCRFD) is the next option if hard-face
  *recall* ever proves the bottleneck — see [`MODELS.md`](MODELS.md).
