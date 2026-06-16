# Photo Atlas

Navigate years of photos by **who's in them**, **what they show**, **where**
and **when** — with a modern deep-learning face pipeline, offline reverse
geocoding, automatic scene tagging and a clean web UI.

Point it at a folder (or 15 years of folders), let it build a catalog, then
browse and filter your library and put names to faces. New imports are
recognised automatically once a person has been named.

![pipeline](https://img.shields.io/badge/faces-YuNet%20%2B%20SFace-4c8dff) ![offline](https://img.shields.io/badge/geocoding-offline-2ea043)

---

## Features

- **Deep face recognition** — OpenCV's [YuNet](https://github.com/opencv/opencv_zoo)
  detector + [SFace](https://github.com/opencv/opencv_zoo) 128-d embeddings.
  On real photos same-person pairs sit at ~0.05 cosine distance and different
  people at ~0.9, so identities separate cleanly.
- **Name people once** — unrecognised faces are grouped into clusters; name a
  cluster and every matching photo becomes filterable. Future imports are
  auto-recognised against the people you've named.
- **Filter by anything** — person, scene type (`people` / `animals` /
  `landscape` / `plants` / `food` / `vehicle` / `building` / `document` /
  `screenshot` / `other`), country, city, place (trip/folder), year, camera, or
  filename — combined.
- **Natural-language semantic search** — describe a photo in words ("kids on the
  beach at sunset", "my red car") and rank the library by visual similarity — no
  tags, no folders. Each photo's [SigLIP](https://huggingface.co/Xenova/siglip-base-patch16-224)
  image embedding is stored at index time and a free-text query is embedded into
  the same space at search time; the ✨ **Smart** toggle in the search bar switches
  it on, and it ANDs with every other filter.
- **Offline reverse geocoding** — GPS EXIF → city + country using a bundled
  dataset (no network). Install `reverse_geocoder` for ~150k-city resolution.
- **Rich metadata** — capture date (EXIF, with file-mtime fallback), camera,
  dimensions, GPS, thumbnails.
- **Folder-name mining** — for libraries organised like `2012/2012_05_Sardegna`,
  the year, month and trip/place label are recovered from the folder names and
  used to fill in dates EXIF lacks (synthesised as `YYYY-MM-01`,
  `taken_source='folder'`) and to add a filterable **Place** facet. EXIF always
  wins; only dated folders (with a 4-digit year) are trusted.
- **Web UI** — gallery with lazy thumbnails, a detail lightbox, an interactive
  **map** of your geotagged photos, a People page and a "Name faces" workflow.
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

`uv` resolves a compatible Python (3.10+) automatically and records exact
versions in `uv.lock`. Plain `pip install -e .` still works for the runtime
package if you prefer to manage your own environment.

The YuNet + SFace ONNX weights (~38 MB) are downloaded on first use to
`~/.photo_atlas/models`. For offline/air-gapped setups, point
`PHOTO_ATLAS_YUNET` / `PHOTO_ATLAS_SFACE` at local model files.

### Scene tagging

Scenes are tagged **zero-shot** by a small **SigLIP** vision encoder (a modern
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

Only SigLIP's *vision* tower runs at index time (a ~95 MB quantised ONNX,
downloaded on first use via [ONNX Runtime](https://onnxruntime.ai/) — **no
PyTorch**). The per-label *text* embeddings are pre-computed once and shipped as
a tiny bundled matrix (`data/scene_labels.npz`), so there is no text encoder or
tokenizer at runtime — just a dot product. To swap in a different / newer
encoder (e.g. SigLIP 2 or MobileCLIP2), rebuild that matrix with
`scripts/build_scene_embeddings.py --model <hf-repo>` and point
`PHOTO_ATLAS_SCENE_MODEL` at the matching vision ONNX.

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
photo-atlas retag-scenes            # recompute scene tags in place (no re-index)
photo-atlas stats                   # quick catalog summary
photo-atlas prune                   # drop entries whose files were deleted/moved
photo-atlas export-labels           # write person names to portable XMP sidecars
```

`index` is incremental — already-known photos are skipped (use `--recompute` to
force). Choose the face backend with `--faces {auto,yunet,dlib,synthetic,none}`
(default `auto` → YuNet/SFace). Indexing fans out over worker processes
(`--workers N`, default = CPU count; `--workers 1` for serial): each file is
decoded once and face detection runs on a downscaled copy, while the single main
process performs all database writes. Byte-identical duplicates (same photo in
two folders) are detected by SHA-1 and skipped; video files are recognised and
reported but not catalogued.

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
         ├─ faces.py      YuNet detect → SFace embed → DBSCAN cluster
         ├─ classify.py   scene tag (SigLIP zero-shot)
         ├─ embed.py      SigLIP image/text embeddings for semantic search
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
| `GET /api/image\|preview\|thumb/{id}`, `GET /api/face/{id}` | media (preview = bounded lightbox derivative; `thumb?size=` for retina) |
| `GET /api/persons`, `PATCH/DELETE /api/persons/{id}` | manage people |
| `GET /api/persons/{id}/faces`, `PUT .../cover`, `POST .../merge` | cover picker · merge two people |
| `GET /api/clusters`, `POST /api/clusters/{id}/assign` | name a face group |
| `POST /api/faces/{id}/assign` | name a single face |

## Tests

```bash
uv run pytest                            # offline suite (deterministic, no network)
uv run pytest tests/test_deep_faces.py   # deep YuNet/SFace test on real faces*
```

The offline suite is deterministic, runs on every push/PR via GitHub Actions
(Python 3.10–3.12) and enforces **≥80 % coverage** (`--cov-fail-under=80`); it
currently sits near 98 %. A coverage summary prints after each run.

\* downloads the models + a few public sample faces; **skips** (never fails)
when offline. It asserts the deep model puts the same person far closer than two
different people, and that clustering groups repeat photos of one identity.

## Notes

- Built and verified against `opencv-python 4.11–4.13`, which ships the DNN face
  module (`FaceDetectorYN` / `FaceRecognizerSF`). The code targets `opencv>=4.10`
  and is forward compatible.
- **OpenCV 5 (as of June 2026):** still not released on PyPI — 4.13.0 (Dec 2025)
  is the latest published wheel and the 5.0 milestone is feature-complete but
  unreleased. When 5.0 lands it keeps the same `FaceDetectorYN`/`FaceRecognizerSF`
  API, so no code change is expected; the YuNet/SFace zoo weights are unchanged.
  Note: `opencv-python-headless` dropped macOS-13 Intel wheels at 4.13 (see the
  uv constraint in `pyproject.toml`).
- **Higher-accuracy embeddings.** SFace is 128-d and fast but modest by 2025
  standards. The strongest easy-to-use upgrade is [InsightFace](https://github.com/deepinsight/insightface)'s
  `buffalo_l` pack (ArcFace `w600k_r50`, 512-d, ~99.85 % LFW vs SFace's ~99.6 %),
  which runs via ONNX Runtime with no dlib/CMake build. It would slot in as a new
  `FaceBackend` (detection via its SCRFD + 512-d embeddings); `face_match_threshold`
  / `cluster_eps` would need re-tuning for the new embedding space. Ask if you'd
  like this wired in as an optional `[insightface]` extra.
