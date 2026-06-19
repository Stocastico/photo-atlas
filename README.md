# Photo Atlas

**A local-first photo library you navigate by _who's in them_, _what they show_, _where_ and _when_ — plus natural-language search — all on your own machine.**

Point it at a folder (or 15 years of folders). Photo Atlas builds a single SQLite
catalog with thumbnails, face crops and embeddings, then serves a fast web gallery
where you can filter, search, name people, find duplicates, relive memories and map
your trips. A modern deep-learning face pipeline recognises people you've named in
future imports automatically. Your photos never leave your computer.

![faces](https://img.shields.io/badge/faces-YuNet%20%2B%20ArcFace%20R100-4c8dff)
![search](https://img.shields.io/badge/search-SigLIP%202%20semantic-8957e5)
![geocoding](https://img.shields.io/badge/geocoding-offline-2ea043)
![ui](https://img.shields.io/badge/web%20UI-no%20build%20step-555)
![tests](https://img.shields.io/badge/coverage-%E2%89%A580%25-2ea043)

---

## Table of contents

- [Highlights](#highlights)
- [Quick start (30 seconds)](#quick-start-30-seconds)
- [Installation](#installation) · [the everything-ready setup](#the-everything-ready-setup)
- [Index your library](#index-your-library)
- [Using the web UI](#using-the-web-ui)
- [Search: filters, semantic & grounded](#search-filters-semantic--grounded)
- [Command reference](#command-reference)
- [HTTP API](#http-api)
- [How it works](#how-it-works)
- [Configuration & environment variables](#configuration--environment-variables)
- [Documentation](#documentation)
- [Development & testing](#development--testing)
- [Privacy](#privacy) · [License](#license)

---

## Highlights

| | |
| --- | --- |
| 🧑‍🤝‍🧑 **Deep face recognition** | YuNet detection + **ArcFace R100** 512-d embeddings (glint360k). Same-person pairs sit at ~0.03 cosine distance, different people at ~1.0 — identities separate cleanly. Name a face once and future imports are recognised by **k-NN vote** (robust as a look drifts over years). |
| 🏷️ **Zero-shot scene tags** | A **SigLIP 2** vision encoder tags every photo (`people`, `animals`, `landscape`, `plants`, `food`, `vehicle`, `building`, `document`, `screenshot`, `other`) — no training, no PyTorch. |
| ✨ **Natural-language search** | Describe a photo ("kids on the beach at sunset", "my red car") and rank the library by visual similarity, ANDed with every structured filter. |
| 🎯 **Grounded hybrid queries** | "Stefano eating food" is decomposed into a **person filter** + a **visual residual** that's scored against the region around *Stefano's* face — not the whole frame. |
| 🗺️ **Place, time & trips** | Offline GPS→city/country reverse geocoding, an interactive map, an **On-this-day** Memories tab, and auto-detected **Trips** (split on time gaps + GPS jumps). |
| 🧩 **Near-duplicate & burst grouping** | A perceptual hash groups near-identical shots; keep the best-of-N cover and hide/delete the rest. |
| 🔎 **More like this** | From the lightbox, page visually similar photos, or — per face — more shots of the same person (even for an *unnamed* face). |
| ⭐ **Favorites, albums & bulk actions** | Star photos, save any filter set as a **Smart album**, and multi-select to favorite / hide / **export originals** in one go. |
| 🎞️ **Videos** | With `ffmpeg`/`ffprobe` installed, clips are catalogued with a poster frame + capture date/GPS and play inline. |
| 🔒 **Everything local** | One SQLite catalog + derivatives under `~/.photo_atlas`. No accounts, no cloud, no telemetry. |

---

## Quick start (30 seconds)

No photos handy? Generate a synthetic, geotagged demo library and explore the whole UI:

```bash
uv sync                 # set up the environment (see Installation)
uv run photo-atlas demo # paint & index ~20 cartoon photos with EXIF dates, GPS & 3 recurring "people"
uv run photo-atlas serve # open http://127.0.0.1:8000
```

The demo exercises filtering, clustering, naming, the map, Memories and Trips
without any real photos — a safe way to learn the interface.

---

## Installation

Photo Atlas uses [**uv**](https://docs.astral.sh/uv/) for environment and dependency
management. `uv` picks a compatible Python (3.11+) automatically and pins exact
versions in `uv.lock`.

```bash
uv sync                     # core app + dev tooling (pytest/ruff/mypy) into .venv
```

The core install already includes face recognition, scene tagging and semantic
search (ONNX Runtime + tokenizers are core dependencies — **no PyTorch anywhere**).

### Optional extras

Some formats and backends are opt-in to keep the base install lean:

```bash
uv sync --extra geo         # high-resolution offline reverse geocoding (~150k cities)
uv sync --extra heic        # HEIC/HEIF decoding — the default iPhone format
uv sync --extra dlib        # legacy face_recognition (dlib) backend (needs CMake/C++)
```

> **HEIC is important for iPhone libraries.** Without the `heic` extra, HEIC photos
> (often a fifth of an iPhone library) won't decode for thumbnails *or* face
> detection. Install it with `uv sync --extra heic`.

### The everything-ready setup

To enable **every feature at once** — high-resolution geocoding, HEIC decoding and
the dlib backend — combine the extras and add `ffmpeg` for video support:

```bash
# 1. all optional Python features
uv sync --extra geo --extra heic --extra dlib

# 2. video support (poster frames + capture date/GPS) — system package, not pip:
#    macOS:        brew install ffmpeg
#    Debian/Ubuntu: sudo apt install ffmpeg
#    Windows:      winget install Gyan.FFmpeg   (or scoop install ffmpeg)
```

That's it — semantic search and grounding are on by default and need no extra
package. The first `index` run downloads the model weights (below) on demand.

### Models (downloaded on first use)

| Models | Size | Cached to | Offline override |
| --- | --- | --- | --- |
| YuNet detector + **ArcFace R100** | ~260 MB | `~/.photo_atlas/models` | `PHOTO_ATLAS_YUNET`, `PHOTO_ATLAS_ARCFACE` |
| Legacy YuNet + **SFace** (`--faces sface`) | ~38 MB | ″ | `PHOTO_ATLAS_SFACE` |
| **SigLIP 2** vision tower (scenes + image embeddings) | ~90 MB | ″ | `PHOTO_ATLAS_SCENE_MODEL` |
| **SigLIP 2** text tower + tokenizer (query-time semantic search) | text tower + tokenizer | ″ | `PHOTO_ATLAS_TEXT_MODEL`, `PHOTO_ATLAS_SCENE_TOKENIZER` |

For air-gapped machines, point the env vars at local copies and everything runs
offline. (Plain `pip install -e .` also works if you'd rather manage your own
environment; the extras map to pip extras of the same name.)

---

## Index your library

```bash
# Recommended first run: index, store embeddings for semantic search, and clean up in one pass
photo-atlas index ~/Pictures --embed --prune
photo-atlas cluster          # group the still-unnamed faces so you can name them
photo-atlas serve            # browse, filter, search and name people
```

`index` is **incremental and crash-safe**: already-known photos are skipped (so an
interrupted run resumes cleanly), byte-identical duplicates are detected by SHA-1
and skipped, and a perceptual hash is computed for every photo for the Duplicates
tab. Useful flags:

- `--embed` — also store a SigLIP image embedding per photo (and a region embedding
  per face) so **semantic search** and **grounding** work. Already indexed? Backfill
  later without re-detecting faces: `photo-atlas embed`.
- `--prune` — after indexing, drop rows for deleted/moved files and sweep orphaned
  thumbnail/preview/crop files (otherwise run `photo-atlas prune` on demand).
- `--recompute` — re-index already-known photos (refreshes detection/embeddings;
  **manual face names are preserved**).
- `--faces {auto,arcface,yunet,sface,dlib,synthetic,none}` — face backend
  (default `auto` → YuNet + ArcFace R100).
- `--workers N` — decode/detect across N worker processes (default = CPU count;
  `--workers 1` for serial). Every database write still funnels through one
  connection, so there's no contention.

> **City labels need the `geo` extra.** Without `reverse_geocoder`, GPS is matched
> against a tiny bundled table and `index` warns that city/country labels will be coarse.

After indexing you can keep the catalog fresh without a full re-index:

```bash
photo-atlas embed          # backfill semantic-search embeddings (+ per-face grounding)
photo-atlas dedup          # backfill perceptual hashes for the Duplicates tab
photo-atlas retag-scenes   # recompute scene tags in place
photo-atlas stats          # catalog summary + date-source breakdown
photo-atlas export-labels  # write person names to portable XMP sidecars
```

`export-labels` writes a `<photo>.xmp` sidecar (or use `--dest DIR`) recording names
as `dc:subject` + `People|Name` keywords, readable by digiKam, Lightroom and Bridge,
so naming work survives a catalog loss. Originals are never modified.

---

## Using the web UI

Launch it with `photo-atlas serve` (defaults to `http://127.0.0.1:8000`; override
with `--host`/`--port`). The interface is a no-build-step vanilla-JS app with a
filter sidebar on the left and seven tabs across the top.

### Tabs

- **Photos** — the main virtualised gallery. Only a viewport-sized window of cards
  is ever in the DOM, so it stays fast at 100k+ photos. Infinite-scroll paging, lazy
  thumbnails with retina `srcset`, and a hover/keyboard ★ star on every card.
- **Memories** — "On this day": photos taken on today's calendar date in earlier
  years, one film-strip per year ("3 years ago"). Pick any date.
- **Trips** — auto-detected trips (contiguous runs split on capture-time gaps and big
  GPS jumps), each labelled by place with a "Browse all →" into the grid.
- **Duplicates** — near-identical shots grouped into bursts. Each set pre-keeps a
  best-of-N cover (★) and selects the rest for **Hide** (reversible) or **Delete**
  (irreversible, removes files from disk behind a confirm).
- **Map** — an interactive Leaflet map of your geotagged photos, respecting the
  active filters. Click a point to open the photo.
- **People** — one card per named person, with inline **Rename**, **Merge** (fold one
  person into another) and a **Cover** photo picker.
- **Name faces** — the labelling workflow: unnamed face **clusters** to name in one
  go, plus a **"Review guesses"** queue of low-confidence auto-tags to confirm or
  reject (each correction teaches recognition via negative feedback).

### Filtering

The sidebar shows **filter-aware facet counts** (Photos-app style — each count
reflects the *other* active filters). Filter by **person** (with an any/all toggle for
multiple people), **type of picture** (portrait/group + scene tags), **known-people**
count, **place / city / country**, **year**, a **date range**, **camera**, and quick
chips for **👤 Has people**, **★ Favorites** and **🙈 Hidden**. Active filters appear
as removable pills above the grid with a "Clear all". Sort by newest/oldest, filename
A–Z/Z–A or recently indexed. Everything is reflected in the URL, so any view is
**shareable and bookmarkable** and back/forward works.

### Select mode & bulk actions

Click **Select** to turn grid clicks into a multi-selection (Shift-click extends a
range). The selection bar applies **★ Favorite / Unfavorite**, **🙈 Hide / Unhide**,
or **⬇ Export…** (copy the originals to a folder) to the whole set.

### The lightbox

Click any photo (or press Enter on a focused card) to open the detail lightbox:

- **Navigate** with the on-screen arrows or `←` / `→` (it pages further results at the
  end of the loaded set). `Esc` closes.
- **Zoom & pan** with the scroll wheel or `+` / `−`, `0` to reset, double-click to
  toggle, drag to pan past 1×.
- **Slideshow** — `Space` or the ▶ button auto-advances, pulling further pages.
- **EXIF panel** (ℹ︎) shows aperture / ISO / shutter / focal length / lens on demand.
- **Name faces** inline — type a name on a detected face to assign it (Enter saves), or
  **✕** to send it back to unknown.
- **✨ More like this** pages visually similar photos; the per-face **🧑** button finds
  more shots of that same person.
- **★** stars the photo; **View full size ↗** opens the untouched original.
- Press **?** (or `/`) any time for the keyboard-shortcut legend.

### Keyboard shortcuts (lightbox)

| Key | Action | Key | Action |
| --- | --- | --- | --- |
| `←` / `→` | Previous / next photo | `Space` | Play/pause slideshow |
| `Esc` | Close (legend first, then lightbox) | `+` / `−` | Zoom in / out |
| `?` or `/` | Toggle shortcut legend | `0` | Reset zoom |

---

## Search: filters, semantic & grounded

The search bar has a ✨ **Smart** toggle. Off, it does a fast substring search across
filename, city, country, place label, folder/trip and camera. On, it runs
**natural-language semantic search**.

**Semantic search** embeds your phrase into the same SigLIP 2 space as the stored
photo embeddings and ranks by cosine similarity — so "kids on the beach at sunset" or
"my red car" finds photos by *content*, with no tags or folders. It ANDs with every
structured filter (the filters narrow the set, the query orders what's left) and is
capped at the most relevant `semantic_top_k` (default 200) matches. Requires
embeddings (`index --embed` or `photo-atlas embed`).

**Grounded hybrid queries.** SigLIP has no idea *who* "Stefano" is — identity is the
face pipeline's job — so a query like **"Stefano eating food"** is decomposed
server-side rather than fed whole to the model:

- known **person names** are peeled into a person filter (2+ names → People AND-mode);
- coarse **count phrases** ("alone", "with other people", "in a group") map to the
  number-of-people buckets;
- the **residual** ("eating food") goes to SigLIP.

When the named person has region embeddings, the residual is scored against the
**region around _their_ face** (per-person grounding) rather than the whole frame — so
the photo where Stefano is the one eating outranks one where eating merely happens
elsewhere. The UI shows how your words were split and a **🎯 "focused on the person"**
hint when grounding kicks in. A query that reduces to pure filters ("Stefano alone") is
answered with **zero ML at query time**.

Storing one ~768-d float32 vector per photo adds ~3 KB/photo; per-face region
embeddings add a little more for grounding.

---

## Command reference

```text
photo-atlas index PATH [--embed] [--prune] [--recompute] [--faces …] [--workers N]
photo-atlas cluster                 # group unnamed faces into nameable clusters
photo-atlas embed [--recompute]     # backfill semantic-search + grounding embeddings
photo-atlas dedup [--recompute]     # backfill perceptual hashes (Duplicates tab)
photo-atlas retag-scenes            # recompute scene tags in place (no re-index)
photo-atlas prune                   # drop dead rows + sweep orphaned derivatives
photo-atlas export-labels [--dest]  # write names to portable XMP sidecars
photo-atlas stats                   # catalog summary + date-source provenance
photo-atlas demo                    # generate + index a synthetic library
photo-atlas serve [--host] [--port] # launch the web UI
```

Global `--home DIR` (or `PHOTO_ATLAS_HOME`) selects the library directory
(default `~/.photo_atlas`).

---

## HTTP API

`photo-atlas serve` exposes a small JSON API (see `src/photo_atlas/api.py`):

| Method & path | Purpose |
| --- | --- |
| `GET /api/facets` | counts for the filter sidebar (filter-aware) |
| `GET /api/capabilities` | feature flags for the UI (e.g. `{"semantic": true}`) |
| `GET /api/photos?person_id=&scene=&country=&city=&place=&year=&camera=&q=` | filtered list |
| `GET /api/photos?text=...` | natural-language semantic / grounded search (ranked; ANDs with filters) |
| `GET /api/map?...` | geotagged `{id, lat, lon, year}` points for the map (same filters) |
| `GET /api/photos/{id}` | photo detail + faces |
| `GET /api/photos/{id}/similar`, `GET /api/faces/{id}/similar` | "more like this" (visual) · "more of this person" |
| `GET /api/memories?month=&day=`, `GET /api/trips`, `GET /api/duplicates` | on-this-day · trips · near-duplicate groups |
| `GET /api/exif/{id}` | on-demand capture settings for the lightbox info panel |
| `GET /api/image\|preview\|thumb/{id}`, `GET /api/face/{id}` | media (preview = bounded lightbox derivative; `thumb?size=` for retina) |
| `PUT /api/photos/{id}/favorite`, `POST /api/photos/bulk` | star a photo · bulk favorite/hide a selection |
| `POST /api/photos/export`, `POST /api/photos/delete` | copy a selection's originals · hard-delete (rows + files) |
| `GET/POST /api/albums`, `DELETE /api/albums/{id}` | smart albums (saved searches) |
| `GET /api/persons`, `PATCH/DELETE /api/persons/{id}` | manage people |
| `GET /api/persons/{id}/faces`, `PUT .../cover`, `POST .../merge` | cover picker · merge two people |
| `GET /api/clusters`, `POST /api/clusters/{id}/assign` | name a face group |
| `POST /api/faces/{id}/assign\|unassign`, `GET /api/faces/review` | name/unname a face · low-confidence review queue |

State-changing endpoints (POST/PUT/PATCH/DELETE) are guarded against cross-origin
requests; see `_same_origin_writes` in `api.py`.

---

## How it works

```
indexer ─┬─ metadata.py     EXIF date / camera / GPS  + thumbnails + perceptual hash
         ├─ filename_date.py capture date mined from real filename conventions
         ├─ folder_meta.py   year / month / place mined from folder names
         ├─ geocode.py       GPS → city, country (offline nearest-city)
         ├─ faces.py         YuNet detect → ArcFace embed → DBSCAN cluster → k-NN recognise
         ├─ classify.py      scene tag (SigLIP 2 zero-shot)
         ├─ embed.py         SigLIP 2 image/text + per-face region embeddings
         └─ db.py            SQLite catalog (photos / persons / faces)

api.py (FastAPI)  →  web/  (gallery · filters · people · name-faces · ✨ smart search · 🗺 map)
   ├─ search.py    filter→SQL + SemanticIndex / RegionIndex relevance ranking
   └─ planner.py   decompose NL queries → person/people filters + (grounded) visual text
```

- **Recognition.** Each new face is matched by **k-nearest-neighbour vote** against
  every already-named face (the `recognition_k` closest within `face_match_threshold`
  cosine distance vote). Matching nearest individual examples — rather than one averaged
  centroid — is robust when a person's look drifts over years; corrections feed
  **negative feedback** so a rejected identity is penalised next time.
- **Clustering.** Unnamed faces are grouped with DBSCAN over cosine distance so you can
  name a whole group at once.
- **Decode-once pipeline.** Each file is decoded a single time and reused across
  metadata, faces, thumbnail, scene tag and embeddings; detection runs on a downscaled
  copy. Work fans out over a process pool while one connection owns all DB writes.

---

## Configuration & environment variables

Runtime tunables live in `photo_atlas.config.AtlasConfig` (thumbnail/preview sizes,
recognition/clustering thresholds, trip & duplicate parameters, `semantic_top_k`, …).

| Variable | Purpose |
| --- | --- |
| `PHOTO_ATLAS_HOME` | library directory (default `~/.photo_atlas`) |
| `PHOTO_ATLAS_YUNET` / `PHOTO_ATLAS_ARCFACE` / `PHOTO_ATLAS_SFACE` | local face-model paths (offline) |
| `PHOTO_ATLAS_SCENE_MODEL` / `PHOTO_ATLAS_SCENE_INPUT_SIZE` | SigLIP 2 vision ONNX + its input resolution |
| `PHOTO_ATLAS_TEXT_MODEL` / `PHOTO_ATLAS_SCENE_TOKENIZER` | SigLIP 2 text tower + tokenizer (query-time search) |
| `PHOTO_ATLAS_FFMPEG` / `PHOTO_ATLAS_FFPROBE` | ffmpeg/ffprobe binaries if not on `PATH` |

To swap in a different scene/semantic encoder, rebuild the bundled label matrix with
`scripts/build_scene_embeddings.py --model <hf-repo>` and point the env vars at the
matching ONNX (see [`MODELS.md`](MODELS.md) and [`SIGLIP2_MIGRATION.md`](SIGLIP2_MIGRATION.md)).

---

## Documentation

- [**MODELS.md**](MODELS.md) — the deep-learning models (detection, recognition,
  scene/semantic), why they were chosen, and upgrade/migration mechanics.
- [**PERFORMANCE.md**](PERFORMANCE.md) — indexing throughput, memory behaviour at scale
  and tuning notes.
- [**SIGLIP2_MIGRATION.md**](SIGLIP2_MIGRATION.md) — the SigLIP → SigLIP 2 migration and
  the gaps it closed.
- [**CLAUDE.md**](CLAUDE.md) — architecture map and contributor conventions (the golden
  rules: offline tests, coverage gate, ruff/mypy clean).
- [**TODO.md**](TODO.md) — the full feature backlog and the rationale behind what
  shipped (and what's intentionally out of scope).

---

## Development & testing

```bash
uv run pytest                 # offline suite (deterministic, no network), enforces ≥80% coverage
uv run ruff check src tests   # lint
uv run mypy                   # type-check (src only)
uv run pytest tests/test_deep_faces.py   # optional deep YuNet/ArcFace test on real faces*
```

CI mirrors this on every push/PR via GitHub Actions: a **lint** job (ruff + mypy) and a
**test** matrix (Python 3.11 / 3.12). The offline suite is deterministic and never
touches the network; SigLIP/face code is exercised with stub encoders or gated behind
env vars. Coverage currently sits near 95 %.

\* downloads the models + a few public sample faces and **skips** (never fails) when
offline. It asserts the deep model puts the same person far closer than two different
people, and that clustering groups repeat photos of one identity.

> **OpenCV / wheels.** Built and verified against `opencv-python 4.11–4.13` (ships the
> DNN face module). ArcFace recognition runs on ONNX Runtime, so no OpenCV recognition
> module is needed. `opencv-python-headless` and `onnxruntime` dropped macOS-13 Intel
> wheels at 4.13 / 1.24 — see the uv constraints in `pyproject.toml`.

---

## Privacy

Everything runs locally. Photos, faces, embeddings and the catalog all live under
`~/.photo_atlas` and never leave your machine — there are no accounts, no cloud and no
telemetry. The only network access is **on demand**: model weights are downloaded once
on first use (and can be pre-staged for air-gapped machines, see
[Models](#models-downloaded-on-first-use)), and the **Map** tab fetches OpenStreetMap
basemap tiles — only tile coordinates are requested; no photo or location data is uploaded.

## License

See [LICENSE](LICENSE).
