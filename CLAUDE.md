# CLAUDE.md

Guidance for Claude Code working in this repo. Keep it accurate — update it when
the architecture or workflow changes.

## What this is

Photo Atlas: a local-first tool to navigate years of photos by **person, scene,
place, date** and **natural-language semantic search**. A CLI indexes a photo
tree into a single SQLite catalog (+ thumbnails/face crops/previews) and a
FastAPI app serves a no-build-step vanilla-JS web UI. Everything runs locally;
photos never leave the machine.

## Environment & commands

Uses [uv](https://docs.astral.sh/uv/). The `dev` dependency group (pytest, httpx,
ruff, mypy) is installed by default.

```bash
uv sync                       # core app + dev tooling into .venv (incl. SigLIP/ONNX)
uv sync --extra geo|heic|dlib # other optional features
uv run pytest -q              # full test suite (coverage gate: --cov-fail-under=80)
uv run ruff check src tests   # lint (must pass)
uv run mypy                   # type check, src only (must pass)
uv run photo-atlas --help     # the CLI
```

CI mirrors this: a `lint` job (`ruff check src tests` + `mypy`) and a `test`
matrix (Python 3.11/3.12) running `uv run pytest -q`. Always run all three
locally before committing.

In the remote (web) environment a SessionStart hook (`.claude/hooks/session-start.sh`)
pip-installs `.[dev]` and exports `PYTHONPATH=src`.

## Golden rules

- **Tests must stay offline and green.** No network, no model downloads in the
  default suite. For SigLIP/ONNX code, either test the pure logic with hand-built
  vectors / **stub encoders**, or gate a live round-trip behind an env var (e.g.
  `PHOTO_ATLAS_SCENE_MODEL`) with `pytest.mark.skipif`. Scene tagging is now
  SigLIP-only, so any path that *indexes* must inject the picklable
  `tests/scene_stub.StubTagger` (an autouse conftest fixture patches
  `indexer.get_tagger` for in-process paths; the parallel/spawn path takes
  `index_path(..., tagger=StubTagger())` since a monkeypatch can't cross processes).
  `onnxruntime`/`tokenizers` are core deps but imported lazily inside
  functions/`__init__`, never at module top level, so importing a module never
  triggers a model download.
- **Coverage gate is 80%** (enforced via pyproject `addopts`). New code needs tests.
- **ruff + mypy clean.** ruff selects `E,F,I,B,UP`, line-length 100; `B008` is
  ignored (FastAPI's `Depends()`/`Query()` default idiom). mypy checks `src/` only.
- Models (YuNet detector + **ArcFace R100** recogniser by default, SigLIP 2
  vision/text towers, tokenizer) download on demand to `~/.photo_atlas/models`;
  each has a `PHOTO_ATLAS_*` env override for offline use (incl.
  `PHOTO_ATLAS_ARCFACE`, `PHOTO_ATLAS_SCENE_INPUT_SIZE`). Add new downloads via
  `models._resolve`. The legacy SFace recogniser stays selectable (`--faces sface`).

## Module map (`src/photo_atlas/`)

| Module | Role |
| --- | --- |
| `cli.py` | argparse entry point (`index`, `embed`, `dedup`, `cluster`, `retag-scenes`, `serve`, `stats`, `prune`, `export-labels`, `demo`) |
| `config.py` | `AtlasConfig` — library paths + tunables (`~/.photo_atlas`, `PHOTO_ATLAS_HOME`) |
| `db.py` | SQLite schema, additive migrations (`_migrate`), embedding (de)serialisation. `PHOTO_COLUMNS` is the single source of truth for writable photo columns |
| `indexer.py` | the ingest pipeline; decode-once per file, fan-out over a `ProcessPoolExecutor` (main process does all DB writes). Also `embed_library`, `backfill_phashes`, `retag_scenes`, `prune_library`, `delete_photos` (hard delete: rows + files + derivatives), `export_photos` (copy a selection's originals to a folder), `cluster_library` |
| `metadata.py` | EXIF/dimensions/thumbnails, `cached_resized` derivatives (atomic temp+replace), HEIF opener. Resolves `taken_at` as **exif → filename → mtime** (folder hint slots in via the indexer) |
| `filename_date.py` | `parse_filename_date` — ordered regex registry mining a capture date/time from real filename conventions (Android/iOS/WP/WhatsApp/`YYYY-MM-DD HH.MM.SS`/bare compact/Italian text), validated against a sane calendar range so counters/resolutions aren't misread |
| `video.py` | optional ffmpeg/ffprobe poster-frame + capture-date/GPS extraction for videos (`index_video`); pure `_parse_probe` is the offline-testable seam |
| `faces.py` | shared `_YuNetBackend` detection + `YuNetArcFaceBackend` (default, 512-d via onnxruntime, 5-pt `norm_crop` alignment) / `YuNetSFaceBackend` (legacy 128-d) embed backends, DBSCAN clustering, negative-aware k-NN recognition (`Enrollment` carries positives + "not this person" negatives) |
| `classify.py` | scene tagging: SigLIP 2-only `ZeroShotSceneTagger` (shares the vision encoder with embeddings) |
| `embed.py` | `SigLipImageEncoder` / `SigLipTextEncoder` for semantic search; `configure_text_tokenizer` honours the tokenizer's embedded pad config (SigLIP 2 Gemma) |
| `search.py` | filter dict → SQL (`_where`), facets, plus `SemanticIndex`/`semantic_search` (cosine ranking ANDed with filters), `FaceIndex`/`similar_faces` ("more like this person" over ArcFace embeddings), trip/memory grouping, and `find_burst_groups` (perceptual+temporal near-duplicate detection) |
| `planner.py` | model-free decomposition of NL queries → person/people filters + residual visual text |
| `geocode.py` / `folder_meta.py` | GPS→city/country; year/place mined from folder names (incl. a month from a yearless named-month subfolder, e.g. `2026/01-gennaio`) |
| `library.py` | person/cluster management (rename/merge/cover/assign) |
| `api.py` | FastAPI app (`create_app`); media + JSON endpoints; cross-origin write guard; caches the semantic index + text encoder |
| `models.py` | on-demand model downloads (YuNet/ArcFace/SFace + SigLIP 2) with env overrides; `ensure_arcface_models` (default), `ensure_models` (SFace), `ensure_scene_input_size` |
| `web/` | `index.html` + `app.js` + `styles.css`, Leaflet vendored locally; **no build step** |

## Conventions & gotchas

- **DB writes funnel through one connection.** In parallel indexing, workers only do
  CPU-bound prep (`_prepare_photo`, picklable `_PreparedPhoto`); the main process
  commits. Worker-built objects (ONNX sessions) aren't pickled — they're built in
  `_worker_init`.
- **Photo embeddings live in `photos.embedding`/`embed_dim` but are deliberately
  NOT in `PHOTO_COLUMNS`** — they'd bloat the grid/list payload. They're written
  separately (`db.set_photo_embedding` from `embed`, or `db.set_photo_embedding_blob`
  from the index pipeline's pre-serialised blob — **both** bump
  `meta['embeddings_version']`) and loaded by `SemanticIndex`. The API caches that index
  keyed on `(count, max_id, embeddings_version)`, so an in-place `embed`/`index --embed
  --recompute` (same count + max id) still invalidates the cache and a running `serve`
  reloads it.
- **The perceptual hash (`photos.phash`, dHash hex) is also out of `PHOTO_COLUMNS`**
  (written via `db.set_phash`, kept off `_LIST_COLUMNS`), but unlike embeddings it's
  *always* computed at index time (it's cheap) — `_commit_prepared` refreshes it on
  every re-index, and `indexer.backfill_phashes` (CLI `dedup`) fills it for older
  catalogs. `search.find_burst_groups` reads it for the Duplicates tab; videos carry
  no phash so they fall out naturally.
- **`favorite`, `is_video` and `hidden` are also kept out of `PHOTO_COLUMNS`** so a
  re-index (an `ON CONFLICT DO UPDATE` over only those columns) never resets them; all
  are appended to `_LIST_COLUMNS` by hand and written via their own UPDATEs (bulk via
  `db.set_*_bulk`, the `POST /api/photos/bulk` multi-select action). `hidden` is a
  tri-state `_where` filter — absent (no clause, keeps `_where({})` empty), `False`
  (exclude, the API browsing default), `True` (only hidden, the 🙈 review chip). Videos are
  ingested by `index_video` (poster frame stored content-addressed under
  `posters_dir`; `path` stays the playable file so `/api/image` streams it), gated on
  `video.ffmpeg_available()` — no ffmpeg means videos are counted but not indexed.
  `extract_poster` writes the poster to an atomic `.part` temp, so it **must** pass
  `-f image2` to pin the muxer (ffmpeg otherwise infers the format from the `.part`
  extension and fails); the `at=` seek falls back to the opening frame for short clips.
- **Facet filters accept a scalar or a list** (OR within a facet, AND across facets).
  Semantic search is a *ranking* layered on top, via the `text` query param.
- **Face active learning:** correcting an auto-tag (un/reassign) records a
  `face_negatives` row; `_load_enrollment` feeds those into `Enrollment` so
  `knn_person_match` penalises a rejected identity. A human `assign_face` sets
  `confidence=1.0`; the "Review guesses" list (`/api/faces/review`) shows auto-tags
  below `config.review_confidence`. **`index --recompute` preserves manual naming:**
  before `replace_faces` blanket-deletes a photo's faces, `_carry_human_labels`
  (IoU-matched by bbox) copies any prior `confidence>=1.0` assignment onto the
  re-detected face, so re-indexing refreshes detection/embeddings without wiping names.
- **Frontend has no test runner**; pure JS helpers (grid windowing, URL state) are
  exercised by Node harnesses in `tests/js/` via `tests/test_web_js.py`, which skips
  when Node isn't installed.
- Backlog lives in `TODO.md`; performance notes in `PERFORMANCE.md`.

## Git

Develop on a feature branch, never commit directly to `main`. Run ruff + mypy +
pytest before every commit. Open PRs against `main`; don't create a PR unless asked.
