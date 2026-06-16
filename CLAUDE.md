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
matrix (Python 3.10/3.11/3.12) running `uv run pytest -q`. Always run all three
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
- Models (YuNet/SFace face stack, SigLIP vision/text towers, tokenizer) download on
  demand to `~/.photo_atlas/models`; each has a `PHOTO_ATLAS_*` env override for
  offline use. Add new downloads via `models._resolve`.

## Module map (`src/photo_atlas/`)

| Module | Role |
| --- | --- |
| `cli.py` | argparse entry point (`index`, `embed`, `cluster`, `retag-scenes`, `serve`, `stats`, `prune`, `export-labels`, `demo`) |
| `config.py` | `AtlasConfig` — library paths + tunables (`~/.photo_atlas`, `PHOTO_ATLAS_HOME`) |
| `db.py` | SQLite schema, additive migrations (`_migrate`), embedding (de)serialisation. `PHOTO_COLUMNS` is the single source of truth for writable photo columns |
| `indexer.py` | the ingest pipeline; decode-once per file, fan-out over a `ProcessPoolExecutor` (main process does all DB writes). Also `embed_library`, `retag_scenes`, `prune_library`, `cluster_library` |
| `metadata.py` | EXIF/dimensions/thumbnails, `cached_resized` derivatives (atomic temp+replace), HEIF opener |
| `faces.py` | YuNet detect + SFace embed backends, DBSCAN clustering, k-NN recognition (`Enrollment`) |
| `classify.py` | scene tagging: SigLIP-only `ZeroShotSceneTagger` (shares the vision encoder with embeddings) |
| `embed.py` | `SigLipImageEncoder` / `SigLipTextEncoder` for semantic search |
| `search.py` | filter dict → SQL (`_where`), facets, plus `SemanticIndex` + `semantic_search` (cosine ranking ANDed with filters) |
| `planner.py` | model-free decomposition of NL queries → person/people filters + residual visual text |
| `geocode.py` / `folder_meta.py` | GPS→city/country; year/place mined from folder names |
| `library.py` | person/cluster management (rename/merge/cover/assign) |
| `api.py` | FastAPI app (`create_app`); media + JSON endpoints; cross-origin write guard; caches the semantic index + text encoder |
| `models.py` | on-demand model downloads (face + SigLIP) with env overrides |
| `web/` | `index.html` + `app.js` + `styles.css`, Leaflet vendored locally; **no build step** |

## Conventions & gotchas

- **DB writes funnel through one connection.** In parallel indexing, workers only do
  CPU-bound prep (`_prepare_photo`, picklable `_PreparedPhoto`); the main process
  commits. Worker-built objects (ONNX sessions) aren't pickled — they're built in
  `_worker_init`.
- **Photo embeddings live in `photos.embedding`/`embed_dim` but are deliberately
  NOT in `PHOTO_COLUMNS`** — they'd bloat the grid/list payload. They're written
  separately (`db.set_photo_embedding`) and loaded by `SemanticIndex`.
- **Facet filters accept a scalar or a list** (OR within a facet, AND across facets).
  Semantic search is a *ranking* layered on top, via the `text` query param.
- **Frontend has no test runner**; pure JS helpers (grid windowing, URL state) are
  exercised by Node harnesses in `tests/js/` via `tests/test_web_js.py`, which skips
  when Node isn't installed.
- Backlog lives in `TODO.md`; performance notes in `PERFORMANCE.md`.

## Git

Develop on a feature branch, never commit directly to `main`. Run ruff + mypy +
pytest before every commit. Open PRs against `main`; don't create a PR unless asked.
