# Photo Atlas — Browse & Filter UX backlog

Improvements identified during the UX review. The "Core UX" bundle below is
**done**; the rest is queued for later. Browser-only is the target, so the
responsive/mobile work is intentionally deprioritised.

## Done (core UX bundle)
- [x] **Infinite scroll** — grid pages through the whole library via `offset`
  instead of hard-capping at 120 photos.
- [x] **Lightbox navigation** — `←/→` step between photos, `Esc` closes, plus
  on-screen prev/next arrows (disabled at the ends).
- [x] **Active-filter pills** — applied filters show as removable pills above
  the grid, with a "Clear all".
- [x] **Broader search** — `q` now matches filename, city, country, place
  label, folder/trip and camera make/model (was filename-only).
- [x] Minor: loading row, Enter-to-save on lightbox face inputs.
- [x] **Filter-aware facet counts.** `/api/facets` now accepts the active
  filters and each facet's counts reflect the other filters but not its own
  dimension (Photos-app style); `total` stays the library size. Sidebar passes
  the current filters and refreshes counts as you filter/search.

## Queued

### Filtering correctness & power
- [x] **Multi-select within a facet.** Facet filters (`person_id`, `scene`,
  `country`, `city`, `place`, `year`, `camera`) accept a value or a list:
  OR-within-facet, AND-across-facets. SQL uses `IN (...)` (and OR-of-LIKE for
  camera, an `IN` join for people); the API takes repeated query params; the
  sidebar toggles values and each gets its own removable pill.
  Possible follow-up: an *AND* mode for people ("photos containing A **and** B").
- [x] **Surface `has_faces`.** "👤 Has people" quick-filter chip in the sidebar;
  facets now return a filter-aware `with_faces` count for it.
- [x] **Date range.** Two inclusive `date_taken` inputs (bounded by the
  library's `date_min`/`date_max`); `_where` compares on the date prefix so
  same-day photos with a time component are included. A timeline/scrubber is a
  possible future upgrade.
- [x] **Facet "show more".** Sidebar sections cap at 14 items with a
  "+N more / Show less" toggle (per-facet expand state) instead of silently
  truncating.
- [x] **More sort options.** Sort dropdown now offers newest/oldest, filename
  A–Z / Z–A and "recently indexed"; every key carries an `id` tiebreaker so
  `LIMIT/OFFSET` paging stays stable when the sort value ties.

### Richer people / content filters (requested 2026-06-15)
A batch of filtering ideas to make "find this kind of photo" sharper. Note that
**person + place** (and any cross-facet combination) already works today — facets
AND across dimensions — and **multiple people (OR)** is covered by multi-select;
these items are the genuinely new pieces.
- [x] **People AND-mode.** `person_mode=all` matches only photos containing
  *every* selected person — one AND-ed `EXISTS` per person — vs the default `any`
  (a single `EXISTS … IN (…)`). The People facet shows a "Match: any/all of them"
  toggle once 2+ people are selected; the mode round-trips in the URL as a scalar
  modifier (no pill). Unit + DB tested.
- [x] **Filter by number of people portrayed (incl. portrait / group).** A
  "Number of people" facet buckets photos by `face_count` (`0` / `1` / `2-4` / `5+`)
  via a SQL `CASE` — no schema change, no re-index. Bucket `1` is a **portrait** and
  `2-4`/`5+` are **groups**, so this also covers the portrait/group picture-type ask.
  Implemented as `search.PEOPLE_BUCKETS` + a filter-aware `people` facet; the API
  takes repeated `people=` params and the sidebar renders friendly bucket chips with
  removable pills. Unit + DB + API (chip-count == result-count) tested.
  Possible follow-up: fold the scene tags and these people buckets into one unified
  "type of picture" facet.
- [x] **Filter by number of *known* (named) people.** A "Known people" facet
  buckets photos by how many faces are assigned to a named person
  (`0` / `1` / `2+`), via a correlated subquery over `faces` (cheap through
  `idx_faces_photo`). `search.KNOWN_BUCKETS` + a filter-aware `known` facet; API
  takes repeated `known=` params; sidebar shows friendly chips/pills. Unit + DB +
  API (chip-count == result-count) tested.
  Perf follow-up: if this ever lands on a hot path, denormalise a
  `named_face_count` column (maintained on assign/unassign/merge/delete) to drop
  the per-row subquery.

### Performance & memory at scale
- [x] **Virtualize / window the photo grid.** Implemented option 3 (true
  windowing): the grid is now a positioned canvas whose height spans the whole
  result set, and `renderWindow` keeps only a viewport-sized window (+4 buffer
  rows) of absolutely-positioned, recycled card nodes in the DOM. Node count and
  decoded-bitmap memory stay flat regardless of library size; scroll/resize are
  rAF-throttled, and infinite-scroll loading + lightbox indexing are preserved.
  The layout math (`gridLayout`/`cardOffset`/`windowRange`) is unit-tested via a
  Node harness (`tests/test_web_js.py`). Cards keep `content-visibility: auto`
  as a belt-and-suspenders.
- [x] **Cap the lightbox image size.** The lightbox now loads a bounded
  preview derivative (`preview_size`, default 1600px) from `GET /api/preview/{id}`,
  generated on first request and cached content-addressed under
  `~/.photo_atlas/previews`. The true full-resolution original stays behind a
  "View full size ↗" link to `/api/image/{id}`.
- [x] **Thumbnail `srcset` / sizing.** Thumbnails carry `width`/`height`
  intrinsic hints + `decoding="async"`, and a real `srcset` (`320w` default +
  `640w` retina). The 2x variant is generated and cached on demand via
  `GET /api/thumb/{id}?size=640` (`metadata.cached_resized`, shared with the
  lightbox preview), so hi-DPI screens get crisp thumbs without a re-index.

### Navigation & state
- [x] **URL / history state.** Filters, view and sort are reflected in the
  querystring (`pushState`); the back/forward buttons restore them via
  `popstate`, and a link is shareable/bookmarkable. Covered by a Node-driven
  fake-DOM harness (`tests/test_web_url_state.py`, skips without Node).
- [x] **Infinite scroll near the lightbox end.** Stepping "next" past the last
  loaded photo now pulls the next page (when more remain) and continues; the
  on-screen next arrow stays enabled while more pages exist on the server.

### People / management
- [x] **Rename in the People page.** Each person card has an inline **Rename**
  (Enter saves, Esc cancels) backed by `PATCH /api/persons/{id}`.
- [x] **Merge people** (two clusters of the same person) and **reassign a face**
  to a different/again-unknown person from the lightbox. A card's **Merge**
  control folds it into another person (`POST /api/persons/{id}/merge`); in the
  lightbox, typing a name reassigns a face and a **✕** sends it back to unknown
  (`POST /api/faces/{id}/unassign`).
- [x] **Person cover photo picker.** The **Cover** control lists the person's
  face crops (`GET /api/persons/{id}/faces`) and pins the chosen one
  (`PUT /api/persons/{id}/cover`).

### Robustness & polish
- [x] **Error states.** `api()` is now a thin wrapper that catches network
  failures and non-2xx responses, surfaces them via an `aria-live` toast and
  throws (so callers skip their re-render) instead of breaking silently.
- [x] **Empty-library onboarding.** When the library is genuinely empty (no
  photos and no active filters) a first-run panel shows the
  `photo-atlas index` / `cluster` / `serve` commands; a filtered no-match still
  shows the plain "No photos match" message.
- [x] **Accessibility.** Lightbox is a focus-trapped `role="dialog"` that
  restores focus to the opening card on close; arrow keys no longer hijack
  typing in face inputs. Cards are keyboard-operable (`role=button`, Enter/Space),
  chips expose `aria-pressed`, pills have `aria-label`s, and a visible
  `:focus-visible` outline was added.

### Explicitly deferred (browser-only target)
- [ ] **Responsive / mobile layout.** `.layout` is a fixed `250px 1fr` grid with
  a sticky full-height sidebar; on small screens it needs a collapsible drawer.
  Skipped for now per usage (desktop browser only).

## Correctness / scale / quality (2026-06-15 final review)

A full-app review at ~27k images + ~600 videos drove a round of hardening.

### Done
- [x] **HEIC face detection.** `cv2.imread` can't decode HEIC (pillow-heif only
  patches Pillow), so ~19% of an iPhone library got zero faces. `faces._read_bgr`
  now falls back to Pillow. Needs the `heic` extra installed.
- [x] **Decode-once + downscaled detection.** Each file is decoded a single time
  and reused across metadata/thumbnail/scene/crops (was 4–5×); YuNet detects on a
  ≤1280px copy and maps boxes back. Faster indexing at scale.
- [x] **Geocoder resolution warning.** `index` warns when GPS is matched against
  the bundled ~120-city table (install `--extra geo` for real city labels).
- [x] **Videos surfaced.** Recognised, counted and reported (not catalogued);
  the walk no longer re-ingests the library's own thumbs/crops/previews.
- [x] **`prune`.** Removes catalog rows whose source files were deleted/moved,
  plus the orphaned thumbnail/crops.
- [x] **SHA-1 dedup.** Byte-identical copies (same photo in two folders) are
  skipped instead of duplicated.
- [x] **Map point cap.** Raised 20k → 50k and made configurable
  (`config.map_point_limit`).
- [x] **`export-labels`.** Person names exported to portable XMP sidecars
  (`dc:subject` + `People|Name`), readable by digiKam/Lightroom/Bridge, so the
  naming work survives a catalog loss.
- [x] **ruff + mypy in CI.** Both clean; new `lint` job. JS harness test made
  tolerant of Node's intermittent exit-time SIGSEGV.

### Deferred (bigger design changes)
- [x] **Parallel / multiprocess indexing.** Decode + YuNet inference now fan out
  over a `ProcessPoolExecutor` (`index --workers N`, default = CPU count). Each
  file's CPU-bound work (`_prepare_photo`: decode-once, detect, thumbnail, scene
  tag, crop-encode) runs in a worker; the single main-process SQLite connection
  performs every write (`_commit_prepared`), so there's no DB contention. Only
  `workers*4` files are in flight (bounded memory), commits are batched, the ONNX
  weights are pre-fetched once before fan-out (no download race), and `spawn`
  workers keep OpenCV/ONNX native libs clean. SHA-1 dedup / scan-skip bookkeeping
  stays in the main walk. Serial path (`workers<=1`) preserved for the demo/tests;
  parity is unit-tested (`test_parallel_indexing_matches_serial`).
- [x] **Better scene tagging.** Added an opt-in **zero-shot** tagger
  (`classify.ZeroShotSceneTagger`, `config.scene_backend`, `index --scene`) that
  runs a small **SigLIP** vision encoder (quantised ONNX, ~95 MB) via ONNX Runtime
  — a modern CLIP successor, no PyTorch. Only the vision tower runs at index time;
  the per-label *text* embeddings are pre-computed once
  (`scripts/build_scene_embeddings.py`) and shipped as a tiny bundled matrix
  (`data/scene_labels.npz`), so there's no text encoder/tokenizer at runtime. The
  catch-all `other` is a learned-bias logit (single argmax, no separate threshold);
  a detected face nudges `people`. It also tags a richer class set than the
  heuristic — people/animals/landscape/plants/food/vehicle/building/document/
  screenshot (+other) — which just appear as extra scene-filter options (no schema
  change; the facet is built from DB values). Same `tag()` contract, so the
  indexer/DB/facets are unchanged. The heuristic stays the zero-dep default and the
  fallback when the extra/model isn't present. Each class was validated on real
  ground-truth photos (correct + well separated; out-of-vocabulary images route to
  `other`, and a detected face recovers `people`); scoring logic, label matrix and
  fallback are unit-tested, with an optional live round-trip gated on
  `PHOTO_ATLAS_SCENE_MODEL`. The architecture is model-agnostic — point
  `--model`/`PHOTO_ATLAS_SCENE_MODEL` at a SigLIP 2 or MobileCLIP2 export and
  rebuild the matrix to upgrade.
- [x] **Recognition beyond a single centroid.** Auto-recognition now matches each
  new face by **k-NN majority vote** (`faces.knn_person_match`, `config.recognition_k`,
  default 5) over every named ("enrolled") face, instead of one averaged centroid
  per person — robust when a look drifts over a 15-year span (child→adult, beards,
  glasses), since a far-apart pair of enrolments no longer pulls a centroid into the
  empty space between them. Enrolled faces are loaded once per run (`_load_enrollment`)
  and shipped to the parallel workers as a picklable `Enrollment` (matrix + ids).
  Unit-tested incl. a "centroid would miss, k-NN finds" case.
- [ ] **Video thumbnails/metadata.** Videos are only counted today; extracting a
  poster frame + capture date (ffmpeg) would make them browsable on the timeline
  and map.

## Deep review (2026-06-16): bugs, scale & bold ideas

A full-codebase audit (backend correctness/security, frontend/UX, data model at
scale). Items below were verified against the code; claims that turned out to be
non-issues (int-id file endpoints aren't path-traversable; `known_facet` has no
KeyError; `delete_person` detaching faces is intentional) were dropped.

### Bugs / correctness (verified)
- [x] **EXIF-orientation face crops are sideways.** **Done:** the indexer now bakes
  in the EXIF orientation once at decode (`ImageOps.exif_transpose` in
  `_prepare_photo`) and runs metadata/detect/thumb/crop off that single upright
  image, so portrait-photo face crops are no longer rotated. Covered by
  `tests/test_orientation.py`. (A re-index refreshes pre-existing crops.)
- [x] **Renaming a person to an existing name 500s.** **Done:** `library.rename_person`
  catches `sqlite3.IntegrityError` and raises `ValueError`, which the API maps to a
  clean 409.
- [x] **Stale person cover → broken avatar.** **Done:** `list_persons` now validates
  the pinned cover in-query (the first `COALESCE` arm only uses `cover_face_id` when
  that face still exists, still belongs to the person and still has a crop) and falls
  back to the person's first valid crop otherwise — so a dangling pin no longer
  serves a 404 avatar.
- [x] **Silent face-crop save failure is unrecoverable.** **Done:** a crop write
  error at index time still stores `crop_path=NULL` (the face keeps its embedding +
  bbox), but `/api/face/{id}` no longer 404s forever — when the crop is missing it
  calls `indexer.regenerate_face_crop`, which rebuilds the crop from the source photo
  (re-decoding with the same EXIF-transpose, cropping the stored bbox) and persists
  the new path. This also recovers crops whose files were later deleted. Covered by
  `tests/test_api_errors.py` (regenerate-on-demand + the source-gone 404 path).
- [x] **Thin input validation.** **Done:** `offset` is `Query(0, ge=0)` (and `limit`
  is bounded `ge=1, le=500`); `date_from`/`date_to` are constrained to an ISO
  `^\d{4}-\d{2}-\d{2}$` pattern, so bad params 422 instead of silently mis-filtering.
- [x] **`cached_resized` TOCTTOU.** **Done:** the derivative is written to a
  per-pid `.part` temp file and atomically `os.replace()`d into place, so concurrent
  first-requests can't double-write a half-encoded file.
- [x] **Local API is unauthenticated + CORS-open.** **Done:** a `_same_origin_writes`
  middleware rejects state-changing requests (POST/PUT/PATCH/DELETE) whose `Origin`
  doesn't match the `Host` with a 403, while still allowing same-origin UI calls and
  non-browser clients (no Origin). GETs are never blocked.

### Scale & efficiency
- [x] **Re-tag scenes without a full re-index.** **Done:** the `photo-atlas
  retag-scenes` command (`indexer.retag_scenes`) decodes each still-present photo
  once and upserts only `scene_type`/`scene_scores` (reusing the stored
  `face_count`), so switching heuristic↔zero-shot or tuning it needs no re-detect.
- [ ] **Resumable / crash-safe indexing.** An interrupted run leaves a mixed state
  and orphans; `prune` is a separate manual step. Checkpoint progress and
  auto-prune orphaned rows + derivative files.
- [ ] **Hot-path denormalisation & composite indexes.**
  - [x] **Composite indexes for the browse/filter access patterns.** Added
    `(scene_type, taken_at)` and `(folder_place, taken_at)` on `photos` so a facet
    filter + the default `taken_at DESC` sort is served from one index (no separate
    sort step), and `(person_id, photo_id)` on `faces` so the person `EXISTS` subquery
    is a covering seek (`taken_at` lives on `photos`, so the cross-table
    `(person_id, taken_at)` isn't expressible). Their leading columns supersede the
    old single-column `idx_photos_scene`/`idx_photos_folder`/`idx_faces_person`, which
    are dropped (in `_migrate`, so existing catalogs shed the now-redundant indexes).
    Verified via `EXPLAIN QUERY PLAN`; index presence/migration unit-tested.
  - [ ] **Backfill a `named_face_count` column** (already flagged) to kill the
    per-row "known people" subquery, maintained on assign/unassign/merge/delete.

### Bold features
- [x] ⭐ **Natural-language semantic search** — the headline opportunity now that a
  SigLIP encoder is in the pipeline. **Done (core):** each photo's SigLIP image
  embedding is persisted (new `photos.embedding`/`embed_dim` BLOB columns) at index
  time (`index --embed`) or via a decode-once backfill (`photo-atlas embed`, no face
  re-detect) — and reused for free from the zero-shot scene pass when both run (one
  vision inference per photo). A free-text query is embedded into the same space by a
  new `embed.SigLipTextEncoder` (text tower + tokenizer, downloaded on demand; the
  `scene` extra now also pulls `tokenizers`); `search.SemanticIndex` caches the matrix
  and ranks by cosine, ANDed with the structured filters and capped at
  `config.semantic_top_k`. Exposed as `GET /api/photos?text=` (+ `/api/capabilities`),
  with a ✨ **Smart** toggle on the search bar. Unit + DB + API tested (stub encoders,
  no model download); an optional live round-trip is still gated on the model env vars.
  - [x] **Hybrid person + semantic queries** ("Stefano eating food", "Stefano with
    other people"). **Done:** a small, model-free `planner.plan_query` decomposes the
    text — peels known person-names (`persons` table) → a person filter (2+ names →
    the existing **People AND-mode**), maps "alone / with other people / in a group"
    → the **number-of-people buckets**, and leaves the residual ("eating food", "at
    the beach") for SigLIP. `/api/photos?text=` runs the planner, ANDs the structured
    legs with the visual ranking, and echoes the `plan` back so the UI shows how the
    words were split. A query that reduces to pure filters ("Stefano alone") needs no
    model at all. Caveat unchanged: the visual score is whole-image (photo
    *containing* Stefano that *looks like* the residual), not per-person grounding —
    running SigLIP on the per-person crop is the heavier follow-up. Planner + API
    (person-AND-visual and structured-only) unit-tested.
- [ ] **Near-duplicate & burst grouping.** Only exact SHA-1 dedup exists today; real
  libraries have 5–20-frame bursts. Add a perceptual hash (dHash/pHash) column,
  group near-identical shots, collapse them in the grid behind a "best of N" cover,
  and offer bulk-delete of the rest. Huge real-world decluttering win.
- [ ] **"On this day" / Memories + trip auto-detection.** A 15-year archive is made
  for "this week, 8 years ago". Add day/week-of-year endpoints and auto-group trips
  from date gaps + place/GPS proximity (folders already hint at it).
- [ ] **Favorites + Smart Albums (saved searches).** Star shots (`favorite` column +
  facet) and persist any filter set as a named, shareable album.
- [ ] **Multi-select + bulk actions.** Shift/Ctrl-click in the grid → assign a
  person, favorite, export, or hide a whole selection at once.
- [ ] **"More like this."** Reuse the embeddings we already compute: SFace for
  "same person", SigLIP for "same vibe/scene" — a similarity button in the lightbox.
- [ ] **Face active-learning (negative feedback).** Reassigning/unassigning an
  auto-tag is thrown away today. Record "not this person" negatives and feed them
  into the k-NN vote (penalise), and surface low-confidence auto-tags for review.
- [ ] **Lightbox power tools.** Scroll/drag zoom + pan, an EXIF panel (ƒ/ISO/shutter/
  lens), a slideshow auto-advance, and a `?` keyboard-shortcut legend.
- [ ] **RAW ingest.** A photographer's 15-year library has `.CR2/.NEF/.ARW`; add an
  optional `rawpy` extra to pull the embedded preview + EXIF (currently dropped).
