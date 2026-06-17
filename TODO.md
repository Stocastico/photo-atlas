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

### Won't do (browser-only target)
- [ ] ~~**Responsive / mobile layout.**~~ **Won't do** (2026-06-16). `.layout` is a
  fixed `250px 1fr` grid with a sticky full-height sidebar; on small screens it would
  need a collapsible drawer. The target is a desktop browser only, so this is out of
  scope.

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
- [x] **Video thumbnails/metadata.** **Done:** videos are now ingested as
  browsable rows. A new optional `video.py` (ffmpeg/ffprobe) pulls a full-resolution
  **poster frame** + **capture date**/**GPS** from the container; `indexer.index_video`
  stores a `photos` row with `is_video=1` whose `path` is the playable file and whose
  thumbnail/preview come from the poster (content-addressed under `posters_dir`). The
  indexing walk siphons videos off the worker fan-out and ingests them in the main
  process after the photo pass; without ffmpeg they're counted but not indexed (the old
  behaviour). `is_video` is kept out of `PHOTO_COLUMNS` (like `favorite`) so a re-index
  never clears it. Media endpoints serve the poster's thumb/preview derivatives while
  `/api/image` streams the raw video, so the grid shows a ▶ badge + poster and the
  lightbox plays the clip inline (`<video>`). The pure `ffprobe`-JSON parsing
  (`_parse_probe`/`_parse_iso6709`/`_parse_creation_time`) and the indexing path (via an
  injected stub poster + probe) are tested offline in `tests/test_video.py`, with a live
  ffmpeg round-trip gated on the binary being installed.

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
- [x] **Resumable / crash-safe indexing.** **Done:** indexing is already
  resume-on-rerun (per-file commits + skip-already-indexed by path), so the new
  pieces are orphan reclamation and one-step reconciliation.
  `indexer.sweep_orphan_derivatives` deletes derivative files no row references —
  content-addressed thumbnails / preview+retina variants whose SHA-1 is no longer in
  the catalog, leftover `.part` temps from interrupted atomic writes, and face-crop
  dirs for vanished photo ids — and `prune_library` now runs it after dropping dead
  rows (returns `{removed, kept, orphans}`). `index --prune` folds the whole
  reconciliation into the index run so it's no longer a separate manual step. Unit +
  CLI tested (`tests/test_prune_orphans.py`: referenced-kept/orphan-removed,
  idempotence, prune integration, `--prune` flag).
- [x] **Hot-path denormalisation & composite indexes.**
  - [x] **Composite indexes for the browse/filter access patterns.** Added
    `(scene_type, taken_at)` and `(folder_place, taken_at)` on `photos` so a facet
    filter + the default `taken_at DESC` sort is served from one index (no separate
    sort step), and `(person_id, photo_id)` on `faces` so the person `EXISTS` subquery
    is a covering seek (`taken_at` lives on `photos`, so the cross-table
    `(person_id, taken_at)` isn't expressible). Their leading columns supersede the
    old single-column `idx_photos_scene`/`idx_photos_folder`/`idx_faces_person`, which
    are dropped (in `_migrate`, so existing catalogs shed the now-redundant indexes).
    Verified via `EXPLAIN QUERY PLAN`; index presence/migration unit-tested.
  - [x] **Backfill a `named_face_count` column** to kill the per-row "known
    people" subquery. The column is **trigger-maintained** (AFTER INSERT/DELETE/
    UPDATE-OF-person_id on `faces`), so every write path stays exact — index-time
    auto-recognition, assign/unassign/cluster-assign, merge (named→named, no
    change), delete-person, re-index via `replace_faces`, and prune's FK cascade —
    without any Python call site having to remember to update it. `_migrate` adds
    the column and backfills it once for existing catalogs before the triggers take
    over; `search.KNOWN_BUCKETS` now reads `p.named_face_count` directly instead of
    a correlated subquery. Maintenance across all paths + facet/result agreement +
    migration backfill are unit-tested (`tests/test_named_face_count.py`).

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
- [x] **Near-duplicate & burst grouping.** **Done:** a perceptual hash (dHash,
  `metadata.dhash`) is stored hex in `photos.phash` — computed for free at index time
  (kept out of `PHOTO_COLUMNS` like the embedding, refreshed on every re-index) and
  backfillable for older catalogs via `indexer.backfill_phashes` / the `photo-atlas
  dedup` command. `search.find_burst_groups` groups near-identical shots that are both
  perceptually close (dHash Hamming ≤ `config.dup_max_distance`) **and** captured within
  `config.dup_max_gap_seconds` of a neighbour — a union-find over a time-sorted sliding
  window, so it's `O(N·window)` and a burst survives the odd off frame. Each group picks
  a **best-of-N** cover (favorite → highest resolution → earliest) and is served over
  `GET /api/duplicates`. A new **Duplicates** tab lists each set with the cover pre-kept
  (★) and the rest checkbox-selected for removal: **Hide selected** (reversible, reuses
  the bulk action) or **Delete selected** (irreversible hard delete of rows + source
  files + derivatives via `POST /api/photos/delete` → `indexer.delete_photos`, behind a
  confirm + the same-origin guard). dHash/Hamming, grouping (burst/time-gap/cover/hidden/
  undated), backfill, hard delete and both endpoints are unit + API tested
  (`tests/test_dedup.py`); the CLI backfill in `tests/test_cli.py`. Follow-up: inline
  grid collapse behind the cover (the virtualised grid rewrite) was deferred in favour of
  the dedicated review tab.
- [x] **"On this day" / Memories.** **Done:** a new **Memories** tab surfaces photos
  taken on the same calendar date in earlier years. `search.on_this_day` slices on
  the `taken_at` month/day prefix and groups by year (newest first, each group a
  full `count` + a capped photo sample); `GET /api/memories?month=&day=` defaults to
  the server's current date. The UI renders one horizontal film-strip per past year
  ("3 years ago"), each thumb opening in the lightbox. db + API tested
  (`tests/test_memories.py`, incl. the 422 on a bad date).
  - [x] **Trip auto-detection.** **Done:** a new **Trips** tab auto-groups the library
    into trips. `search.detect_trips` walks every dated photo in order and splits a new
    trip on a capture-time break longer than `config.trip_gap_days` (default 2) *or* a
    GPS jump farther than `config.trip_gap_km` (default 200 km) between consecutive
    geotagged shots — so a same-week hop to a far city reads as its own leg — and drops
    clusters smaller than `config.trip_min_photos` (default 4). Each trip carries its
    date span, photo `count`, a `place` label (most common place label → folder → city/
    country), a GPS centroid + geotagged cover, and a capped photo sample. Derived on
    the fly from `taken_at`/GPS (no schema change, tracks re-indexing for free) and
    surfaced over `GET /api/trips`; the UI shows one film-strip per trip (newest first)
    with a "Browse all →" that loads the whole trip into the grid via its date range.
    db + API tested (`tests/test_trips.py`: time-gap split, far-GPS split, nearby
    no-split, min-photos drop, undated-ignored, label fallback, the endpoint).
- [x] **Favorites.** Star shots: a `favorite` 0/1 column (kept out of
  `PHOTO_COLUMNS` so a re-index never clears it; written via `db.set_favorite` and
  `PUT /api/photos/{id}/favorite`, guarded by the same-origin write middleware), a
  filter-aware **`favorites`** facet count, and a `favorite` filter on
  `/api/photos`. The UI adds a "★ Favorites" quick-filter chip, a hover/keyboard
  star overlay on every grid card, and an inline star in the lightbox; all star
  buttons for a photo stay in sync. URL round-trip + db/search/facet/API tested
  (`tests/test_favorites.py`, `tests/js/url_state_harness.mjs`).
  - [x] **Smart Albums (saved searches).** **Done:** any filter set can be saved
    under a name and restored later. A `saved_searches` table (name UNIQUE + the
    filter querystring) with `db.create_saved_search` (upsert-by-name, so re-saving
    overwrites), `list_saved_searches`, `delete_saved_search`, surfaced over
    `GET/POST /api/albums` and `DELETE /api/albums/{id}` (writes behind the
    same-origin guard; empty name → 400). The sidebar gains a "Smart albums" section
    with a "💾 Save current search" button and per-album load/delete chips; loading
    pushes the saved querystring and restores it through the existing URL-state
    machinery (`applyQuery`), so filters/view/sort all come back. db + API tested
    (`tests/test_albums.py`, incl. upsert, unknown-id no-op, table-create migration).
- [x] **Multi-select + bulk actions.** **Done:** a **Select** mode in the photos
  toolbar turns grid clicks into selection (Shift-click extends a range from the
  anchor); selection is keyed by photo id so it survives the virtualised grid's window
  recycling. A selection bar applies a bulk action to the whole set —
  **Favorite / Unfavorite / Hide / Unhide** — via `POST /api/photos/bulk {ids, action}`
  (behind the same-origin guard; `db.set_favorite_bulk`/`set_hidden_bulk`). Hiding uses a
  new `hidden` 0/1 column (kept out of `PHOTO_COLUMNS` like `favorite`, so a re-index
  never un-hides): user-hidden photos are excluded from browsing everywhere by default
  (a tri-state `hidden` filter in `_where`, API default `False`), and a filter-aware
  **🙈 Hidden** quick-filter chip flips the grid to "only hidden" so they can be
  reviewed/unhidden. db (bulk + tri-state where + facet), search and the API
  (hide→gone→unhide, bulk favorite, only-hidden view, 400 on a bad action) are tested
  (`tests/test_multiselect.py`).
  Possible follow-ups (dropped from this slice): bulk **assign a person** (ambiguous at
  the photo level — faces are the unit) and bulk **export** (file copy).
- [x] **"More like this."** **Done:** a ✨ **More like this** button in the lightbox
  pages the new `GET /api/photos/{id}/similar` endpoint, which cosine-ranks the
  library by the photo's own stored SigLIP image embedding (`search.similar_photos`
  + `SemanticIndex.vector_for`, the target always excluded). No text encoder/model
  download — it reuses the embeddings `embed`/`index --embed` already wrote, so it
  works offline whenever the library is embedded. The grid enters a dedicated
  "similar" mode (filters cleared, a removable "✨ Similar to …" banner pill,
  infinite-scroll paging preserved); picking any filter/search exits it. Backend
  (ranking, self-exclusion, paging, 404/409 edge cases) is unit + API tested
  (`tests/test_similar.py`) and the request-URL switch via a Node harness
  (`tests/js/similar_harness.mjs`). Follow-up: SFace "same person" similarity is
  still open (the person filter already covers exact-match).
- [x] **Face active-learning (negative feedback).** **Done:** correcting an auto-tag
  now teaches recognition. Unassigning a face — or reassigning it to someone else —
  records a "not this person" **negative** (new `face_negatives` table, both FKs
  cascade, `UNIQUE(face_id, person_id)`); `library.assign_face`/`unassign_face` write
  them and a human assignment sets `confidence=1.0` (so a confirmed face is certain).
  `Enrollment` now carries the negatives and `knn_person_match` is **negative-aware**:
  each identity's net vote is its positive neighbours minus its near negatives, so a
  probe that looks like a rejected example is penalised — or vetoed when the negatives
  outweigh the positives — with behaviour identical to plain k-NN when there are none.
  Low-confidence guesses are surfaced in a **"Review guesses"** section of the Name-faces
  tab (`GET /api/faces/review`, `config.review_confidence` default 0.6) where each can be
  confirmed (→ 1.0, drops out) or rejected (→ unassign + negative). Negative-aware k-NN,
  the DB/cascade, the correction flow and the review API are unit + API tested
  (`tests/test_active_learning.py`).
- [x] **Lightbox power tools.** **Done:** the lightbox gained scroll-wheel / `+`/`-`/`0`
  **zoom** with drag-to-**pan** past 1× (double-click toggles; the centre-anchored
  transform `nextZoom` is pure + Node-harness tested), a **slideshow** auto-advance
  (▶/⏸ button or Space, pulling further pages via `lightboxStep` and stopping at the
  end of the library), an **EXIF info panel** (ℹ︎) showing ƒ/ISO/shutter/focal-length/
  lens read **on demand** from a new `GET /api/exif/{id}` (`metadata.exif_settings`,
  formatted server-side — no schema change, no re-index; a moved file or EXIF-free
  image just yields `{}`), and a `?` **keyboard-shortcut legend** overlay. Zoom resets
  per photo and on close; Esc closes the legend before the lightbox. Backend
  (formatting + endpoint 404/missing-file/empty) unit + API tested
  (`tests/test_exif.py`, `tests/test_api_errors.py`); zoom math via
  `tests/js/lightbox_harness.mjs`.
- [ ] ~~**RAW ingest.**~~ **Won't do** (2026-06-16). A photographer's 15-year library
  has `.CR2/.NEF/.ARW`; pulling the embedded preview + EXIF via an optional `rawpy`
  extra was considered but is out of scope for now — the library targets
  already-developed JPEG/HEIC/PNG, and RAW workflows live in dedicated tools.

## New (2026-06-16)

- [x] **Drop the heuristic scene tagger — SigLIP only.** **Done** (chosen option:
  remove the heuristic + make SigLIP a core dependency). `SceneTagger`,
  `config.scene_backend` and the `--scene` CLI flags are gone; `classify.get_tagger`
  always returns the `ZeroShotSceneTagger`, and `onnxruntime`/`tokenizers` moved from
  the `scene` extra into core `dependencies` (the extra is removed). The SigLIP
  vision model downloads on demand for every index (prefetched once before the
  worker fan-out). Offline tests are preserved via dependency injection: a picklable
  `tests/scene_stub.StubTagger`, wired by an autouse conftest fixture that patches
  `indexer.get_tagger` for in-process paths and passed explicitly via
  `index_path(..., tagger=...)` for the parallel/spawn path (a monkeypatch can't
  cross processes). Docs (README/CLAUDE.md) updated; full suite + ruff + mypy green
  at 94% coverage.
- [x] **Investigate newer / better models everywhere a deep-learning net is used.**
  **Done (investigation):** written comparison in [`MODELS.md`](MODELS.md) covering all
  three nets — face detection (YuNet → latest Zoo / SCRFD / RetinaFace), face
  recognition (SFace → ArcFace R100 / AdaFace), and scene+semantic (SigLIP →
  **SigLIP 2** / MobileCLIP2) — each with the ONNX/no-PyTorch path, size/speed,
  licensing risk and migration mechanics (env overrides + rebuild `scene_labels.npz`
  + re-embed on a dim change). Recommended order: SigLIP 2 first (biggest quality/effort
  win, no architecture change), then a YuNet Zoo bump, then ArcFace (highest ceiling but
  most invasive). No swap made yet — each needs a local A/B eval first. Implementation of
  any actual swap remains open as a follow-up.

## Optional follow-ups (open, prioritised)

Every feature/bug item above is done; what's left are *optional* enhancements, each
noted inline on its parent item but consolidated here with upside / effort / risk so
the next slice is easy to pick. Ordered by value-per-effort.

- [ ] **Adopt SigLIP 2** (scene tags + semantic search). **Upside: high** — strictly
  better zero-shot tags *and* NL retrieval from one swap, no architecture change.
  **Effort: medium** (~2–3 days + a real A/B eval). **Risk: medium** — the SigLIP 2
  text tower may use the Gemma tokenizer (the hardcoded `</s>`/pad-64 assumption needs
  verifying). **Fully scoped in [`SIGLIP2_MIGRATION.md`](SIGLIP2_MIGRATION.md)**; the
  offline enabling refactors are already merged — Gap 1 (vision input-size auto-detect),
  Gap 2 (output-tensor-name resolution) and Gap 4 (dim-mismatch guard). Remaining before
  a swap: verify the tokenizer (Gap 3), run the A/B eval, rebuild `scene_labels.npz` +
  re-embed.
- [ ] **YuNet → latest Zoo revision.** **Upside: low–medium** (possibly-free face-recall
  gain). **Effort: trivial** (bump the `.onnx` name/URL in `models.py`, same
  `FaceDetectorYN` API). **Risk: very low.** Gate on a quick recall A/B. See
  [`MODELS.md`](MODELS.md) §1.
- [ ] **SFace → ArcFace R100 (or AdaFace for low-quality).** **Upside: high** —
  recognition is the product's backbone; 512-d raises the "same person across 15 years"
  ceiling. **Effort: high / most invasive** — embedding dim 128→512 ⇒ re-embed + re-cluster
  migration, ArcFace-standard 5-pt alignment. **Risk: medium–high.** Do it behind the
  `PHOTO_ATLAS_SFACE` override with a verification-accuracy A/B first. See
  [`MODELS.md`](MODELS.md) §2.
- [ ] **YuNet → SCRFD.** **Upside: medium** but only realised if hard-face *recall* is
  proven to be the bottleneck. **Effort: medium** (new anchor decode in `faces.py`, not a
  `FaceDetectorYN` drop-in). **Risk: medium.** Lower priority than ArcFace unless an eval
  says detection (not embedding) is the limiter.
- [ ] **Per-person semantic grounding.** Run SigLIP on the per-person face crop so
  "Stefano eating food" scores the region containing Stefano, not the whole frame.
  **Upside: medium** (sharper hybrid queries). **Effort: medium–high** (per-crop
  embedding store + ranking). **Risk: medium.** Follow-up to the hybrid planner.
- [ ] **Inline duplicate-grid collapse.** Collapse a burst behind its cover *in the main
  photo grid* (badge + expand) instead of only the dedicated Duplicates tab. **Upside:
  medium** (nicer browsing). **Effort: high** — reworks the virtualised windowing grid
  (variable-height rows / group headers). **Risk: medium** (grid perf regressions).
- [ ] **"More like this person" (SFace similarity).** A face-embedding "same person"
  ranking alongside the existing whole-image "more like this". **Upside: low–medium**
  (the exact person filter already covers named people). **Effort: medium.** **Risk: low.**
- [ ] **Unified "type of picture" facet.** Fold scene tags + people-count buckets into one
  facet. **Upside: low** (UX tidy-up). **Effort: low–medium.** **Risk: low.**
- [ ] **Bulk assign-a-person / bulk export.** Dropped from the multi-select slice.
  Assign is ambiguous at the photo level (faces are the unit); export is a file copy.
  **Upside: low–medium.** **Effort: low–medium.** **Risk: low** (export touches the
  filesystem — confirm + dry-run).
- [ ] **Semantic-index cache signature.** `api._embed_signature` keys on `(count, max_id)`,
  so an in-place `embed --recompute` while `serve` is running won't reload the matrix
  (restart works today). **Upside: low** (only matters for live model swaps). **Effort:
  low** (hash/model-tag the signature). **Risk: low.** See `SIGLIP2_MIGRATION.md` Gap 5.

### Won't do (already decided)
- ~~Responsive / mobile layout~~ — desktop-browser target (2026-06-16).
- ~~RAW ingest~~ (`.CR2/.NEF/.ARW`) — library targets developed JPEG/HEIC/PNG (2026-06-16).
