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
- [ ] **Filter by number of *known* (named) people.** How many faces in a photo are
  assigned to a named person (vs unknown). Needs a per-photo count of faces with
  `person_id IS NOT NULL` — either a correlated subquery/`HAVING` at query time or a
  denormalised `named_face_count` column maintained alongside `face_count`. Enables
  "photos where everyone is identified" / "photos with at least one stranger".

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
- [ ] **Better scene tagging.** `classify.py` is a colour/brightness heuristic
  that mislabels real photos (sunsets→food, snow→document). Replace with a small
  zero-shot model (MobileCLIP/CLIP) or narrow the labels to what's reliable.
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
