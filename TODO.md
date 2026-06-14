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
- [ ] **More sort options** (e.g. by filename / by recently indexed).

### Navigation & state
- [ ] **URL / history state.** Reflect filters + view in the querystring so the
  back button undoes a filter and views are shareable/bookmarkable.
- [ ] **Infinite scroll near the lightbox end.** Stepping "next" past the last
  loaded photo should trigger the next page load instead of stopping.

### People / management
- [ ] **Rename in the People page.** `PATCH /api/persons/{id}` exists but the UI
  only offers View/Delete — add inline rename.
- [ ] **Merge people** (two clusters of the same person) and **reassign a face**
  to a different/again-unknown person from the lightbox.
- [ ] **Person cover photo picker.**

### Robustness & polish
- [ ] **Error states.** `api()` assumes JSON; a failed request currently breaks
  silently. Add a thin wrapper with try/catch + a toast/inline message.
- [ ] **Empty-library onboarding.** First-run hint pointing at
  `photo-atlas index` / `cluster` when the catalog is empty.
- [ ] **Accessibility.** Focus-trap the lightbox, restore focus on close, add
  `aria` labels/roles to chips and pills, keyboard-operable cards.

### Explicitly deferred (browser-only target)
- [ ] **Responsive / mobile layout.** `.layout` is a fixed `250px 1fr` grid with
  a sticky full-height sidebar; on small screens it needs a collapsible drawer.
  Skipped for now per usage (desktop browser only).
