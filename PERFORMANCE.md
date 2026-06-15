# Photo Atlas — Performance & memory review (2026-06-15)

A code-level audit of the hot paths after the parallel-indexing work landed.
Findings are grounded in `file:line` and sized against the stated target of
**~27k images + ~600 videos** (so roughly 30–50k detected faces). Ordered by
impact, not by area.

Legend: **P0** = breaks at scale (OOM / unusable) · **P1** = large, easy win ·
**P2** = worth doing · **P3** = minor / polish.

---

## P0 — Clustering builds a dense O(n²) distance matrix (will OOM)

`faces.cluster_embeddings` (`src/photo_atlas/faces.py:326-333`):

```python
matrix = np.vstack([l2_normalize(e) for e in embeddings])   # n × 128
similarity = np.clip(matrix @ matrix.T, -1.0, 1.0)           # n × n  (!)
distance = 1.0 - similarity                                  # n × n  (!)
labels = DBSCAN(..., metric="precomputed").fit_predict(distance)
```

`cluster_library` (`indexer.py:395`) feeds it **every unnamed face at once**, so
`n × n` is the whole face set. At 40k faces that is `40000² × 4 bytes ≈ 6.4 GB`
per matrix, and there are two of them live at once (~13 GB) before DBSCAN even
copies the precomputed matrix. `photo-atlas cluster` will OOM on a real library.

**Fix (drop-in, no accuracy loss).** Embeddings are L2-normalised, so cosine
distance is monotonic in Euclidean distance: `‖a−b‖² = 2 − 2·cos(a,b)`. Cluster
with a tree-based metric that never materialises the full matrix:

```python
eps_euclid = math.sqrt(2.0 * eps)        # eps is the cosine epsilon
labels = DBSCAN(eps=eps_euclid, min_samples=min_samples,
                metric="euclidean", algorithm="ball_tree").fit_predict(matrix)
```

Memory drops from `O(n²)` to `O(n·128)` (~20 MB at 40k) and DBSCAN's ball-tree
neighbour queries are far faster than a 40k×40k scan. For very large sets,
`HDBSCAN` or a blocked/incremental clustering is the next step, but the metric
swap alone removes the OOM.

---

## P1 — Every HTTP request re-creates the schema

`get_conn` (`api.py:50`) calls `db.connect` per request, and `db.connect`
(`db.py:95-109`) runs on **every** call:

- `PRAGMA journal_mode = WAL`, `PRAGMA busy_timeout`,
- `_migrate` → `PRAGMA table_info(photos)`,
- `executescript(SCHEMA)` → 3× `CREATE TABLE IF NOT EXISTS` + **9× `CREATE INDEX
  IF NOT EXISTS`**.

A single filter toggle in the UI fans out to `/api/facets` (9 aggregations),
`/api/photos` and `/api/map` — each opening a fresh connection and re-running the
whole schema/migration script. The DDL is a no-op data-wise but still parses and
checks 12 objects against `sqlite_master` on every request.

**Fix.** Run schema creation/migration **once** at startup (in `create_app`),
and give `db.connect` an `ensure_schema: bool = True` flag that request-time
connections pass as `False`. Better still, keep one connection per worker thread
(SQLite + WAL is happy with `check_same_thread=False` + a short-lived cursor) so
the per-request cost is a cursor, not a connect + DDL. Expect a clear drop in
p50 latency for every interaction.

---

## P1 — `SELECT DISTINCT p.*` on every photo query

`search_photos` (`search.py:122-128`) always uses `DISTINCT`:

```python
total = ... f"SELECT COUNT(DISTINCT p.id) {base}"
rows  = ... f"SELECT DISTINCT p.* {base} ORDER BY {order} LIMIT ? OFFSET ?"
```

`DISTINCT` is only needed because the optional `person_id` filter `JOIN`s `faces`
and can fan out rows. When no person filter is active (the common case) the join
is absent, yet `DISTINCT p.*` still forces SQLite to de-dup on **every column**,
including the `scene_scores` JSON text — a full hash/sort of wide rows per page.

**Fix.**
1. Only emit `DISTINCT` when the person join is present (`_where` already returns
   `join`, so the caller knows).
2. For the person filter, prefer `WHERE EXISTS (SELECT 1 FROM faces f WHERE
   f.photo_id = p.id AND f.person_id IN (...))` over `JOIN ... DISTINCT`. `EXISTS`
   short-circuits and never produces duplicate rows, eliminating `DISTINCT`
   entirely and letting `idx_faces_person` do the work.

---

## P1 — `COUNT(DISTINCT)` recomputed on every infinite-scroll page

`total` is page-invariant, but `search_photos` recomputes the full
`COUNT(DISTINCT p.id)` for **each** page the grid pulls (`renderPhotos`,
`app.js:421`, fires once per scroll window). On a 27k library a filtered count
scans the whole filtered set every few hundred pixels of scroll.

**Fix.** Skip the count when `offset > 0` and have the client keep the `total`
from the first page (it already stores `state.total`, `app.js:429`). Or return
`total` only on page 0. Halves the queries per scroll step.

---

## P2 — Missing indexes for two real query patterns

Schema indexes (`db.py:70-77`) cover `taken_at`, `scene_type`, country, city,
folder and the face FKs — but not:

- **`indexed_at`** — `sort=indexed` (`search.py:108`) does
  `ORDER BY p.indexed_at DESC` with no index → full sort of the result set.
- **`camera_model`** — used both as a filter (`search.py:76`) and a `GROUP BY`
  facet (`search.py:229`); currently a scan.

**Fix.** Add `idx_photos_indexed (indexed_at)` and `idx_photos_camera
(camera_model)`. Two cheap additive migrations.

---

## P2 — The facet sidebar issues ~9 aggregations per render

`facets()` (`search.py:168-234`) runs one `GROUP BY` per dimension (scenes,
countries, cities, places, years, cameras), a person aggregation, a
`with_faces` count, a min/max date and a total — **~11 queries**, each re-running
the shared `_where` join/scan. `renderSidebar` (`app.js:179`) calls it on every
filter change. It's correct and tolerable at 27k, but it's the single most
expensive UI interaction.

**Options.** Combine the independent single-column facets into one pass with
conditional aggregation, or cache the facet payload keyed by the active-filter
signature on the client (the sidebar only needs to change when filters change).
At minimum, the `total` and `date_min/max` are filter-independent and can be
computed once per process, not per request.

---

## P2 — `scene_scores` JSON shipped to the grid but never used

`search_photos` returns `p.*`, so each `/api/photos` row carries the
`scene_scores` JSON blob (`db.py` column) for 60 photos/page. The grid only uses
id/thumb/dimensions/filename. It's bandwidth + JSON-decode the browser throws
away. **Fix:** select an explicit column list for the grid; keep `p.*` for the
single-photo detail endpoint.

---

## P2 — Map builds 50k marker popups eagerly

`renderMap` (`app.js:626-636`) iterates up to `map_point_limit` (50k) points and,
for **every** one, constructs an `L.marker` *and* a detached popup `<div>` with an
`<img>` before anything is clicked. That's 50k DOM subtrees held in memory even
though at most one popup is ever open.

**Fix.** Bind the popup lazily — pass a function/`popupopen` handler that builds
the `<div>` on demand, or use `bindPopup(() => html)`. `markerClusterGroup`
already does `chunkedLoading`, so the marker objects are the only unavoidable
cost; the eager popups are not.

---

## P3 — Smaller items

- **`list_persons` N+1** (`library.py:22-31`): one extra `SELECT ... LIMIT 1` per
  person without a pinned cover. Fold into the main query with a correlated
  subquery or a `GROUP BY` min(face id).
- **Serial indexing commits per file** (`indexer.py`, the `workers<=1` branch):
  fine for the demo, but the parallel path's batched commit (every 64) is the
  pattern to copy if the serial path is ever used at scale.
- **`map_points` / `search` build `[dict(r) for r in rows]`** — materialises the
  full result list; fine within the 50k cap, just note it scales with the cap.
- **`_person_centroids`** (`indexer.py:76`) loads all named-face embeddings into
  memory at index start. Bounded by named faces (small) — no action, just
  flagged for completeness.

---

## Suggested order of work

1. **P0 clustering metric swap** — removes the OOM; ~5-line change + a memory/
   correctness test on a synthetic embedding set.
2. **P1 schema-once + connection reuse** — biggest UI latency win, low risk.
3. **P1 conditional `DISTINCT` / `EXISTS`** + **skip count after page 0**.
4. **P2 indexes** (`indexed_at`, `camera_model`) — trivial migrations.
5. **P2 lazy map popups** + **explicit grid column list**.
6. **P2 facet consolidation / client cache**, **P3 cleanups** as polish.

Items 1–4 are the high-leverage set: they turn `cluster` from "OOMs" into
"works", and cut the per-interaction DB cost of the web UI substantially.
