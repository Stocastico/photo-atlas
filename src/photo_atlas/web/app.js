// Photo Atlas — single page front-end (no build step, vanilla JS).
const PAGE = 120;
const state = {
  view: "photos",
  filters: {}, // person_id, scene, country, city, place, year, camera, q
  sort: "newest",
  photos: [],        // currently loaded photos (drives infinite scroll + lightbox)
  offset: 0,
  total: 0,
  loading: false,
  facetData: null,
  lightboxIndex: null,
  facetExpanded: {}, // facet key -> show all values
  rendered: new Map(), // index -> live card node (virtualised grid window)
  layout: null,        // last computed grid layout
};

// Grid windowing constants — GAP/MIN must match the CSS .grid/.card rules.
const GRID_GAP = 10;
const CARD_MIN = 160;     // grid-template min column width
const BUFFER_ROWS = 4;    // rows rendered above/below the viewport

const FILTER_NAMES = {
  person_id: "Person", scene: "Scene", country: "Country", city: "City",
  place: "Place", year: "Year", camera: "Camera", q: "Search",
  date_from: "From", date_to: "To", people: "People", known: "Known",
};
// Friendly labels for the number-of-people buckets (tokens come from the API).
const PEOPLE_LABELS = {
  "0": "No people", "1": "1 (portrait)", "2-4": "2–4", "5+": "5+",
};
// Friendly labels for the number-of-known-(named)-people buckets.
const KNOWN_LABELS = {
  "0": "None identified", "1": "1 identified", "2+": "2+ identified",
};
const FACET_CAP = 14;

const $ = (s) => document.querySelector(s);
const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

// Transient status message (errors by default). aria-live announces it.
let toastTimer;
function toast(msg, kind = "error") {
  const t = $("#toast");
  if (!t) return;
  t.textContent = msg;
  t.className = `toast show ${kind}`;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => (t.className = "toast"), 4500);
}

// Facet payloads are filter-aware but otherwise stable, so cache them by the
// active-filter signature: revisiting a filter state (toggling a chip off and
// on, back/forward navigation, switching views) then skips the ~11-query
// /api/facets round-trip. Any mutating request (assign/rename/merge/…) can
// change the counts, so it clears the cache (see api()).
const facetCache = new Map();
const FACET_CACHE_CAP = 50;

// Thin fetch wrapper: surfaces network failures and non-2xx responses as a
// toast and throws, so a failed request never breaks the UI silently. Callers
// that await it are skipped (no re-render) when it throws.
async function api(url, opts) {
  let res;
  try {
    res = await fetch(url, opts);
  } catch (e) {
    toast("Network error — is the server still running?");
    throw e;
  }
  if (!res.ok) {
    let detail = res.statusText;
    try { detail = (await res.json()).detail || detail; } catch (_) { /* non-JSON body */ }
    toast(`Error ${res.status}: ${detail}`);
    throw new Error(`${res.status} ${detail}`);
  }
  // A successful write may shift facet counts (e.g. naming a cluster), so drop
  // the cached facet payloads rather than show stale numbers.
  if (opts && opts.method && opts.method !== "GET") facetCache.clear();
  try {
    return await res.json();
  } catch (_) {
    return null; // tolerate empty bodies
  }
}

function fmtDate(iso) {
  if (!iso) return "—";
  return new Date(iso).toLocaleDateString(undefined, { year: "numeric", month: "short", day: "numeric" });
}

// ---- filters / sidebar ----------------------------------------------------
// Facet filters hold an array of values (OR within a facet, AND across facets).
function isActive(key, value) {
  const cur = state.filters[key];
  if (Array.isArray(cur)) return cur.some((v) => v == value);
  return cur != null && cur == value;
}

function toggleFilter(key, value) {
  const arr = Array.isArray(state.filters[key])
    ? state.filters[key].slice()
    : (state.filters[key] != null ? [state.filters[key]] : []);
  const i = arr.findIndex((v) => v == value);
  if (i >= 0) arr.splice(i, 1);
  else arr.push(value);
  if (arr.length) state.filters[key] = arr;
  else delete state.filters[key];
  refresh();
}

function removeFilterValue(key, value) {
  const cur = state.filters[key];
  if (Array.isArray(cur)) {
    const arr = cur.filter((v) => v != value);
    if (arr.length) state.filters[key] = arr;
    else delete state.filters[key];
  } else {
    delete state.filters[key];
  }
  if (key === "q") $("#search").value = "";
  refresh();
}

function toggleHasFaces() {
  if (state.filters.has_faces) delete state.filters.has_faces;
  else state.filters.has_faces = true;
  refresh();
}

function chip(label, count, active, onClick) {
  const el = document.createElement("button");
  el.className = "chip" + (active ? " active" : "");
  el.setAttribute("aria-pressed", active ? "true" : "false");
  el.innerHTML = `${esc(label)}${count != null ? ` <span class="n">${count}</span>` : ""}`;
  el.onclick = onClick;
  return el;
}

function filterParams() {
  const params = new URLSearchParams();
  for (const [k, v] of Object.entries(state.filters)) {
    if (v == null || v === "") continue;
    if (Array.isArray(v)) v.forEach((x) => x != null && x !== "" && params.append(k, x));
    else params.set(k, v);
  }
  return params;
}

// ---- URL / history state --------------------------------------------------
// Reflect filters + view + sort in the querystring so the back button undoes a
// filter and a filtered view is shareable/bookmarkable.
const SCALAR_FILTERS = new Set(["q", "date_from", "date_to", "person_mode"]);
let restoringState = false;

function buildQuery() {
  const params = filterParams();
  if (state.view !== "photos") params.set("view", state.view);
  if (state.sort && state.sort !== "newest") params.set("sort", state.sort);
  return params.toString();
}

function syncURL() {
  if (restoringState) return;
  const qs = buildQuery();
  const target = qs ? "?" + qs : location.pathname;
  const current = location.search || location.pathname;
  if (target !== current) history.pushState(null, "", target);
}

function applyQuery() {
  const params = new URLSearchParams(location.search);
  const filters = {};
  for (const [k, v] of params.entries()) {
    if (k === "view" || k === "sort") continue;
    if (k === "has_faces") filters.has_faces = true;
    else if (SCALAR_FILTERS.has(k)) filters[k] = v;
    else (filters[k] = filters[k] || []).push(v);
  }
  state.filters = filters;
  state.view = params.get("view") || "photos";
  state.sort = params.get("sort") || "newest";
  const search = $("#search"); if (search) search.value = filters.q || "";
  const sort = $("#sort"); if (sort) sort.value = state.sort;
}

// Flatten state.filters into [key, value] pairs (one per selected value).
function activeFilterPairs() {
  const pairs = [];
  for (const [k, v] of Object.entries(state.filters)) {
    if (v == null || v === "") continue;
    if (k === "person_mode") continue; // a modifier on the person filter, not a value
    if (Array.isArray(v)) v.forEach((x) => (x != null && x !== "") && pairs.push([k, x]));
    else pairs.push([k, v]);
  }
  return pairs;
}

// Fetch the facet payload for a filter signature, served from the cache on a
// repeat signature. Kept a tiny pure-ish helper (no DOM) so it's unit-testable.
async function fetchFacets(key) {
  let f = facetCache.get(key);
  if (!f) {
    f = await api("/api/facets?" + key);
    if (f) {
      if (facetCache.size >= FACET_CACHE_CAP) facetCache.clear();
      facetCache.set(key, f);
    }
  }
  return f;
}

async function renderSidebar() {
  const f = await fetchFacets(filterParams().toString());
  state.facetData = f;
  const side = $("#sidebar");
  side.innerHTML = "";

  const heading = (title) => {
    const h = document.createElement("h3");
    h.textContent = title;
    side.appendChild(h);
  };

  const section = (title, items, key, labelFn = (i) => i.value, valFn = (i) => i.value) => {
    if (!items || !items.length) return;
    heading(title);
    const wrap = document.createElement("div");
    wrap.className = "facet";
    const expanded = state.facetExpanded[key];
    const shown = expanded ? items : items.slice(0, FACET_CAP);
    shown.forEach((i) => {
      const v = valFn(i);
      wrap.appendChild(chip(labelFn(i), i.count, isActive(key, v), () => toggleFilter(key, v)));
    });
    side.appendChild(wrap);
    if (items.length > FACET_CAP) {
      const more = document.createElement("button");
      more.className = "show-more";
      more.textContent = expanded ? "Show less" : `+${items.length - FACET_CAP} more`;
      more.onclick = () => { state.facetExpanded[key] = !expanded; renderSidebar(); };
      side.appendChild(more);
    }
  };

  const hdr = document.createElement("div");
  hdr.style.display = "flex"; hdr.style.justifyContent = "space-between"; hdr.style.alignItems = "center";
  hdr.innerHTML = `<h3 style="margin:0">${f.total} photos</h3>`;
  const clear = document.createElement("button");
  clear.className = "clear"; clear.textContent = "Clear all";
  clear.onclick = clearAllFilters;
  hdr.appendChild(clear);
  side.appendChild(hdr);

  // Quick toggle: only photos with at least one detected face.
  heading("Quick filters");
  const quick = document.createElement("div");
  quick.className = "facet";
  quick.appendChild(chip("👤 Has people", f.with_faces, !!state.filters.has_faces, toggleHasFaces));
  side.appendChild(quick);

  section("People", f.persons, "person_id", (i) => `${i.name}`, (i) => i.id);
  renderPeopleModeToggle(side);
  section("Number of people", f.people, "people", (i) => PEOPLE_LABELS[i.value] || i.value);
  section("Known people", f.known, "known", (i) => KNOWN_LABELS[i.value] || i.value);
  section("Scene", f.scenes, "scene");
  section("Country", f.countries, "country");
  section("City", f.cities, "city");
  section("Place", f.places, "place");
  section("Year", f.years, "year");
  dateSection(side, f);
  section("Camera", f.cameras, "camera");

  // Person names just became available — refresh the pills' labels.
  renderActiveFilters();
}

// Two date-taken bounds (inclusive). Inputs read/write state.filters directly.
function dateSection(side, f) {
  const h = document.createElement("h3");
  h.textContent = "Date taken";
  side.appendChild(h);
  const wrap = document.createElement("div");
  wrap.className = "date-range";
  const mk = (key, label) => {
    const inp = document.createElement("input");
    inp.type = "date";
    inp.className = "date-input";
    inp.value = state.filters[key] || "";
    inp.setAttribute("aria-label", `${label} date`);
    if (f.date_min) inp.min = f.date_min;
    if (f.date_max) inp.max = f.date_max;
    inp.onchange = () => {
      if (inp.value) state.filters[key] = inp.value;
      else delete state.filters[key];
      refresh();
    };
    return inp;
  };
  wrap.appendChild(mk("date_from", "From"));
  const dash = document.createElement("span");
  dash.className = "date-dash";
  dash.textContent = "–";
  wrap.appendChild(dash);
  wrap.appendChild(mk("date_to", "To"));
  side.appendChild(wrap);
}

// When 2+ people are selected, offer an any/all switch: "any" (default) matches
// photos containing any of them, "all" only photos containing every one.
function renderPeopleModeToggle(side) {
  const sel = state.filters.person_id;
  const count = Array.isArray(sel) ? sel.length : sel != null ? 1 : 0;
  if (count < 2) return;
  const all = state.filters.person_mode === "all";
  const btn = document.createElement("button");
  btn.className = "chip" + (all ? " active" : "");
  btn.setAttribute("aria-pressed", all ? "true" : "false");
  btn.textContent = all ? "Match: all of them" : "Match: any of them";
  btn.title = "Toggle whether a photo must contain all selected people or any of them";
  btn.onclick = () => {
    if (state.filters.person_mode === "all") delete state.filters.person_mode;
    else state.filters.person_mode = "all";
    refresh();
  };
  side.appendChild(btn);
}

function clearAllFilters() {
  state.filters = {};
  $("#search").value = "";
  refresh();
}

// ---- active filter pills --------------------------------------------------
function filterValueLabel(key, value) {
  if (key === "person_id") {
    const p = (state.facetData?.persons || []).find((x) => x.id == value);
    return p ? p.name : `#${value}`;
  }
  if (key === "people") return PEOPLE_LABELS[value] || value;
  if (key === "known") return KNOWN_LABELS[value] || value;
  return value;
}

function renderActiveFilters() {
  const bar = $("#active-filters");
  if (!bar) return;
  bar.innerHTML = "";
  const pairs = activeFilterPairs();
  bar.style.display = pairs.length ? "flex" : "none";
  for (const [k, v] of pairs) {
    const pill = document.createElement("button");
    pill.className = "filter-pill";
    const text = k === "has_faces" ? "Has people" : `${FILTER_NAMES[k] || k}: ${filterValueLabel(k, v)}`;
    pill.innerHTML = `<span>${esc(text)}</span><span class="x">✕</span>`;
    pill.title = "Remove filter";
    pill.setAttribute("aria-label", `Remove filter ${text}`);
    pill.onclick = () => removeFilterValue(k, v);
    bar.appendChild(pill);
  }
  if (pairs.length) {
    const clr = document.createElement("button");
    clr.className = "filter-pill clear-pill";
    clr.textContent = "Clear all";
    clr.onclick = clearAllFilters;
    bar.appendChild(clr);
  }
}

// ---- photos grid (infinite scroll) ----------------------------------------
function photoCard(p, index) {
  const card = document.createElement("div");
  card.className = "card";
  const placeText = p.place_label ? p.place_label.split(",")[0] : p.folder_place;
  const place = placeText ? `<span>${esc(placeText)}</span>` : "<span></span>";
  // srcset lets the browser pull the 2x (retina) thumb only on hi-DPI screens;
  // the base 320px thumb is pre-generated, the 640px one is cached on demand.
  card.innerHTML = `
    <img loading="lazy" decoding="async" width="320" height="320"
      src="/api/thumb/${p.id}"
      srcset="/api/thumb/${p.id} 320w, /api/thumb/${p.id}?size=640 640w"
      sizes="220px"
      alt="${esc(p.filename)}" />
    ${p.face_count ? `<span class="badge">👤 ${p.face_count}</span>` : ""}
    <div class="meta">${place}<span>${(p.taken_at || "").slice(0, 4)}</span></div>`;
  card.setAttribute("role", "button");
  card.setAttribute("tabindex", "0");
  card.setAttribute("aria-label", `Open ${p.filename}`);
  card.onclick = () => openLightbox(index);
  card.onkeydown = (e) => {
    if (e.key === "Enter" || e.key === " ") { e.preventDefault(); openLightbox(index); }
  };
  return card;
}

// -- pure windowing math (unit-tested via tests/js/grid_window_harness.mjs) --
// The grid takes over its own layout (cards are absolutely positioned) so only
// a viewport-sized window of card nodes ever exists in the DOM, bounding both
// node count and decoded-bitmap memory regardless of library size.
function gridLayout(containerWidth, n) {
  const w = Math.max(containerWidth, CARD_MIN);
  const cols = Math.max(1, Math.floor((w + GRID_GAP) / (CARD_MIN + GRID_GAP)));
  const cardW = (w - (cols - 1) * GRID_GAP) / cols;
  const cardH = cardW; // square cards (aspect-ratio 1)
  const rows = Math.ceil(n / cols);
  const totalH = rows > 0 ? rows * cardH + (rows - 1) * GRID_GAP : 0;
  return { cols, cardW, cardH, rows, totalH };
}

function cardOffset(i, layout) {
  const row = Math.floor(i / layout.cols);
  const col = i % layout.cols;
  return { left: col * (layout.cardW + GRID_GAP), top: row * (layout.cardH + GRID_GAP) };
}

function windowRange(scrollTop, viewportH, layout, n) {
  const unit = layout.cardH + GRID_GAP;
  const firstRow = Math.max(0, Math.floor(Math.max(0, scrollTop) / unit) - BUFFER_ROWS);
  const visRows = Math.ceil(viewportH / unit) + BUFFER_ROWS * 2;
  const start = firstRow * layout.cols;
  const end = Math.min(n, (firstRow + visRows) * layout.cols);
  return { start: Math.min(start, n), end };
}

// Reconcile the DOM with the currently-visible window: drop cards that scrolled
// out, create+position the ones that scrolled in, reposition the rest.
function renderWindow() {
  if (state.view !== "photos") return;
  const grid = $("#grid");
  const n = state.photos.length;
  const layout = gridLayout(grid.clientWidth || 0, n);
  state.layout = layout;
  grid.style.height = layout.totalH + "px";

  const gridTop = grid.getBoundingClientRect().top + window.scrollY;
  const scrollTop = window.scrollY - gridTop;
  const { start, end } = windowRange(scrollTop, window.innerHeight, layout, n);

  for (const [i, el] of state.rendered) {
    if (i < start || i >= end) { el.remove(); state.rendered.delete(i); }
  }
  for (let i = start; i < end; i++) {
    let el = state.rendered.get(i);
    if (!el) {
      el = photoCard(state.photos[i], i);
      grid.appendChild(el);
      state.rendered.set(i, el);
    }
    const off = cardOffset(i, layout);
    el.style.width = layout.cardW + "px";
    el.style.height = layout.cardH + "px";
    el.style.transform = `translate(${off.left}px, ${off.top}px)`;
  }
}

function clearGrid() {
  for (const el of state.rendered.values()) el.remove();
  state.rendered.clear();
  $("#grid").style.height = "0px";
}

async function renderPhotos(reset = true) {
  if (state.loading) return;
  state.loading = true;
  if (reset) {
    state.offset = 0;
    state.photos = [];
    clearGrid();
  }
  $("#grid-loading").style.display = "block";

  const params = filterParams();
  if (state.sort && state.sort !== "newest") params.set("sort", state.sort);
  params.set("limit", String(PAGE));
  params.set("offset", String(state.offset));

  let data;
  try {
    data = await api("/api/photos?" + params.toString());
  } catch (e) {
    $("#grid-loading").textContent = "Could not load photos.";
    state.loading = false;
    return;
  }

  state.photos = state.photos.concat(data.photos);
  // Only the first page carries a total (later pages send null to skip the
  // server-side COUNT); keep the previously known total across those pages.
  if (data.total != null) state.total = data.total;
  state.offset += data.photos.length;

  $("#result-count").textContent = `${state.total} result${state.total === 1 ? "" : "s"}`;
  // Distinguish a genuinely empty library (first-run onboarding) from a filter
  // that simply matched nothing.
  const empty = state.photos.length === 0;
  const libraryEmpty = empty && activeFilterPairs().length === 0;
  $("#onboarding").style.display = libraryEmpty ? "block" : "none";
  $("#grid-empty").style.display = empty && !libraryEmpty ? "block" : "none";

  renderWindow();

  $("#grid-loading").style.display = state.photos.length < state.total ? "block" : "none";
  $("#grid-loading").textContent = "Loading…";
  state.loading = false;
  maybeLoadMore();
}

function nearBottom() {
  return window.innerHeight + window.scrollY >= document.body.offsetHeight - 600;
}

function maybeLoadMore() {
  if (state.view !== "photos" || state.loading) return;
  if (state.photos.length >= state.total) return;
  // Keep loading until the viewport is filled (so short result sets behave).
  if (document.body.offsetHeight <= window.innerHeight + 600) renderPhotos(false);
}

let frameScheduled = false;
function onViewportChange() {
  if (frameScheduled) return;
  frameScheduled = true;
  requestAnimationFrame(() => {
    frameScheduled = false;
    if (state.view !== "photos") return;
    renderWindow();
    if (!state.loading && state.photos.length < state.total && nearBottom()) renderPhotos(false);
  });
}

window.addEventListener("scroll", onViewportChange, { passive: true });
window.addEventListener("resize", onViewportChange);

// ---- lightbox / detail ----------------------------------------------------
function closeLightbox() {
  $("#lightbox").classList.remove("open");
  state.lightboxIndex = null;
  // Restore focus to whatever opened the lightbox (the photo card).
  if (state.lastFocus && state.lastFocus.focus) state.lastFocus.focus();
  state.lastFocus = null;
}

function focusableInLightbox() {
  const lb = $("#lightbox");
  return [...lb.querySelectorAll('button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])')]
    .filter((el) => !el.disabled && el.offsetParent !== null);
}

async function lightboxStep(delta) {
  const i = state.lightboxIndex + delta;
  if (i < 0) return;
  if (i >= state.photos.length) {
    // Stepping past the last loaded photo pulls the next page (if any) so the
    // lightbox keeps going instead of dead-ending mid-library.
    if (state.photos.length >= state.total) return;
    await renderPhotos(false);
    if (i >= state.photos.length) return;
  }
  openLightbox(i);
}

// Render the detail side-panel (metadata + editable faces) for a photo. The
// ``reopen`` thunk is how a face edit refreshes the panel in place — it differs
// for grid (index-based) vs map (id-based) entry points.
function renderLightboxSide(p, id, reopen) {
  $("#lb-img").src = `/api/preview/${id}`;
  const side = $("#lb-side");
  const kv = (k, v) => (v ? `<div class="kv"><span>${k}</span><span>${esc(v)}</span></div>` : "");
  side.innerHTML = `
    <h2>${esc(p.filename)}</h2>
    <p class="tagline">${esc(p.scene_type || "")} · ${fmtDate(p.taken_at)}</p>
    ${kv("Place", p.place_label)}
    ${kv("Trip", p.folder_place)}
    ${kv("Camera", [p.camera_make, p.camera_model].filter(Boolean).join(" "))}
    ${kv("Size", p.width && p.height ? `${p.width}×${p.height}` : "")}
    ${kv("Coordinates", p.lat != null ? `${p.lat.toFixed(4)}, ${p.lon.toFixed(4)}` : "")}
    <p class="lb-actions"><a href="/api/image/${id}" target="_blank" rel="noopener">View full size ↗</a></p>
    <h3 style="margin-top:16px">Faces (${p.faces.length})</h3>
    <div class="face-list" id="lb-faces"></div>`;

  const list = $("#lb-faces");
  if (!p.faces.length) list.innerHTML = `<span class="tagline">No faces detected.</span>`;
  for (const face of p.faces) {
    const item = document.createElement("div");
    item.className = "face-item";
    // A named face also gets a "✕" to reassign it back to unknown; typing a
    // different name and saving reassigns it to that (new or existing) person.
    item.innerHTML = `
      <img src="/api/face/${face.id}" onerror="this.style.visibility='hidden'" />
      <input placeholder="name…" value="${esc(face.person_name || "")}" aria-label="Assign face to a person" />
      <button class="primary">Save</button>
      ${face.person_id ? `<button class="ghost icon" data-act="clear" title="Reassign to unknown" aria-label="Reassign face to unknown">✕</button>` : ""}`;
    const input = item.querySelector("input");
    const refresh = () => { renderSidebar(); reopen(); };
    item.querySelector('.primary').onclick = async () => {
      if (!input.value.trim()) return;
      await api(`/api/faces/${face.id}/assign`, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name: input.value.trim() }),
      });
      refresh();
    };
    const clear = item.querySelector('[data-act="clear"]');
    if (clear) clear.onclick = async () => {
      await api(`/api/faces/${face.id}/unassign`, { method: "POST" });
      refresh();
    };
    input.addEventListener("keydown", (e) => e.key === "Enter" && item.querySelector('.primary').click());
    list.appendChild(item);
  }
}

async function openLightbox(index) {
  const wasOpen = $("#lightbox").classList.contains("open");
  if (!wasOpen) state.lastFocus = document.activeElement;
  state.lightboxIndex = index;
  $("#lb-prev").disabled = index <= 0;
  // "Next" stays enabled at the last loaded photo when more pages remain on the
  // server, so the arrow can trigger the next load (see lightboxStep).
  $("#lb-next").disabled = index >= state.photos.length - 1 && state.photos.length >= state.total;

  const base = state.photos[index];
  $("#lightbox").classList.add("open");
  if (!wasOpen) $("#lb-close").focus();
  const p = await api(`/api/photos/${base.id}`);
  // Guard against a fast prev/next click landing on a different photo.
  if (state.lightboxIndex !== index) return;
  // Bounded preview derivative keeps memory flat while flicking; full original
  // stays behind the "View full size" link (see renderLightboxSide).
  renderLightboxSide(p, base.id, () => openLightbox(index));
}

// Open a single photo by id (used by map markers). There's no surrounding
// result list to page through, so prev/next are disabled.
async function openPhotoById(id) {
  const wasOpen = $("#lightbox").classList.contains("open");
  if (!wasOpen) state.lastFocus = document.activeElement;
  state.lightboxIndex = null;
  $("#lb-prev").disabled = true;
  $("#lb-next").disabled = true;
  $("#lightbox").classList.add("open");
  if (!wasOpen) $("#lb-close").focus();
  const p = await api(`/api/photos/${id}`);
  if (!$("#lightbox").classList.contains("open")) return; // closed while loading
  renderLightboxSide(p, id, () => openPhotoById(id));
}

// ---- map ------------------------------------------------------------------
let _map = null, _markers = null, _leafletIcons = false;

function ensureLeafletIcons() {
  if (_leafletIcons || !window.L) return;
  // Point Leaflet at the locally-vendored marker images (no CDN; offline-safe).
  delete L.Icon.Default.prototype._getIconUrl;
  L.Icon.Default.mergeOptions({
    iconRetinaUrl: "/static/vendor/leaflet/images/marker-icon-2x.png",
    iconUrl: "/static/vendor/leaflet/images/marker-icon.png",
    shadowUrl: "/static/vendor/leaflet/images/marker-shadow.png",
  });
  _leafletIcons = true;
}

async function renderMap() {
  if (!window.L) { toast("Map library failed to load."); return; }
  ensureLeafletIcons();
  if (!_map) {
    _map = L.map("map", { worldCopyJump: true }).setView([20, 0], 2);
    L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
      maxZoom: 19, attribution: "© OpenStreetMap contributors",
    }).addTo(_map);
  }
  // The container was display:none until the tab opened; re-measure it.
  setTimeout(() => _map && _map.invalidateSize(), 0);

  let data;
  try { data = await api("/api/map?" + filterParams().toString()); }
  catch (e) { return; }

  if (_markers) { _map.removeLayer(_markers); _markers = null; }
  _markers = L.markerClusterGroup
    ? L.markerClusterGroup({ chunkedLoading: true })
    : L.layerGroup();

  $("#map-empty").style.display = data.points.length ? "none" : "block";
  const bounds = [];
  for (const pt of data.points) {
    if (pt.lat == null || pt.lon == null) continue;
    const marker = L.marker([pt.lat, pt.lon]);
    // Build the popup (a thumbnail + caption) lazily, only when the marker is
    // actually clicked. With up to map_point_limit (50k) markers, eagerly
    // creating a DOM subtree per point would hold tens of thousands of unused
    // nodes in memory; at most one popup is ever open.
    marker.bindPopup(() => {
      const pop = document.createElement("div");
      pop.className = "map-pop";
      pop.innerHTML = `<img src="/api/thumb/${pt.id}" alt="" loading="lazy" />
        <div class="cap">${esc(String(pt.year || ""))} · open ↗</div>`;
      pop.onclick = () => openPhotoById(pt.id);
      return pop;
    });
    _markers.addLayer(marker);
    bounds.push([pt.lat, pt.lon]);
  }
  _map.addLayer(_markers);
  if (bounds.length) _map.fitBounds(bounds, { maxZoom: 12, padding: [30, 30] });
}

// ---- people ---------------------------------------------------------------
async function renderPeople() {
  const { persons } = await api("/api/persons");
  const wrap = $("#people");
  wrap.innerHTML = "";
  $("#people-empty").style.display = persons.length ? "none" : "block";
  for (const person of persons) wrap.appendChild(personCard(person, persons));
}

function personCard(person, allPersons) {
  const el = document.createElement("div");
  el.className = "person-card";
  const avatar = person.cover_face_id
    ? `<img class="avatar" src="/api/face/${person.cover_face_id}" onerror="this.style.visibility='hidden'"/>`
    : `<div class="avatar"></div>`;
  el.innerHTML = `
    ${avatar}
    <div class="name" data-role="name">${esc(person.name)}</div>
    <div class="sub">${person.photo_count} photos · ${person.face_count} faces</div>
    <div class="row">
      <button class="ghost" data-act="view" aria-label="View ${esc(person.name)}'s photos">View</button>
      <button class="ghost" data-act="rename" aria-label="Rename ${esc(person.name)}">Rename</button>
    </div>
    <div class="row">
      <button class="ghost" data-act="cover" aria-label="Choose cover photo">Cover</button>
      <button class="ghost" data-act="merge" aria-label="Merge ${esc(person.name)} into another person">Merge</button>
      <button class="ghost danger" data-act="del" aria-label="Delete ${esc(person.name)}">Delete</button>
    </div>
    <div class="person-panel" data-role="panel" hidden></div>`;

  const panel = el.querySelector('[data-role="panel"]');
  const closePanel = () => { panel.hidden = true; panel.innerHTML = ""; };

  el.querySelector('[data-act="view"]').onclick = () => {
    state.filters = { person_id: [person.id] }; setView("photos");
  };
  el.querySelector('[data-act="rename"]').onclick = () => startRename(el, person);
  el.querySelector('[data-act="del"]').onclick = async () => {
    if (!confirm(`Remove ${person.name}? Faces are kept for re-clustering.`)) return;
    await api(`/api/persons/${person.id}`, { method: "DELETE" });
    renderPeople(); renderSidebar();
  };
  el.querySelector('[data-act="cover"]').onclick = () =>
    panel.hidden ? openCoverPicker(panel, person) : closePanel();
  el.querySelector('[data-act="merge"]').onclick = () =>
    panel.hidden ? openMergePicker(panel, person, allPersons) : closePanel();
  return el;
}

function startRename(card, person) {
  const nameEl = card.querySelector('[data-role="name"]');
  if (card.querySelector(".rename-box")) return; // already editing
  const box = document.createElement("div");
  box.className = "rename-box row";
  box.innerHTML = `<input value="${esc(person.name)}" aria-label="New name" />
    <button class="primary">Save</button>`;
  nameEl.replaceWith(box);
  const input = box.querySelector("input");
  input.focus(); input.select();
  const save = async () => {
    const name = input.value.trim();
    if (!name || name === person.name) return renderPeople();
    await api(`/api/persons/${person.id}`, {
      method: "PATCH", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name }),
    });
    renderPeople(); renderSidebar();
  };
  box.querySelector("button").onclick = save;
  input.addEventListener("keydown", (e) => {
    if (e.key === "Enter") save();
    else if (e.key === "Escape") renderPeople();
  });
}

async function openCoverPicker(panel, person) {
  panel.hidden = false;
  panel.innerHTML = `<div class="tagline">Pick a cover photo…</div>`;
  const { faces } = await api(`/api/persons/${person.id}/faces`);
  if (!faces.length) { panel.innerHTML = `<div class="tagline">No face crops available.</div>`; return; }
  panel.innerHTML = "";
  const grid = document.createElement("div");
  grid.className = "cover-grid";
  faces.forEach((face) => {
    const img = document.createElement("img");
    img.src = `/api/face/${face.id}`;
    img.alt = "face crop";
    img.title = "Set as cover";
    if (face.id === person.cover_face_id) img.classList.add("selected");
    img.onclick = async () => {
      await api(`/api/persons/${person.id}/cover`, {
        method: "PUT", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ face_id: face.id }),
      });
      renderPeople();
    };
    grid.appendChild(img);
  });
  panel.appendChild(grid);
}

function openMergePicker(panel, person, allPersons) {
  const others = allPersons.filter((p) => p.id !== person.id);
  panel.hidden = false;
  if (!others.length) { panel.innerHTML = `<div class="tagline">No other people to merge into.</div>`; return; }
  panel.innerHTML = `
    <div class="tagline">Merge <b>${esc(person.name)}</b> into…</div>
    <div class="row">
      <select aria-label="Merge target">
        ${others.map((p) => `<option value="${p.id}">${esc(p.name)}</option>`).join("")}
      </select>
      <button class="primary">Merge</button>
    </div>`;
  const select = panel.querySelector("select");
  panel.querySelector("button").onclick = async () => {
    const target = Number(select.value);
    const targetName = others.find((p) => p.id === target)?.name || "that person";
    if (!confirm(`Merge ${person.name} into ${targetName}? This can't be undone.`)) return;
    await api(`/api/persons/${target}/merge`, {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ source_id: person.id }),
    });
    renderPeople(); renderSidebar();
  };
}

// ---- clusters (name faces) ------------------------------------------------
async function renderClusters() {
  const { clusters } = await api("/api/clusters");
  const wrap = $("#clusters");
  wrap.innerHTML = "";
  $("#clusters-empty").style.display = clusters.length ? "none" : "block";
  for (const c of clusters) {
    const el = document.createElement("div");
    el.className = "cluster-card";
    el.innerHTML = `
      <div class="cluster-faces">
        ${c.samples.map((s) => `<img src="/api/face/${s.id}" onerror="this.style.visibility='hidden'"/>`).join("")}
      </div>
      <div class="sub">${c.size} faces</div>
      <div class="row">
        <input placeholder="Who is this?" />
        <button class="primary">Name</button>
      </div>`;
    const input = el.querySelector("input");
    el.querySelector("button").onclick = async () => {
      if (!input.value.trim()) return;
      await api(`/api/clusters/${c.cluster_id}/assign`, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name: input.value.trim() }),
      });
      renderClusters(); renderSidebar();
    };
    input.addEventListener("keydown", (e) => e.key === "Enter" && el.querySelector("button").click());
    wrap.appendChild(el);
  }
}

// ---- view switching -------------------------------------------------------
function setView(view) {
  state.view = view;
  document.querySelectorAll(".tab").forEach((t) => {
    const on = t.dataset.view === view;
    t.classList.toggle("active", on);
    t.setAttribute("aria-selected", on ? "true" : "false");
    t.tabIndex = on ? 0 : -1; // roving tabindex for the WAI-ARIA tabs pattern
  });
  $("#view-photos").style.display = view === "photos" ? "block" : "none";
  $("#view-map").style.display = view === "map" ? "block" : "none";
  $("#view-people").style.display = view === "people" ? "block" : "none";
  $("#view-clusters").style.display = view === "clusters" ? "block" : "none";
  refresh();
}

function refresh() {
  syncURL();
  renderActiveFilters();
  renderSidebar();
  if (state.view === "photos") renderPhotos(true);
  else if (state.view === "map") renderMap();
  else if (state.view === "people") renderPeople();
  else renderClusters();
}

// ---- wiring ---------------------------------------------------------------
const _tabs = [...document.querySelectorAll(".tab")];
_tabs.forEach((t) => (t.onclick = () => setView(t.dataset.view)));
// Arrow/Home/End move focus across the tablist (WAI-ARIA tabs keyboard model).
const _tabbar = document.querySelector(".tabbar");
if (_tabbar) _tabbar.addEventListener("keydown", (e) => {
  const i = _tabs.indexOf(document.activeElement);
  if (i < 0) return;
  let j = i;
  if (e.key === "ArrowRight") j = (i + 1) % _tabs.length;
  else if (e.key === "ArrowLeft") j = (i - 1 + _tabs.length) % _tabs.length;
  else if (e.key === "Home") j = 0;
  else if (e.key === "End") j = _tabs.length - 1;
  else return;
  e.preventDefault();
  _tabs[j].focus();
  setView(_tabs[j].dataset.view);
});
$("#lb-close").onclick = closeLightbox;
$("#lb-prev").onclick = () => lightboxStep(-1);
$("#lb-next").onclick = () => lightboxStep(1);
$("#lightbox").onclick = (e) => { if (e.target.id === "lightbox") closeLightbox(); };
$("#sort").onchange = (e) => { state.sort = e.target.value; syncURL(); renderPhotos(true); };

document.addEventListener("keydown", (e) => {
  if (!$("#lightbox").classList.contains("open")) return;
  const typing = /^(INPUT|TEXTAREA|SELECT)$/.test(document.activeElement?.tagName || "");
  if (e.key === "Escape") closeLightbox();
  else if (e.key === "ArrowLeft" && !typing) lightboxStep(-1);
  else if (e.key === "ArrowRight" && !typing) lightboxStep(1);
  else if (e.key === "Tab") {
    // Focus trap: keep Tab cycling within the dialog.
    const f = focusableInLightbox();
    if (!f.length) return;
    const first = f[0], last = f[f.length - 1];
    if (e.shiftKey && document.activeElement === first) { e.preventDefault(); last.focus(); }
    else if (!e.shiftKey && document.activeElement === last) { e.preventDefault(); first.focus(); }
  }
});

let searchTimer;
$("#search").oninput = (e) => {
  clearTimeout(searchTimer);
  searchTimer = setTimeout(() => {
    const v = e.target.value.trim();
    if (v) state.filters.q = v; else delete state.filters.q;
    refresh();
  }, 250);
};

// Restore filters/view from the URL on back/forward navigation.
window.addEventListener("popstate", () => {
  restoringState = true;
  applyQuery();
  setView(state.view); // refresh()'s syncURL is skipped while restoring
  restoringState = false;
});

// Initial load: hydrate from any querystring (shared/bookmarked link).
applyQuery();
setView(state.view);
