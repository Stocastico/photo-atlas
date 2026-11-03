// Virtualised photo grid, infinite scroll, multi-select and bulk actions.
import { api } from "./api.js";
import { makeStar } from "./favorites.js";
import { inSimilarMode, photosRequestURL } from "./filters.js";
import { bubbleText, buildLayout, itemAt, visibleRange } from "./layout.js";
import { openLightbox } from "./lightbox.js";
import { renderSidebar } from "./sidebar.js";
import { BUFFER_PX, GRID_GAP, HEADER_H, PEOPLE_LABELS, state } from "./state.js";
import { activeFilterPairs } from "./urlstate.js";
import { $, esc, toast } from "./util.js";

// ---- photos grid (justified rows, grouped, infinite scroll) ----------------
// Cards are deliberately clean: the thumbnail, a hover star + checkbox, and a
// video badge. The tint is the photo's average colour, shown while the thumb
// loads and then cross-faded/de-blurred (blur-up).
function photoCard(p, index) {
  const card = document.createElement("div");
  card.className = "card" + (state.selected.has(p.id) ? " selected" : "");
  if (p.tint) card.style.backgroundColor = "#" + p.tint;
  // srcset lets the browser pull the 2x (retina) thumb only on hi-DPI screens;
  // the base 320px thumb is pre-generated, the 640px one is cached on demand.
  card.innerHTML = `
    <img loading="lazy" decoding="async"
      src="/api/thumb/${p.id}"
      srcset="/api/thumb/${p.id} 320w, /api/thumb/${p.id}?size=640 640w"
      sizes="${Math.ceil(state.rowHeight * 1.6)}px"
      alt="${esc(p.filename)}" />
    <span class="select-check" role="checkbox" aria-label="Select ${esc(p.filename)}"
      aria-checked="${state.selected.has(p.id)}">✓</span>
    ${p.is_video ? `<span class="badge video-badge" aria-label="Video">▶</span>` : ""}`;
  const img = card.querySelector("img");
  if (img) {
    const done = () => card.classList.add("loaded");
    img.onload = done;
    img.onerror = done;
    if (img.complete) done();
  }
  card.setAttribute("role", "button");
  card.setAttribute("tabindex", "0");
  card.setAttribute("aria-label", `Open ${p.filename}`);
  const activate = (e) => state.selected.size ? toggleSelect(index, e.shiftKey) : openLightbox(index);
  card.onclick = activate;
  card.onkeydown = (e) => {
    if (e.key === "Enter" || e.key === " ") { e.preventDefault(); activate(e); }
  };
  const check = card.querySelector(".select-check");
  if (check) check.onclick = (e) => { e.stopPropagation(); toggleSelect(index, e.shiftKey); };
  card.appendChild(makeStar(p)); // favourite toggle overlay
  return card;
}

// ---- multi-select + bulk actions ------------------------------------------
// Selection is implicit: tick a hover checkbox and the bar appears; while anything
// is selected a plain click toggles instead of opening. Cleared -> back to browsing.
export function setSelectMode(on) {
  if (!on) { state.selected.clear(); state.selectAnchor = null; }
  state.selectMode = state.selected.size > 0;
  repaintSelection();
  updateSelectionBar();
}

// Toggle one card, or extend a shift-click range from the anchor. Selection is
// keyed by photo id so it survives the grid's window recycling.
function toggleSelect(index, shift) {
  if (shift && state.selectAnchor != null) {
    const [a, b] = [state.selectAnchor, index].sort((x, y) => x - y);
    for (let i = a; i <= b; i++) state.selected.add(state.photos[i].id);
  } else {
    const id = state.photos[index].id;
    if (state.selected.has(id)) state.selected.delete(id);
    else state.selected.add(id);
    state.selectAnchor = index;
  }
  repaintSelection();
  updateSelectionBar();
}

export function repaintSelection() {
  for (const [i, el] of state.rendered) {
    const p = state.photos[i];
    if (!p) continue;
    const on = state.selected.has(p.id);
    el.classList.toggle("selected", on);
    const chk = el.querySelector(".select-check");
    if (chk) chk.setAttribute("aria-checked", on ? "true" : "false");
  }
}

export function updateSelectionBar() {
  const n = state.selected.size;
  state.selectMode = n > 0;
  const bar = $("#selection-bar");
  if (bar) bar.hidden = n === 0;
  const grid = $("#grid");
  if (grid) grid.classList.toggle("selecting", n > 0);
  const label = $("#selection-count");
  if (label) label.textContent = `${n} selected`;
  document.querySelectorAll("#selection-bar [data-bulk]").forEach((b) => (b.disabled = n === 0));
  const exp = $("#select-export");
  if (exp) exp.disabled = n === 0;
}

export async function bulkAction(action) {
  const ids = [...state.selected];
  if (!ids.length) return;
  try {
    const res = await api("/api/photos/bulk", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ ids, action }),
    });
    toast(`${res.updated} photo${res.updated === 1 ? "" : "s"} updated`, "info");
  } catch (e) { return; }
  state.selected.clear();
  state.selectAnchor = null;
  // Hidden photos leave the default grid and favorites/hidden counts change, so
  // reload the grid + sidebar to reflect the new state.
  renderSidebar();
  renderPhotos(true);
  updateSelectionBar();
}

// Bulk export: copy the selection's originals into a server-side folder. Distinct
// from the flag actions (it needs a destination), so it has its own handler.
export async function bulkExport() {
  const ids = [...state.selected];
  if (!ids.length) return;
  const dest = window.prompt(
    `Export ${ids.length} photo${ids.length === 1 ? "" : "s"} to which folder on the server?`
  );
  if (!dest || !dest.trim()) return;
  let res;
  try {
    res = await api("/api/photos/export", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ ids, dest: dest.trim() }),
    });
  } catch (e) { return; }
  const miss = res.missing ? `, ${res.missing} missing` : "";
  toast(`Exported ${res.copied} photo${res.copied === 1 ? "" : "s"}${miss}`, "info");
}

// -- windowed rendering -----------------------------------------------------
// layout.js computes every tile's rectangle (justified rows + group headers); only
// the rectangles intersecting the viewport (plus a buffer) get DOM nodes, bounding
// node count and decoded-bitmap memory regardless of library size.
let layoutKey = "";
function currentLayout(grid) {
  const width = Math.max(grid.clientWidth || 0, 200);
  const complete = state.photos.length >= state.total;
  const key = [state.photos.length, width, state.rowHeight, state.groupMode, complete].join("|");
  if (key !== layoutKey || !state.layout) {
    state.layout = buildLayout(state.photos, {
      width, target: state.rowHeight, gap: GRID_GAP, mode: state.groupMode,
      headerH: HEADER_H, complete,
    });
    layoutKey = key; // live nodes are repositioned by index in renderWindow
  }
  return state.layout;
}

function headerNode(h) {
  const el = document.createElement("div");
  el.className = "group-header";
  el.innerHTML = `<span class="gh-title">${esc(h.label)}</span><span class="gh-count"></span>`;
  return el;
}

// Reconcile the DOM with the currently-visible window.
function renderWindow() {
  if (state.view !== "photos") return;
  const grid = $("#grid");
  const layout = currentLayout(grid);
  grid.style.height = layout.totalH + "px";

  const gridTop = grid.getBoundingClientRect().top + window.scrollY;
  const scrollTop = window.scrollY - gridTop;
  const vis = visibleRange(layout, scrollTop, window.innerHeight, BUFFER_PX);

  const keep = new Set(vis.items.map((it) => it.i));
  for (const [i, el] of state.rendered) {
    if (!keep.has(i)) { el.remove(); state.rendered.delete(i); }
  }
  for (const it of vis.items) {
    let el = state.rendered.get(it.i);
    if (!el) {
      el = photoCard(state.photos[it.i], it.i);
      grid.appendChild(el);
      state.rendered.set(it.i, el);
    }
    el.style.width = it.w + "px";
    el.style.height = it.h + "px";
    el.style.transform = `translate(${it.left}px, ${it.top}px)`;
  }

  const hkeep = new Set(vis.headers.map((h) => h.key));
  for (const [k, el] of state.renderedHeaders) {
    if (!hkeep.has(k)) { el.remove(); state.renderedHeaders.delete(k); }
  }
  for (const h of vis.headers) {
    let el = state.renderedHeaders.get(h.key);
    if (!el) {
      el = headerNode(h);
      grid.appendChild(el);
      state.renderedHeaders.set(h.key, el);
    }
    el.style.height = h.h + "px";
    el.style.transform = `translateY(${h.top}px)`;
    const c = el.querySelector(".gh-count");
    if (c) c.textContent = h.count == null ? "" : `${h.count} photo${h.count === 1 ? "" : "s"}`;
  }
  updateBubble(layout, scrollTop);
}

// Floating date shown while scrolling fast through a long library.
let bubbleTimer = null;
function updateBubble(layout, scrollTop) {
  const el = $("#date-bubble");
  if (!el) return;
  const it = itemAt(layout, scrollTop + 120);
  const text = it ? (state.groupMode === "year"
    ? (state.photos[it.i].taken_at || "").slice(0, 4)
    : bubbleText(state.photos[it.i].taken_at)) : "";
  if (!text || scrollTop < 200) { el.classList.remove("show"); return; }
  el.textContent = text;
  el.classList.add("show");
  clearTimeout(bubbleTimer);
  bubbleTimer = setTimeout(() => el.classList.remove("show"), 900);
}

function clearNodes() {
  for (const el of state.rendered.values()) el.remove();
  state.rendered.clear();
  for (const el of state.renderedHeaders.values()) el.remove();
  state.renderedHeaders.clear();
}

function clearGrid() {
  clearNodes();
  layoutKey = "";
  state.layout = null;
  $("#grid").style.height = "0px";
}

/** Re-flow after a display preference (grouping / size) changed — no refetch. */
export function relayout() {
  layoutKey = "";
  renderWindow();
}

export async function renderPhotos(reset = true) {
  if (state.loading) return;
  state.loading = true;
  if (reset) {
    state.offset = 0;
    state.photos = [];
    clearGrid();
  }
  $("#grid-loading").style.display = "block";

  let data;
  try {
    data = await api(photosRequestURL(state.offset));
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
  renderSearchPlan(data.plan);
  // Distinguish a genuinely empty library (first-run onboarding) from a filter
  // that simply matched nothing.
  const empty = state.photos.length === 0;
  const libraryEmpty =
    empty && activeFilterPairs().length === 0 && !inSimilarMode();
  $("#onboarding").style.display = libraryEmpty ? "block" : "none";
  $("#grid-empty").style.display = empty && !libraryEmpty ? "block" : "none";

  renderWindow();

  $("#grid-loading").style.display = state.photos.length < state.total ? "block" : "none";
  $("#grid-loading").textContent = "Loading…";
  state.loading = false;
  maybeLoadMore();
}

// Show how the server decomposed a natural-language query ("Stefano eating food"
// -> 👤 Stefano + “eating food”) so the hybrid planning is transparent.
function renderSearchPlan(plan) {
  const el = $("#search-plan");
  if (!el) return;
  if (!plan) { el.textContent = ""; el.title = ""; return; }
  const bits = [];
  (plan.persons || []).forEach((n) => bits.push(`👤 ${n}`));
  if ((plan.persons || []).length >= 2 && plan.person_mode === "all") bits.push("(all of them)");
  const people = plan.people || [];
  if (people.length === 1 && people[0] === "1") bits.push("alone");
  else if (people.includes("2-4") && people.includes("5+")) bits.push("with others");
  else people.forEach((t) => bits.push(PEOPLE_LABELS[t] || t));
  if (plan.text) bits.push(`“${plan.text}”`);
  // A 🎯 marks that the visual residual was scored against the named person's own
  // region (per-person grounding) rather than the whole photo.
  if (plan.grounded && plan.text) bits.push("🎯 focused on the person");
  el.textContent = bits.length ? `Interpreting: ${bits.join(" · ")}` : "";
  el.title = plan.grounded
    ? "Scored the region around the named person, not the whole photo (grounding)"
    : "How your words were split into people/scene filters and a visual query";
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
