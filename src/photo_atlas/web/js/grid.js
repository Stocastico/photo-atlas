// Virtualised photo grid, infinite scroll, multi-select and bulk actions.
import { api } from "./api.js";
import { makeStar } from "./favorites.js";
import { inSimilarMode, photosRequestURL } from "./filters.js";
import { openLightbox } from "./lightbox.js";
import { renderSidebar } from "./sidebar.js";
import { BUFFER_ROWS, CARD_MIN, GRID_GAP, PEOPLE_LABELS, state } from "./state.js";
import { activeFilterPairs } from "./urlstate.js";
import { $, esc, toast } from "./util.js";

// ---- photos grid (infinite scroll) ----------------------------------------
function photoCard(p, index) {
  const card = document.createElement("div");
  card.className = "card" + (state.selected.has(p.id) ? " selected" : "");
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
    <span class="select-check" aria-hidden="true">✓</span>
    ${p.face_count ? `<span class="badge">👤 ${p.face_count}</span>` : ""}
    ${p.is_video ? `<span class="badge video-badge" aria-label="Video">▶</span>` : ""}
    <div class="meta">${place}<span>${(p.taken_at || "").slice(0, 4)}</span></div>`;
  card.setAttribute("role", "button");
  card.setAttribute("tabindex", "0");
  card.setAttribute("aria-label", `Open ${p.filename}`);
  const activate = (e) => state.selectMode ? toggleSelect(index, e.shiftKey) : openLightbox(index);
  card.onclick = activate;
  card.onkeydown = (e) => {
    if (e.key === "Enter" || e.key === " ") { e.preventDefault(); activate(e); }
  };
  card.appendChild(makeStar(p)); // favourite toggle overlay
  return card;
}

// ---- multi-select + bulk actions ------------------------------------------
export function setSelectMode(on) {
  state.selectMode = on;
  if (!on) { state.selected.clear(); state.selectAnchor = null; }
  const btn = $("#select-toggle");
  if (btn) { btn.classList.toggle("active", on); btn.setAttribute("aria-pressed", on ? "true" : "false"); }
  $("#selection-bar").hidden = !on;
  $("#grid").classList.toggle("selecting", on);
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
    if (p) el.classList.toggle("selected", state.selected.has(p.id));
  }
}

export function updateSelectionBar() {
  const n = state.selected.size;
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

// -- pure windowing math (unit-tested via tests/js/grid_window_harness.mjs) --
// The grid takes over its own layout (cards are absolutely positioned) so only
// a viewport-sized window of card nodes ever exists in the DOM, bounding both
// node count and decoded-bitmap memory regardless of library size.
export function gridLayout(containerWidth, n) {
  const w = Math.max(containerWidth, CARD_MIN);
  const cols = Math.max(1, Math.floor((w + GRID_GAP) / (CARD_MIN + GRID_GAP)));
  const cardW = (w - (cols - 1) * GRID_GAP) / cols;
  const cardH = cardW; // square cards (aspect-ratio 1)
  const rows = Math.ceil(n / cols);
  const totalH = rows > 0 ? rows * cardH + (rows - 1) * GRID_GAP : 0;
  return { cols, cardW, cardH, rows, totalH };
}

export function cardOffset(i, layout) {
  const row = Math.floor(i / layout.cols);
  const col = i % layout.cols;
  return { left: col * (layout.cardW + GRID_GAP), top: row * (layout.cardH + GRID_GAP) };
}

export function windowRange(scrollTop, viewportH, layout, n) {
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
