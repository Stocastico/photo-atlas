// Filter state helpers: toggles, request params, "more like this" mode.

import { closeLightbox } from "./lightbox.js";
import { PAGE, state } from "./state.js";
import { $, esc } from "./util.js";
import { refresh, showViewPanel } from "./views.js";

// ---- filters / sidebar ----------------------------------------------------
// Facet filters hold an array of values (OR within a facet, AND across facets).
export function isActive(key, value) {
  const cur = state.filters[key];
  if (Array.isArray(cur)) return cur.some((v) => v == value);
  return cur != null && cur == value;
}

export function toggleFilter(key, value) {
  state.similarTo = state.similarFace = null; // picking a filter exits "more like this" mode
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

export function removeFilterValue(key, value) {
  const cur = state.filters[key];
  if (Array.isArray(cur)) {
    const arr = cur.filter((v) => v != value);
    if (arr.length) state.filters[key] = arr;
    else delete state.filters[key];
  } else {
    delete state.filters[key];
  }
  if (key === "q" || key === "text") $("#search").value = "";
  refresh();
}

export function toggleHasFaces() {
  state.similarTo = state.similarFace = null;
  if (state.filters.has_faces) delete state.filters.has_faces;
  else state.filters.has_faces = true;
  refresh();
}

export function toggleFavoriteFilter() {
  state.similarTo = state.similarFace = null;
  if (state.filters.favorite) delete state.filters.favorite;
  else state.filters.favorite = true;
  refresh();
}

export function toggleHiddenFilter() {
  // The 🙈 chip flips the grid into "show only hidden" so the user can review and
  // unhide; off, hidden photos are excluded everywhere (the API default).
  state.similarTo = state.similarFace = null;
  if (state.filters.hidden) delete state.filters.hidden;
  else state.filters.hidden = true;
  refresh();
}

export function chip(label, count, active, onClick) {
  const el = document.createElement("button");
  el.className = "chip" + (active ? " active" : "");
  el.setAttribute("aria-pressed", active ? "true" : "false");
  el.innerHTML = `${esc(label)}${count != null ? ` <span class="n">${count}</span>` : ""}`;
  el.onclick = onClick;
  return el;
}

export function filterParams() {
  const params = new URLSearchParams();
  for (const [k, v] of Object.entries(state.filters)) {
    if (v == null || v === "") continue;
    if (Array.isArray(v)) v.forEach((x) => x != null && x !== "" && params.append(k, x));
    else params.set(k, v);
  }
  return params;
}

// Build the /api/photos request URL for one page of the grid. In "more like
// this" mode (state.similarTo set) it pages the similarity endpoint and ignores
// the structured filters/sort; otherwise the normal filtered+sorted query.
export function photosRequestURL(offset) {
  if (state.similarFace != null || state.similarTo != null) {
    const p = new URLSearchParams();
    p.set("limit", String(PAGE));
    p.set("offset", String(offset));
    const base = state.similarFace != null
      ? `/api/faces/${state.similarFace}/similar`
      : `/api/photos/${state.similarTo}/similar`;
    return base + "?" + p.toString();
  }
  const params = filterParams();
  if (state.sort && state.sort !== "newest") params.set("sort", state.sort);
  params.set("limit", String(PAGE));
  params.set("offset", String(offset));
  return "/api/photos?" + params.toString();
}

// Enter a "similar" mode (the endpoint ignores filters), so the filter set is
// cleared; exitSimilar restores the normal filtered grid. ``kind`` is "photo"
// (visual SigLIP) or "face" (SFace "same person").
function enterSimilar(kind, id, label) {
  state.similarTo = kind === "photo" ? id : null;
  state.similarFace = kind === "face" ? id : null;
  state.similarLabel = label || "";
  state.filters = {};
  state.searchMode = "filter";
  const search = $("#search"); if (search) search.value = "";
  closeLightbox();
  state.view = "photos";
  document.querySelectorAll(".tab").forEach((t) => {
    const on = t.dataset.view === "photos";
    t.classList.toggle("active", on);
    t.setAttribute("aria-selected", on ? "true" : "false");
  });
  showViewPanel("photos");
  refresh();
}

export function moreLikeThis(id, label) { enterSimilar("photo", id, label); }
export function moreLikeThisFace(faceId, label) { enterSimilar("face", faceId, label); }

export function inSimilarMode() {
  return state.similarTo != null || state.similarFace != null;
}

export function exitSimilar() {
  state.similarTo = state.similarFace = null;
  state.similarLabel = "";
  refresh();
}
