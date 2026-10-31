// URL / history state: filters + view + sort in the querystring.
import { filterParams } from "./filters.js";
import { applySearchMode } from "./search.js";
import { state } from "./state.js";
import { $ } from "./util.js";

// ---- URL / history state --------------------------------------------------
// Reflect filters + view + sort in the querystring so the back button undoes a
// filter and a filtered view is shareable/bookmarkable.
const SCALAR_FILTERS = new Set(["q", "text", "date_from", "date_to", "person_mode"]);
let restoringState = false;

export function buildQuery() {
  const params = filterParams();
  if (state.view !== "photos") params.set("view", state.view);
  if (state.sort && state.sort !== "newest") params.set("sort", state.sort);
  return params.toString();
}

export function syncURL() {
  if (restoringState) return;
  const qs = buildQuery();
  const target = qs ? "?" + qs : location.pathname;
  const current = location.search || location.pathname;
  if (target !== current) history.pushState(null, "", target);
}

export function applyQuery() {
  const params = new URLSearchParams(location.search);
  const filters = {};
  for (const [k, v] of params.entries()) {
    if (k === "view" || k === "sort") continue;
    if (k === "has_faces" || k === "favorite" || k === "hidden") filters[k] = true;
    else if (SCALAR_FILTERS.has(k)) filters[k] = v;
    else (filters[k] = filters[k] || []).push(v);
  }
  state.filters = filters;
  // Similar mode isn't part of the URL state, so any history navigation (or a
  // fresh load) returns to the normal filtered grid.
  state.similarTo = state.similarFace = null;
  state.similarLabel = "";
  state.view = params.get("view") || "photos";
  state.sort = params.get("sort") || "newest";
  // A `text` filter implies the semantic search mode; otherwise plain substring.
  state.searchMode = filters.text != null ? "semantic" : "filter";
  const search = $("#search");
  if (search) search.value = (filters.text != null ? filters.text : filters.q) || "";
  applySearchMode();
  const sort = $("#sort"); if (sort) sort.value = state.sort;
}

// Flatten state.filters into [key, value] pairs (one per selected value).
export function activeFilterPairs() {
  const pairs = [];
  for (const [k, v] of Object.entries(state.filters)) {
    if (v == null || v === "") continue;
    if (k === "person_mode") continue; // a modifier on the person filter, not a value
    if (Array.isArray(v)) v.forEach((x) => (x != null && x !== "") && pairs.push([k, x]));
    else pairs.push([k, v]);
  }
  return pairs;
}


// Set from the popstate handler (an imported binding can't be reassigned elsewhere).
export function setRestoringState(v) { restoringState = v; }
