// Search box modes: substring filter vs natural-language semantic.
import { state } from "./state.js";
import { $ } from "./util.js";
import { refresh } from "./views.js";

// ---- search: substring filter vs natural-language semantic ----------------
// One input drives two modes. "filter" sets `q` (a substring match); "semantic"
// sets `text` (SigLIP image-embedding ranking). The ✨ toggle appears only when
// the server reports semantic search is available (embeddings + the scene extra).
export function applySearchMode() {
  const input = $("#search");
  const btn = $("#search-mode");
  const semantic = state.searchMode === "semantic";
  if (input) input.placeholder = semantic
    ? "Describe a photo… (e.g. kids on the beach)"
    : "Search name, place, camera…";
  if (btn) {
    btn.classList.toggle("active", semantic);
    btn.setAttribute("aria-pressed", semantic ? "true" : "false");
    // Reveal the toggle once we know semantic search is available, or whenever a
    // semantic query is already active (e.g. opening a shared ?text= link).
    if (semantic || state.semanticAvailable) btn.hidden = false;
  }
}

export function setSearchMode(mode) {
  if (state.searchMode === mode) return;
  state.searchMode = mode;
  // Carry whatever's typed across to the new mode's filter key so toggling
  // doesn't silently drop the query.
  const v = ($("#search").value || "").trim();
  delete state.filters.q;
  delete state.filters.text;
  if (v) state.filters[mode === "semantic" ? "text" : "q"] = v;
  applySearchMode();
  refresh();
}
