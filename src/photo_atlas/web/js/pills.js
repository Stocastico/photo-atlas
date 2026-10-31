// Active-filter pills.
import { exitSimilar, inSimilarMode, removeFilterValue } from "./filters.js";
import { clearAllFilters } from "./sidebar.js";
import { FILTER_NAMES, KIND_LABELS, KNOWN_LABELS, PEOPLE_LABELS, state } from "./state.js";
import { activeFilterPairs } from "./urlstate.js";
import { $, esc } from "./util.js";

// ---- active filter pills --------------------------------------------------
function filterValueLabel(key, value) {
  if (key === "person_id") {
    const p = (state.facetData?.persons || []).find((x) => x.id == value);
    return p ? p.name : `#${value}`;
  }
  if (key === "people") return PEOPLE_LABELS[value] || value;
  if (key === "known") return KNOWN_LABELS[value] || value;
  if (key === "kind") return KIND_LABELS[value] || value;
  return value;
}

export function renderActiveFilters() {
  const bar = $("#active-filters");
  if (!bar) return;
  bar.innerHTML = "";
  const pairs = activeFilterPairs();
  const similar = inSimilarMode();
  bar.style.display = pairs.length || similar ? "flex" : "none";
  if (similar) {
    const icon = state.similarFace != null ? "🧑" : "✨";
    const fallback = "#" + (state.similarFace ?? state.similarTo);
    const text = `${icon} Similar to ${state.similarLabel || fallback}`;
    const pill = document.createElement("button");
    pill.className = "filter-pill";
    pill.innerHTML = `<span>${esc(text)}</span><span class="x">✕</span>`;
    pill.title = "Exit similar photos";
    pill.setAttribute("aria-label", "Exit similar photos");
    pill.onclick = exitSimilar;
    bar.appendChild(pill);
  }
  for (const [k, v] of pairs) {
    const pill = document.createElement("button");
    pill.className = "filter-pill";
    const text = k === "has_faces" ? "Has people"
      : k === "favorite" ? "★ Favorites"
      : k === "hidden" ? "🙈 Hidden"
      : `${FILTER_NAMES[k] || k}: ${filterValueLabel(k, v)}`;
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
