// View switching and the central refresh().
import { renderClusters } from "./clusters.js";
import { renderDuplicates } from "./duplicates.js";
import { renderPhotos } from "./grid.js";
import { renderMap } from "./map.js";
import { renderMemories } from "./memories.js";
import { renderPeople } from "./people.js";
import { renderActiveFilters } from "./pills.js";
import { renderSidebar } from "./sidebar.js";
import { state } from "./state.js";
import { renderTrips } from "./trips.js";
import { syncURL } from "./urlstate.js";
import { $ } from "./util.js";

// ---- view switching -------------------------------------------------------
const VIEWS = ["photos", "memories", "trips", "duplicates", "map", "people", "clusters"];

export function showViewPanel(view) {
  for (const v of VIEWS) {
    const el = $("#view-" + v);
    if (el) el.style.display = v === view ? "block" : "none";
  }
}

export function setView(view) {
  state.view = view;
  document.querySelectorAll(".tab").forEach((t) => {
    const on = t.dataset.view === view;
    t.classList.toggle("active", on);
    t.setAttribute("aria-selected", on ? "true" : "false");
    t.tabIndex = on ? 0 : -1; // roving tabindex for the WAI-ARIA tabs pattern
  });
  showViewPanel(view);
  refresh();
}

export function refresh() {
  syncURL();
  renderActiveFilters();
  renderSidebar();
  if (state.view === "photos") renderPhotos(true);
  else if (state.view === "memories") renderMemories();
  else if (state.view === "trips") renderTrips();
  else if (state.view === "duplicates") renderDuplicates();
  else if (state.view === "map") renderMap();
  else if (state.view === "people") renderPeople();
  else renderClusters();
}
