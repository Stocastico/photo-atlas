// Facet sidebar, smart albums, date range and people-mode controls.
import { api, fetchFacets } from "./api.js";
import { chip, filterParams, isActive, toggleFavoriteFilter, toggleFilter, toggleHasFaces, toggleHiddenFilter } from "./filters.js";
import { renderActiveFilters } from "./pills.js";
import { FACET_CAP, KIND_LABELS, KNOWN_LABELS, state } from "./state.js";
import { applyQuery, buildQuery } from "./urlstate.js";
import { $, toast } from "./util.js";
import { refresh, setView } from "./views.js";


export async function renderSidebar() {
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

  // Quick toggles: starred photos, and photos with at least one detected face.
  heading("Quick filters");
  const quick = document.createElement("div");
  quick.className = "facet";
  quick.appendChild(chip("★ Favorites", f.favorites, !!state.filters.favorite, toggleFavoriteFilter));
  quick.appendChild(chip("👤 Has people", f.with_faces, !!state.filters.has_faces, toggleHasFaces));
  if (f.hidden || state.filters.hidden)
    quick.appendChild(chip("🙈 Hidden", f.hidden, !!state.filters.hidden, toggleHiddenFilter));
  side.appendChild(quick);

  section("People", f.persons, "person_id", (i) => `${i.name}`, (i) => i.id);
  renderPeopleModeToggle(side);
  // Unified "type of picture": folds the old "Number of people" + "Scene" sections
  // (portrait/group + scene tags) into one facet.
  section("Type of picture", f.kinds, "kind", (i) => KIND_LABELS[i.value] || i.value);
  section("Known people", f.known, "known", (i) => KNOWN_LABELS[i.value] || i.value);
  section("Country", f.countries, "country");
  section("City", f.cities, "city");
  section("Place", f.places, "place");
  section("Year", f.years, "year");
  dateSection(side, f);
  section("Camera", f.cameras, "camera");

  await renderAlbums(side);

  // Person names just became available — refresh the pills' labels.
  renderActiveFilters();
}

// ---- smart albums (saved searches) ----------------------------------------
// Persist the current filter set under a name and restore it later. The stored
// "query" is the same querystring the URL uses (buildQuery), so loading an album
// reuses the URL-state machinery (applyQuery) to repopulate filters/view/sort.
async function renderAlbums(side) {
  const h = document.createElement("h3");
  h.textContent = "Smart albums";
  side.appendChild(h);

  const save = document.createElement("button");
  save.className = "show-more";
  save.textContent = "💾 Save current search";
  save.onclick = saveCurrentSearch;
  side.appendChild(save);

  let data;
  try { data = await api("/api/albums"); } catch (e) { return; }
  const albums = (data && data.albums) || [];
  if (!albums.length) return;
  const wrap = document.createElement("div");
  wrap.className = "facet album-list";
  for (const al of albums) {
    const row = document.createElement("div");
    row.className = "album-row";
    const open = document.createElement("button");
    open.className = "chip";
    open.textContent = al.name;
    open.title = "Load this saved search";
    open.onclick = () => loadAlbum(al);
    const del = document.createElement("button");
    del.className = "chip icon";
    del.textContent = "✕";
    del.title = "Delete album";
    del.setAttribute("aria-label", `Delete album ${al.name}`);
    del.onclick = (e) => { e.stopPropagation(); deleteAlbum(al.id); };
    row.appendChild(open);
    row.appendChild(del);
    wrap.appendChild(row);
  }
  side.appendChild(wrap);
}

async function saveCurrentSearch() {
  const query = buildQuery(); // filters + non-default view/sort, same as the URL
  const name = (window.prompt("Name this album:", "") || "").trim();
  if (!name) return;
  await api("/api/albums", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ name, query }),
  });
  toast(`Saved album “${name}”.`, "info");
  renderSidebar();
}

function loadAlbum(al) {
  // Push the saved querystring and restore it through the URL-state machinery.
  history.pushState(null, "", al.query ? "?" + al.query : location.pathname);
  applyQuery();
  setView(state.view); // updates the tab highlight and re-renders the view
}

async function deleteAlbum(id) {
  await api(`/api/albums/${id}`, { method: "DELETE" });
  renderSidebar();
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

export function clearAllFilters() {
  state.filters = {};
  state.similarTo = state.similarFace = null;
  $("#search").value = "";
  refresh();
}
