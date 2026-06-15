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
};

const FILTER_NAMES = {
  person_id: "Person", scene: "Scene", country: "Country", city: "City",
  place: "Place", year: "Year", camera: "Camera", q: "Search",
  date_from: "From", date_to: "To",
};
const FACET_CAP = 14;

const $ = (s) => document.querySelector(s);
const api = async (url, opts) => (await fetch(url, opts)).json();
const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

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

// Flatten state.filters into [key, value] pairs (one per selected value).
function activeFilterPairs() {
  const pairs = [];
  for (const [k, v] of Object.entries(state.filters)) {
    if (v == null || v === "") continue;
    if (Array.isArray(v)) v.forEach((x) => (x != null && x !== "") && pairs.push([k, x]));
    else pairs.push([k, v]);
  }
  return pairs;
}

async function renderSidebar() {
  const f = await api("/api/facets?" + filterParams().toString());
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
  card.innerHTML = `
    <img loading="lazy" src="/api/thumb/${p.id}" alt="${esc(p.filename)}" />
    ${p.face_count ? `<span class="badge">👤 ${p.face_count}</span>` : ""}
    <div class="meta">${place}<span>${(p.taken_at || "").slice(0, 4)}</span></div>`;
  card.onclick = () => openLightbox(index);
  return card;
}

async function renderPhotos(reset = true) {
  if (state.loading) return;
  state.loading = true;
  if (reset) {
    state.offset = 0;
    state.photos = [];
    $("#grid").innerHTML = "";
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

  const baseIndex = state.photos.length;
  state.photos = state.photos.concat(data.photos);
  state.total = data.total;
  state.offset += data.photos.length;

  $("#result-count").textContent = `${data.total} result${data.total === 1 ? "" : "s"}`;
  $("#grid-empty").style.display = state.photos.length ? "none" : "block";

  const grid = $("#grid");
  data.photos.forEach((p, i) => grid.appendChild(photoCard(p, baseIndex + i)));

  $("#grid-loading").style.display = state.photos.length < state.total ? "block" : "none";
  $("#grid-loading").textContent = "Loading…";
  state.loading = false;
  maybeLoadMore();
}

function maybeLoadMore() {
  if (state.view !== "photos" || state.loading) return;
  if (state.photos.length >= state.total) return;
  // Keep loading until the viewport is filled (so short result sets behave).
  if (document.body.offsetHeight <= window.innerHeight + 600) renderPhotos(false);
}

window.addEventListener("scroll", () => {
  if (state.view !== "photos" || state.loading) return;
  if (state.photos.length >= state.total) return;
  if (window.innerHeight + window.scrollY >= document.body.offsetHeight - 600) renderPhotos(false);
});

// ---- lightbox / detail ----------------------------------------------------
function closeLightbox() {
  $("#lightbox").classList.remove("open");
  state.lightboxIndex = null;
}

function lightboxStep(delta) {
  const i = state.lightboxIndex + delta;
  if (i < 0 || i >= state.photos.length) return;
  openLightbox(i);
}

async function openLightbox(index) {
  state.lightboxIndex = index;
  $("#lb-prev").disabled = index <= 0;
  $("#lb-next").disabled = index >= state.photos.length - 1;

  const base = state.photos[index];
  $("#lightbox").classList.add("open");
  const p = await api(`/api/photos/${base.id}`);
  // Guard against a fast prev/next click landing on a different photo.
  if (state.lightboxIndex !== index) return;

  $("#lb-img").src = `/api/image/${base.id}`;
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
    <h3 style="margin-top:16px">Faces (${p.faces.length})</h3>
    <div class="face-list" id="lb-faces"></div>`;

  const list = $("#lb-faces");
  if (!p.faces.length) list.innerHTML = `<span class="tagline">No faces detected.</span>`;
  for (const face of p.faces) {
    const item = document.createElement("div");
    item.className = "face-item";
    item.innerHTML = `
      <img src="/api/face/${face.id}" onerror="this.style.visibility='hidden'" />
      <input placeholder="name…" value="${esc(face.person_name || "")}" />
      <button class="primary">Save</button>`;
    const input = item.querySelector("input");
    item.querySelector("button").onclick = async () => {
      if (!input.value.trim()) return;
      await api(`/api/faces/${face.id}/assign`, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name: input.value.trim() }),
      });
      renderSidebar();
    };
    input.addEventListener("keydown", (e) => e.key === "Enter" && item.querySelector("button").click());
    list.appendChild(item);
  }
}

// ---- people ---------------------------------------------------------------
async function renderPeople() {
  const { persons } = await api("/api/persons");
  const wrap = $("#people");
  wrap.innerHTML = "";
  $("#people-empty").style.display = persons.length ? "none" : "block";
  for (const person of persons) {
    const el = document.createElement("div");
    el.className = "person-card";
    const avatar = person.cover_face_id
      ? `<img class="avatar" src="/api/face/${person.cover_face_id}" onerror="this.style.visibility='hidden'"/>`
      : `<div class="avatar"></div>`;
    el.innerHTML = `
      ${avatar}
      <div class="name">${esc(person.name)}</div>
      <div class="sub">${person.photo_count} photos · ${person.face_count} faces</div>
      <div class="row">
        <button class="ghost" data-act="view">View photos</button>
        <button class="ghost" data-act="del">Delete</button>
      </div>`;
    el.querySelector('[data-act="view"]').onclick = () => {
      state.filters = { person_id: [person.id] }; setView("photos");
    };
    el.querySelector('[data-act="del"]').onclick = async () => {
      if (!confirm(`Remove ${person.name}? Faces are kept for re-clustering.`)) return;
      await api(`/api/persons/${person.id}`, { method: "DELETE" });
      renderPeople(); renderSidebar();
    };
    wrap.appendChild(el);
  }
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
  document.querySelectorAll(".tab").forEach((t) => t.classList.toggle("active", t.dataset.view === view));
  $("#view-photos").style.display = view === "photos" ? "block" : "none";
  $("#view-people").style.display = view === "people" ? "block" : "none";
  $("#view-clusters").style.display = view === "clusters" ? "block" : "none";
  refresh();
}

function refresh() {
  renderActiveFilters();
  renderSidebar();
  if (state.view === "photos") renderPhotos(true);
  else if (state.view === "people") renderPeople();
  else renderClusters();
}

// ---- wiring ---------------------------------------------------------------
document.querySelectorAll(".tab").forEach((t) => (t.onclick = () => setView(t.dataset.view)));
$("#lb-close").onclick = closeLightbox;
$("#lb-prev").onclick = () => lightboxStep(-1);
$("#lb-next").onclick = () => lightboxStep(1);
$("#lightbox").onclick = (e) => { if (e.target.id === "lightbox") closeLightbox(); };
$("#sort").onchange = (e) => { state.sort = e.target.value; renderPhotos(true); };

document.addEventListener("keydown", (e) => {
  if (!$("#lightbox").classList.contains("open")) return;
  if (e.key === "Escape") closeLightbox();
  else if (e.key === "ArrowLeft") lightboxStep(-1);
  else if (e.key === "ArrowRight") lightboxStep(1);
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

setView("photos");
