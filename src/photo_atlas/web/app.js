// Photo Atlas — single page front-end (no build step, vanilla JS).
const state = {
  view: "photos",
  filters: {}, // person_id, scene, country, city, place, year, camera, q
  sort: "newest",
};

const $ = (s) => document.querySelector(s);
const api = async (url, opts) => (await fetch(url, opts)).json();
const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

function fmtDate(iso) {
  if (!iso) return "—";
  return new Date(iso).toLocaleDateString(undefined, { year: "numeric", month: "short", day: "numeric" });
}

// ---- filters / sidebar ----------------------------------------------------
function toggleFilter(key, value) {
  if (state.filters[key] === value) delete state.filters[key];
  else state.filters[key] = value;
  refresh();
}

function chip(label, count, active, onClick) {
  const el = document.createElement("button");
  el.className = "chip" + (active ? " active" : "");
  el.innerHTML = `${esc(label)}${count != null ? ` <span class="n">${count}</span>` : ""}`;
  el.onclick = onClick;
  return el;
}

async function renderSidebar() {
  const f = await api("/api/facets");
  const side = $("#sidebar");
  side.innerHTML = "";

  const section = (title, items, key, labelFn = (i) => i.value, valFn = (i) => i.value) => {
    if (!items || !items.length) return;
    const h = document.createElement("h3");
    h.textContent = title;
    side.appendChild(h);
    const wrap = document.createElement("div");
    wrap.className = "facet";
    items.slice(0, 14).forEach((i) => {
      const v = valFn(i);
      wrap.appendChild(chip(labelFn(i), i.count, state.filters[key] == v, () => toggleFilter(key, v)));
    });
    side.appendChild(wrap);
  };

  const hdr = document.createElement("div");
  hdr.style.display = "flex"; hdr.style.justifyContent = "space-between"; hdr.style.alignItems = "center";
  hdr.innerHTML = `<h3 style="margin:0">${f.total} photos</h3>`;
  const clear = document.createElement("button");
  clear.className = "clear"; clear.textContent = "Clear all";
  clear.onclick = () => { state.filters = {}; $("#search").value = ""; refresh(); };
  hdr.appendChild(clear);
  side.appendChild(hdr);

  section("People", f.persons, "person_id", (i) => `${i.name}`, (i) => i.id);
  section("Scene", f.scenes, "scene");
  section("Country", f.countries, "country");
  section("City", f.cities, "city");
  section("Place", f.places, "place");
  section("Year", f.years, "year");
  section("Camera", f.cameras, "camera");
}

// ---- photos grid ----------------------------------------------------------
async function renderPhotos() {
  const params = new URLSearchParams();
  Object.entries(state.filters).forEach(([k, v]) => v != null && v !== "" && params.set(k, v));
  if (state.sort === "oldest") params.set("sort", "oldest");
  params.set("limit", "120");
  const data = await api("/api/photos?" + params.toString());

  $("#result-count").textContent = `${data.total} result${data.total === 1 ? "" : "s"}`;
  const grid = $("#grid");
  grid.innerHTML = "";
  $("#grid-empty").style.display = data.photos.length ? "none" : "block";

  for (const p of data.photos) {
    const card = document.createElement("div");
    card.className = "card";
    const placeText = p.place_label ? p.place_label.split(",")[0] : p.folder_place;
    const place = placeText ? `<span>${esc(placeText)}</span>` : "<span></span>";
    card.innerHTML = `
      <img loading="lazy" src="/api/thumb/${p.id}" alt="${esc(p.filename)}" />
      ${p.face_count ? `<span class="badge">👤 ${p.face_count}</span>` : ""}
      <div class="meta">${place}<span>${(p.taken_at || "").slice(0, 4)}</span></div>`;
    card.onclick = () => openLightbox(p.id);
    grid.appendChild(card);
  }
}

// ---- lightbox / detail ----------------------------------------------------
async function openLightbox(id) {
  const p = await api(`/api/photos/${id}`);
  $("#lb-img").src = `/api/image/${id}`;
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
    list.appendChild(item);
  }
  $("#lightbox").classList.add("open");
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
      state.filters = { person_id: person.id }; setView("photos");
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
  renderSidebar();
  if (state.view === "photos") renderPhotos();
  else if (state.view === "people") renderPeople();
  else renderClusters();
}

// ---- wiring ---------------------------------------------------------------
document.querySelectorAll(".tab").forEach((t) => (t.onclick = () => setView(t.dataset.view)));
$("#lb-close").onclick = () => $("#lightbox").classList.remove("open");
$("#lightbox").onclick = (e) => { if (e.target.id === "lightbox") $("#lightbox").classList.remove("open"); };
$("#sort").onchange = (e) => { state.sort = e.target.value; renderPhotos(); };
let searchTimer;
$("#search").oninput = (e) => {
  clearTimeout(searchTimer);
  searchTimer = setTimeout(() => {
    const v = e.target.value.trim();
    if (v) state.filters.q = v; else delete state.filters.q;
    renderPhotos();
  }, 250);
};

setView("photos");
