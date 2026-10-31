// People view.
import { api } from "./api.js";
import { renderSidebar } from "./sidebar.js";
import { state } from "./state.js";
import { $, esc } from "./util.js";
import { setView } from "./views.js";

// ---- people ---------------------------------------------------------------
export async function renderPeople() {
  const { persons } = await api("/api/persons");
  const wrap = $("#people");
  wrap.innerHTML = "";
  $("#people-empty").style.display = persons.length ? "none" : "block";
  for (const person of persons) wrap.appendChild(personCard(person, persons));
}

function personCard(person, allPersons) {
  const el = document.createElement("div");
  el.className = "person-card";
  const avatar = person.cover_face_id
    ? `<img class="avatar" src="/api/face/${person.cover_face_id}" onerror="this.style.visibility='hidden'"/>`
    : `<div class="avatar"></div>`;
  el.innerHTML = `
    ${avatar}
    <div class="name" data-role="name">${esc(person.name)}</div>
    <div class="sub">${person.photo_count} photos · ${person.face_count} faces</div>
    <div class="row">
      <button class="ghost" data-act="view" aria-label="View ${esc(person.name)}'s photos">View</button>
      <button class="ghost" data-act="rename" aria-label="Rename ${esc(person.name)}">Rename</button>
    </div>
    <div class="row">
      <button class="ghost" data-act="cover" aria-label="Choose cover photo">Cover</button>
      <button class="ghost" data-act="merge" aria-label="Merge ${esc(person.name)} into another person">Merge</button>
      <button class="ghost danger" data-act="del" aria-label="Delete ${esc(person.name)}">Delete</button>
    </div>
    <div class="person-panel" data-role="panel" hidden></div>`;

  const panel = el.querySelector('[data-role="panel"]');
  const closePanel = () => { panel.hidden = true; panel.innerHTML = ""; };

  el.querySelector('[data-act="view"]').onclick = () => {
    state.filters = { person_id: [person.id] }; setView("photos");
  };
  el.querySelector('[data-act="rename"]').onclick = () => startRename(el, person);
  el.querySelector('[data-act="del"]').onclick = async () => {
    if (!confirm(`Remove ${person.name}? Faces are kept for re-clustering.`)) return;
    await api(`/api/persons/${person.id}`, { method: "DELETE" });
    renderPeople(); renderSidebar();
  };
  el.querySelector('[data-act="cover"]').onclick = () =>
    panel.hidden ? openCoverPicker(panel, person) : closePanel();
  el.querySelector('[data-act="merge"]').onclick = () =>
    panel.hidden ? openMergePicker(panel, person, allPersons) : closePanel();
  return el;
}

function startRename(card, person) {
  const nameEl = card.querySelector('[data-role="name"]');
  if (card.querySelector(".rename-box")) return; // already editing
  const box = document.createElement("div");
  box.className = "rename-box row";
  box.innerHTML = `<input value="${esc(person.name)}" aria-label="New name" />
    <button class="primary">Save</button>`;
  nameEl.replaceWith(box);
  const input = box.querySelector("input");
  input.focus(); input.select();
  const save = async () => {
    const name = input.value.trim();
    if (!name || name === person.name) return renderPeople();
    await api(`/api/persons/${person.id}`, {
      method: "PATCH", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name }),
    });
    renderPeople(); renderSidebar();
  };
  box.querySelector("button").onclick = save;
  input.addEventListener("keydown", (e) => {
    if (e.key === "Enter") save();
    else if (e.key === "Escape") renderPeople();
  });
}

async function openCoverPicker(panel, person) {
  panel.hidden = false;
  panel.innerHTML = `<div class="tagline">Pick a cover photo…</div>`;
  const { faces } = await api(`/api/persons/${person.id}/faces`);
  if (!faces.length) { panel.innerHTML = `<div class="tagline">No face crops available.</div>`; return; }
  panel.innerHTML = "";
  const grid = document.createElement("div");
  grid.className = "cover-grid";
  faces.forEach((face) => {
    const img = document.createElement("img");
    img.src = `/api/face/${face.id}`;
    img.alt = "face crop";
    img.title = "Set as cover";
    if (face.id === person.cover_face_id) img.classList.add("selected");
    img.onclick = async () => {
      await api(`/api/persons/${person.id}/cover`, {
        method: "PUT", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ face_id: face.id }),
      });
      renderPeople();
    };
    grid.appendChild(img);
  });
  panel.appendChild(grid);
}

function openMergePicker(panel, person, allPersons) {
  const others = allPersons.filter((p) => p.id !== person.id);
  panel.hidden = false;
  if (!others.length) { panel.innerHTML = `<div class="tagline">No other people to merge into.</div>`; return; }
  panel.innerHTML = `
    <div class="tagline">Merge <b>${esc(person.name)}</b> into…</div>
    <div class="row">
      <select aria-label="Merge target">
        ${others.map((p) => `<option value="${p.id}">${esc(p.name)}</option>`).join("")}
      </select>
      <button class="primary">Merge</button>
    </div>`;
  const select = panel.querySelector("select");
  panel.querySelector("button").onclick = async () => {
    const target = Number(select.value);
    const targetName = others.find((p) => p.id === target)?.name || "that person";
    if (!confirm(`Merge ${person.name} into ${targetName}? This can't be undone.`)) return;
    await api(`/api/persons/${target}/merge`, {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ source_id: person.id }),
    });
    renderPeople(); renderSidebar();
  };
}
