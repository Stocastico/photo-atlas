// Name-faces and review views.
import { api } from "./api.js";
import { renderSidebar } from "./sidebar.js";
import { $, esc } from "./util.js";

// ---- clusters (name faces) ------------------------------------------------
// Active-learning review: low-confidence auto-tags the user can confirm or
// reject. Confirming assigns the proposed person (confidence → 1.0); rejecting
// unassigns and records a "not this person" negative for future recognition.
async function renderReview() {
  const section = $("#review-section");
  const wrap = $("#review");
  if (!section || !wrap) return;
  let data;
  try {
    data = await api("/api/faces/review");
  } catch (e) { return; }
  const faces = (data && data.faces) || [];
  section.style.display = faces.length ? "block" : "none";
  wrap.innerHTML = "";
  for (const f of faces) {
    const el = document.createElement("div");
    el.className = "review-card";
    el.innerHTML = `
      <img src="/api/face/${f.id}" onerror="this.style.visibility='hidden'" />
      <div class="sub">Looks like <strong>${esc(f.person_name)}</strong>?</div>
      <div class="row">
        <button class="primary" data-act="confirm">✓ Yes</button>
        <button class="ghost" data-act="reject">✗ Not ${esc(f.person_name)}</button>
      </div>`;
    const refresh = () => { renderReview(); renderSidebar(); };
    el.querySelector('[data-act="confirm"]').onclick = async () => {
      await api(`/api/faces/${f.id}/assign`, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ person_id: f.person_id }),
      });
      refresh();
    };
    el.querySelector('[data-act="reject"]').onclick = async () => {
      await api(`/api/faces/${f.id}/unassign`, { method: "POST" });
      refresh();
    };
    wrap.appendChild(el);
  }
}

export async function renderClusters() {
  renderReview();
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
