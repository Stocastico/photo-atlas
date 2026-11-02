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

// "Looks like Anna" banner for a cluster whose faces resemble someone you've named.
function suggestionHtml(s) {
  if (!s || !s.name) return "";
  const weak = s.agreement < 0.6;
  return `<div class="suggest${weak ? " weak" : ""}">
      Looks like <strong>${esc(s.name)}</strong>
      <span class="conf">${Math.round(s.confidence * 100)}% match${weak ? " · mixed group" : ""}</span><br/>
      <button class="primary" data-act="accept">✓ Yes, it's ${esc(s.name)}</button>
    </div>`;
}

// "Auto-tag": run recognition over the whole catalog from the people you've confirmed.
export function setupRecognize() {
  const btn = $("#recognize-btn");
  if (!btn) return;
  btn.onclick = async () => {
    const status = $("#recognize-status");
    btn.disabled = true;
    status.textContent = "Matching faces…";
    try {
      const r = await api("/api/faces/recognize", { method: "POST" });
      if (!r.enrolled) {
        status.textContent = "Name a few people first (below), then run this again.";
      } else {
        const bits = [`Tagged ${r.tagged} more face${r.tagged === 1 ? "" : "s"}`];
        if (r.people) bits.push(`across ${r.people} ${r.people === 1 ? "person" : "people"}`);
        if (r.retagged) bits.push(`${r.retagged} corrected`);
        if (r.cleared) bits.push(`${r.cleared} weak guess${r.cleared === 1 ? "" : "es"} cleared`);
        if (r.need_review) bits.push(`${r.need_review} to double-check under “Review guesses”`);
        status.textContent = bits.join(" · ") + ".";
      }
      await renderClusters();
      renderSidebar();
    } catch (e) {
      status.textContent = "Couldn't run recognition.";
    } finally {
      btn.disabled = false;
    }
  };
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
      ${suggestionHtml(c.suggestion)}
      <div class="row">
        <input placeholder="Who is this?" />
        <button class="primary">Name</button>
      </div>`;
    const accept = el.querySelector('[data-act="accept"]');
    if (accept) {
      accept.onclick = async () => {
        await api(`/api/clusters/${c.cluster_id}/assign`, {
          method: "POST", headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ person_id: c.suggestion.person_id }),
        });
        renderClusters(); renderSidebar();
      };
    }
    const input = el.querySelector("input");
    el.querySelector(".row button").onclick = async () => {
      if (!input.value.trim()) return;
      await api(`/api/clusters/${c.cluster_id}/assign`, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name: input.value.trim() }),
      });
      renderClusters(); renderSidebar();
    };
    input.addEventListener("keydown", (e) => e.key === "Enter" && el.querySelector(".row button").click());
    wrap.appendChild(el);
  }
}
