// Name-faces and review views.
import { api } from "./api.js";
import { renderSidebar } from "./sidebar.js";
import { $, esc, toast } from "./util.js";

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
  const qr = $("#quick-review-btn");
  if (qr) qr.onclick = () => startQuickReview().catch(() => toast("Couldn't start quick review."));
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

// ---- quick review ----------------------------------------------------------
// Keyboard-first naming: one face group at a time, the likeliest people on keys
// 1-6, N for someone new, S to skip. Pure helpers are unit-tested in Node.
const QR_MAX = 6;

/** Likeliest people for a group: the "looks like" suggestion, then the most photographed. */
export function quickReviewCandidates(people, suggestion) {
  const out = [];
  if (suggestion && suggestion.person_id != null) {
    out.push({ id: suggestion.person_id, name: suggestion.name });
  }
  const rest = [...people].sort((a, b) => (b.photo_count || 0) - (a.photo_count || 0));
  for (const p of rest) {
    if (out.length >= QR_MAX) break;
    if (!out.some((o) => o.id === p.id)) out.push({ id: p.id, name: p.name });
  }
  return out.slice(0, QR_MAX);
}

/** Map a key press to an action, or null. `n` = number of candidates on screen. */
export function quickReviewAction(key, n) {
  if (/^[1-6]$/.test(key)) { const i = +key - 1; return i < n ? { type: "pick", index: i } : null; }
  const k = key.toLowerCase();
  if (k === "n") return { type: "new" };
  if (k === "s" || key === "ArrowRight") return { type: "skip" };
  return null;
}

export async function startQuickReview() {
  const [{ clusters }, pr] = await Promise.all([api("/api/clusters"), api("/api/persons")]);
  let people = pr.persons || [];
  if (!clusters.length) { toast("No unnamed groups to review.", "info"); return; }
  let i = 0;
  let overlay = $("#qr");
  if (!overlay) {
    overlay = document.createElement("div");
    overlay.id = "qr";
    overlay.className = "qr";
    overlay.setAttribute("role", "dialog");
    overlay.setAttribute("aria-modal", "true");
    overlay.setAttribute("aria-label", "Quick review");
    document.body.appendChild(overlay);
  }
  overlay.hidden = false;
  let cands = [];

  const close = () => {
    overlay.hidden = true;
    document.removeEventListener("keydown", onKey, true);
    renderClusters(); renderSidebar();
  };
  const next = () => { i += 1; if (i >= clusters.length) close(); else show(); };
  const assign = async (body) => {
    await api(`/api/clusters/${clusters[i].cluster_id}/assign`, {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
    });
    if (body.name) people = (await api("/api/persons")).persons || []; // new person: pick up its real id
    next();
  };
  const show = () => {
    const c = clusters[i];
    cands = quickReviewCandidates(people, c.suggestion);
    overlay.innerHTML = `
      <div class="qr-card">
        <div class="qr-progress">${i + 1} / ${clusters.length} · ${c.size} faces
          <button class="ghost" id="qr-close" type="button">Close (Esc)</button></div>
        <div class="qr-faces">${c.samples.map((s) =>
          `<img src="/api/face/${s.id}" alt="" onerror="this.style.visibility='hidden'"/>`).join("")}</div>
        <div class="qr-cands">${cands.map((p, k) =>
          `<button class="primary" data-k="${k}"><kbd>${k + 1}</kbd> ${esc(p.name)}${
            c.suggestion && c.suggestion.person_id === p.id ? " ✨" : ""}</button>`).join("")}</div>
        <div class="qr-new"><kbd>N</kbd> <input id="qr-name" placeholder="New person…" />
          <button class="ghost" id="qr-skip" type="button"><kbd>S</kbd> Skip</button></div>
      </div>`;
    overlay.querySelectorAll("[data-k]").forEach((b) =>
      (b.onclick = () => assign({ person_id: cands[+b.dataset.k].id })));
    $("#qr-close").onclick = close;
    $("#qr-skip").onclick = next;
    const inp = $("#qr-name");
    inp.onkeydown = (e) => {
      e.stopPropagation();
      if (e.key === "Escape") inp.blur();
      if (e.key === "Enter" && inp.value.trim()) {
        assign({ name: inp.value.trim() });
      }
    };
  };
  const onKey = (e) => {
    if (overlay.hidden) return;
    if (e.key === "Escape") { e.preventDefault(); close(); return; }
    if (e.target && e.target.id === "qr-name") return;
    const a = quickReviewAction(e.key, cands.length);
    if (!a) return;
    e.preventDefault();
    if (a.type === "pick") assign({ person_id: cands[a.index].id });
    else if (a.type === "new") $("#qr-name").focus();
    else if (a.type === "skip") next();
  };
  document.addEventListener("keydown", onKey, true);
  show();
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
