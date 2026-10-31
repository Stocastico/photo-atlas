// Duplicates / bursts view.
import { api } from "./api.js";
import { openPhotoById } from "./lightbox.js";
import { $, esc, toast } from "./util.js";

// ---- duplicates / bursts --------------------------------------------------
// The "remove set" for a group is every shot whose checkbox is ticked — the
// cover (best of N) starts unticked, the rest ticked, but the user can override.
function dupSelectedIds(section) {
  return [...section.querySelectorAll("input[type=checkbox]:checked")]
    .map((c) => Number(c.dataset.id));
}

async function dupAction(section, action) {
  const ids = dupSelectedIds(section);
  if (!ids.length) { toast("Nothing selected to remove."); return; }
  if (action === "delete") {
    const ok = window.confirm(
      `Delete ${ids.length} file${ids.length === 1 ? "" : "s"} from the library? `
      + "Originals are moved to the Photo Atlas trash (empty it with `photo-atlas empty-trash`).");
    if (!ok) return;
    await api("/api/photos/delete", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ ids }),
    });
    toast(`Moved ${ids.length} file${ids.length === 1 ? "" : "s"} to trash.`);
  } else {
    await api("/api/photos/bulk", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ ids, action: "hide" }),
    });
    toast(`Hid ${ids.length} shot${ids.length === 1 ? "" : "s"}.`);
  }
  renderDuplicates(); // groups shift once members are removed/hidden
}

export async function renderDuplicates() {
  const wrap = $("#duplicates");
  if (!wrap) return;
  let data;
  try { data = await api("/api/duplicates"); }
  catch (e) { return; }

  wrap.innerHTML = "";
  const groups = (data && data.groups) || [];
  $("#duplicates-empty").style.display = groups.length ? "none" : "block";
  const summary = $("#dup-summary");
  if (summary) {
    summary.textContent = groups.length
      ? `${groups.length} set${groups.length === 1 ? "" : "s"} · ${data.redundant} redundant shot${data.redundant === 1 ? "" : "s"}`
      : "";
  }

  for (const g of groups) {
    const sec = document.createElement("section");
    sec.className = "dup-group";
    const head = document.createElement("div");
    head.className = "trip-head";
    const h = document.createElement("h3");
    h.textContent = `${g.count} near-identical shots`;
    head.appendChild(h);
    const meta = document.createElement("span");
    meta.className = "trip-meta";
    meta.textContent = "keep ★, remove the rest";
    head.appendChild(meta);
    const hideBtn = document.createElement("button");
    hideBtn.className = "ghost";
    hideBtn.type = "button";
    hideBtn.textContent = "🙈 Hide selected";
    hideBtn.onclick = () => dupAction(sec, "hide");
    const delBtn = document.createElement("button");
    delBtn.className = "ghost danger";
    delBtn.type = "button";
    delBtn.textContent = "🗑 Delete selected";
    delBtn.onclick = () => dupAction(sec, "delete");
    head.appendChild(hideBtn);
    head.appendChild(delBtn);
    sec.appendChild(head);

    const strip = document.createElement("div");
    strip.className = "memory-strip";
    for (const p of g.photos) {
      const isCover = p.id === g.cover_id;
      const card = document.createElement("label");
      card.className = "dup-card" + (isCover ? " cover" : "");
      card.innerHTML = `
        <input type="checkbox" data-id="${p.id}" ${isCover ? "" : "checked"}
          aria-label="Mark ${esc(p.filename)} for removal" />
        ${isCover ? '<span class="dup-best" title="Best of the set — kept">★</span>' : ""}
        <img loading="lazy" decoding="async" src="/api/thumb/${p.id}" alt="${esc(p.filename)}" />`;
      // Clicking the thumbnail opens the lightbox; the checkbox handles selection.
      card.querySelector("img").onclick = (e) => { e.preventDefault(); openPhotoById(p.id); };
      strip.appendChild(card);
    }
    sec.appendChild(strip);
    wrap.appendChild(sec);
  }
}
