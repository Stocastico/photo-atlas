// "On this day" view.
import { api } from "./api.js";
import { openPhotoById } from "./lightbox.js";
import { $, esc } from "./util.js";

// ---- memories ("on this day") ---------------------------------------------
export async function renderMemories() {
  const wrap = $("#memories");
  if (!wrap) return;
  const today = new Date();
  let data;
  try {
    data = await api(`/api/memories?month=${today.getMonth() + 1}&day=${today.getDate()}`);
  } catch (e) { return; }

  const title = $("#memories-title");
  if (title) {
    const when = today.toLocaleDateString(undefined, { month: "long", day: "numeric" });
    title.textContent = `On this day · ${when}`;
  }
  wrap.innerHTML = "";
  const groups = (data && data.groups) || [];
  $("#memories-empty").style.display = groups.length ? "none" : "block";

  for (const g of groups) {
    const sec = document.createElement("section");
    sec.className = "memory-year";
    const yearsAgo = today.getFullYear() - Number(g.year);
    const label = yearsAgo > 0
      ? `${g.year} · ${yearsAgo} year${yearsAgo === 1 ? "" : "s"} ago`
      : `${g.year}`;
    const h = document.createElement("h3");
    h.textContent = `${label} · ${g.count} photo${g.count === 1 ? "" : "s"}`;
    sec.appendChild(h);
    const strip = document.createElement("div");
    strip.className = "memory-strip";
    for (const p of g.photos) {
      const card = document.createElement("div");
      card.className = "memory-card";
      card.tabIndex = 0;
      card.setAttribute("role", "button");
      card.setAttribute("aria-label", `Open ${p.filename}`);
      card.innerHTML = `<img loading="lazy" decoding="async" src="/api/thumb/${p.id}" alt="${esc(p.filename)}" />`;
      card.onclick = () => openPhotoById(p.id);
      card.onkeydown = (e) => {
        if (e.key === "Enter" || e.key === " ") { e.preventDefault(); openPhotoById(p.id); }
      };
      strip.appendChild(card);
    }
    sec.appendChild(strip);
    wrap.appendChild(sec);
  }
}
