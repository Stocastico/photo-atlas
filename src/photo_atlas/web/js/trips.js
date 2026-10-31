// Trips view.
import { api } from "./api.js";
import { openPhotoById } from "./lightbox.js";
import { state } from "./state.js";
import { $, esc } from "./util.js";
import { setView } from "./views.js";

// ---- trips (auto-detected from date gaps + GPS) ---------------------------
function fmtTripRange(start, end) {
  const opts = { year: "numeric", month: "short", day: "numeric" };
  const s = new Date(start + "T00:00:00").toLocaleDateString(undefined, opts);
  if (start === end) return s;
  const e = new Date(end + "T00:00:00").toLocaleDateString(undefined, opts);
  return `${s} – ${e}`;
}

export async function renderTrips() {
  const wrap = $("#trips");
  if (!wrap) return;
  let data;
  try {
    data = await api("/api/trips");
  } catch (e) { return; }

  wrap.innerHTML = "";
  const trips = (data && data.trips) || [];
  $("#trips-empty").style.display = trips.length ? "none" : "block";

  for (const t of trips) {
    const sec = document.createElement("section");
    sec.className = "memory-year";
    const head = document.createElement("div");
    head.className = "trip-head";
    const h = document.createElement("h3");
    h.textContent = t.place || "Unknown place";
    const meta = document.createElement("span");
    meta.className = "trip-meta";
    meta.textContent = `${fmtTripRange(t.start, t.end)} · ${t.count} photo${t.count === 1 ? "" : "s"}`;
    head.appendChild(h);
    head.appendChild(meta);
    // "Browse all" loads the whole trip into the grid via its date range.
    const browse = document.createElement("button");
    browse.className = "ghost";
    browse.type = "button";
    browse.textContent = "Browse all →";
    browse.onclick = () => {
      state.similarTo = state.similarFace = null;
      state.filters = { date_from: t.start, date_to: t.end };
      setView("photos");
    };
    head.appendChild(browse);
    sec.appendChild(head);

    const strip = document.createElement("div");
    strip.className = "memory-strip";
    for (const p of t.photos) {
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
