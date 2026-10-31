// Lightbox: detail panel, zoom, slideshow, EXIF, shortcuts.
import { api } from "./api.js";
import { makeStar } from "./favorites.js";
import { moreLikeThis, moreLikeThisFace } from "./filters.js";
import { renderPhotos } from "./grid.js";
import { renderSidebar } from "./sidebar.js";
import { state } from "./state.js";
import { $, esc, fmtDate } from "./util.js";

// ---- lightbox / detail ----------------------------------------------------
// Power tools: scroll/drag zoom + pan, a slideshow, an EXIF info panel and a
// "?" shortcut legend.

const LB_MAX_ZOOM = 6;
// Pure zoom model (centre-anchored): given the image's current {scale, tx, ty}
// and a multiplicative `factor`, return the next transform. Scale is clamped to
// [1, max]; at 1× the image snaps back to centred, and the pan is rescaled with
// the zoom so the focused point stays put. Unit-tested via a Node harness.
export function nextZoom(view, factor, max = LB_MAX_ZOOM) {
  const scale = Math.min(max, Math.max(1, view.scale * factor));
  if (scale === 1) return { scale: 1, tx: 0, ty: 0 };
  const ratio = scale / view.scale;
  return { scale, tx: view.tx * ratio, ty: view.ty * ratio };
}

export let lbZoom = { scale: 1, tx: 0, ty: 0 };
export function applyLbZoom() {
  const img = $("#lb-img");
  if (!img) return;
  img.style.transform = `translate(${lbZoom.tx}px, ${lbZoom.ty}px) scale(${lbZoom.scale})`;
  img.style.cursor = lbZoom.scale > 1 ? "grab" : "zoom-in";
  const stage = $(".lb-img");
  if (stage) stage.classList.toggle("zoomed", lbZoom.scale > 1);
}
export function resetLbZoom() { lbZoom = { scale: 1, tx: 0, ty: 0 }; applyLbZoom(); }
export function zoomBy(factor) { lbZoom = nextZoom(lbZoom, factor); applyLbZoom(); }

// Slideshow: auto-advance through the loaded result set (pulling further pages
// via lightboxStep), stopping at the end of the library or when the lightbox
// closes. Only meaningful for the grid path (an index-addressed list).
let slideTimer = null;
function stopSlideshow() {
  if (slideTimer) { clearInterval(slideTimer); slideTimer = null; }
  const b = $("#lb-play");
  if (b) { b.textContent = "▶"; b.setAttribute("aria-pressed", "false"); b.title = "Start slideshow (Space)"; }
}
export function toggleSlideshow() {
  if (slideTimer) { stopSlideshow(); return; }
  if (state.lightboxIndex == null) return; // single-photo (map) view: nothing to advance
  slideTimer = setInterval(async () => {
    const before = state.lightboxIndex;
    await lightboxStep(1);
    if (state.lightboxIndex === before) stopSlideshow(); // reached the end
  }, 3500);
  const b = $("#lb-play");
  if (b) { b.textContent = "⏸"; b.setAttribute("aria-pressed", "true"); b.title = "Pause slideshow (Space)"; }
}

// Keyboard-shortcut legend, built lazily and toggled with "?".
function helpPanel() {
  let el = $("#lb-help-panel");
  if (!el) {
    el = document.createElement("div");
    el.id = "lb-help-panel";
    el.className = "lb-help";
    el.hidden = true;
    el.innerHTML = `<div class="lb-help-card" role="dialog" aria-label="Keyboard shortcuts">
      <h3>Keyboard shortcuts</h3>
      <dl>
        <dt>← / →</dt><dd>Previous / next photo</dd>
        <dt>Space</dt><dd>Play / pause slideshow</dd>
        <dt>+ / − / 0</dt><dd>Zoom in / out / reset</dd>
        <dt>Double-click</dt><dd>Toggle zoom</dd>
        <dt>Drag</dt><dd>Pan when zoomed in</dd>
        <dt>?</dt><dd>Toggle this help</dd>
        <dt>Esc</dt><dd>Close</dd>
      </dl></div>`;
    el.onclick = () => { el.hidden = true; };
    $("#lightbox").appendChild(el);
  }
  return el;
}
export function helpOpen() { const el = $("#lb-help-panel"); return !!el && !el.hidden; }
export function toggleHelp() { const el = helpPanel(); el.hidden = !el.hidden; }
export function closeHelp() { const el = $("#lb-help-panel"); if (el) el.hidden = true; }

export function closeLightbox() {
  $("#lightbox").classList.remove("open");
  state.lightboxIndex = null;
  stopSlideshow();
  closeHelp();
  resetLbZoom();
  const vid = $("#lb-video"); // stop playback (and audio) when closing
  if (vid && vid.pause) vid.pause();
  // Restore focus to whatever opened the lightbox (the photo card).
  if (state.lastFocus && state.lastFocus.focus) state.lastFocus.focus();
  state.lastFocus = null;
}

export function focusableInLightbox() {
  const lb = $("#lightbox");
  return [...lb.querySelectorAll('button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])')]
    .filter((el) => !el.disabled && el.offsetParent !== null);
}

export async function lightboxStep(delta) {
  const i = state.lightboxIndex + delta;
  if (i < 0) return;
  if (i >= state.photos.length) {
    // Stepping past the last loaded photo pulls the next page (if any) so the
    // lightbox keeps going instead of dead-ending mid-library.
    if (state.photos.length >= state.total) return;
    await renderPhotos(false);
    if (i >= state.photos.length) return;
  }
  openLightbox(i);
}

// Render the detail side-panel (metadata + editable faces) for a photo. The
// ``reopen`` thunk is how a face edit refreshes the panel in place — it differs
// for grid (index-based) vs map (id-based) entry points.
function renderLightboxSide(p, id, reopen) {
  // Videos play inline (poster from the preview frame, stream from the original);
  // photos use the reusable <img>. Swap the stage element so navigating between a
  // video and a photo restores the right one.
  const stage = $(".lb-img");
  if (p.is_video) {
    stage.innerHTML = `<video id="lb-video" controls preload="metadata" playsinline
      poster="/api/preview/${id}" src="/api/image/${id}"></video>`;
  } else {
    if (!$("#lb-img")) stage.innerHTML = `<img id="lb-img" src="" alt="" />`;
    $("#lb-img").src = `/api/preview/${id}`;
    resetLbZoom(); // a fresh image starts un-zoomed
  }
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
    <p class="lb-actions" id="lb-actions"><a href="/api/image/${id}" target="_blank" rel="noopener">View full size ↗</a></p>
    <h3 style="margin-top:16px">Faces (${p.faces.length})</h3>
    <div class="face-list" id="lb-faces"></div>`;

  // A favourite toggle for the open photo; shares paint/sync logic with the grid.
  $("#lb-actions").prepend(makeStar(p, { inline: true }));

  // EXIF info panel: capture settings are read on demand (not in the grid
  // payload), so fetch them the first time the panel is opened.
  const info = document.createElement("button");
  info.className = "ghost";
  info.type = "button";
  info.textContent = "ℹ︎ Info";
  info.title = "Camera settings (EXIF)";
  info.setAttribute("aria-pressed", "false");
  const exifBox = document.createElement("div");
  exifBox.className = "lb-exif";
  exifBox.hidden = true;
  let exifLoaded = false;
  info.onclick = async () => {
    exifBox.hidden = !exifBox.hidden;
    info.setAttribute("aria-pressed", exifBox.hidden ? "false" : "true");
    if (exifLoaded || exifBox.hidden) return;
    exifLoaded = true;
    const s = await api(`/api/exif/${id}`).catch(() => ({}));
    const rows = [
      ["Aperture", s.aperture], ["Shutter", s.shutter], ["ISO", s.iso],
      ["Focal length", s.focal_length], ["Lens", s.lens],
    ].filter(([, v]) => v);
    exifBox.innerHTML = rows.length
      ? rows.map(([k, v]) => `<div class="kv"><span>${k}</span><span>${esc(v)}</span></div>`).join("")
      : `<span class="tagline">No camera settings in EXIF.</span>`;
  };
  $("#lb-actions").appendChild(info);
  $("#lb-actions").after(exifBox);

  // "More like this" reuses the stored SigLIP embeddings, so only offer it when
  // the library is embedded (same signal the Smart-search toggle uses).
  if (state.semanticAvailable) {
    const sim = document.createElement("button");
    sim.className = "ghost";
    sim.type = "button";
    sim.textContent = "✨ More like this";
    sim.title = "Find visually similar photos";
    sim.onclick = () => moreLikeThis(id, p.filename);
    $("#lb-actions").appendChild(sim);
  }

  const list = $("#lb-faces");
  if (!p.faces.length) list.innerHTML = `<span class="tagline">No faces detected.</span>`;
  for (const face of p.faces) {
    const item = document.createElement("div");
    item.className = "face-item";
    // A named face also gets a "✕" to reassign it back to unknown; typing a
    // different name and saving reassigns it to that (new or existing) person.
    item.innerHTML = `
      <img src="/api/face/${face.id}" onerror="this.style.visibility='hidden'" />
      <input placeholder="name…" value="${esc(face.person_name || "")}" aria-label="Assign face to a person" />
      <button class="primary">Save</button>
      ${face.person_id ? `<button class="ghost icon" data-act="clear" title="Reassign to unknown" aria-label="Reassign face to unknown">✕</button>` : ""}
      <button class="ghost icon" data-act="similar" title="Find more photos of this person" aria-label="Find more photos of this person">🧑</button>`;
    const input = item.querySelector("input");
    const refresh = () => { renderSidebar(); reopen(); };
    // "More like this person": SFace similarity gathers other shots of the same
    // face — the only way to collect an *unnamed* person (the filter needs a name).
    item.querySelector('[data-act="similar"]').onclick = () =>
      moreLikeThisFace(face.id, face.person_name || "this face");
    item.querySelector('.primary').onclick = async () => {
      if (!input.value.trim()) return;
      await api(`/api/faces/${face.id}/assign`, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name: input.value.trim() }),
      });
      refresh();
    };
    const clear = item.querySelector('[data-act="clear"]');
    if (clear) clear.onclick = async () => {
      await api(`/api/faces/${face.id}/unassign`, { method: "POST" });
      refresh();
    };
    input.addEventListener("keydown", (e) => e.key === "Enter" && item.querySelector('.primary').click());
    list.appendChild(item);
  }
}

export async function openLightbox(index) {
  const wasOpen = $("#lightbox").classList.contains("open");
  if (!wasOpen) state.lastFocus = document.activeElement;
  state.lightboxIndex = index;
  $("#lb-prev").disabled = index <= 0;
  // "Next" stays enabled at the last loaded photo when more pages remain on the
  // server, so the arrow can trigger the next load (see lightboxStep).
  $("#lb-next").disabled = index >= state.photos.length - 1 && state.photos.length >= state.total;

  const base = state.photos[index];
  $("#lightbox").classList.add("open");
  if (!wasOpen) $("#lb-close").focus();
  const p = await api(`/api/photos/${base.id}`);
  // Guard against a fast prev/next click landing on a different photo.
  if (state.lightboxIndex !== index) return;
  // Bounded preview derivative keeps memory flat while flicking; full original
  // stays behind the "View full size" link (see renderLightboxSide).
  renderLightboxSide(p, base.id, () => openLightbox(index));
}

// Open a single photo by id (used by map markers). There's no surrounding
// result list to page through, so prev/next are disabled.
export async function openPhotoById(id) {
  const wasOpen = $("#lightbox").classList.contains("open");
  if (!wasOpen) state.lastFocus = document.activeElement;
  state.lightboxIndex = null;
  $("#lb-prev").disabled = true;
  $("#lb-next").disabled = true;
  $("#lightbox").classList.add("open");
  if (!wasOpen) $("#lb-close").focus();
  const p = await api(`/api/photos/${id}`);
  if (!$("#lightbox").classList.contains("open")) return; // closed while loading
  renderLightboxSide(p, id, () => openPhotoById(id));
}
