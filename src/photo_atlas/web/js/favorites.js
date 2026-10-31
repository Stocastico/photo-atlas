// Per-photo favourite star.
import { api } from "./api.js";
import { state } from "./state.js";

import { refresh } from "./views.js";

// ---- favorites (per-photo star) -------------------------------------------
// Paint a star button to match a photo's favourite state (used by both the grid
// overlay and the lightbox); keeps the icon, pressed-state and label in sync.
function paintStar(btn, fav) {
  btn.classList.toggle("on", !!fav);
  btn.textContent = fav ? "★" : "☆";
  btn.title = "Toggle favorite";
  btn.setAttribute("aria-pressed", fav ? "true" : "false");
  btn.setAttribute("aria-label", fav ? "Remove from favorites" : "Add to favorites");
}

export function makeStar(photo, { inline = false } = {}) {
  const btn = document.createElement("button");
  btn.className = "star" + (inline ? " star-inline" : "");
  btn.dataset.favId = photo.id;
  paintStar(btn, photo.favorite);
  btn.onclick = (e) => { e.stopPropagation(); toggleFavorite(photo); };
  return btn;
}

// Repaint every star button in the DOM that points at this photo, so the grid
// overlay and the lightbox toggle never disagree.
function syncStars(id, fav) {
  document.querySelectorAll(`.star[data-fav-id="${id}"]`).forEach((b) => paintStar(b, fav));
}

async function toggleFavorite(photo) {
  const next = !photo.favorite;
  try {
    await api(`/api/photos/${photo.id}/favorite`, {
      method: "PUT", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ favorite: next }),
    });
  } catch (e) {
    return; // api() already toasted; leave the star as-is
  }
  const fav = next ? 1 : 0;
  photo.favorite = fav;
  // Keep every in-memory copy in sync (the grid list and the lightbox detail
  // can be distinct objects for the same photo).
  for (const gp of state.photos) if (gp.id === photo.id) gp.favorite = fav;
  syncStars(photo.id, fav);
  // When the Favorites filter is on, un-starring removes the photo from the set,
  // so re-run the query to drop it from the grid.
  if (state.filters.favorite && !next) refresh();
}
