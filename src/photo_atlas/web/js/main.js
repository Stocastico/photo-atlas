// Entry point: event wiring and initial load.
import { api } from "./api.js";
import { bulkAction, bulkExport, renderPhotos, repaintSelection, setSelectMode, updateSelectionBar } from "./grid.js";
import { applyLbZoom, closeHelp, closeLightbox, focusableInLightbox, helpOpen, lbZoom, lightboxStep, resetLbZoom, toggleHelp, toggleSlideshow, zoomBy } from "./lightbox.js";
import { setSearchMode } from "./search.js";
import { state } from "./state.js";
import { applyQuery, setRestoringState, syncURL } from "./urlstate.js";
import { $ } from "./util.js";
import { refresh, setView } from "./views.js";

// ---- wiring ---------------------------------------------------------------
const _tabs = [...document.querySelectorAll(".tab")];
_tabs.forEach((t) => (t.onclick = () => setView(t.dataset.view)));
// Arrow/Home/End move focus across the tablist (WAI-ARIA tabs keyboard model).
const _tabbar = document.querySelector(".tabbar");
if (_tabbar) _tabbar.addEventListener("keydown", (e) => {
  const i = _tabs.indexOf(document.activeElement);
  if (i < 0) return;
  let j = i;
  if (e.key === "ArrowRight") j = (i + 1) % _tabs.length;
  else if (e.key === "ArrowLeft") j = (i - 1 + _tabs.length) % _tabs.length;
  else if (e.key === "Home") j = 0;
  else if (e.key === "End") j = _tabs.length - 1;
  else return;
  e.preventDefault();
  _tabs[j].focus();
  setView(_tabs[j].dataset.view);
});
$("#lb-close").onclick = closeLightbox;
$("#lb-prev").onclick = () => lightboxStep(-1);
$("#lb-next").onclick = () => lightboxStep(1);
$("#lightbox").onclick = (e) => { if (e.target.id === "lightbox") closeLightbox(); };
if ($("#lb-play")) $("#lb-play").onclick = toggleSlideshow;
if ($("#lb-help")) $("#lb-help").onclick = toggleHelp;

// Scroll/drag zoom + pan on the image stage. Wired once; the handlers no-op
// unless the lightbox is open. Drag panning only kicks in past 1× zoom.
(function setupLbZoom() {
  const stage = $(".lb-img");
  if (!stage || !stage.addEventListener) return;
  stage.addEventListener("wheel", (e) => {
    if (!$("#lightbox").classList.contains("open")) return;
    e.preventDefault();
    zoomBy(e.deltaY < 0 ? 1.15 : 1 / 1.15);
  }, { passive: false });
  stage.addEventListener("dblclick", () => zoomBy(lbZoom.scale > 1 ? 0.001 : 2.5));
  let dragging = false, lastX = 0, lastY = 0;
  stage.addEventListener("pointerdown", (e) => {
    if (lbZoom.scale <= 1) return;
    dragging = true; lastX = e.clientX; lastY = e.clientY;
    stage.setPointerCapture?.(e.pointerId);
    const img = $("#lb-img"); if (img) img.style.cursor = "grabbing";
  });
  stage.addEventListener("pointermove", (e) => {
    if (!dragging) return;
    lbZoom.tx += e.clientX - lastX; lbZoom.ty += e.clientY - lastY;
    lastX = e.clientX; lastY = e.clientY;
    applyLbZoom();
  });
  const endDrag = () => {
    dragging = false;
    const img = $("#lb-img"); if (img) img.style.cursor = lbZoom.scale > 1 ? "grab" : "zoom-in";
  };
  stage.addEventListener("pointerup", endDrag);
  stage.addEventListener("pointercancel", endDrag);
})();
$("#sort").onchange = (e) => { state.sort = e.target.value; syncURL(); renderPhotos(true); };

// Multi-select wiring.
if ($("#select-toggle")) $("#select-toggle").onclick = () => setSelectMode(!state.selectMode);
if ($("#select-clear")) $("#select-clear").onclick = () => {
  state.selected.clear(); state.selectAnchor = null; repaintSelection(); updateSelectionBar();
};
if ($("#select-all")) $("#select-all").onclick = () => {
  for (const p of state.photos) state.selected.add(p.id);
  repaintSelection(); updateSelectionBar();
};
document.querySelectorAll("#selection-bar [data-bulk]").forEach((b) =>
  (b.onclick = () => bulkAction(b.dataset.bulk)));
if ($("#select-export")) $("#select-export").onclick = bulkExport;

document.addEventListener("keydown", (e) => {
  if (!$("#lightbox").classList.contains("open")) return;
  const typing = /^(INPUT|TEXTAREA|SELECT)$/.test(document.activeElement?.tagName || "");
  if (e.key === "Escape") { if (helpOpen()) closeHelp(); else closeLightbox(); }
  else if (e.key === "ArrowLeft" && !typing) lightboxStep(-1);
  else if (e.key === "ArrowRight" && !typing) lightboxStep(1);
  else if ((e.key === "?" || (e.key === "/" && e.shiftKey)) && !typing) { e.preventDefault(); toggleHelp(); }
  else if (e.key === " " && !typing) { e.preventDefault(); toggleSlideshow(); }
  else if ((e.key === "+" || e.key === "=") && !typing) { e.preventDefault(); zoomBy(1.25); }
  else if (e.key === "-" && !typing) { e.preventDefault(); zoomBy(1 / 1.25); }
  else if (e.key === "0" && !typing) { e.preventDefault(); resetLbZoom(); }
  else if (e.key === "Tab") {
    // Focus trap: keep Tab cycling within the dialog.
    const f = focusableInLightbox();
    if (!f.length) return;
    const first = f[0], last = f[f.length - 1];
    if (e.shiftKey && document.activeElement === first) { e.preventDefault(); last.focus(); }
    else if (!e.shiftKey && document.activeElement === last) { e.preventDefault(); first.focus(); }
  }
});

let searchTimer;
$("#search").oninput = (e) => {
  clearTimeout(searchTimer);
  searchTimer = setTimeout(() => {
    const v = e.target.value.trim();
    state.similarTo = state.similarFace = null; // a new search exits "more like this" mode
    const key = state.searchMode === "semantic" ? "text" : "q";
    const other = key === "text" ? "q" : "text";
    delete state.filters[other];
    if (v) state.filters[key] = v; else delete state.filters[key];
    refresh();
  }, state.searchMode === "semantic" ? 350 : 250);
};

const _searchModeBtn = $("#search-mode");
if (_searchModeBtn) _searchModeBtn.onclick = () =>
  setSearchMode(state.searchMode === "semantic" ? "filter" : "semantic");

// Reveal the semantic toggle only when the backend can serve it.
api("/api/capabilities").then((caps) => {
  state.semanticAvailable = !!(caps && caps.semantic);
  const btn = $("#search-mode");
  if (btn && state.semanticAvailable) btn.hidden = false;
}).catch(() => { /* capabilities are best-effort; leave the toggle hidden */ });

// Restore filters/view from the URL on back/forward navigation.
window.addEventListener("popstate", () => {
  setRestoringState(true);
  applyQuery();
  setView(state.view); // refresh()'s syncURL is skipped while restoring
  setRestoringState(false);
});

// Initial load: hydrate from any querystring (shared/bookmarked link).
applyQuery();
setView(state.view);
