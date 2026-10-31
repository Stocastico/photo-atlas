// Map view (Leaflet).
import { api } from "./api.js";
import { filterParams } from "./filters.js";
import { openPhotoById } from "./lightbox.js";
import { $, esc, toast } from "./util.js";

// ---- map ------------------------------------------------------------------
let _map = null, _markers = null, _leafletIcons = false;

function ensureLeafletIcons() {
  if (_leafletIcons || !window.L) return;
  // Point Leaflet at the locally-vendored marker images (no CDN; offline-safe).
  delete L.Icon.Default.prototype._getIconUrl;
  L.Icon.Default.mergeOptions({
    iconRetinaUrl: "/static/vendor/leaflet/images/marker-icon-2x.png",
    iconUrl: "/static/vendor/leaflet/images/marker-icon.png",
    shadowUrl: "/static/vendor/leaflet/images/marker-shadow.png",
  });
  _leafletIcons = true;
}

export async function renderMap() {
  if (!window.L) { toast("Map library failed to load."); return; }
  ensureLeafletIcons();
  if (!_map) {
    _map = L.map("map", { worldCopyJump: true }).setView([20, 0], 2);
    L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
      maxZoom: 19, attribution: "© OpenStreetMap contributors",
    }).addTo(_map);
  }
  // The container was display:none until the tab opened; re-measure it.
  setTimeout(() => _map && _map.invalidateSize(), 0);

  let data;
  try { data = await api("/api/map?" + filterParams().toString()); }
  catch (e) { return; }

  if (_markers) { _map.removeLayer(_markers); _markers = null; }
  _markers = L.markerClusterGroup
    ? L.markerClusterGroup({ chunkedLoading: true })
    : L.layerGroup();

  $("#map-empty").style.display = data.points.length ? "none" : "block";
  const bounds = [];
  for (const pt of data.points) {
    if (pt.lat == null || pt.lon == null) continue;
    const marker = L.marker([pt.lat, pt.lon]);
    // Build the popup (a thumbnail + caption) lazily, only when the marker is
    // actually clicked. With up to map_point_limit (50k) markers, eagerly
    // creating a DOM subtree per point would hold tens of thousands of unused
    // nodes in memory; at most one popup is ever open.
    marker.bindPopup(() => {
      const pop = document.createElement("div");
      pop.className = "map-pop";
      pop.innerHTML = `<img src="/api/thumb/${pt.id}" alt="" loading="lazy" />
        <div class="cap">${esc(String(pt.year || ""))} · open ↗</div>`;
      pop.onclick = () => openPhotoById(pt.id);
      return pop;
    });
    _markers.addLayer(marker);
    bounds.push([pt.lat, pt.lon]);
  }
  _map.addLayer(_markers);
  if (bounds.length) _map.fitBounds(bounds, { maxZoom: 12, padding: [30, 30] });
}
