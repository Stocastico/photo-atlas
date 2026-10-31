// Shared UI state and constants (labels, grid geometry).

export const PAGE = 120;
export const state = {
  view: "photos",
  filters: {}, // person_id, scene, country, city, place, year, camera, q
  sort: "newest",
  photos: [],        // currently loaded photos (drives infinite scroll + lightbox)
  offset: 0,
  total: 0,
  loading: false,
  facetData: null,
  lightboxIndex: null,
  facetExpanded: {}, // facet key -> show all values
  rendered: new Map(), // index -> live card node (virtualised grid window)
  layout: null,        // last computed grid layout
  searchMode: "filter", // "filter" (substring q) or "semantic" (natural-language text)
  semanticAvailable: false,
  similarTo: null,    // photo id when the grid is in "more like this" mode
  similarFace: null,  // face id when in "more like this person" mode
  similarLabel: "",   // its filename / face label, for the banner pill
  selectMode: false,  // multi-select: clicks toggle selection instead of opening
  selected: new Set(),// selected photo ids (survives grid windowing/recycling)
  selectAnchor: null, // last-clicked index, for shift-click range selection
};

// Grid windowing constants — GAP/MIN must match the CSS .grid/.card rules.
export const GRID_GAP = 10;
export const CARD_MIN = 160;     // grid-template min column width
export const BUFFER_ROWS = 4;    // rows rendered above/below the viewport

export const FILTER_NAMES = {
  person_id: "Person", scene: "Scene", country: "Country", city: "City",
  place: "Place", year: "Year", camera: "Camera", q: "Search", text: "Smart",
  date_from: "From", date_to: "To", people: "People", known: "Known", kind: "Type",
};
// Friendly labels for the number-of-people buckets (tokens come from the API).
export const PEOPLE_LABELS = {
  "0": "No people", "1": "1 (portrait)", "2-4": "2–4", "5+": "5+",
};
// Friendly labels for the number-of-known-(named)-people buckets.
export const KNOWN_LABELS = {
  "0": "None identified", "1": "1 identified", "2+": "2+ identified",
};
// Friendly labels for the unified "type of picture" facet (portrait/group from
// face_count + the scene tags). Tokens come from the API (search.PICTURE_TYPES).
export const KIND_LABELS = {
  portrait: "👤 Portrait", group: "👥 Group", animals: "🐾 Animals",
  landscape: "🏞 Landscape", plants: "🌿 Plants", food: "🍽 Food",
  vehicle: "🚗 Vehicle", building: "🏠 Building", document: "📄 Document",
  screenshot: "📱 Screenshot", other: "Other",
};
export const FACET_CAP = 14;
