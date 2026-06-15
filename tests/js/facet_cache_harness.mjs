// Headless harness for the client-side facet cache in ``photo_atlas/web/app.js``.
// Like the URL-state harness, it evaluates the script in a minimal fake DOM and
// then exercises the pure cache helper ``fetchFacets`` (plus ``api``'s cache
// invalidation) with a counting ``fetch`` to assert that:
//   * a repeated filter signature is served from cache (no second round-trip),
//   * a different signature misses the cache (a new round-trip),
//   * a mutating request through ``api`` invalidates the cache, a GET does not.
//
// Run via ``node tests/js/facet_cache_harness.mjs`` (or the pytest wrapper in
// ``tests/test_web_js.py``, which skips when Node isn't installed).
import fs from "fs";
import path from "path";
import url from "url";
import vm from "vm";

const here = path.dirname(url.fileURLToPath(import.meta.url));
const appjs = path.resolve(here, "../../src/photo_atlas/web/app.js");

function fakeEl() {
  return {
    style: {}, dataset: {}, children: [],
    classList: { add() {}, remove() {}, toggle() {}, contains() { return false; } },
    setAttribute() {}, getAttribute() { return null; }, removeAttribute() {},
    appendChild() {}, replaceWith() {}, querySelector() { return fakeEl(); },
    querySelectorAll() { return []; }, addEventListener() {}, focus() {}, select() {},
    getBoundingClientRect() { return { top: 0, left: 0, width: 0, height: 0 }; },
    onclick: null, onkeydown: null, onchange: null, oninput: null,
    value: "", textContent: "", innerHTML: "", disabled: false, offsetParent: {},
  };
}

const elements = {};
const loc = { pathname: "/", search: "" };
const hist = { pushState() {} };

// A counting fetch. ``/api/facets`` returns an empty-but-valid facets payload;
// any other URL returns an empty photos page so the load-time render can't throw.
let fetchCount = 0;
const sandbox = {
  document: {
    querySelector(sel) { return elements[sel] || (elements[sel] = fakeEl()); },
    querySelectorAll() { return []; },
    addEventListener() {}, createElement() { return fakeEl(); },
    body: { offsetHeight: 1000 }, activeElement: { tagName: "BODY" },
  },
  window: { addEventListener() {}, innerHeight: 800, scrollY: 0, location: loc, history: hist },
  location: loc, history: hist, URLSearchParams, console,
  requestAnimationFrame() { return 0; },
  fetch: async (u) => {
    fetchCount++;
    const facets = { total: 0, with_faces: 0, persons: [], scenes: [] };
    const body = String(u).startsWith("/api/facets") ? facets : { total: 0, photos: [] };
    return { ok: true, json: async () => body };
  },
  setTimeout: () => 0, clearTimeout: () => {},
};
vm.createContext(sandbox);

const src = fs.readFileSync(appjs, "utf8") +
  "\nthis.__fetchFacets=fetchFacets;this.__api=api;this.__facetCache=facetCache;";
vm.runInContext(src, sandbox);

const { __fetchFacets: fetchFacets, __api: api, __facetCache: facetCache } = sandbox;

let pass = 0, fail = 0;
const check = (name, cond) => { cond ? pass++ : (fail++, console.error("FAIL:", name)); };

async function main() {
  // The script runs an initial render at load (async). Let those promises fully
  // settle on the real timer queue, then start from a clean slate so the
  // load-time fetches don't pollute the counts below.
  await new Promise((r) => globalThis.setTimeout(r, 50));
  facetCache.clear();
  fetchCount = 0;

  // First time a signature is seen -> network + cache.
  await fetchFacets("scene=food");
  check("first signature fetches", fetchCount === 1);
  check("payload cached", facetCache.size === 1);

  // Same signature again -> served from cache, no new round-trip.
  await fetchFacets("scene=food");
  check("repeat signature is cached", fetchCount === 1);

  // Different signature -> cache miss -> fetch again.
  await fetchFacets("scene=landscape");
  check("new signature fetches", fetchCount === 2);
  check("both signatures cached", facetCache.size === 2);

  // A mutating request through api() clears the cache; a later fetch re-hits
  // the network (rather than serving the now-stale cached payload).
  await api("/api/persons/1", { method: "PATCH", body: "{}" });
  check("mutation clears cache", facetCache.size === 0);
  const afterMutation = fetchCount;
  await fetchFacets("scene=food");
  check("fetch after mutation refetches", fetchCount === afterMutation + 1);

  // A GET through api() must NOT clear the cache.
  const before = facetCache.size;
  await api("/api/photos");
  check("GET does not clear cache", facetCache.size === before);

  console.log(`${pass} passed, ${fail} failed`);
  if (fail) process.exit(1);
}

main();
