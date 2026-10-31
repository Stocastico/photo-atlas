// Headless harness for the client-side facet cache in ``photo_atlas/web/js/*.js``.
// Like the URL-state harness, it imports the real modules over a minimal fake DOM and
// then exercises the pure cache helper ``fetchFacets`` (plus ``api``'s cache
// invalidation) with a counting ``fetch`` to assert that:
//   * a repeated filter signature is served from cache (no second round-trip),
//   * a different signature misses the cache (a new round-trip),
//   * a mutating request through ``api`` invalidates the cache, a GET does not.
//
// Run via ``node tests/js/facet_cache_harness.mjs`` (or the pytest wrapper in
// ``tests/test_web_js.py``, which skips when Node isn't installed).
import { installDom, webModule } from "./fake_dom.mjs";

// A counting fetch. ``/api/facets`` returns an empty-but-valid facets payload.
let fetchCount = 0;
installDom({
  fetch: async (u) => {
    fetchCount++;
    const facets = { total: 0, with_faces: 0, persons: [], scenes: [] };
    const body = String(u).startsWith("/api/facets") ? facets : { total: 0, photos: [] };
    return { ok: true, json: async () => body };
  },
});
const { api, facetCache, fetchFacets } = await import(webModule("api.js"));

let pass = 0, fail = 0;
const check = (name, cond) => { cond ? pass++ : (fail++, console.error("FAIL:", name)); };

async function main() {
  // Start from a clean slate (nothing renders at import time any more).
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
