// fetch wrapper with error toasts, plus the client-side facet cache.
import { toast } from "./util.js";

// Facet payloads are filter-aware but otherwise stable, so cache them by the
// active-filter signature: revisiting a filter state (toggling a chip off and
// on, back/forward navigation, switching views) then skips the ~11-query
// /api/facets round-trip. Any mutating request (assign/rename/merge/…) can
// change the counts, so it clears the cache (see api()).
export const facetCache = new Map();
export const FACET_CACHE_CAP = 50;

// Thin fetch wrapper: surfaces network failures and non-2xx responses as a
// toast and throws, so a failed request never breaks the UI silently. Callers
// that await it are skipped (no re-render) when it throws.
export async function api(url, opts) {
  let res;
  try {
    res = await fetch(url, opts);
  } catch (e) {
    toast("Network error — is the server still running?");
    throw e;
  }
  if (!res.ok) {
    let detail = res.statusText;
    try { detail = (await res.json()).detail || detail; } catch (_) { /* non-JSON body */ }
    toast(`Error ${res.status}: ${detail}`);
    throw new Error(`${res.status} ${detail}`);
  }
  // A successful write may shift facet counts (e.g. naming a cluster), so drop
  // the cached facet payloads rather than show stale numbers.
  if (opts && opts.method && opts.method !== "GET") facetCache.clear();
  try {
    return await res.json();
  } catch (_) {
    return null; // tolerate empty bodies
  }
}

// Fetch the facet payload for a filter signature, served from the cache on a
// repeat signature. Kept a tiny pure-ish helper (no DOM) so it's unit-testable.
export async function fetchFacets(key) {
  let f = facetCache.get(key);
  if (!f) {
    f = await api("/api/facets?" + key);
    if (f) {
      if (facetCache.size >= FACET_CACHE_CAP) facetCache.clear();
      facetCache.set(key, f);
    }
  }
  return f;
}
