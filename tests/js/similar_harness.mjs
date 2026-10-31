// Headless harness for the "more like this" request-URL helper in
// ``photo_atlas/web/js/*.js``. The grid is normally driven by the filter set, but
// when ``state.similarTo`` is set it pages the ``/api/photos/{id}/similar``
// endpoint instead. ``photosRequestURL`` is the pure switch between the two, so
// it's exercised here in a minimal fake DOM (like the URL-state harness).
//
// Run via ``node tests/js/similar_harness.mjs`` (or the pytest wrapper in
// ``tests/test_web_js.py``, which skips when Node isn't installed).
import { installDom, webModule } from "./fake_dom.mjs";

installDom();
const { state } = await import(webModule("state.js"));
const { photosRequestURL, exitSimilar } = await import(webModule("filters.js"));

let pass = 0, fail = 0;
const check = (name, cond) => { cond ? pass++ : (fail++, console.error("FAIL:", name)); };

// Filter mode: builds the normal /api/photos query (facets + sort + paging).
state.similarTo = null;
state.filters = { scene: ["food"], country: ["Italy"] };
state.sort = "oldest";
let u = new URL("http://x" + photosRequestURL(0));
check("filter path", u.pathname === "/api/photos");
check("filter scene", u.searchParams.get("scene") === "food");
check("filter sort", u.searchParams.get("sort") === "oldest");
check("filter limit", u.searchParams.get("limit") === "120");
check("filter offset", u.searchParams.get("offset") === "0");

// Default sort is omitted from the query.
state.sort = "newest";
u = new URL("http://x" + photosRequestURL(0));
check("default sort omitted", u.searchParams.get("sort") === null);

// Similar mode: pages the /api/photos/{id}/similar endpoint, ignoring filters/sort.
state.similarTo = 42;
state.sort = "oldest";
u = new URL("http://x" + photosRequestURL(120));
check("similar path", u.pathname === "/api/photos/42/similar");
check("similar ignores filters", u.searchParams.get("scene") === null);
check("similar ignores sort", u.searchParams.get("sort") === null);
check("similar offset", u.searchParams.get("offset") === "120");
check("similar limit", u.searchParams.get("limit") === "120");

// Face-similar mode ("more like this person"): pages /api/faces/{id}/similar.
state.similarTo = null;
state.similarFace = 7;
u = new URL("http://x" + photosRequestURL(60));
check("face-similar path", u.pathname === "/api/faces/7/similar");
check("face-similar ignores filters", u.searchParams.get("scene") === null);
check("face-similar offset", u.searchParams.get("offset") === "60");

// exitSimilar clears both similar targets (refresh is a no-op in this fake DOM).
exitSimilar();
check("exitSimilar clears photo target", state.similarTo === null);
check("exitSimilar clears face target", state.similarFace === null);
u = new URL("http://x" + photosRequestURL(0));
check("back to filter path", u.pathname === "/api/photos");

console.log(`${pass} passed, ${fail} failed`);
process.exit(fail ? 1 : 0);
