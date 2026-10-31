// Headless harness for the front-end URL/history-state helpers in
// ``photo_atlas/web/js/*.js``. The repo has no browser test runner, so we load
// the script in a minimal fake DOM (just enough for it to evaluate) and then
// exercise the pure query (de)serialisation: buildQuery / applyQuery / syncURL.
//
// Run via ``node tests/js/url_state_harness.mjs`` (or the pytest wrapper in
// ``tests/test_web_js.py``, which skips when Node isn't installed).
import { installDom, webModule } from "./fake_dom.mjs";

// pushState mirrors the href's querystring into ``loc.search`` like a browser.
const { loc } = installDom({
  pushState(l, _s, _t, href) {
    const q = href.indexOf("?");
    l.search = q >= 0 ? href.slice(q) : "";
  },
});
const { state } = await import(webModule("state.js"));
const { buildQuery, applyQuery, syncURL } = await import(webModule("urlstate.js"));

let pass = 0, fail = 0;
const check = (name, cond) => { cond ? pass++ : (fail++, console.error("FAIL:", name)); };

// buildQuery serialises multi-value facets, scalars, the boolean toggles, view and sort.
state.filters = {
  country: ["Italy", "France"], scene: ["food"], q: "beach",
  has_faces: true, favorite: true,
};
state.view = "people"; state.sort = "filename";
const p = new URLSearchParams(buildQuery());
check("country repeats", p.getAll("country").join(",") === "Italy,France");
check("scene scalar", p.get("scene") === "food");
check("q scalar", p.get("q") === "beach");
check("has_faces flag", p.get("has_faces") != null);
check("favorite flag", p.get("favorite") != null);
check("view", p.get("view") === "people");
check("sort", p.get("sort") === "filename");

// Full round-trip: syncURL writes the querystring, applyQuery restores it.
loc.search = ""; syncURL();
check("syncURL writes search", loc.search.length > 1);
state.filters = {}; state.view = "photos"; state.sort = "newest";
applyQuery();
check("restore multi-value", JSON.stringify(state.filters.country) === JSON.stringify(["Italy", "France"]));
check("restore scalar", state.filters.q === "beach");
check("restore boolean", state.filters.has_faces === true);
check("restore favorite", state.filters.favorite === true);
check("restore view", state.view === "people");
check("restore sort", state.sort === "filename");

// An empty/default state produces an empty query (so the URL clears cleanly).
state.filters = {}; state.view = "photos"; state.sort = "newest";
check("empty query", buildQuery() === "");

console.log(`${pass} passed, ${fail} failed`);
process.exit(fail ? 1 : 0);
