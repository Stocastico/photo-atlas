// Headless harness for the front-end URL/history-state helpers in
// ``photo_atlas/web/app.js``. The repo has no browser test runner, so we load
// the script in a minimal fake DOM (just enough for it to evaluate) and then
// exercise the pure query (de)serialisation: buildQuery / applyQuery / syncURL.
//
// Run via ``node tests/js/url_state_harness.mjs`` (or the pytest wrapper in
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
    onclick: null, onkeydown: null, onchange: null, oninput: null,
    value: "", textContent: "", innerHTML: "", disabled: false, offsetParent: {},
  };
}

const elements = {};
const loc = { pathname: "/", search: "" };
const hist = {
  pushState(_s, _t, href) {
    const q = href.indexOf("?");
    loc.search = q >= 0 ? href.slice(q) : "";
  },
};
const sandbox = {
  document: {
    querySelector(sel) { return elements[sel] || (elements[sel] = fakeEl()); },
    querySelectorAll() { return []; },
    addEventListener() {}, createElement() { return fakeEl(); },
    body: { offsetHeight: 1000 }, activeElement: { tagName: "BODY" },
  },
  window: { addEventListener() {}, innerHeight: 800, scrollY: 0, location: loc, history: hist },
  location: loc, history: hist, URLSearchParams, console,
  fetch: async () => ({ ok: true, json: async () => ({}) }),
  setTimeout: () => 0, clearTimeout: () => {},
};
vm.createContext(sandbox);

const src = fs.readFileSync(appjs, "utf8") +
  "\nthis.__state=state;this.__buildQuery=buildQuery;this.__applyQuery=applyQuery;this.__syncURL=syncURL;";
vm.runInContext(src, sandbox);

const { __state: state, __buildQuery: buildQuery, __applyQuery: applyQuery, __syncURL: syncURL } = sandbox;

let pass = 0, fail = 0;
const check = (name, cond) => { cond ? pass++ : (fail++, console.error("FAIL:", name)); };

// buildQuery serialises multi-value facets, scalars, the boolean toggle, view and sort.
state.filters = { country: ["Italy", "France"], scene: ["food"], q: "beach", has_faces: true };
state.view = "people"; state.sort = "filename";
const p = new URLSearchParams(buildQuery());
check("country repeats", p.getAll("country").join(",") === "Italy,France");
check("scene scalar", p.get("scene") === "food");
check("q scalar", p.get("q") === "beach");
check("has_faces flag", p.get("has_faces") != null);
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
check("restore view", state.view === "people");
check("restore sort", state.sort === "filename");

// An empty/default state produces an empty query (so the URL clears cleanly).
state.filters = {}; state.view = "photos"; state.sort = "newest";
check("empty query", buildQuery() === "");

console.log(`${pass} passed, ${fail} failed`);
process.exit(fail ? 1 : 0);
