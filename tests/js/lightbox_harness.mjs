// Headless harness for the lightbox zoom math in ``photo_atlas/web/app.js``.
// The repo has no browser test runner, so we load the script in a minimal fake
// DOM and exercise the pure ``nextZoom`` transform helper (centre-anchored zoom
// with clamping). Run via ``node tests/js/lightbox_harness.mjs`` (or the pytest
// wrapper in ``tests/test_web_js.py``, which skips when Node isn't installed).
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
    appendChild() {}, replaceWith() {}, after() {}, prepend() {},
    querySelector() { return fakeEl(); }, querySelectorAll() { return []; },
    addEventListener() {}, focus() {}, select() {}, setPointerCapture() {},
    onclick: null, onkeydown: null, onchange: null, oninput: null,
    value: "", textContent: "", innerHTML: "", disabled: false, hidden: false, offsetParent: {},
  };
}

const elements = {};
const loc = { pathname: "/", search: "" };
const hist = { pushState() {}, replaceState() {} };
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
  setTimeout: () => 0, clearTimeout: () => {}, setInterval: () => 0, clearInterval: () => {},
};
vm.createContext(sandbox);

const src = fs.readFileSync(appjs, "utf8") + "\nthis.__nextZoom=nextZoom;";
vm.runInContext(src, sandbox);
const nextZoom = sandbox.__nextZoom;

let pass = 0, fail = 0;
const check = (name, cond) => { cond ? pass++ : (fail++, console.error("FAIL:", name)); };
const near = (a, b) => Math.abs(a - b) < 1e-9;

// Zooming in past 1× raises the scale and rescales the pan by the same ratio.
let v = nextZoom({ scale: 1, tx: 0, ty: 0 }, 2);
check("zoom in scales", near(v.scale, 2));
check("zoom in keeps centred pan", near(v.tx, 0) && near(v.ty, 0));

v = nextZoom({ scale: 2, tx: 10, ty: -4 }, 2);
check("pan rescales with zoom", near(v.scale, 4) && near(v.tx, 20) && near(v.ty, -8));

// Clamps: never below 1× (and 1× snaps the pan back to centred)…
v = nextZoom({ scale: 2, tx: 30, ty: 30 }, 0.1);
check("clamp to min 1", near(v.scale, 1));
check("min resets pan", near(v.tx, 0) && near(v.ty, 0));

// …and never above the max.
v = nextZoom({ scale: 5, tx: 0, ty: 0 }, 10, 6);
check("clamp to max", near(v.scale, 6));

// A no-op factor leaves the transform unchanged.
v = nextZoom({ scale: 3, tx: 7, ty: 9 }, 1);
check("factor 1 is a no-op", near(v.scale, 3) && near(v.tx, 7) && near(v.ty, 9));

console.log(`${pass} passed, ${fail} failed`);
// Exit now: loading the full app.js kicks off a fire-and-forget renderPhotos()
// whose fetch stub returns no photos, so its deferred rejection would otherwise
// crash Node *after* this verdict. The other harnesses exit the same way.
process.exit(fail ? 1 : 0);
