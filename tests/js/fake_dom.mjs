// Shared fake DOM for the front-end harnesses.
//
// The UI is a set of native ES modules (src/photo_atlas/web/js). The harnesses
// install minimal browser globals here, then ``await import(webModule("x.js"))``
// so they exercise the real modules (pure helpers, URL state, facet cache) under
// Node with no browser. Real-browser coverage lives in browser_smoke.cjs.
import path from "path";
import url from "url";

const here = path.dirname(url.fileURLToPath(import.meta.url));

/** file:// URL of a module under src/photo_atlas/web/js, for dynamic import(). */
export const webModule = (name) =>
  url.pathToFileURL(path.resolve(here, "../../src/photo_atlas/web/js", name)).href;

export function fakeEl() {
  return {
    style: {}, dataset: {}, children: [],
    classList: { add() {}, remove() {}, toggle() {}, contains() { return false; } },
    setAttribute() {}, getAttribute() { return null; }, removeAttribute() {},
    appendChild() {}, replaceWith() {}, after() {}, prepend() {},
    querySelector() { return fakeEl(); }, querySelectorAll() { return []; },
    addEventListener() {}, focus() {}, select() {}, setPointerCapture() {},
    getBoundingClientRect() { return { top: 0, left: 0, width: 0, height: 0 }; },
    onclick: null, onkeydown: null, onchange: null, oninput: null,
    value: "", textContent: "", innerHTML: "", disabled: false, hidden: false,
    offsetParent: {}, clientWidth: 1000,
  };
}

/**
 * Install browser-ish globals. Returns the shared ``loc`` / ``hist`` objects so a
 * harness can inspect URL writes. ``fetch`` defaults to an empty-JSON stub.
 */
export function installDom({ fetch, pushState } = {}) {
  const elements = {};
  const loc = { pathname: "/", search: "" };
  const hist = {
    pushState: pushState ? (...a) => pushState(loc, ...a) : () => {},
    replaceState() {},
  };
  const noop = () => {};
  globalThis.document = {
    querySelector(sel) { return elements[sel] || (elements[sel] = fakeEl()); },
    querySelectorAll() { return []; },
    addEventListener: noop, createElement: fakeEl,
    body: { offsetHeight: 1000 }, activeElement: { tagName: "BODY" },
  };
  globalThis.window = {
    addEventListener: noop, innerHeight: 800, scrollY: 0, location: loc, history: hist,
  };
  globalThis.location = loc;
  globalThis.history = hist;
  globalThis.requestAnimationFrame = () => 0;
  globalThis.fetch = fetch || (async () => ({ ok: true, json: async () => ({}) }));
  return { loc, hist, elements };
}
