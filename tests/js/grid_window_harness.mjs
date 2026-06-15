// Headless harness for the virtualised-grid math in ``photo_atlas/web/app.js``
// (gridLayout / cardOffset / windowRange). These are pure functions, so we load
// the script in a tiny fake DOM and assert the layout/window arithmetic that
// keeps only a bounded window of cards in the DOM.
//
// Run via ``node tests/js/grid_window_harness.mjs`` or the pytest wrapper
// (``tests/test_web_grid_window.py``), which skips when Node isn't installed.
import fs from "fs";
import path from "path";
import url from "url";
import vm from "vm";

const here = path.dirname(url.fileURLToPath(import.meta.url));
const appjs = path.resolve(here, "../../src/photo_atlas/web/app.js");

const noop = () => {};
function fakeEl() {
  return {
    style: {}, dataset: {}, classList: { add: noop, remove: noop, toggle: noop, contains: () => false },
    setAttribute: noop, getAttribute: () => null, appendChild: noop, querySelector: () => fakeEl(),
    querySelectorAll: () => [], addEventListener: noop, getBoundingClientRect: () => ({ top: 0 }),
    clientWidth: 1000, value: "", textContent: "", innerHTML: "", onclick: null, onkeydown: null,
  };
}
const sandbox = {
  document: {
    querySelector: () => fakeEl(), querySelectorAll: () => [], addEventListener: noop,
    createElement: () => fakeEl(), body: { offsetHeight: 1000 }, activeElement: { tagName: "BODY" },
  },
  window: { addEventListener: noop, innerHeight: 800, scrollY: 0 },
  location: { pathname: "/", search: "" }, history: { pushState: noop },
  URLSearchParams, console, fetch: async () => ({ ok: true, json: async () => ({}) }),
  setTimeout: () => 0, clearTimeout: noop, requestAnimationFrame: noop,
};
vm.createContext(sandbox);
const src = fs.readFileSync(appjs, "utf8") +
  "\nthis.__gridLayout=gridLayout;this.__cardOffset=cardOffset;this.__windowRange=windowRange;";
vm.runInContext(src, sandbox);
const { __gridLayout: gridLayout, __cardOffset: cardOffset, __windowRange: windowRange } = sandbox;

let pass = 0, fail = 0;
const check = (name, cond) => { cond ? pass++ : (fail++, console.error("FAIL:", name)); };

// 1000px wide, 50 items -> 5 columns of 192px squares (gap 10).
const layout = gridLayout(1000, 50);
check("cols", layout.cols === 5);
check("cardW", Math.abs(layout.cardW - 192) < 0.01);
check("square cards", layout.cardW === layout.cardH);
check("rows", layout.rows === 10);
check("totalH", Math.abs(layout.totalH - (10 * 192 + 9 * 10)) < 0.01);

// item 7 sits at row 1, col 2.
const off = cardOffset(7, layout);
check("offset left", Math.abs(off.left - 2 * (192 + 10)) < 0.01);
check("offset top", Math.abs(off.top - 1 * (192 + 10)) < 0.01);

// A huge library keeps the rendered window bounded near the scroll position.
const big = gridLayout(1000, 10000);
const w = windowRange(100000, 800, big, 10000);
check("window bounded", w.end - w.start <= (Math.ceil(800 / (big.cardH + 10)) + 8) * big.cols);
check("window non-trivial offset", w.start > 2000 && w.end < 3000);
check("window within bounds", w.start >= 0 && w.end <= 10000);

// Top of the list starts at index 0.
const top = windowRange(0, 800, big, 10000);
check("top starts at 0", top.start === 0);

// Narrow containers never drop below a single column.
check("min one column", gridLayout(50, 10).cols >= 1);

// Empty result set is inert.
const empty = gridLayout(1000, 0);
check("empty totalH", empty.totalH === 0);
const ew = windowRange(0, 800, empty, 0);
check("empty window", ew.start === 0 && ew.end === 0);

console.log(`${pass} passed, ${fail} failed`);
process.exit(fail ? 1 : 0);
