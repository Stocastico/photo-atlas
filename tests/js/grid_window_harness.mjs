// Headless harness for the virtualised-grid math in ``photo_atlas/web/js/*.js``
// (gridLayout / cardOffset / windowRange). These are pure functions, so we load
// the script in a tiny fake DOM and assert the layout/window arithmetic that
// keeps only a bounded window of cards in the DOM.
//
// Run via ``node tests/js/grid_window_harness.mjs`` or the pytest wrapper
// (``tests/test_web_js.py``), which skips when Node isn't installed.
import { installDom, webModule } from "./fake_dom.mjs";

installDom();
const { gridLayout, cardOffset, windowRange } = await import(webModule("grid.js"));

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
