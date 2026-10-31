// Headless harness for the lightbox zoom math in ``photo_atlas/web/js/*.js``.
// The repo has no browser test runner, so we load the script in a minimal fake
// DOM and exercise the pure ``nextZoom`` transform helper (centre-anchored zoom
// with clamping). Run via ``node tests/js/lightbox_harness.mjs`` (or the pytest
// wrapper in ``tests/test_web_js.py``, which skips when Node isn't installed).
import { installDom, webModule } from "./fake_dom.mjs";

installDom();
const { nextZoom } = await import(webModule("lightbox.js"));

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
process.exit(fail ? 1 : 0);
