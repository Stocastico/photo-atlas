// Pure tests for the justified-row + grouped layout engine (web/js/layout.js).
import { installDom, webModule } from "./fake_dom.mjs";

installDom();
const L = await import(webModule("layout.js"));

let pass = 0, fail = 0;
const check = (name, cond) => { cond ? pass++ : (fail++, console.error("FAIL:", name)); };

const mk = (n, w = 300, h = 200, taken = "2024-05-10T10:00:00", path = "/a/b/x.jpg") =>
  Array.from({ length: n }, (_, i) => ({ id: i + 1, width: w, height: h, taken_at: taken, path }));

// ---- grouping keys / labels
const p = { taken_at: "2024-05-10T10:00:00", path: "/pics/Roma 2024/x.jpg" };
check("key none", L.groupKey(p, "none") === "");
check("key year", L.groupKey(p, "year") === "2024");
check("key month", L.groupKey(p, "month") === "2024-05");
check("key day", L.groupKey(p, "day") === "2024-05-10");
check("key folder", L.groupKey(p, "folder") === "/pics/Roma 2024");
check("key folder windows", L.groupKey({ path: "C:\\pics\\Roma\\x.jpg" }, "folder") === "C:/pics/Roma");
check("key undated", L.groupKey({ path: "/x.jpg" }, "month") === "undated");
check("label month", /May/.test(L.groupLabel("2024-05", "month")) && /2024/.test(L.groupLabel("2024-05", "month")));
check("label year", L.groupLabel("2024", "year") === "2024");
check("label folder", L.groupLabel("/pics/Roma 2024", "folder") === "Roma 2024");
check("label undated", L.groupLabel("undated", "month") === "Undated");

// ---- justified rows, no grouping
const W = 1000, GAP = 4, T = 200;
let lay = L.buildLayout(mk(30), { width: W, target: T, gap: GAP, mode: "none" });
check("no headers when ungrouped", lay.headers.length === 0);
check("every photo placed", lay.items.length === 30);
const byRow = new Map();
for (const it of lay.items) (byRow.get(it.row) ?? byRow.set(it.row, []).get(it.row)).push(it);
const rows = [...byRow.values()];
// all rows but the last fill the width exactly (justified)
const full = rows.slice(0, -1).every((r) => {
  const right = Math.max(...r.map((i) => i.left + i.w));
  return Math.abs(right - W) < 0.5;
});
check("non-last rows justified to width", full);
// heights near target
check("row heights near target", rows.slice(0, -1).every((r) => r[0].h > T * 0.6 && r[0].h < T * 1.4));
// last row is not stretched
const last = rows[rows.length - 1];
check("last row not stretched", last[0].h <= T * 1.001);
check("totalH covers last row", lay.totalH >= last[0].top + last[0].h - 0.01);
// preserve aspect ratio
check("aspect preserved", lay.items.every((i) => Math.abs(i.w / i.h - 1.5) < 0.02));
// order preserved
check("index order", lay.items.every((it, k) => it.i === k));

// ---- grouping inserts headers and breaks rows at group boundaries
const photos = [...mk(5, 300, 200, "2024-05-10T10:00:00"), ...mk(5, 300, 200, "2024-04-02T10:00:00")]
  .map((p, i) => ({ ...p, id: i + 1 }));
lay = L.buildLayout(photos, { width: W, target: T, gap: GAP, mode: "month", headerH: 40 });
check("two headers", lay.headers.length === 2);
check("header counts", lay.headers[0].count === 5 && lay.headers[1].count === 5);
const firstOfSecond = lay.items[5], lastOfFirst = lay.items[4];
check("new group starts a new row", firstOfSecond.row !== lastOfFirst.row);
check("second header below first group", lay.headers[1].top >= lastOfFirst.top + lastOfFirst.h);
check("items of group 2 below its header", firstOfSecond.top >= lay.headers[1].top + 40 - 0.01);

// ---- incomplete trailing group does not claim a final count
lay = L.buildLayout(photos, { width: W, target: T, gap: GAP, mode: "month", headerH: 40, complete: false });
check("trailing group count unknown while loading", lay.headers[1].count === null);
check("earlier group count known", lay.headers[0].count === 5);

// ---- panoramas / extreme aspect ratios stay bounded
const odd = [{ id: 1, width: 10000, height: 100, path: "/x" }, { id: 2, width: 100, height: 10000, path: "/y" }];
lay = L.buildLayout(odd, { width: W, target: T, gap: GAP, mode: "none" });
check("extreme aspect clamped", lay.items.every((i) => i.w <= W + 0.5 && i.h <= T * 3));
// missing dimensions fall back to square
lay = L.buildLayout([{ id: 1, path: "/x" }], { width: W, target: T, gap: GAP, mode: "none" });
check("missing dims -> square", Math.abs(lay.items[0].w - lay.items[0].h) < 0.01);

// ---- empty
lay = L.buildLayout([], { width: W, target: T, gap: GAP, mode: "month" });
check("empty", lay.totalH === 0 && lay.items.length === 0 && lay.headers.length === 0);

// ---- windowing: binary search returns only visible rows + buffer
const big = L.buildLayout(mk(5000), { width: W, target: T, gap: GAP, mode: "none" });
const win = L.visibleRange(big, 100000, 800, 400);
check("window bounded", win.items.length > 0 && win.items.length < 200);
check("window covers viewport", win.items.some((i) => i.top <= 100400 && i.top + i.h >= 100000));
check("window near scroll", win.items.every((i) => i.top + i.h >= 100000 - 400 && i.top <= 100000 + 800 + 400));
const topWin = L.visibleRange(big, 0, 800, 400);
check("top window starts at 0", topWin.items[0].i === 0);
check("visible headers returned", L.visibleRange(
  L.buildLayout(photos, { width: W, target: T, gap: GAP, mode: "month", headerH: 40 }), 0, 800, 0
).headers.length >= 1);

// ---- date at a scroll offset (for the floating bubble)
const dl = L.buildLayout(photos, { width: W, target: T, gap: GAP, mode: "none" });
check("dateAt first", L.itemAt(dl, 0).i === 0);
check("dateAt beyond end -> last row", L.itemAt(dl, 1e9).row === dl.rows.length - 1);
check("dateAt empty", L.itemAt(L.buildLayout([], { width: W, target: T, gap: GAP, mode: "none" }), 5) === null);

// ---- month/year bubble text
check("bubble text", /2024/.test(L.bubbleText("2024-05-10T10:00:00")) && L.bubbleText(null) === "");

console.log(`${pass} passed, ${fail} failed`);
process.exit(fail ? 1 : 0);
