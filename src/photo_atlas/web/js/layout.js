// Pure layout engine for the photo grid: justified rows (every row fills the
// width, tiles keep their aspect ratio) with optional date/folder grouping.
// No DOM access, so it is unit-tested under Node (tests/js/layout_harness.mjs).

export const GROUP_MODES = ["none", "year", "month", "day", "folder"];
const MONTHS = ["January", "February", "March", "April", "May", "June", "July",
  "August", "September", "October", "November", "December"];
const AR_MIN = 0.4, AR_MAX = 3; // keeps panoramas / slivers from wrecking a row
const GROUP_GAP = 18; // extra air before a group header

const normPath = (p) => String(p || "").replace(/\\/g, "/");

export function groupKey(p, mode) {
  if (mode === "folder") {
    const s = normPath(p.path);
    const i = s.lastIndexOf("/");
    return i > 0 ? s.slice(0, i) : "/";
  }
  const t = p.taken_at || "";
  const n = { year: 4, month: 7, day: 10 }[mode];
  if (!n) return "";
  return t.length >= n ? t.slice(0, n) : "undated";
}

export function groupLabel(key, mode) {
  if (key === "undated") return "Undated";
  if (mode === "year") return key;
  if (mode === "month") {
    const [y, m] = key.split("-");
    return `${MONTHS[+m - 1] || m} ${y}`;
  }
  if (mode === "day") {
    const [y, m, d] = key.split("-");
    return `${+d} ${MONTHS[+m - 1] || m} ${y}`;
  }
  if (mode === "folder") {
    const parts = key.split("/").filter(Boolean);
    return parts[parts.length - 1] || "/";
  }
  return "";
}

/** "May 2024" for the floating scroll bubble; "" when undated. */
export function bubbleText(takenAt) {
  if (!takenAt || takenAt.length < 7) return "";
  return groupLabel(takenAt.slice(0, 7), "month");
}

const aspect = (p) => {
  const ar = p.width && p.height ? p.width / p.height : 1;
  return Math.min(AR_MAX, Math.max(AR_MIN, ar));
};

/**
 * Lay out `photos` (already in display order) into justified rows.
 * opts: { width, target (row height), gap, mode, headerH, complete }
 * -> { items:[{i,left,top,w,h,row}], rows:[{top,h,first,last}],
 *      headers:[{key,label,top,h,count}], totalH }
 * `complete:false` means more pages are still coming, so the trailing group's
 * count is unknown (null) rather than a misleading partial number.
 */
export function buildLayout(photos, opts) {
  const { width, target, gap, mode = "none", headerH = 44, complete = true } = opts;
  const items = [], rows = [], headers = [];
  let y = 0;

  const flushRow = (first, last, sumAr, final) => {
    const n = last - first + 1;
    const avail = width - gap * (n - 1);
    let h = avail / sumAr;
    if (final && h > target) h = target; // never stretch a short last row
    let x = 0;
    for (let k = first; k <= last; k++) {
      const w = aspect(photos[k]) * h;
      items.push({ i: k, left: x, top: y, w, h, row: rows.length });
      x += w + gap;
    }
    rows.push({ top: y, h, first, last });
    y += h + gap;
  };

  let i = 0;
  while (i < photos.length) {
    const key = groupKey(photos[i], mode);
    let end = i;
    while (end + 1 < photos.length && groupKey(photos[end + 1], mode) === key) end++;
    if (mode !== "none") {
      if (headers.length) y += GROUP_GAP;
      headers.push({
        key, label: groupLabel(key, mode), top: y, h: headerH,
        count: end === photos.length - 1 && !complete ? null : end - i + 1,
      });
      y += headerH;
    }
    let first = i, sum = 0;
    for (let k = i; k <= end; k++) {
      sum += aspect(photos[k]);
      const n = k - first + 1;
      if (sum * target + gap * (n - 1) >= width) {
        flushRow(first, k, sum, false);
        first = k + 1; sum = 0;
      }
    }
    if (first <= end) flushRow(first, end, sum, true);
    i = end + 1;
  }
  const totalH = rows.length ? y - gap : 0;
  return { items, rows, headers, totalH };
}

// Index of the first row whose bottom edge is at or below `y`.
function rowAt(layout, y) {
  let lo = 0, hi = layout.rows.length - 1, ans = layout.rows.length;
  while (lo <= hi) {
    const mid = (lo + hi) >> 1;
    const r = layout.rows[mid];
    if (r.top + r.h >= y) { ans = mid; hi = mid - 1; } else lo = mid + 1;
  }
  return ans;
}

/** Items + headers intersecting [scrollTop - buffer, scrollTop + viewportH + buffer]. */
export function visibleRange(layout, scrollTop, viewportH, buffer = 0) {
  const top = scrollTop - buffer, bottom = scrollTop + viewportH + buffer;
  const out = { items: [], headers: [] };
  for (let r = rowAt(layout, top); r < layout.rows.length; r++) {
    const row = layout.rows[r];
    if (row.top > bottom) break;
    for (let k = row.first; k <= row.last; k++) out.items.push(layout.items[k]);
  }
  out.headers = layout.headers.filter((h) => h.top + h.h >= top && h.top <= bottom);
  return out;
}

/** The item at (or nearest below) vertical offset y — drives the date bubble. */
export function itemAt(layout, y) {
  if (!layout.rows.length) return null;
  const r = Math.min(rowAt(layout, y), layout.rows.length - 1);
  return layout.items[layout.rows[r].first];
}
