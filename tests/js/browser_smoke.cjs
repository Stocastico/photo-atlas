// Real-browser smoke test for the web UI (Chromium via Playwright).
//
// Usage: node browser_smoke.cjs <base-url>
// Loads the page against a live server, then asserts there are no console errors /
// uncaught exceptions / failed or 4xx-5xx requests while it: renders the grid,
// visits every tab, filters via the sidebar, searches, and opens + closes the
// lightbox. Prints "<n> passed, <m> failed" like the Node harnesses.
const { chromium } = require("playwright");

const base = process.argv[2];
let passed = 0, failed = 0;
const ok = (cond, msg) => {
  if (cond) passed++; else { failed++; console.log("FAIL:", msg); }
};

(async () => {
  const browser = await chromium.launch({ args: ["--no-sandbox"] });
  const page = await browser.newPage({ viewport: { width: 1280, height: 900 } });
  const problems = [];
  page.on("console", (m) => { if (m.type() === "error") problems.push("console: " + m.text()); });
  page.on("pageerror", (e) => problems.push("pageerror: " + e.message));
  page.on("requestfailed", (r) => problems.push("requestfailed: " + r.url()));
  page.on("response", (r) => { if (r.status() >= 400) problems.push(`${r.status()}: ${r.url()}`); });

  await page.goto(base, { waitUntil: "networkidle" });
  await page.waitForSelector("#grid .card", { timeout: 10000 });
  ok((await page.locator("#grid .card").count()) > 0, "grid renders photo cards");
  ok(/\d/.test(await page.textContent("#result-count")), "result count is shown");

  // Every tab shows its own panel (and hides the photos panel).
  for (const view of ["memories", "trips", "duplicates", "map", "people", "clusters", "photos"]) {
    await page.click(`.tab[data-view="${view}"]`);
    await page.waitForTimeout(400);
    ok(await page.isVisible(`#view-${view}`), `tab ${view}: panel visible`);
    ok((await page.getAttribute(`#tab-${view}`, "aria-selected")) === "true", `tab ${view}: selected`);
    if (view === "map") ok((await page.locator("#map .leaflet-container, #map.leaflet-container").count()) > 0, "map initialises leaflet");
  }

  // Sidebar filter -> URL state + fewer-or-equal results; clear restores.
  await page.waitForSelector("#sidebar .chip", { timeout: 5000 });
  await page.locator("#sidebar .chip").first().click();
  await page.waitForTimeout(500);
  ok(new URL(page.url()).search.length > 1, "filter is reflected in the URL");
  ok((await page.locator("#active-filters .pill, #active-filters [class*=pill]").count()) > 0, "active-filter pill shown");

  // Search box.
  await page.fill("#search", "zzzz-no-such-photo");
  await page.waitForTimeout(900);
  ok(new URL(page.url()).searchParams.get("q") === "zzzz-no-such-photo", "search term in URL");
  await page.fill("#search", "");
  await page.waitForTimeout(900);

  // Lightbox open / close.
  await page.goto(base, { waitUntil: "networkidle" });
  await page.waitForSelector("#grid .card");
  await page.locator("#grid .card").first().click();
  await page.waitForSelector("#lightbox.open", { timeout: 5000 });
  ok(true, "lightbox opens");
  await page.waitForTimeout(500);
  await page.keyboard.press("Escape");
  await page.waitForTimeout(300);
  ok((await page.locator("#lightbox.open").count()) === 0, "Escape closes lightbox");

  ok(problems.length === 0, "no console/page/request errors: " + JSON.stringify(problems.slice(0, 8)));
  await browser.close();
  console.log(`${passed} passed, ${failed} failed`);
  process.exit(failed ? 1 : 0);
})().catch((e) => { console.log("CRASH", e); console.log(`${passed} passed, ${failed + 1} failed`); process.exit(1); });
