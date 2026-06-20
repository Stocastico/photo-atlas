# Packaging Photo Atlas as an installable app

*Investigation — no implementation yet.* The question: can Photo Atlas be
distributed so a non-technical user installs and runs it **without a terminal,
Python, or `uv`**? This documents the realistic options, the constraints that
shape them, and a recommendation.

## What we're packaging

Photo Atlas is **already a local web app**: a FastAPI/uvicorn server that serves
a no-build-step vanilla-JS UI (`web/index.html` + `app.js` + `styles.css`,
Leaflet vendored locally). So "make it an app" means *wrap the server + show the
UI in a window* — no GUI rewrite is needed. That's the central lever every
option below pulls.

## Constraints that shape the choice

1. **Heavy native dependencies.** `onnxruntime`, `opencv-python-headless`,
   `numpy`, `scikit-learn`, `pillow` are large, platform-specific binary wheels.
   A frozen bundle is realistically **300–600 MB+** before any models, and a
   freezer (PyInstaller) needs explicit help to collect onnxruntime/opencv native
   libs or they go missing at runtime.
2. **Models download on first run.** `models._resolve` pulls SigLIP 2 (vision +
   text), ArcFace R100, and YuNet on demand to `~/.photo_atlas/models` (hundreds
   of MB). Either keep lazy download (needs network on first launch) or pre-bundle
   them (bigger installer, offline-ready). Either way a **first-run progress
   screen** matters for UX.
3. **Optional system `ffmpeg`.** Video indexing is gated on a system ffmpeg
   (`video.ffmpeg_available()`). Bundle a static binary or document it as
   optional.
4. **Code signing / notarization.** Unsigned `.app`/`.exe` trips Gatekeeper /
   SmartScreen — a hard wall for non-technical users. Signing needs an Apple
   Developer account (~$99/yr) and a Windows cert.
5. **Per-user data already lives outside the app** (`~/.photo_atlas`,
   `PHOTO_ATLAS_HOME`), so the bundle stays read-only and upgrades cleanly.

## Options (roughly best-fit first)

### 1. Native window wrapping the existing server — *recommended*
Add a thin launcher that boots uvicorn in-process on a random localhost port and
opens a desktop window pointed at it.

- **[pywebview](https://pywebview.flowrl.com/)** — tiny dep, uses the OS native
  webview (WebKit / Edge WebView2). The current `web/` assets load unchanged;
  the launcher is ~20 lines.
- Freeze with **[PyInstaller](https://pyinstaller.org/)** → `.app` / `.exe`, then
  wrap in a `.dmg` (macOS) and an Inno Setup / MSIX installer (Windows).
- UX: double-click an icon, a window opens. No terminal, no `uv`.
- Main work: PyInstaller hidden-imports / data-files tuning for onnxruntime +
  opencv, plus the `web/` and `data/` package-data.

Lowest-effort path to a real "installed app" feel that reuses 100% of the current
architecture.

### 2. BeeWare Briefcase
[Briefcase](https://briefcase.readthedocs.io/) builds signed native installers
(`.dmg`, `.msi`, Linux packages) from a Python project, with code-signing /
notarization baked into its workflow. Same pywebview-style launcher applies.
Best graduation path once signed, polished distributables matter.

### 3. Tauri / Electron with Python as a "sidecar"
Ship a real front-end shell (Tauri = Rust, tiny binaries + good auto-update;
Electron = bigger) that spawns the packaged Python server as a sidecar process.
Nicest result and best update story, but the most moving parts and a second
toolchain to maintain. Overkill for a personal tool.

### 4. Docker + a one-click launcher
`docker compose up` behind a desktop shortcut. Cleanly sidesteps the native-deps
mess (reproducible Linux image), but **requires Docker Desktop** — itself a heavy,
technical prerequisite. Fails the "nothing technical" bar.

### 5. `pipx` / `uv tool install`
`uv tool install photo-atlas` or `pipx install photo-atlas` gives a one-liner
global install. Still a terminal command, so it misses the "no terminal" goal —
but it's the cheapest thing to offer power users today.

## Recommendation

For a personal project you want others to try: **pywebview launcher + PyInstaller**,
distributed as a `.dmg` and an Inno Setup `.exe`, with **lazy model download + a
first-run progress screen**. It is the least new surface area, reuses the entire
web UI, and gets to "double-click to run." If it gains traction and you want
signed / auto-updating installers, graduate to **Briefcase** (easiest signing) or
a **Tauri sidecar** (nicest shell).

A sensible first slice (kept behind an optional `app`/`desktop` extra so it never
touches the core install): the pywebview launcher module + a PyInstaller spec, so
the end-to-end bundle build can be seen before committing to signing/installers.
