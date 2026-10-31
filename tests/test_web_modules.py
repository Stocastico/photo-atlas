"""Static checks on the ES-module front-end (no browser, no Node needed)."""

from __future__ import annotations

import mimetypes
import re
from pathlib import Path

from fastapi.testclient import TestClient

from photo_atlas.api import create_app
from photo_atlas.config import AtlasConfig

WEB = Path(__file__).parent.parent / "src" / "photo_atlas" / "web"
JS = WEB / "js"
IMPORT = re.compile(r'^import\s*\{([^}]*)\}\s*from\s*"\./([\w-]+)\.js";', re.M)
EXPORT = re.compile(r"^export\s+(?:async\s+)?(?:function|const|let)\s+([\w$]+)", re.M)


def test_index_loads_the_module_entry_point():
    html = (WEB / "index.html").read_text(encoding="utf-8")
    assert '<script type="module" src="/static/js/main.js"></script>' in html
    assert "/static/app.js" not in html
    assert not (WEB / "app.js").exists()


def test_every_import_resolves_to_a_real_export():
    modules = {p.stem: p.read_text(encoding="utf-8") for p in JS.glob("*.js")}
    assert "main" in modules
    exports = {name: set(EXPORT.findall(text)) for name, text in modules.items()}
    for name, text in modules.items():
        for names, target in IMPORT.findall(text):
            assert target in modules, f"{name}.js imports missing module {target}.js"
            for imported in (n.strip() for n in names.split(",") if n.strip()):
                assert imported in exports[target], f"{name}.js: {target}.js lacks {imported}"


def test_static_js_is_served_as_javascript(tmp_path):
    client = TestClient(create_app(AtlasConfig(home=tmp_path)))
    res = client.get("/static/js/main.js")
    assert res.status_code == 200
    assert res.headers["content-type"].startswith("text/javascript")
    assert mimetypes.guess_type("x.js")[0] == "text/javascript"


def test_create_app_repairs_a_bad_js_mime_registration(tmp_path, monkeypatch):
    """Windows registries sometimes map .js to text/plain, which browsers refuse
    for <script type=module>; the app pins it."""

    monkeypatch.setattr(mimetypes, "types_map", dict(mimetypes.types_map))
    mimetypes.add_type("text/plain", ".js")
    assert mimetypes.guess_type("x.js")[0] == "text/plain"
    create_app(AtlasConfig(home=tmp_path))
    assert mimetypes.guess_type("x.js")[0] == "text/javascript"


def test_js_modules_are_packaged():
    """`web/*` doesn't match `web/js/*`; without this an installed wheel has no UI."""

    import tomllib

    pyproject = tomllib.loads(
        (WEB.parent.parent.parent / "pyproject.toml").read_text(encoding="utf-8")
    )
    patterns = pyproject["tool"]["setuptools"]["package-data"]["photo_atlas"]
    assert "web/js/*" in patterns
