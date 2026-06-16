"""End-to-end tests for the command-line entry point.

These drive :mod:`photo_atlas.cli` exactly as a user would (``main(argv)``)
against a throwaway ``--home`` library built from the synthetic demo backend,
so no network or real photos are needed. They also cover :mod:`photo_atlas.demo`
transitively.
"""

from __future__ import annotations

import pytest

from photo_atlas import cli, db
from photo_atlas.config import AtlasConfig


def test_build_parser_requires_subcommand():
    parser = cli.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([])  # subcommand is required


def test_demo_then_stats_and_cluster(tmp_path, capsys):
    home = tmp_path / "lib"

    assert cli.main(["--home", str(home), "demo", "--count", "8"]) == 0
    out = capsys.readouterr().out
    assert "Indexed" in out and "face groups" in out

    # The catalog now exists and has photos.
    conn = db.connect(AtlasConfig(home=home).db_path)
    try:
        n = conn.execute("SELECT COUNT(*) FROM photos").fetchone()[0]
    finally:
        conn.close()
    assert n == 8

    assert cli.main(["--home", str(home), "stats"]) == 0
    stats_out = capsys.readouterr().out
    assert "Photos: 8" in stats_out
    assert "Years:" in stats_out

    # Re-running cluster on the populated library is idempotent and prints a count.
    assert cli.main(["--home", str(home), "cluster"]) == 0
    assert "Clustered" in capsys.readouterr().out


def test_is_loopback_classification():
    assert cli._is_loopback("127.0.0.1")
    assert cli._is_loopback("localhost")
    assert cli._is_loopback("::1")
    assert not cli._is_loopback("0.0.0.0")
    assert not cli._is_loopback("192.168.1.5")
    assert not cli._is_loopback("")  # binds all interfaces


def test_serve_warns_when_bound_off_loopback(tmp_path, capsys, monkeypatch):
    """Binding off-loopback exposes the unauthenticated API; serve must warn.
    ``uvicorn.run`` is stubbed so the test never actually starts a server."""
    started = {}
    monkeypatch.setitem(
        __import__("sys").modules, "uvicorn",
        type("U", (), {"run": staticmethod(lambda app, host, port: started.update(host=host))}),
    )
    home = tmp_path / "lib"
    cli.main(["--home", str(home), "serve", "--host", "0.0.0.0", "--port", "9123"])
    err = capsys.readouterr().err
    assert "WARNING" in err and "0.0.0.0" in err
    assert started["host"] == "0.0.0.0"


def test_serve_no_warning_on_loopback(tmp_path, capsys, monkeypatch):
    monkeypatch.setitem(
        __import__("sys").modules, "uvicorn",
        type("U", (), {"run": staticmethod(lambda app, host, port: None)}),
    )
    home = tmp_path / "lib"
    cli.main(["--home", str(home), "serve"])  # default 127.0.0.1
    assert "WARNING" not in capsys.readouterr().err


def test_index_real_directory(tmp_path, capsys):
    """`index` on a folder of demo JPEGs, with faces disabled for speed."""
    from photo_atlas import demo

    photos = tmp_path / "pics"
    demo.generate(photos, count=5, seed=3)
    home = tmp_path / "lib"

    rc = cli.main(["--home", str(home), "index", str(photos), "--faces", "none"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "5 indexed" in out

    # Indexing again skips everything (incremental).
    cli.main(["--home", str(home), "index", str(photos), "--faces", "none"])
    assert "0 indexed, 5 skipped" in capsys.readouterr().out


def test_prune_command_removes_deleted_files(tmp_path, capsys):
    from pathlib import Path

    from photo_atlas import demo

    photos = tmp_path / "pics"
    paths = demo.generate(photos, count=3, seed=4)
    home = tmp_path / "lib"
    cli.main(["--home", str(home), "index", str(photos), "--faces", "none"])
    capsys.readouterr()

    Path(paths[0]).unlink()
    rc = cli.main(["--home", str(home), "prune"])
    assert rc == 0
    assert "Pruned 1" in capsys.readouterr().out


def test_export_labels_command_writes_sidecars(tmp_path, capsys):

    from photo_atlas import db
    from photo_atlas.config import AtlasConfig

    home = tmp_path / "lib"
    cfg = AtlasConfig(home=home).ensure_dirs()
    conn = db.connect(cfg.db_path)
    try:
        pid = db.upsert_photo(
            conn, {"path": str(tmp_path / "p.jpg"), "filename": "p.jpg"}
        )
        person = db.get_or_create_person(conn, "Grace")
        db.replace_faces(conn, pid, [{"person_id": person}])
        conn.commit()
    finally:
        conn.close()

    out = tmp_path / "xmp"
    rc = cli.main(["--home", str(home), "export-labels", "--dest", str(out)])
    assert rc == 0
    assert "1 XMP sidecar" in capsys.readouterr().out
    assert (out / "p.jpg.xmp").exists()


def test_dedup_command_backfills_hashes(tmp_path, capsys):
    from photo_atlas import demo

    photos = tmp_path / "pics"
    demo.generate(photos, count=4, seed=5)
    home = tmp_path / "lib"
    cli.main(["--home", str(home), "index", str(photos), "--faces", "none"])
    # Clear the hashes computed at index time to simulate a pre-phash catalog.
    conn = db.connect(AtlasConfig(home=home).db_path)
    conn.execute("UPDATE photos SET phash=NULL")
    conn.commit()
    conn.close()
    capsys.readouterr()

    rc = cli.main(["--home", str(home), "dedup"])
    assert rc == 0
    assert "4 photo(s) hashed" in capsys.readouterr().out


def test_index_missing_path_returns_error(tmp_path, capsys):
    rc = cli.main(["--home", str(tmp_path / "lib"), "index", str(tmp_path / "nope")])
    assert rc == 2
    assert "path not found" in capsys.readouterr().err


def test_index_reports_failed_files_on_stderr(tmp_path, capsys):
    pics = tmp_path / "pics"
    pics.mkdir()
    (pics / "broken.jpg").write_bytes(b"not a real jpeg")
    rc = cli.main(["--home", str(tmp_path / "lib"), "index", str(pics), "--faces", "none"])
    assert rc == 0
    err = capsys.readouterr().err
    assert "failed" in err and "broken.jpg" in err


def test_serve_invokes_uvicorn(tmp_path, monkeypatch, capsys):
    """`serve` builds the app and hands it to uvicorn (which we stub out)."""
    calls = {}

    def fake_run(app, host, port):
        calls["app"] = app
        calls["host"] = host
        calls["port"] = port

    import uvicorn

    monkeypatch.setattr(uvicorn, "run", fake_run)
    rc = cli.main(["--home", str(tmp_path / "lib"), "serve", "--port", "9123"])
    assert rc == 0
    assert calls["port"] == 9123
    assert calls["host"] == "127.0.0.1"
    assert calls["app"] is not None
    assert "Serving Photo Atlas" in capsys.readouterr().out
