"""Command line entry point for Photo Atlas.

Examples::

    photo-atlas demo                 # create a synthetic library to play with
    photo-atlas index ~/Pictures     # ingest real photos
    photo-atlas cluster              # group unnamed faces
    photo-atlas serve                # launch the web UI
    photo-atlas stats                # print a summary of the catalog
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import db
from .config import AtlasConfig
from .indexer import cluster_library, index_path, prune_library
from .search import facets


def _config(args) -> AtlasConfig:
    home = Path(args.home).expanduser() if getattr(args, "home", None) else None
    config = AtlasConfig(home=home) if home else AtlasConfig()
    return config.ensure_dirs()


def _cmd_index(args) -> int:
    config = _config(args)
    root = Path(args.path).expanduser()
    if not root.exists():
        print(f"error: path not found: {root}", file=sys.stderr)
        return 2

    def progress(path: Path, stats) -> None:
        sys.stdout.write(
            f"\r  scanned={stats.scanned} indexed={stats.indexed} "
            f"skipped={stats.skipped} faces={stats.faces} recognized={stats.recognized}"
        )
        sys.stdout.flush()

    print(f"Indexing {root} ...")
    stats = index_path(
        config, root,
        backend_name=args.faces,
        geocode=not args.no_geocode,
        recompute=args.recompute,
        progress=progress,
    )
    print()
    print(
        f"Done: {stats.indexed} indexed, {stats.skipped} skipped, "
        f"{stats.duplicates} duplicate(s), {stats.failed} failed, "
        f"{stats.faces} faces ({stats.recognized} auto-recognized)."
    )
    if stats.videos:
        print(
            f"Skipped {stats.videos} video file(s) — Photo Atlas indexes still "
            "images only (videos aren't catalogued)."
        )
    if stats.errors:
        print(f"{stats.failed} file(s) failed; first errors:", file=sys.stderr)
        for line in stats.errors[:10]:
            print(f"  - {line}", file=sys.stderr)
    if stats.faces:
        print("Tip: run `photo-atlas cluster` then name people in the web UI.")
    return 0


def _cmd_cluster(args) -> int:
    config = _config(args)
    result = cluster_library(config)
    print(f"Clustered {result['faces']} unnamed faces into {result['clusters']} groups.")
    print("Open the web UI (`photo-atlas serve`) to name them.")
    return 0


def _cmd_prune(args) -> int:
    config = _config(args)
    result = prune_library(config)
    print(
        f"Pruned {result['removed']} photo(s) whose files are gone; "
        f"{result['kept']} still present."
    )
    return 0


def _cmd_export_labels(args) -> int:
    from .export import write_xmp_sidecars

    config = _config(args)
    dest = Path(args.dest).expanduser() if args.dest else None
    result = write_xmp_sidecars(config, dest=dest)
    where = f" into {dest}" if dest else " next to each photo"
    print(
        f"Wrote {result['written']} XMP sidecar(s){where} "
        f"for {result['photos']} photo(s) with named people."
    )
    return 0


def _cmd_demo(args) -> int:
    from .demo import generate

    config = _config(args)
    dest = Path(args.dest).expanduser() if args.dest else config.home / "demo_photos"
    print(f"Generating {args.count} demo photos in {dest} ...")
    generate(dest, count=args.count)
    stats = index_path(config, dest, backend_name="synthetic", geocode=True)
    cluster = cluster_library(config)
    print(
        f"Indexed {stats.indexed} photos, {stats.faces} faces, "
        f"{cluster['clusters']} face groups to name."
    )
    print("Now run `photo-atlas serve` and open http://127.0.0.1:8000")
    return 0


def _cmd_stats(args) -> int:
    config = _config(args)
    conn = db.connect(config.db_path)
    try:
        data = facets(conn)
    finally:
        conn.close()

    print(f"Photos: {data['total']}")
    print(f"People named: {len(data['persons'])}")
    if data["years"]:
        years = [y["value"] for y in data["years"]]
        print(f"Years: {min(years)} – {max(years)}")
    for key in ("scenes", "countries"):
        top = ", ".join(f"{r['value']} ({r['count']})" for r in data[key][:6])
        print(f"{key.capitalize()}: {top}")
    if data["persons"]:
        people = ", ".join(f"{p['name']} ({p['count']})" for p in data["persons"][:10])
        print(f"Top people: {people}")
    return 0


def _cmd_serve(args) -> int:
    import uvicorn

    from .api import create_app

    config = _config(args)
    app = create_app(config)
    print(f"Serving Photo Atlas on http://{args.host}:{args.port}  (library: {config.home})")
    uvicorn.run(app, host=args.host, port=args.port)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="photo-atlas", description="Navigate years of photos.")
    parser.add_argument("--home", help="Library directory (default: ~/.photo_atlas)")
    sub = parser.add_subparsers(dest="command", required=True)

    p_index = sub.add_parser("index", help="Index a directory of photos")
    p_index.add_argument("path", help="Directory (or file) to index")
    p_index.add_argument("--faces", default="auto",
                         choices=["auto", "yunet", "dlib", "synthetic", "none"],
                         help="Face detection backend (yunet = deep YuNet+SFace)")
    p_index.add_argument("--no-geocode", action="store_true", help="Skip reverse geocoding")
    p_index.add_argument("--recompute", action="store_true", help="Re-index already known photos")
    p_index.set_defaults(func=_cmd_index)

    p_cluster = sub.add_parser("cluster", help="Cluster unnamed faces")
    p_cluster.set_defaults(func=_cmd_cluster)

    p_prune = sub.add_parser("prune", help="Remove catalog entries for deleted files")
    p_prune.set_defaults(func=_cmd_prune)

    p_export = sub.add_parser(
        "export-labels", help="Write person names to portable XMP sidecars"
    )
    p_export.add_argument(
        "--dest", help="Write sidecars into this directory instead of next to photos"
    )
    p_export.set_defaults(func=_cmd_export_labels)

    p_demo = sub.add_parser("demo", help="Generate and index a synthetic demo library")
    p_demo.add_argument("--count", type=int, default=24, help="Number of demo photos")
    p_demo.add_argument("--dest", help="Where to write demo photos")
    p_demo.set_defaults(func=_cmd_demo)

    p_stats = sub.add_parser("stats", help="Print a catalog summary")
    p_stats.set_defaults(func=_cmd_stats)

    p_serve = sub.add_parser("serve", help="Launch the web UI")
    p_serve.add_argument("--host", default="127.0.0.1")
    p_serve.add_argument("--port", type=int, default=8000)
    p_serve.set_defaults(func=_cmd_serve)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
