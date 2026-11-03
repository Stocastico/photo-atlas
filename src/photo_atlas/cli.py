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
import os
import sys
from pathlib import Path

from . import db
from .config import AtlasConfig
from .indexer import (
    backfill_phashes,
    backfill_tints,
    cluster_library,
    embed_face_regions,
    embed_library,
    index_path,
    prune_library,
    retag_scenes,
)
from .search import facets, taken_source_counts


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

    workers = args.workers if args.workers is not None else _default_workers()
    print(f"Indexing {root} ... (workers={workers})")
    try:
        stats = index_path(
            config, root,
            backend_name=args.faces,
            geocode=not args.no_geocode,
            recompute=args.recompute,
            workers=workers,
            embed=args.embed,
            progress=progress,
            accept_model_change=args.accept_model_change,
        )
    except db.ModelMismatchError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print()
    print(
        f"Done: {stats.indexed} indexed, {stats.skipped} skipped, "
        f"{stats.duplicates} duplicate(s), {stats.failed} failed, "
        f"{stats.faces} faces ({stats.recognized} auto-recognized)."
    )
    if stats.videos:
        from . import video

        if video.ffmpeg_available():
            print(
                f"Found {stats.videos} video file(s); indexed a poster frame + "
                "capture date for each (install nothing — ffmpeg detected)."
            )
        else:
            print(
                f"Found {stats.videos} video file(s) but ffmpeg/ffprobe aren't on "
                "PATH, so they were counted but not indexed. Install ffmpeg for "
                "poster-frame thumbnails + capture dates."
            )
    if stats.errors:
        print(f"{stats.failed} file(s) failed; first errors:", file=sys.stderr)
        for line in stats.errors[:10]:
            print(f"  - {line}", file=sys.stderr)
        log = config.home / "index_errors.log"
        log.write_text("\n".join(stats.error_log) + "\n", encoding="utf-8")
        print(f"Full list of {len(stats.error_log)} error(s): {log}", file=sys.stderr)
    if stats.faces:
        print("Tip: run `photo-atlas cluster` then name people in the web UI.")
    if getattr(args, "prune", False):
        result = prune_library(config)
        print(
            f"Pruned {result['removed']} photo(s) whose files are gone "
            f"({result['kept']} still present); swept {result['orphans']} "
            "orphaned derivative file(s)."
        )
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
        f"Pruned {result['removed']} photo(s) whose files are gone "
        f"({result['kept']} still present); swept {result['orphans']} "
        "orphaned derivative file(s)."
    )
    return 0


def _cmd_empty_trash(args) -> int:
    from .indexer import empty_trash

    print(f"Permanently deleted {empty_trash(_config(args))} trashed file(s).")
    return 0


def _cmd_restore(args) -> int:
    from .indexer import restore_trash

    result = restore_trash(_config(args), args.ids or None)
    msg = f"Restored {result['restored']} file(s)"
    if result["conflicts"]:
        msg += (f"; skipped {result['conflicts']} because the original path is taken "
                "(left in the trash)")
    if result["missing"]:
        msg += f"; {result['missing']} trashed file(s) were already gone"
    print(msg + ".")
    if result["restored"]:
        print("Run `photo-atlas index <folder>` to add them back to the catalog.")
    return 1 if result["conflicts"] else 0


def _cmd_face_quality(args) -> int:
    from .housekeeping import backfill_face_quality

    config = _config(args)

    def progress(done: int, total: int) -> None:
        sys.stdout.write(f"\r  scored {done}/{total}")
        sys.stdout.flush()

    n = backfill_face_quality(config, progress=progress)
    print(f"\nScored {n} face(s). Best faces now lead cluster previews and person covers.")
    return 0


def _cmd_recognize(args) -> int:
    from .housekeeping import recognize_faces

    config = _config(args)
    r = recognize_faces(config, dry_run=args.dry_run)
    if r["enrolled"] == 0:
        print("No confirmed faces yet: name some people first (web UI → Name faces), "
              "then run this again.")
        return 1
    verb = "Would tag" if args.dry_run else "Tagged"
    print(f"{verb} {r['tagged']} previously unnamed face(s) across {r['people']} people "
          f"(learned from {r['enrolled']} confirmed faces); retagged {r['retagged']}, "
          f"cleared {r['cleared']} guess(es) that no longer match.")
    if r["need_review"]:
        print(f"{r['need_review']} guess(es) are below {r['review_confidence']:.0%} "
              "confidence: check them in the web UI (Name faces → Review guesses).")
    if r["skipped_dim"]:
        print(f"Skipped {r['skipped_dim']} face(s) from a different face model "
              "(re-index them with --recompute --accept-model-change).")
    return 0


def _cmd_calibrate_faces(args) -> int:
    from .housekeeping import calibrate_faces

    config = _config(args)
    r = calibrate_faces(config)
    if r is None:
        print(
            "Not enough data to calibrate: name at least two people in the web UI "
            "(each with 2+ faces you assigned or confirmed yourself), then run this again."
        )
        return 1
    print(f"Calibrated on {r['faces']} named faces of {r['people']} people "
          f"({r['genuine_pairs']} same-person and {r['impostor_pairs']} different-person pairs).")
    print(f"  same person      distance: median {r['genuine_p50']:.2f}, 90th pct "
          f"{r['genuine_p90']:.2f}")
    print(f"  different people distance: 1st pct {r['impostor_p1']:.2f}, median "
          f"{r['impostor_p50']:.2f}")
    print(f"  at {r['threshold']}: keeps {r['recall']:.1%} of same-person pairs, accepts "
          f"{r['false_accept']:.2%} of different-person pairs")
    print(f"  current settings: match {config.face_match_threshold}, cluster "
          f"{config.cluster_eps}, review below {config.review_confidence}")
    print("\nTo use the suggested values (then re-run `index --recompute` / `cluster`):")
    print(f"  export PHOTO_ATLAS_FACE_THRESHOLD={r['threshold']}")
    print(f"  export PHOTO_ATLAS_CLUSTER_EPS={r['threshold']}")
    print(f"  export PHOTO_ATLAS_REVIEW_CONFIDENCE={r['review_confidence']}")
    return 0


def _run_eval(args, build, run, report) -> int:
    """Shared driver for the eval-* commands: build candidates, run, print, optional JSON."""

    import json

    from . import evaluate

    config = _config(args)
    try:
        candidates = build(config)
        print(f"Evaluating {len(candidates)} candidate(s): "
              f"{', '.join(c.name for c in candidates)} ...")
        result = run(config, candidates)
    except evaluate.EvaluationError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except (ValueError, FileNotFoundError, OSError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(report(result))
    if args.json:
        Path(args.json).write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"\nFull results written to {args.json}")
    return 0


def _cmd_eval_faces(args) -> int:
    from . import evaluate

    return _run_eval(
        args,
        lambda config: evaluate.build_face_candidates(
            config, args.candidate, args.preset, include_default=not args.no_default
        ),
        lambda config, candidates: evaluate.evaluate_faces(
            config, candidates, detector=evaluate.default_face_detector(config),
            max_faces=args.max_faces, max_per_person=args.max_per_person,
        ),
        evaluate.format_faces_report,
    )


def _cmd_eval_search(args) -> int:
    from . import evaluate

    return _run_eval(
        args,
        lambda config: evaluate.build_search_candidates(
            config, args.candidate, args.preset, include_default=not args.no_default
        ),
        lambda config, candidates: evaluate.evaluate_search(
            config, candidates, queries_path=args.queries, bursts=not args.no_bursts,
            max_photos=args.max_photos,
        ),
        evaluate.format_search_report,
    )


def _cmd_eval_init_queries(args) -> int:
    from . import evaluate

    path = Path(args.file)
    if path.exists():
        print(f"error: {path} already exists; not overwriting it.", file=sys.stderr)
        return 1
    evaluate.write_query_template(path)
    print(f"Wrote a query template to {path}. Edit it, then run:\n"
          f"  photo-atlas eval-search --queries {path}")
    return 0


def _cmd_retag_scenes(args) -> int:
    config = _config(args)

    def progress(done: int, total: int) -> None:
        sys.stdout.write(f"\r  retagged {done}/{total}")
        sys.stdout.flush()

    print("Re-tagging scenes with the SigLIP zero-shot tagger ...")
    n = retag_scenes(config, progress=progress)
    print(f"\nDone: {n} photo(s) retagged.")
    return 0


def _cmd_embed(args) -> int:
    config = _config(args)

    def progress(done: int, total: int) -> None:
        sys.stdout.write(f"\r  embedded {done}/{total}")
        sys.stdout.flush()

    def region_progress(done: int, total: int) -> None:
        sys.stdout.write(f"\r  grounded {done}/{total}")
        sys.stdout.flush()

    print("Computing SigLIP image embeddings for semantic search ...")
    try:
        n = embed_library(
            config, recompute=args.recompute, progress=progress,
            accept_model_change=args.accept_model_change,
        )
        print(f"\nDone: {n} photo(s) embedded.")
        # Per-person grounding: also embed the region around each named/unnamed face
        # so a hybrid query ("Stefano eating food") can score the person's region.
        print("Computing per-face region embeddings for per-person grounding ...")
        m = embed_face_regions(
            config, recompute=args.recompute, progress=region_progress,
            accept_model_change=args.accept_model_change,
        )
    except db.ModelMismatchError as exc:
        print(f"\nerror: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:  # noqa: BLE001 - surface a clean message, not a traceback
        print(
            f"\nerror: could not build embeddings ({type(exc).__name__}: {exc}). "
            "Check that the SigLIP model can be downloaded (or set "
            "PHOTO_ATLAS_SCENE_MODEL / PHOTO_ATLAS_SCENE_TEXT_MODEL to local files).",
            file=sys.stderr,
        )
        return 1
    print(
        f"\nDone: {m} face region(s) embedded. Semantic search + grounding are now "
        "available in `serve`."
    )
    return 0


def _cmd_dedup(args) -> int:
    config = _config(args)

    def progress(done: int, total: int) -> None:
        sys.stdout.write(f"\r  hashed {done}/{total}")
        sys.stdout.flush()

    print("Computing perceptual hashes for near-duplicate detection ...")
    n = backfill_phashes(config, recompute=args.recompute, progress=progress)
    print(
        f"\nDone: {n} photo(s) hashed. Open the Duplicates tab in `photo-atlas serve` "
        "to review bursts and near-duplicates."
    )
    return 0


def _cmd_tints(args) -> int:
    config = _config(args)

    def progress(done: int, total: int) -> None:
        sys.stdout.write(f"\r  tinted {done}/{total}")
        sys.stdout.flush()

    print("Computing placeholder colours for the photo grid ...")
    n = backfill_tints(config, recompute=args.recompute, progress=progress)
    print(f"\nDone: {n} photo(s) tinted.")
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
        date_sources = taken_source_counts(conn)
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
    if date_sources:
        # Provenance of capture dates, best source first; 'mtime' is the weakest
        # (backup/copy time), so a large filename/folder share is the win here.
        order = ["exif", "filename", "folder", "video", "mtime"]
        ranked = sorted(date_sources.items(), key=lambda kv: (order.index(kv[0])
                        if kv[0] in order else len(order), -kv[1]))
        breakdown = ", ".join(f"{src} ({n})" for src, n in ranked)
        print(f"Date sources: {breakdown}")
    return 0


def _is_loopback(host: str) -> bool:
    """True when ``host`` keeps the server bound to the local machine only."""

    import ipaddress

    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _cmd_serve(args) -> int:
    import uvicorn

    from .api import create_app

    config = _config(args)
    app = create_app(config, allow_any_host=not _is_loopback(args.host))
    if not _is_loopback(args.host):
        # The API has no authentication; writes are same-origin-guarded but every
        # GET (photos, thumbnails, metadata) is open. Binding off-loopback exposes
        # the whole library to anything that can reach this host.
        print(
            f"WARNING: --host {args.host} exposes Photo Atlas on the network with no "
            "authentication — any device that can reach this host can browse your "
            "entire library. Use the default 127.0.0.1 unless you mean to share it.",
            file=sys.stderr,
        )
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
                         choices=["auto", "arcface", "yunet", "sface", "dlib", "synthetic", "none"],
                         help="Face backend (auto/arcface = YuNet+ArcFace R100; sface = legacy)")
    p_index.add_argument("--no-geocode", action="store_true", help="Skip reverse geocoding")
    p_index.add_argument("--recompute", action="store_true", help="Re-index already known photos")
    p_index.add_argument(
        "--prune", action="store_true",
        help="After indexing, reconcile the catalog: drop rows for deleted files and "
             "sweep orphaned thumbnail/preview/crop files (no separate prune step)",
    )
    p_index.add_argument(
        "--embed", action="store_true",
        help="Also store a SigLIP image embedding per photo for natural-language "
             "semantic search",
    )
    p_index.add_argument(
        "--accept-model-change", action="store_true",
        help="Allow switching the catalog's recorded face/embedding model "
             "(use when deliberately rebuilding everything with the new model)",
    )
    p_index.add_argument(
        "--workers", type=int, default=None,
        help="Worker processes for decode/detect (default: CPU count; 1 = serial)",
    )
    p_index.set_defaults(func=_cmd_index)

    p_cluster = sub.add_parser("cluster", help="Cluster unnamed faces")
    p_cluster.set_defaults(func=_cmd_cluster)

    p_prune = sub.add_parser("prune", help="Remove catalog entries for deleted files")
    p_prune.set_defaults(func=_cmd_prune)

    p_trash = sub.add_parser(
        "empty-trash", help="Permanently delete files removed via the web UI (kept in trash)"
    )
    p_trash.set_defaults(func=_cmd_empty_trash)

    p_restore = sub.add_parser(
        "restore", help="Move trashed originals back to their original location"
    )
    p_restore.add_argument(
        "ids", nargs="*", type=int, help="Photo ids to restore (default: everything in the trash)"
    )
    p_restore.set_defaults(func=_cmd_restore)

    # -- model evaluation (see MODELS.md) --
    def _add_eval_args(p, kind: str) -> None:
        from . import models

        presets = models.FACE_PRESETS if kind == "faces" else models.SEARCH_PRESETS
        spec = "NAME=RECOGNISER.onnx" if kind == "faces" else "NAME=VISION,TEXT[,TOKENIZER][,SIZE]"
        p.add_argument("--candidate", action="append", default=[], metavar=spec,
                       help="a model file(s) to evaluate (repeatable)")
        p.add_argument("--preset", action="append", default=[], choices=sorted(presets),
                       help="a downloadable, checksum-pinned candidate (repeatable)")
        p.add_argument("--no-default", action="store_true",
                       help="don't include the app's current default model as the baseline")
        p.add_argument("--json", metavar="FILE", help="also write the full results as JSON")

    p_ef = sub.add_parser(
        "eval-faces",
        help="Compare face recognisers on the people you have named (see MODELS.md)",
    )
    _add_eval_args(p_ef, "faces")
    p_ef.add_argument("--max-faces", type=int, default=2000, help="cap on faces used")
    p_ef.add_argument("--max-per-person", type=int, default=40, help="cap per person")
    p_ef.set_defaults(func=_cmd_eval_faces)

    p_es = sub.add_parser(
        "eval-search",
        help="Compare image-text encoders on your queries and on photo bursts (see MODELS.md)",
    )
    _add_eval_args(p_es, "search")
    p_es.add_argument("--queries", metavar="FILE",
                      help="JSONL of {query, relevant[]} (create one with eval-init-queries)")
    p_es.add_argument("--no-bursts", action="store_true",
                      help="skip the label-free near-duplicate image->image check")
    p_es.add_argument("--max-photos", type=int, default=2000, help="photo pool size")
    p_es.set_defaults(func=_cmd_eval_search)

    p_eq = sub.add_parser("eval-init-queries", help="Write a template queries file for eval-search")
    p_eq.add_argument("file", help="where to write the template")
    p_eq.set_defaults(func=_cmd_eval_init_queries)

    p_fq = sub.add_parser(
        "face-quality", help="Score faces indexed before quality existed (from their saved crops)"
    )
    p_fq.set_defaults(func=_cmd_face_quality)

    p_rec = sub.add_parser(
        "recognize",
        help="Name the faces already indexed using the people you've confirmed (no re-index)",
    )
    p_rec.add_argument("--dry-run", action="store_true", help="show what would change")
    p_rec.set_defaults(func=_cmd_recognize)

    p_cal = sub.add_parser(
        "calibrate-faces",
        help="Suggest face-match thresholds from the people you have already named",
    )
    p_cal.set_defaults(func=_cmd_calibrate_faces)

    p_retag = sub.add_parser("retag-scenes", help="Recompute scene tags in place (no re-index)")
    p_retag.set_defaults(func=_cmd_retag_scenes)

    p_embed = sub.add_parser(
        "embed", help="Backfill SigLIP image embeddings for semantic search (no re-index)"
    )
    p_embed.add_argument(
        "--recompute", action="store_true", help="Re-embed photos that already have an embedding"
    )
    p_embed.add_argument(
        "--accept-model-change", action="store_true",
        help="Allow switching the catalog's recorded face/embedding model "
             "(use when deliberately rebuilding everything with the new model)",
    )
    p_embed.set_defaults(func=_cmd_embed)

    p_dedup = sub.add_parser(
        "dedup",
        help="Backfill perceptual hashes for near-duplicate / burst detection (no re-index)",
    )
    p_dedup.add_argument(
        "--recompute", action="store_true", help="Re-hash photos that already have a hash"
    )
    p_dedup.set_defaults(func=_cmd_dedup)

    p_tints = sub.add_parser(
        "tints",
        help="Backfill the average colour used as the grid's blur-up placeholder (from thumbnails)",
    )
    p_tints.add_argument(
        "--recompute", action="store_true", help="Recompute photos that already have a tint"
    )
    p_tints.set_defaults(func=_cmd_tints)

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


def _configure_streams() -> None:
    """Never crash on a character the console can't encode.

    A Windows console that is redirected (or an old code page) can't encode e.g. "→" or
    "✓"; with the default ``strict`` handler a status line would raise
    ``UnicodeEncodeError`` mid-run. Substitute instead.
    """

    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(errors="replace")
            except (ValueError, OSError):
                pass


def _default_workers() -> int:
    """CPU count, capped at 61 on Windows (the most processes ``ProcessPoolExecutor`` allows)."""

    count = os.cpu_count() or 1
    return min(count, 61) if sys.platform == "win32" else count


def main(argv: list[str] | None = None) -> int:
    _configure_streams()
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
