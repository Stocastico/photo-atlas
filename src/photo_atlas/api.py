"""FastAPI application for Photo Atlas.

Exposes the catalog as a small JSON API plus media endpoints (full image,
thumbnail, face crop) and serves the single-page web UI from
:mod:`photo_atlas.web`.

Create the app with :func:`create_app` (used by the CLI ``serve`` command) so a
custom :class:`~photo_atlas.config.AtlasConfig` can be injected; importing
``app`` uses the default library location.
"""

from __future__ import annotations

import datetime
import sqlite3
from collections.abc import Iterator
from pathlib import Path
from urllib.parse import urlparse

from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import db, library, metadata, planner, search
from .config import AtlasConfig

WEB_DIR = Path(__file__).parent / "web"


def _union(existing, derived: list) -> list:
    """Merge ``derived`` values into ``existing`` (a scalar/list/None), de-duped."""

    out: list = list(existing) if isinstance(existing, (list, tuple)) else (
        [existing] if existing not in (None, "") else []
    )
    for value in derived:
        if value not in out:
            out.append(value)
    return out

# Inclusive ``date_taken`` range bounds are compared on the YYYY-MM-DD prefix, so
# reject anything that isn't that shape (a malformed value would otherwise compare
# lexically and silently mis-filter). Applied to date_from/date_to everywhere.
_DATE_PATTERN = r"^\d{4}-\d{2}-\d{2}$"


class AssignRequest(BaseModel):
    name: str | None = None
    person_id: int | None = None


class RenameRequest(BaseModel):
    name: str


class MergeRequest(BaseModel):
    source_id: int


class CoverRequest(BaseModel):
    face_id: int


class FavoriteRequest(BaseModel):
    favorite: bool


class AlbumRequest(BaseModel):
    name: str
    query: str = ""


def create_app(config: AtlasConfig | None = None) -> FastAPI:
    config = (config or AtlasConfig()).ensure_dirs()
    app = FastAPI(title="Photo Atlas", version="0.1.0")

    # Create / migrate the schema once, here, so the per-request connections below
    # can skip the (idempotent but not free) DDL script on every single call.
    db.connect(config.db_path).close()

    @app.middleware("http")
    async def _same_origin_writes(request, call_next):
        # The catalog has no auth (it's a personal, loopback-bound tool), so guard
        # the state-changing endpoints against cross-site requests: a browser
        # attaches an ``Origin`` to such calls, and a malicious page's Origin won't
        # match our Host. Same-origin (our own UI) and non-browser clients (no
        # Origin, e.g. curl/tests) are allowed; GETs are never blocked.
        if request.method in ("POST", "PUT", "PATCH", "DELETE"):
            origin = request.headers.get("origin")
            if origin and urlparse(origin).netloc != request.headers.get("host"):
                return JSONResponse({"detail": "cross-origin request forbidden"}, 403)
        return await call_next(request)

    def get_conn() -> Iterator[sqlite3.Connection]:
        conn = db.connect(config.db_path, ensure_schema=False)
        try:
            yield conn
        finally:
            conn.close()

    # -- semantic search (lazy, cached) -----------------------------------
    # The embedding matrix and the text encoder are both expensive to build, so
    # cache them on the app. The matrix is rebuilt only when the set of embedded
    # photos changes (a concurrent `index`/`embed` run); the text encoder is built
    # once on first use (it downloads the SigLIP text model + tokenizer).
    _semantic: dict = {"index": None, "sig": None, "encoder": None, "encoder_tried": False}

    def _embed_signature(conn: sqlite3.Connection) -> tuple[int, int]:
        row = conn.execute(
            "SELECT COUNT(*), COALESCE(MAX(id), 0) FROM photos WHERE embedding IS NOT NULL"
        ).fetchone()
        return int(row[0]), int(row[1])

    def _semantic_index(conn: sqlite3.Connection):
        sig = _embed_signature(conn)
        if _semantic["sig"] != sig:
            _semantic["index"] = search.SemanticIndex.load(conn)
            _semantic["sig"] = sig
        return _semantic["index"]

    def _text_encoder():
        if _semantic["encoder"] is None and not _semantic["encoder_tried"]:
            _semantic["encoder_tried"] = True
            try:
                from .embed import SigLipTextEncoder

                _semantic["encoder"] = SigLipTextEncoder.from_config(config)
            except Exception:
                _semantic["encoder"] = None  # extra/model unavailable
        return _semantic["encoder"]

    @app.get("/api/capabilities")
    def api_capabilities(conn: sqlite3.Connection = Depends(get_conn)):
        # Semantic search needs both embedded photos and the runtime libraries to
        # embed a query. Checked cheaply (lib presence, not a model download) so the
        # UI can show/hide the control without blocking on a download.
        import importlib.util  # noqa: PLC0415

        has_embeddings = _embed_signature(conn)[0] > 0
        libs = all(importlib.util.find_spec(m) for m in ("onnxruntime", "tokenizers"))
        return {"semantic": bool(has_embeddings and libs)}

    # -- discovery --------------------------------------------------------
    @app.get("/api/facets")
    def api_facets(
        conn: sqlite3.Connection = Depends(get_conn),
        person_id: list[int] | None = Query(None),
        person_mode: str | None = None,
        scene: list[str] | None = Query(None),
        country: list[str] | None = Query(None),
        city: list[str] | None = Query(None),
        place: list[str] | None = Query(None),
        year: list[str] | None = Query(None),
        date_from: str | None = Query(None, pattern=_DATE_PATTERN),
        date_to: str | None = Query(None, pattern=_DATE_PATTERN),
        camera: list[str] | None = Query(None),
        people: list[str] | None = Query(None),
        known: list[str] | None = Query(None),
        has_faces: bool | None = None,
        favorite: bool | None = None,
        q: str | None = None,
    ):
        filters = {
            "person_id": person_id, "person_mode": person_mode,
            "scene": scene, "country": country,
            "city": city, "place": place, "year": year, "date_from": date_from,
            "date_to": date_to, "camera": camera, "people": people, "known": known,
            "has_faces": has_faces, "favorite": favorite, "q": q,
        }
        return search.facets(conn, filters)

    @app.get("/api/photos")
    def api_photos(
        conn: sqlite3.Connection = Depends(get_conn),
        person_id: list[int] | None = Query(None),
        person_mode: str | None = None,
        scene: list[str] | None = Query(None),
        country: list[str] | None = Query(None),
        city: list[str] | None = Query(None),
        place: list[str] | None = Query(None),
        year: list[str] | None = Query(None),
        date_from: str | None = Query(None, pattern=_DATE_PATTERN),
        date_to: str | None = Query(None, pattern=_DATE_PATTERN),
        camera: list[str] | None = Query(None),
        people: list[str] | None = Query(None),
        known: list[str] | None = Query(None),
        has_faces: bool | None = None,
        favorite: bool | None = None,
        q: str | None = None,
        text: str | None = None,
        sort: str | None = None,
        limit: int = Query(60, ge=1, le=500),
        offset: int = Query(0, ge=0),
    ):
        filters = {
            "person_id": person_id, "person_mode": person_mode,
            "scene": scene, "country": country,
            "city": city, "place": place, "year": year, "date_from": date_from,
            "date_to": date_to, "camera": camera, "people": people, "known": known,
            "has_faces": has_faces, "favorite": favorite, "q": q, "text": text, "sort": sort,
        }
        # A natural-language ``text`` query switches to semantic ranking (by image
        # embedding), ANDed with the structured filters. It supersedes ``sort`` —
        # results come back in relevance order — and returns its own (top-k) total.
        if text and text.strip():
            # Hybrid decomposition: peel known person-names and count phrases out of
            # the query into structured filters, leaving the residual for SigLIP.
            persons = [dict(r) for r in conn.execute("SELECT id, name FROM persons")]
            plan = planner.plan_query(text.strip(), persons)
            merged = {k: v for k, v in filters.items() if k != "text"}
            if plan.person_ids:
                merged["person_id"] = _union(merged.get("person_id"), plan.person_ids)
            if plan.person_mode and not merged.get("person_mode"):
                merged["person_mode"] = plan.person_mode
            if plan.people:
                merged["people"] = _union(merged.get("people"), plan.people)
            plan_payload = {
                "persons": plan.person_names, "people": plan.people,
                "person_mode": plan.person_mode, "text": plan.text,
            }

            rows: list[dict]
            total: int | None
            if plan.text:
                index = _semantic_index(conn)
                if index.size == 0:
                    raise HTTPException(
                        409,
                        "No photo embeddings yet — run `photo-atlas embed` (or "
                        "`index --embed`) to enable semantic search.",
                    )
                encoder = _text_encoder()
                if encoder is None:
                    raise HTTPException(
                        501,
                        "Semantic search could not load the SigLIP text encoder — "
                        "check the model can be downloaded (or set "
                        "PHOTO_ATLAS_TEXT_MODEL to a local file).",
                    )
                rows, total = search.semantic_search(
                    conn, merged, encoder.embed_text(plan.text), index,
                    top_k=config.semantic_top_k, limit=limit, offset=offset,
                )
            else:
                # The query reduced to pure structured filters ("Stefano alone"):
                # no visual leg, so no embeddings/model needed — just filter + page.
                rows, total_count = search.search_photos(
                    conn, merged, limit=limit, offset=offset, count=(offset == 0)
                )
                total = total_count if total_count >= 0 else None
            return {
                "total": total, "count": len(rows), "offset": offset,
                "photos": rows, "plan": plan_payload,
            }

        # ``total`` is page-invariant, so only count on the first page; later
        # infinite-scroll pages send ``total: null`` and the client keeps its copy.
        rows, total = search.search_photos(
            conn, filters, limit=limit, offset=offset, count=(offset == 0)
        )
        return {
            "total": total if total >= 0 else None,
            "count": len(rows), "offset": offset, "photos": rows,
        }

    @app.get("/api/memories")
    def api_memories(
        month: int | None = Query(None, ge=1, le=12),
        day: int | None = Query(None, ge=1, le=31),
        conn: sqlite3.Connection = Depends(get_conn),
    ):
        # "On this day": photos from the same calendar date in past years. Defaults
        # to the server's current date when month/day are omitted.
        today = datetime.date.today()
        m = month or today.month
        d = day or today.day
        groups = search.on_this_day(conn, m, d)
        return {
            "month": m, "day": d,
            "total": sum(g["count"] for g in groups), "groups": groups,
        }

    @app.get("/api/map")
    def api_map(
        conn: sqlite3.Connection = Depends(get_conn),
        person_id: list[int] | None = Query(None),
        person_mode: str | None = None,
        scene: list[str] | None = Query(None),
        country: list[str] | None = Query(None),
        city: list[str] | None = Query(None),
        place: list[str] | None = Query(None),
        year: list[str] | None = Query(None),
        date_from: str | None = Query(None, pattern=_DATE_PATTERN),
        date_to: str | None = Query(None, pattern=_DATE_PATTERN),
        camera: list[str] | None = Query(None),
        people: list[str] | None = Query(None),
        known: list[str] | None = Query(None),
        has_faces: bool | None = None,
        favorite: bool | None = None,
        q: str | None = None,
    ):
        filters = {
            "person_id": person_id, "person_mode": person_mode,
            "scene": scene, "country": country,
            "city": city, "place": place, "year": year, "date_from": date_from,
            "date_to": date_to, "camera": camera, "people": people, "known": known,
            "has_faces": has_faces, "favorite": favorite, "q": q,
        }
        return {"points": search.map_points(conn, filters, limit=config.map_point_limit)}

    @app.get("/api/photos/{photo_id}")
    def api_photo(photo_id: int, conn: sqlite3.Connection = Depends(get_conn)):
        photo = search.photo_detail(conn, photo_id)
        if photo is None:
            raise HTTPException(404, "photo not found")
        return photo

    @app.get("/api/photos/{photo_id}/similar")
    def api_similar(
        photo_id: int,
        limit: int = Query(60, ge=1, le=500),
        offset: int = Query(0, ge=0),
        conn: sqlite3.Connection = Depends(get_conn),
    ):
        # "More like this": rank the library by SigLIP image-embedding similarity to
        # this photo. No text encoder/model download needed — it reuses the stored
        # embeddings directly — so it works whenever `embed`/`index --embed` has run.
        if conn.execute("SELECT 1 FROM photos WHERE id=?", (photo_id,)).fetchone() is None:
            raise HTTPException(404, "photo not found")
        index = _semantic_index(conn)
        if index.size == 0:
            raise HTTPException(
                409,
                "No photo embeddings yet — run `photo-atlas embed` (or "
                "`index --embed`) to enable similarity search.",
            )
        rows, total = search.similar_photos(
            conn, photo_id, index, top_k=config.semantic_top_k, limit=limit, offset=offset
        )
        return {"total": total, "count": len(rows), "offset": offset, "photos": rows}

    @app.put("/api/photos/{photo_id}/favorite")
    def api_set_favorite(
        photo_id: int,
        payload: FavoriteRequest,
        conn: sqlite3.Connection = Depends(get_conn),
    ):
        if not db.set_favorite(conn, photo_id, payload.favorite):
            raise HTTPException(404, "photo not found")
        return {"ok": True, "favorite": payload.favorite}

    # -- media ------------------------------------------------------------
    @app.get("/api/image/{photo_id}")
    def api_image(photo_id: int, conn: sqlite3.Connection = Depends(get_conn)):
        row = conn.execute("SELECT path FROM photos WHERE id=?", (photo_id,)).fetchone()
        if row is None or not Path(row["path"]).exists():
            raise HTTPException(404, "image not found")
        return FileResponse(row["path"])

    @app.get("/api/preview/{photo_id}")
    def api_preview(photo_id: int, conn: sqlite3.Connection = Depends(get_conn)):
        row = conn.execute("SELECT path, sha1 FROM photos WHERE id=?", (photo_id,)).fetchone()
        if row is None or not row["path"] or not Path(row["path"]).exists():
            raise HTTPException(404, "image not found")
        src = Path(row["path"])
        sha1 = row["sha1"] or metadata.sha1_of(src)
        try:
            dest = metadata.cached_resized(
                config.previews_dir, src, sha1, config.preview_size, quality=88
            )
        except Exception:
            # Any decode/encode failure (corrupt or exotic format) falls back to
            # streaming the original so the lightbox still works.
            return FileResponse(src)
        return FileResponse(dest)

    @app.get("/api/thumb/{photo_id}")
    def api_thumb(
        photo_id: int,
        size: int | None = Query(None, ge=64, le=1024),
        conn: sqlite3.Connection = Depends(get_conn),
    ):
        row = conn.execute(
            "SELECT thumb_path, path, sha1 FROM photos WHERE id=?", (photo_id,)
        ).fetchone()
        if row is None:
            raise HTTPException(404, "photo not found")
        # A non-default size (e.g. the retina 2x ``srcset`` variant) is generated
        # and cached on demand from the original.
        if size and size != config.thumb_size and row["path"] and Path(row["path"]).exists():
            src = Path(row["path"])
            sha1 = row["sha1"] or metadata.sha1_of(src)
            try:
                return FileResponse(metadata.cached_resized(config.thumbs_dir, src, sha1, size))
            except Exception:
                pass  # fall back to the pre-generated default thumb
        thumb = row["thumb_path"]
        if thumb and Path(thumb).exists():
            return FileResponse(thumb)
        if row["path"] and Path(row["path"]).exists():
            return FileResponse(row["path"])
        raise HTTPException(404, "thumbnail not found")

    @app.get("/api/face/{face_id}")
    def api_face(face_id: int, conn: sqlite3.Connection = Depends(get_conn)):
        row = conn.execute("SELECT crop_path FROM faces WHERE id=?", (face_id,)).fetchone()
        if row is None:
            raise HTTPException(404, "face crop not found")
        crop = row["crop_path"]
        if not crop or not Path(crop).exists():
            # The crop may have failed to write at index time (crop_path=NULL) or
            # been deleted since; rebuild it from the source photo before 404ing.
            from . import indexer

            crop = indexer.regenerate_face_crop(conn, config, face_id)
            if not crop:
                raise HTTPException(404, "face crop not found")
        return FileResponse(crop)

    # -- smart albums (saved searches) ------------------------------------
    @app.get("/api/albums")
    def api_albums(conn: sqlite3.Connection = Depends(get_conn)):
        return {"albums": db.list_saved_searches(conn)}

    @app.post("/api/albums")
    def api_create_album(
        payload: AlbumRequest, conn: sqlite3.Connection = Depends(get_conn)
    ):
        if not payload.name.strip():
            raise HTTPException(400, "name must not be empty")
        album_id = db.create_saved_search(conn, payload.name, payload.query)
        return {"ok": True, "id": album_id}

    @app.delete("/api/albums/{album_id}")
    def api_delete_album(album_id: int, conn: sqlite3.Connection = Depends(get_conn)):
        db.delete_saved_search(conn, album_id)
        return {"ok": True}

    # -- persons ----------------------------------------------------------
    @app.get("/api/persons")
    def api_persons(conn: sqlite3.Connection = Depends(get_conn)):
        return {"persons": library.list_persons(conn)}

    @app.patch("/api/persons/{person_id}")
    def api_rename(
        person_id: int, payload: RenameRequest, conn: sqlite3.Connection = Depends(get_conn)
    ):
        if not payload.name.strip():
            raise HTTPException(400, "name must not be empty")
        try:
            library.rename_person(conn, person_id, payload.name)
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc
        return {"ok": True}

    @app.delete("/api/persons/{person_id}")
    def api_delete_person(person_id: int, conn: sqlite3.Connection = Depends(get_conn)):
        library.delete_person(conn, person_id)
        return {"ok": True}

    @app.get("/api/persons/{person_id}/faces")
    def api_person_faces(person_id: int, conn: sqlite3.Connection = Depends(get_conn)):
        return {"faces": library.list_person_faces(conn, person_id)}

    @app.post("/api/persons/{person_id}/merge")
    def api_merge_person(
        person_id: int, payload: MergeRequest, conn: sqlite3.Connection = Depends(get_conn)
    ):
        try:
            library.merge_persons(conn, payload.source_id, person_id)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        return {"ok": True, "person_id": person_id}

    @app.put("/api/persons/{person_id}/cover")
    def api_set_cover(
        person_id: int, payload: CoverRequest, conn: sqlite3.Connection = Depends(get_conn)
    ):
        try:
            library.set_cover_face(conn, person_id, payload.face_id)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        return {"ok": True}

    # -- clusters & assignment -------------------------------------------
    @app.get("/api/clusters")
    def api_clusters(conn: sqlite3.Connection = Depends(get_conn)):
        return {"clusters": library.list_clusters(conn)}

    @app.post("/api/clusters/{cluster_id}/assign")
    def api_assign_cluster(
        cluster_id: int, payload: AssignRequest, conn: sqlite3.Connection = Depends(get_conn)
    ):
        try:
            person_id = library.assign_cluster(
                conn, cluster_id, name=payload.name, person_id=payload.person_id
            )
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        return {"ok": True, "person_id": person_id}

    @app.post("/api/faces/{face_id}/assign")
    def api_assign_face(
        face_id: int, payload: AssignRequest, conn: sqlite3.Connection = Depends(get_conn)
    ):
        try:
            person_id = library.assign_face(
                conn, face_id, name=payload.name, person_id=payload.person_id
            )
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        return {"ok": True, "person_id": person_id}

    @app.post("/api/faces/{face_id}/unassign")
    def api_unassign_face(face_id: int, conn: sqlite3.Connection = Depends(get_conn)):
        library.unassign_face(conn, face_id)
        return {"ok": True}

    # -- web UI -----------------------------------------------------------
    @app.get("/", response_class=HTMLResponse)
    def index():
        return (WEB_DIR / "index.html").read_text(encoding="utf-8")

    if WEB_DIR.exists():
        app.mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static")

    return app


app = create_app()
