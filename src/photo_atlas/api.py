"""FastAPI application for Photo Atlas.

Exposes the catalog as a small JSON API plus media endpoints (full image,
thumbnail, face crop) and serves the single-page web UI from
:mod:`photo_atlas.web`.

Create the app with :func:`create_app` (used by the CLI ``serve`` command) so a
custom :class:`~photo_atlas.config.AtlasConfig` can be injected; importing
``app`` uses the default library location.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import db, library, search
from .config import AtlasConfig

WEB_DIR = Path(__file__).parent / "web"


class AssignRequest(BaseModel):
    name: str | None = None
    person_id: int | None = None


class RenameRequest(BaseModel):
    name: str


def create_app(config: AtlasConfig | None = None) -> FastAPI:
    config = (config or AtlasConfig()).ensure_dirs()
    app = FastAPI(title="Photo Atlas", version="0.1.0")

    def get_conn() -> sqlite3.Connection:
        conn = db.connect(config.db_path)
        try:
            yield conn
        finally:
            conn.close()

    # -- discovery --------------------------------------------------------
    @app.get("/api/facets")
    def api_facets(
        conn: sqlite3.Connection = Depends(get_conn),
        person_id: list[int] | None = Query(None),
        scene: list[str] | None = Query(None),
        country: list[str] | None = Query(None),
        city: list[str] | None = Query(None),
        place: list[str] | None = Query(None),
        year: list[str] | None = Query(None),
        date_from: str | None = None,
        date_to: str | None = None,
        camera: list[str] | None = Query(None),
        has_faces: bool | None = None,
        q: str | None = None,
    ):
        filters = {
            "person_id": person_id, "scene": scene, "country": country,
            "city": city, "place": place, "year": year, "date_from": date_from,
            "date_to": date_to, "camera": camera, "has_faces": has_faces, "q": q,
        }
        return search.facets(conn, filters)

    @app.get("/api/photos")
    def api_photos(
        conn: sqlite3.Connection = Depends(get_conn),
        person_id: list[int] | None = Query(None),
        scene: list[str] | None = Query(None),
        country: list[str] | None = Query(None),
        city: list[str] | None = Query(None),
        place: list[str] | None = Query(None),
        year: list[str] | None = Query(None),
        date_from: str | None = None,
        date_to: str | None = None,
        camera: list[str] | None = Query(None),
        has_faces: bool | None = None,
        q: str | None = None,
        sort: str | None = None,
        limit: int = Query(60, le=500),
        offset: int = 0,
    ):
        filters = {
            "person_id": person_id, "scene": scene, "country": country,
            "city": city, "place": place, "year": year, "date_from": date_from,
            "date_to": date_to, "camera": camera, "has_faces": has_faces,
            "q": q, "sort": sort,
        }
        rows, total = search.search_photos(conn, filters, limit=limit, offset=offset)
        return {"total": total, "count": len(rows), "offset": offset, "photos": rows}

    @app.get("/api/photos/{photo_id}")
    def api_photo(photo_id: int, conn: sqlite3.Connection = Depends(get_conn)):
        photo = search.photo_detail(conn, photo_id)
        if photo is None:
            raise HTTPException(404, "photo not found")
        return photo

    # -- media ------------------------------------------------------------
    @app.get("/api/image/{photo_id}")
    def api_image(photo_id: int, conn: sqlite3.Connection = Depends(get_conn)):
        row = conn.execute("SELECT path FROM photos WHERE id=?", (photo_id,)).fetchone()
        if row is None or not Path(row["path"]).exists():
            raise HTTPException(404, "image not found")
        return FileResponse(row["path"])

    @app.get("/api/thumb/{photo_id}")
    def api_thumb(photo_id: int, conn: sqlite3.Connection = Depends(get_conn)):
        row = conn.execute("SELECT thumb_path, path FROM photos WHERE id=?", (photo_id,)).fetchone()
        if row is None:
            raise HTTPException(404, "photo not found")
        thumb = row["thumb_path"]
        if thumb and Path(thumb).exists():
            return FileResponse(thumb)
        if row["path"] and Path(row["path"]).exists():
            return FileResponse(row["path"])
        raise HTTPException(404, "thumbnail not found")

    @app.get("/api/face/{face_id}")
    def api_face(face_id: int, conn: sqlite3.Connection = Depends(get_conn)):
        row = conn.execute("SELECT crop_path FROM faces WHERE id=?", (face_id,)).fetchone()
        if row is None or not row["crop_path"] or not Path(row["crop_path"]).exists():
            raise HTTPException(404, "face crop not found")
        return FileResponse(row["crop_path"])

    # -- persons ----------------------------------------------------------
    @app.get("/api/persons")
    def api_persons(conn: sqlite3.Connection = Depends(get_conn)):
        return {"persons": library.list_persons(conn)}

    @app.patch("/api/persons/{person_id}")
    def api_rename(person_id: int, payload: RenameRequest, conn: sqlite3.Connection = Depends(get_conn)):
        library.rename_person(conn, person_id, payload.name)
        return {"ok": True}

    @app.delete("/api/persons/{person_id}")
    def api_delete_person(person_id: int, conn: sqlite3.Connection = Depends(get_conn)):
        library.delete_person(conn, person_id)
        return {"ok": True}

    # -- clusters & assignment -------------------------------------------
    @app.get("/api/clusters")
    def api_clusters(conn: sqlite3.Connection = Depends(get_conn)):
        return {"clusters": library.list_clusters(conn)}

    @app.post("/api/clusters/{cluster_id}/assign")
    def api_assign_cluster(cluster_id: int, payload: AssignRequest, conn: sqlite3.Connection = Depends(get_conn)):
        try:
            person_id = library.assign_cluster(
                conn, cluster_id, name=payload.name, person_id=payload.person_id
            )
        except ValueError as exc:
            raise HTTPException(400, str(exc))
        return {"ok": True, "person_id": person_id}

    @app.post("/api/faces/{face_id}/assign")
    def api_assign_face(face_id: int, payload: AssignRequest, conn: sqlite3.Connection = Depends(get_conn)):
        try:
            person_id = library.assign_face(
                conn, face_id, name=payload.name, person_id=payload.person_id
            )
        except ValueError as exc:
            raise HTTPException(400, str(exc))
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
