"""Exporting person labels to portable XMP sidecars."""

from __future__ import annotations


def _photo_with_person(tmp_path, name="Ada Lovelace", filename="IMG_1.jpg"):
    from PIL import Image

    from photo_atlas import db
    from photo_atlas.config import AtlasConfig

    photo = tmp_path / filename
    Image.new("RGB", (8, 8)).save(photo, "JPEG")
    cfg = AtlasConfig(home=tmp_path / "lib").ensure_dirs()
    conn = db.connect(cfg.db_path)
    try:
        pid = db.upsert_photo(
            conn, {"path": str(photo), "filename": filename, "sha1": "deadbeef"}
        )
        person = db.get_or_create_person(conn, name)
        db.replace_faces(conn, pid, [{"person_id": person}])
        conn.commit()
    finally:
        conn.close()
    return cfg, photo


def test_export_writes_xmp_sidecar_with_person_keyword(tmp_path):
    from photo_atlas import export

    cfg, photo = _photo_with_person(tmp_path)
    result = export.write_xmp_sidecars(cfg)
    assert result["photos"] == 1 and result["written"] == 1

    sidecar = photo.with_name(photo.name + ".xmp")  # IMG_1.jpg.xmp (exiftool convention)
    assert sidecar.exists()
    text = sidecar.read_text(encoding="utf-8")
    assert "Ada Lovelace" in text
    assert "People|Ada Lovelace" in text  # hierarchical keyword
    assert "dc:subject" in text


def test_export_escapes_xml_special_characters(tmp_path):
    from photo_atlas import export

    cfg, photo = _photo_with_person(tmp_path, name="Tom & Jerry")
    export.write_xmp_sidecars(cfg)
    text = (photo.with_name(photo.name + ".xmp")).read_text(encoding="utf-8")
    assert "Tom &amp; Jerry" in text and "Tom & Jerry" not in text


def test_export_dest_directory_redirects_sidecars(tmp_path):
    from photo_atlas import export

    cfg, photo = _photo_with_person(tmp_path)
    out = tmp_path / "labels"
    result = export.write_xmp_sidecars(cfg, dest=out)
    assert result["written"] == 1
    assert (out / (photo.name + ".xmp")).exists()
    # The original photo tree is left untouched.
    assert not photo.with_name(photo.name + ".xmp").exists()


def test_export_skips_photos_without_named_people(tmp_path):
    from photo_atlas import db, export
    from photo_atlas.config import AtlasConfig

    cfg = AtlasConfig(home=tmp_path / "lib").ensure_dirs()
    conn = db.connect(cfg.db_path)
    try:
        db.upsert_photo(conn, {"path": str(tmp_path / "x.jpg"), "filename": "x.jpg"})
        conn.commit()
    finally:
        conn.close()
    assert export.write_xmp_sidecars(cfg) == {"photos": 0, "written": 0}
