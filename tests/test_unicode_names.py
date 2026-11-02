"""macOS hands back NFD file names (e + combining accent); search types NFC.

Names that come from the file system are normalised to NFC for the display/search
columns (the real ``path`` stays byte-exact so the file can still be opened).
"""

from __future__ import annotations

import unicodedata
from pathlib import Path

from PIL import Image

from photo_atlas import db, folder_meta, indexer, search
from photo_atlas.config import AtlasConfig

NFC = "Città"
NFD = unicodedata.normalize("NFD", NFC)


def test_the_two_forms_really_differ():
    assert NFC != NFD and len(NFD) == len(NFC) + 1


def test_folder_place_is_nfc_even_when_the_folder_name_is_nfd():
    meta = folder_meta.extract_folder_meta(Path(f"/photos/2020_{NFD}/IMG_1.jpg"))
    assert meta.place == NFC


def _index_one(tmp_path, name):
    config = AtlasConfig(home=tmp_path / "lib").ensure_dirs()
    photos = tmp_path / "photos"
    photos.mkdir()
    f = photos / name
    Image.new("RGB", (32, 32), (9, 9, 9)).save(f)
    indexer.index_path(config, photos, backend_name="none", geocode=False, workers=1)
    return config, f


def test_indexed_filename_is_nfc_but_path_stays_openable(tmp_path):
    config, f = _index_one(tmp_path, f"{NFD}_mare.png")
    conn = db.connect(config.db_path)
    try:
        row = conn.execute("SELECT filename, path FROM photos").fetchone()
    finally:
        conn.close()
    assert row["filename"] == f"{NFC}_mare.png"
    assert Path(row["path"]).exists()  # the byte-exact path still resolves


def test_search_matches_regardless_of_the_form_typed(tmp_path):
    config, _ = _index_one(tmp_path, f"{NFD}_mare.png")
    conn = db.connect(config.db_path)
    try:
        for typed in (NFC, NFD, NFC.lower()):
            rows, total = search.search_photos(conn, {"q": typed}, limit=10, offset=0)
            assert total == 1, typed
    finally:
        conn.close()
