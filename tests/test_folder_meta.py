"""Tests for mining year / month / place from folder names.

Many real libraries are organised like ``2012/2012_05_Sardegna/IMG_0001.jpg``
even when the files carry no EXIF. We parse those folder names and use them
only to fill the gaps EXIF leaves behind.
"""

from __future__ import annotations

import pytest

from photo_atlas.folder_meta import FolderMeta, extract_folder_meta, parse_component


# -- single-component parsing ---------------------------------------------
@pytest.mark.parametrize(
    "name,expected",
    [
        ("2012_05_Sardegna", FolderMeta(2012, 5, "Sardegna")),
        ("2012-05-Sardegna", FolderMeta(2012, 5, "Sardegna")),
        ("2012_05_12_Sardegna", FolderMeta(2012, 5, "Sardegna")),  # day dropped
        ("2012_Sardegna", FolderMeta(2012, None, "Sardegna")),
        ("Sardegna_2012", FolderMeta(2012, None, "Sardegna")),
        ("2012_05", FolderMeta(2012, 5, None)),
        ("2012", FolderMeta(2012, None, None)),
        ("2012_May_Sardegna", FolderMeta(2012, 5, "Sardegna")),
        ("2012_maggio_Sardegna", FolderMeta(2012, 5, "Sardegna")),  # Italian month
        ("2012_05_New_York", FolderMeta(2012, 5, "New York")),
    ],
)
def test_parse_component(name, expected):
    assert parse_component(name) == expected


@pytest.mark.parametrize(
    "name",
    ["Pictures", "DCIM", "Camera Roll", "backup", "100CANON", ""],
)
def test_parse_component_ignores_undated_folders(name):
    """Folders without a 4-digit year yield nothing, so ordinary directory
    names like ``Pictures`` are never mistaken for a place."""
    assert parse_component(name) == FolderMeta()


def test_parse_component_rejects_out_of_range_month():
    # "13" is not a month: it stays part of the place, not the month slot.
    assert parse_component("2012_13_Trip") == FolderMeta(2012, None, "13 Trip")


# -- walking a full path ----------------------------------------------------
def test_extract_walks_to_nearest_dated_folder(tmp_path):
    photo = tmp_path / "2012" / "2012_05_Sardegna" / "IMG_0001.jpg"
    assert extract_folder_meta(photo) == FolderMeta(2012, 5, "Sardegna")


def test_extract_merges_fields_across_ancestors(tmp_path):
    # year+month live on the closest dated folder, the place one level up.
    photo = tmp_path / "2012_Sardegna" / "2012_05" / "img.jpg"
    assert extract_folder_meta(photo) == FolderMeta(2012, 5, "Sardegna")


def test_extract_returns_empty_without_year(tmp_path):
    photo = tmp_path / "Pictures" / "vacation" / "img.jpg"
    assert extract_folder_meta(photo) == FolderMeta()


def test_extract_month_from_yearless_named_subfolder(tmp_path):
    # A common layout: YYYY/<NN-monthname>/file. The month folder carries no year,
    # but the named month is unambiguous and the year comes from its parent.
    photo = tmp_path / "Kai" / "2026" / "01-gennaio" / "img.jpg"
    assert extract_folder_meta(photo) == FolderMeta(2026, 1, None)


def test_extract_named_month_subfolder_english(tmp_path):
    photo = tmp_path / "2025" / "august" / "img.jpg"
    assert extract_folder_meta(photo) == FolderMeta(2025, 8, None)


def test_extract_ignores_bare_numeric_subfolder(tmp_path):
    # A yearless, purely numeric folder is too ambiguous (day? counter?) to read
    # as a month, so the year folder alone supplies the date.
    photo = tmp_path / "2025" / "05" / "img.jpg"
    assert extract_folder_meta(photo) == FolderMeta(2025, None, None)


def test_extract_explicit_month_beats_yearless_sibling(tmp_path):
    # When the dated folder already names a month, a deeper yearless month folder
    # must not override it.
    photo = tmp_path / "2025_03_Trip" / "marzo" / "img.jpg"
    assert extract_folder_meta(photo) == FolderMeta(2025, 3, "Trip")


# -- integration: indexer / db / search ------------------------------------
def _plain_jpeg(path, color=(20, 40, 60)):
    """Write a tiny JPEG with no EXIF date or GPS."""
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (24, 24), color).save(path, "JPEG")
    return path


def _exif_jpeg(path, when="2019:03:04 05:06:07"):
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    exif = Image.Exif()
    exif[0x8769] = {0x9003: when}  # DateTimeOriginal in the Exif sub-IFD
    Image.new("RGB", (24, 24), (30, 30, 30)).save(path, "JPEG", exif=exif)
    return path


def _index(tmp_path):
    from photo_atlas import db, indexer
    from photo_atlas.config import AtlasConfig

    config = AtlasConfig(home=tmp_path / "lib").ensure_dirs()
    indexer.index_path(config, tmp_path / "src", backend_name="none", geocode=False)
    return db.connect(config.db_path)


def test_folder_date_and_place_fill_when_no_exif(tmp_path):
    _plain_jpeg(tmp_path / "src" / "2013" / "2013_07_Sardegna" / "a.jpg")
    conn = _index(tmp_path)
    row = conn.execute(
        "SELECT taken_at, taken_source, folder_place FROM photos"
    ).fetchone()
    assert row["taken_source"] == "folder"
    assert row["taken_at"].startswith("2013-07-01")
    assert row["folder_place"] == "Sardegna"


def test_exif_date_wins_over_folder(tmp_path):
    _exif_jpeg(tmp_path / "src" / "2013" / "2013_07_Sardegna" / "b.jpg")
    conn = _index(tmp_path)
    row = conn.execute(
        "SELECT taken_at, taken_source, folder_place FROM photos"
    ).fetchone()
    assert row["taken_source"] == "exif"
    assert row["taken_at"].startswith("2019-03-04")
    # The trip label is still recorded even though EXIF supplied the date.
    assert row["folder_place"] == "Sardegna"


def test_place_facet_and_filter(tmp_path):
    from photo_atlas import search

    # Distinct colours so the two trip photos aren't byte-identical (which the
    # indexer would now deduplicate by SHA-1).
    _plain_jpeg(tmp_path / "src" / "2013_05_Sardegna" / "a.jpg", color=(20, 40, 60))
    _plain_jpeg(tmp_path / "src" / "2014_08_Norway" / "b.jpg", color=(60, 40, 20))
    conn = _index(tmp_path)

    places = {f["value"] for f in search.facets(conn)["places"]}
    assert {"Sardegna", "Norway"} <= places

    rows, total = search.search_photos(conn, {"place": "Norway"})
    assert total == 1 and rows[0]["folder_place"] == "Norway"


def test_db_migration_adds_folder_place_column(tmp_path):
    """An existing catalog created before folder_place must gain the column."""
    import sqlite3

    from photo_atlas import db

    db_path = tmp_path / "old.db"
    # Simulate an old DB: the original photos schema, minus folder_place. It
    # includes the columns the schema's indexes reference (taken_at, etc.).
    raw = sqlite3.connect(db_path)
    raw.execute(
        "CREATE TABLE photos (id INTEGER PRIMARY KEY, path TEXT UNIQUE, "
        "filename TEXT, taken_at TEXT, scene_type TEXT, place_country TEXT, "
        "place_city TEXT, camera_model TEXT, indexed_at TEXT)"
    )
    raw.commit()
    raw.close()

    conn = db.connect(db_path)
    cols = {r["name"] for r in conn.execute("PRAGMA table_info(photos)")}
    assert "folder_place" in cols


def _index_names(conn):
    return {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='index'")}


def test_composite_indexes_present_and_supersede_singles(config):
    """Fresh catalog: the browse/filter composites exist and the single-column
    scene/folder/person indexes they supersede are gone."""
    from photo_atlas import db

    conn = db.connect(config.db_path)
    names = _index_names(conn)
    assert {
        "idx_photos_scene_taken",
        "idx_photos_folder_taken",
        "idx_faces_person_photo",
    } <= names
    assert names.isdisjoint({"idx_photos_scene", "idx_photos_folder", "idx_faces_person"})


def test_migration_drops_superseded_single_column_indexes(tmp_path):
    """An old catalog carrying the single-column indexes sheds them on open."""
    import sqlite3

    from photo_atlas import db

    db_path = tmp_path / "old.db"
    raw = sqlite3.connect(db_path)
    raw.executescript(
        "CREATE TABLE photos (id INTEGER PRIMARY KEY, path TEXT UNIQUE, "
        "taken_at TEXT, scene_type TEXT, folder_place TEXT, place_country TEXT, "
        "place_city TEXT, camera_model TEXT, indexed_at TEXT);"
        "CREATE TABLE faces (id INTEGER PRIMARY KEY, photo_id INTEGER, "
        "person_id INTEGER, cluster_id INTEGER);"
        "CREATE INDEX idx_photos_scene ON photos(scene_type);"
        "CREATE INDEX idx_photos_folder ON photos(folder_place);"
        "CREATE INDEX idx_faces_person ON faces(person_id);"
    )
    raw.commit()
    raw.close()

    conn = db.connect(db_path)
    names = _index_names(conn)
    assert names.isdisjoint({"idx_photos_scene", "idx_photos_folder", "idx_faces_person"})
    assert "idx_photos_scene_taken" in names
