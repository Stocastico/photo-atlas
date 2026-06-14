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
