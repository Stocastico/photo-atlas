"""Tests for mining a capture date/time from a photo's *filename*.

Across a 15-year library, phones/cameras/messaging apps stamp the capture date
into the filename in many shapes, and EXIF is often missing on edited, exported
or messaging-app copies. These patterns were derived from the real collection
(see the format survey in TODO.md / CLAUDE.md), so the registry is driven by
what's actually present rather than guesswork. Bare counters (``IMG_7133``,
``DSC_0191``), plain sequence numbers and resolutions must never be mistaken for
dates.
"""

from __future__ import annotations

from datetime import datetime

import pytest

from photo_atlas.filename_date import parse_filename_date


@pytest.mark.parametrize(
    "name,expected",
    [
        # Android camera: IMG_/VID_ YYYYMMDD_HHMMSS
        ("IMG_20180704_153012.jpg", datetime(2018, 7, 4, 15, 30, 12)),
        ("VID_20210530_125454.mp4", datetime(2021, 5, 30, 12, 54, 54)),
        # Google Pixel: PXL_YYYYMMDD_HHMMSSmmm (trailing millis ignored)
        ("PXL_20210101_010101123.jpg", datetime(2021, 1, 1, 1, 1, 1)),
        # iOS export: YYYYMMDD_HHMMSSmmm_iOS (+ optional " 2" dup suffix)
        ("20230721_101826611_iOS.heic", datetime(2023, 7, 21, 10, 18, 26)),
        ("20240902_141239000_iOS 2.jpg", datetime(2024, 9, 2, 14, 12, 39)),
        # Windows Phone: WP_YYYYMMDD_HH_MM_SS_Pro/Selfie/Panorama
        ("WP_20150314_13_48_25_Pro.jpg", datetime(2015, 3, 14, 13, 48, 25)),
        ("WP_20150412_13_44_43_Selfie.jpg", datetime(2015, 4, 12, 13, 44, 43)),
        # Windows Phone counter form (no time): WP_YYYYMMDD_NNN -> date only
        ("WP_20140101_005.jpg", datetime(2014, 1, 1)),
        # WhatsApp (date only, no time): IMG-YYYYMMDD-WAnnnn
        ("IMG-20171125-WA0009.jpg", datetime(2017, 11, 25)),
        ("VID-20180101-WA0001.mp4", datetime(2018, 1, 1)),
        # Standard separated: YYYY-MM-DD HH.MM.SS
        ("2011-08-16 13.31.33.jpg", datetime(2011, 8, 16, 13, 31, 33)),
        ("2015-03-27 12.23.08.jpg", datetime(2015, 3, 27, 12, 23, 8)),
        # Bare compact: YYYYMMDD_HHMMSS (no prefix)
        ("20150328_170838.jpg", datetime(2015, 3, 28, 17, 8, 38)),
        # Italian text date embedded: "DD MonthName YYYY"
        ("Isola D'Elba 25 Aprile 2006 177.jpg", datetime(2006, 4, 25)),
        # Italian numeric date: D.M.YYYY (day-first)
        ("15.5.2006 Compleanno 048.jpg", datetime(2006, 5, 15)),
    ],
)
def test_parse_known_formats(name, expected):
    assert parse_filename_date(name) == expected


@pytest.mark.parametrize(
    "name",
    [
        # Bare counters / sequence numbers must NOT be read as dates.
        "IMG_7133.JPG",
        "DSC_0191.JPG",
        "DSCN0958.JPG",
        "IMGP0192.JPG",
        "1207.JPG",
        "176.JPG",
        "A (215).jpg",
        "mosaico 33.jpg",
        "Urlaub-Junio-0124.JPG",
        # A resolution-looking token is not a date.
        "screenshot_1920x1080.png",
        # Out-of-range "dates" are rejected by validation.
        "20151345_999999.jpg",  # month 13, day 45
        "IMG_20180230_120000.jpg",  # Feb 30 doesn't exist
        "",
    ],
)
def test_rejects_non_dates(name):
    assert parse_filename_date(name) is None


def test_day_first_disambiguates_when_day_gt_12():
    # 25.04.2006 can only be day-first.
    assert parse_filename_date("25.04.2006 gita.jpg") == datetime(2006, 4, 25)


def test_future_year_rejected():
    # A year far in the future is almost certainly a counter, not a capture date.
    assert parse_filename_date("IMG_29990101_000000.jpg") is None
