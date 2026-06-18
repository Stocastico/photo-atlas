"""Recover a capture date/time from a photo's *filename*.

Phones, cameras and messaging apps stamp the capture moment into the filename in
many shapes, and EXIF is frequently stripped on edited / exported / messaging-app
copies. The patterns below were derived from a real 15-year library (the format
survey in ``TODO.md``): Android (``IMG_YYYYMMDD_HHMMSS``), iOS exports
(``YYYYMMDD_HHMMSSmmm_iOS``), Windows Phone (``WP_YYYYMMDD_HH_MM_SS_*``),
WhatsApp (``IMG-YYYYMMDD-WAnnnn``), the ``YYYY-MM-DD HH.MM.SS`` form, the bare
compact ``YYYYMMDD_HHMMSS``, and Italian-language text dates
(``... 25 Aprile 2006 ...`` / ``15.5.2006 ...``).

This loses some generality (it's tuned to one personal collection), which is the
deliberate trade-off. The output is advisory: the indexer ranks it *below* a real
EXIF capture time but *above* the folder hint and filesystem mtime
(see :mod:`photo_atlas.metadata` / :mod:`photo_atlas.indexer`). Bare counters
(``IMG_7133``, ``DSC_0191``), sequence numbers and resolutions must never be
mistaken for a date, so every candidate is validated as a real calendar
date/time within a sane year range.
"""

from __future__ import annotations

import re
from datetime import datetime

from .folder_meta import _MONTHS

# A capture date older than the first consumer digital cameras, or in the
# future, is almost certainly a counter / id / resolution rather than a date.
_MIN_YEAR = 1990


def _max_year() -> int:
    # +1 so a photo taken "today" near New Year (or with a slightly fast clock)
    # isn't rejected; resolved lazily so the bound tracks the wall clock.
    return datetime.now().year + 1


def _make(
    year: int, month: int, day: int, hour: int = 0, minute: int = 0, second: int = 0
) -> datetime | None:
    """Build a :class:`datetime`, or ``None`` if it isn't a sane calendar moment."""

    if not _MIN_YEAR <= year <= _max_year():
        return None
    try:
        return datetime(year, month, day, hour, minute, second)
    except ValueError:
        return None


# Month names (English + Italian, full + abbreviations) for text dates, longest
# first so ``aprile`` wins over ``apr`` and ``settembre`` over ``set``.
_MONTH_ALT = "|".join(sorted(_MONTHS, key=len, reverse=True))

# Ordered, specific-first. Each entry is (compiled regex, builder) where the
# builder turns a match into a datetime-or-None. The first match wins.
_PATTERNS: list[tuple[re.Pattern[str], object]] = [
    # iOS export: YYYYMMDD_HHMMSS(mmm)_iOS  (trailing millis ignored)
    (
        re.compile(r"(?<!\d)(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})(\d{2})\d*_iOS", re.I),
        lambda m: _make(*(int(g) for g in m.groups())),
    ),
    # WhatsApp (date only): IMG-/VID- YYYYMMDD -WAnnnn
    (
        re.compile(r"(?:IMG|VID)-(\d{4})(\d{2})(\d{2})-WA\d+", re.I),
        lambda m: _make(int(m[1]), int(m[2]), int(m[3])),
    ),
    # Prefixed compact w/ time: IMG_/VID_/PXL_/MVIMG_/Screenshot_ YYYYMMDD(_-)HHMMSS
    (
        re.compile(
            r"(?:IMG|VID|PXL|MVIMG|Screenshot)[_-](\d{4})(\d{2})(\d{2})[_-](\d{2})(\d{2})(\d{2})",
            re.I,
        ),
        lambda m: _make(*(int(g) for g in m.groups())),
    ),
    # Windows Phone w/ time: WP_YYYYMMDD_HH_MM_SS_*
    (
        re.compile(r"WP_(\d{4})(\d{2})(\d{2})_(\d{2})_(\d{2})_(\d{2})", re.I),
        lambda m: _make(*(int(g) for g in m.groups())),
    ),
    # Windows Phone counter (date only): WP_YYYYMMDD_NNN
    (
        re.compile(r"WP_(\d{4})(\d{2})(\d{2})_\d+", re.I),
        lambda m: _make(int(m[1]), int(m[2]), int(m[3])),
    ),
    # Standard separated: YYYY-MM-DD HH.MM.SS  (space or underscore before time)
    (
        re.compile(r"(?<!\d)(\d{4})-(\d{2})-(\d{2})[ _](\d{2})\.(\d{2})\.(\d{2})"),
        lambda m: _make(*(int(g) for g in m.groups())),
    ),
    # Bare compact w/ time: YYYYMMDD_HHMMSS  (no trailing digits -> not millis)
    (
        re.compile(r"(?<!\d)(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})(\d{2})(?!\d)"),
        lambda m: _make(*(int(g) for g in m.groups())),
    ),
    # Italian / English text date (date only): "DD MonthName YYYY"
    (
        re.compile(rf"(?<!\d)(\d{{1,2}})[ ._-]({_MONTH_ALT})[ ._-](\d{{4}})", re.I),
        lambda m: _make(int(m[3]), _MONTHS[m[2].lower()], int(m[1])),
    ),
    # Numeric day-first date (date only): D.M.YYYY  (Italian convention)
    (
        re.compile(r"(?<!\d)(\d{1,2})\.(\d{1,2})\.(\d{4})(?!\d)"),
        lambda m: _make(int(m[3]), int(m[2]), int(m[1])),
    ),
]


def parse_filename_date(name: str) -> datetime | None:
    """Return the capture date/time encoded in ``name``, or ``None``.

    ``name`` is a filename (basename); the date may be the whole stem or embedded
    in a longer label. Date-only formats yield midnight. Returns ``None`` for
    counters, sequence numbers and anything that doesn't validate as a real date.
    """

    for pattern, build in _PATTERNS:
        match = pattern.search(name)
        if match is not None:
            dt = build(match)  # type: ignore[operator]
            if dt is not None:
                return dt
    return None
