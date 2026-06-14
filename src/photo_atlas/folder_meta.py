"""Mine year / month / place hints from a photo's folder names.

Many libraries are organised like ``2012/2012_05_Sardegna/IMG_0001.jpg`` even
when the files themselves carry no EXIF date or GPS — a very common situation
for scans, phone exports and old camera dumps. We parse those folder names for
a capture year, an optional month and a trip/place label.

The output is advisory: the indexer uses it only to *fill the gaps* EXIF leaves
behind (see :mod:`photo_atlas.indexer`). To avoid mistaking ordinary directory
names (``Pictures``, ``DCIM``, ``backup``) for places, we only trust a folder
component that contains a four-digit year.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

# Tokens are separated by spaces, underscores, dashes or dots.
_SEP = re.compile(r"[\s_.\-]+")
_YEAR_RE = re.compile(r"^(?:19|20)\d{2}$")

# Month names we recognise in addition to numeric months. English + Italian
# (this project's author organises trips in Italian, e.g. ``maggio``), matched
# case-insensitively against the first three letters as well as the full word.
_MONTHS = {
    "jan": 1, "january": 1, "gen": 1, "gennaio": 1,
    "feb": 2, "february": 2, "febbraio": 2,
    "mar": 3, "march": 3, "marzo": 3,
    "apr": 4, "april": 4, "aprile": 4,
    "may": 5, "mag": 5, "maggio": 5,
    "jun": 6, "june": 6, "giu": 6, "giugno": 6,
    "jul": 7, "july": 7, "lug": 7, "luglio": 7,
    "aug": 8, "august": 8, "ago": 8, "agosto": 8,
    "sep": 9, "sept": 9, "september": 9, "set": 9, "settembre": 9,
    "oct": 10, "october": 10, "ott": 10, "ottobre": 10,
    "nov": 11, "november": 11, "novembre": 11,
    "dec": 12, "december": 12, "dic": 12, "dicembre": 12,
}


@dataclass
class FolderMeta:
    """Date / place hints recovered from folder names. Any field may be ``None``."""

    year: int | None = None
    month: int | None = None
    place: str | None = None


def _as_month(token: str) -> int | None:
    """Return 1-12 if ``token`` is a numeric or named month, else ``None``."""

    if token.isdigit():
        value = int(token)
        return value if 1 <= value <= 12 else None
    return _MONTHS.get(token.lower())


def _is_day(token: str) -> bool:
    return token.isdigit() and 1 <= int(token) <= 31


def parse_component(name: str) -> FolderMeta:
    """Parse a single folder name.

    A component only yields anything when it contains a four-digit year
    (1900-2099). The year may sit anywhere among the tokens, so both
    ``2012_05_Sardegna`` and ``Sardegna_2012`` are understood. A month (numeric
    or named) and an optional day immediately following the year are consumed;
    everything else becomes the place label.
    """

    tokens = [t for t in _SEP.split(name.strip()) if t]
    year_idx = next((i for i, t in enumerate(tokens) if _YEAR_RE.match(t)), None)
    if year_idx is None:
        return FolderMeta()

    year = int(tokens[year_idx])
    before = tokens[:year_idx]
    after = tokens[year_idx + 1:]

    month: int | None = None
    if after:
        m = _as_month(after[0])
        if m is not None:
            month = m
            after = after[1:]
            # A day number can follow the month (e.g. 2012_05_12_Sardegna);
            # drop it — we only keep month-level precision.
            if after and _is_day(after[0]):
                after = after[1:]

    place_tokens = before + after
    place = " ".join(place_tokens).strip() or None
    return FolderMeta(year=year, month=month, place=place)


def extract_folder_meta(path: Path | str) -> FolderMeta:
    """Recover date/place hints from the directories containing ``path``.

    Walks the parent directories from the file outward, considering only
    *dated* folders (those with a year). Fields are filled closest-first: the
    nearest dated folder wins, and shallower dated ancestors only supply fields
    still missing (so ``2012_Sardegna/2012_05/img.jpg`` yields year+month from
    the inner folder and the place from the outer one).
    """

    result = FolderMeta()
    for parent in Path(path).parents:
        fm = parse_component(parent.name)
        if fm.year is None:
            continue
        if result.year is None:
            result.year = fm.year
        if result.month is None:
            result.month = fm.month
        if result.place is None:
            result.place = fm.place
        if result.year is not None and result.month is not None and result.place is not None:
            break
    return result
