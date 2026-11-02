"""Small text helpers."""

from __future__ import annotations

import unicodedata


def nfc(text: str) -> str:
    """Unicode NFC form. macOS file systems hand back decomposed (NFD) names
    ("e" + combining accent), while people type and browsers send composed text, so
    names that feed display/search are normalised to NFC."""

    return unicodedata.normalize("NFC", text)
