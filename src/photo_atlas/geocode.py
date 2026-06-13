"""Offline reverse geocoding.

Given a latitude/longitude pair we return the nearest known place. By default
we use a compact bundled dataset of ~120 world cities (good enough to label a
photo with a city + country). If the optional ``reverse_geocoder`` package is
installed it is used instead for far finer resolution.

The lookup is a brute-force nearest neighbour using the haversine distance.
With a small bundled dataset this is effectively instant and needs no extra
dependencies or network access.
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from functools import lru_cache
from importlib import resources


@dataclass(frozen=True)
class Place:
    city: str
    admin: str
    country: str
    country_code: str
    lat: float
    lon: float

    @property
    def label(self) -> str:
        return f"{self.city}, {self.country}"


@lru_cache(maxsize=1)
def _country_names() -> dict[str, str]:
    """Map ISO country code -> country name, derived from the bundled dataset."""

    return {p.country_code: p.country for p in _bundled_cities() if p.country_code}


@lru_cache(maxsize=1)
def _bundled_cities() -> list[Place]:
    places: list[Place] = []
    data = resources.files("photo_atlas.data").joinpath("cities.csv").read_text(encoding="utf-8")
    for row in csv.DictReader(data.splitlines()):
        places.append(
            Place(
                city=row["name"],
                admin=row["admin"],
                country=row["country"],
                country_code=row["country_code"],
                lat=float(row["lat"]),
                lon=float(row["lon"]),
            )
        )
    return places


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
    return 2 * r * math.asin(min(1.0, math.sqrt(a)))


class Geocoder:
    """Reverse geocoder with an optional high-resolution backend."""

    def __init__(self, prefer_external: bool = True):
        self._rg = None
        if prefer_external:
            try:  # pragma: no cover - exercised only when the package exists
                import reverse_geocoder as rg

                self._rg = rg
            except ModuleNotFoundError:
                self._rg = None

    def lookup(self, lat: float | None, lon: float | None) -> Place | None:
        if lat is None or lon is None:
            return None

        if self._rg is not None:
            result = self._rg.search((lat, lon), mode=1)[0]
            cc = result.get("cc", "")
            return Place(
                city=result.get("name", ""),
                admin=result.get("admin1", ""),
                # reverse_geocoder only returns a country *code*; resolve it to a
                # name (falling back to the code) instead of mislabelling the
                # region (admin1) as the country.
                country=_country_names().get(cc, cc),
                country_code=cc,
                lat=float(result.get("lat", lat)),
                lon=float(result.get("lon", lon)),
            )

        best: Place | None = None
        best_d = math.inf
        for place in _bundled_cities():
            d = _haversine_km(lat, lon, place.lat, place.lon)
            if d < best_d:
                best_d, best = d, place
        return best
