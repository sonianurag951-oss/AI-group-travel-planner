from __future__ import annotations
import math
from pathlib import Path
import pandas as pd

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "pois.csv"
CITY_SPEED_KMPH = {"Goa": 30, "Jaipur": 28, "Manali": 20}  # average road speed
ROAD_FACTOR = 1.35  # straight-line -> road distance
REQUIRED = ["city", "name", "category", "lat", "lon", "cost", "duration", "rating", "open", "close"]


def load_pois(city: str | None = None, path: Path | str | None = None) -> pd.DataFrame:
    df = pd.read_csv(path or DATA_PATH)
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"POI file missing columns: {missing}")
    if city:
        df = df[df.city == city]
        if df.empty:
            raise ValueError(f"No places for city '{city}'")
    return df.reset_index(drop=True)


def list_cities(path: Path | str | None = None) -> list[str]:
    return sorted(pd.read_csv(path or DATA_PATH).city.unique())


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi, dlmb = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def travel_minutes(km: float, speed_kmph: float) -> float:
    return km * ROAD_FACTOR / speed_kmph * 60
