"""Live place data from OpenStreetMap (Nominatim geocoding + Overpass API). Free, no API key.

Important: OSM has NO ratings, prices or reliable hours. This module derives:
  * rating  -> an "importance" proxy (Wikipedia/Wikidata/website/hours tags), NOT a review score
  * cost    -> a typical-price default per category (INR), NOT real pricing
  * hours   -> parsed from `opening_hours` when it is a simple HH:MM-HH:MM, else category defaults
Treat output as a good starting point and verify details before real travel.
"""
from __future__ import annotations
import re
import time
from pathlib import Path

import pandas as pd
import requests

NOMINATIM = "https://nominatim.openstreetmap.org/search"
OVERPASS_URLS = ["https://overpass-api.de/api/interpreter", "https://overpass.kumi.systems/api/interpreter",
                 "https://overpass.private.coffee/api/interpreter"]
HEADERS = {"User-Agent": "group-travel-planner/1.0 (student portfolio project)"}
CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "osm_cache"
PER_CATEGORY = 10
MIN_PLACES = 8

# category -> (typical cost INR, typical duration min, default open, default close)
DEFAULTS = {
    "culture": (100, 90, 9.0, 18.0), "nature": (50, 120, 7.0, 18.0), "food": (600, 90, 11.0, 22.5),
    "adventure": (1500, 150, 9.0, 16.5), "nightlife": (1000, 150, 19.0, 24.0),
    "shopping": (0, 120, 10.0, 21.0), "relaxation": (0, 120, 7.0, 19.0),
}
# Several small queries are far gentler on the public servers than one big one.
QUERY_GROUPS = [
    ['nwr["tourism"~"^(attraction|museum|gallery|viewpoint|theme_park|zoo|aquarium)$"]["name"]({b});',
     'nwr["historic"~"^(castle|fort|monument|ruins|archaeological_site|palace|memorial)$"]["name"]({b});'],
    ['nwr["natural"~"^(waterfall|beach|peak|hot_spring)$"]["name"]({b});',
     'nwr["leisure"~"^(park|nature_reserve|garden|water_park)$"]["name"]({b});'],
    ['nwr["amenity"~"^(restaurant|cafe)$"]["name"]({b});'],
    ['nwr["amenity"~"^(bar|pub|nightclub|marketplace|place_of_worship)$"]["name"]({b});',
     'nwr["shop"="mall"]["name"]({b});'],
]


def classify(tags: dict) -> str | None:
    """Map OSM tags to one of the app's 7 categories (None = ignore)."""
    t = tags
    amenity, tourism, historic = t.get("amenity"), t.get("tourism"), t.get("historic")
    natural, leisure = t.get("natural"), t.get("leisure")
    if amenity in {"bar", "pub", "nightclub"}:
        return "nightlife"
    if tourism in {"theme_park"} or leisure == "water_park" or t.get("sport") in {"climbing", "scuba_diving", "water_sports", "paragliding"}:
        return "adventure"
    if natural in {"beach", "hot_spring"}:
        return "relaxation"
    if natural in {"waterfall", "peak"} or leisure in {"park", "nature_reserve", "garden"} or tourism in {"viewpoint", "zoo", "aquarium"}:
        return "nature"
    if historic or tourism in {"museum", "gallery", "attraction"}:
        return "culture"
    if amenity == "place_of_worship" and ("wikidata" in t or "wikipedia" in t):
        return "culture"  # only notable places of worship, otherwise there are thousands
    if amenity in {"restaurant", "cafe"}:
        return "food"
    if amenity == "marketplace" or t.get("shop") == "mall":
        return "shopping"
    return None


def importance(tags: dict) -> float:
    return (2 * ("wikidata" in tags) + 2 * ("wikipedia" in tags) + ("website" in tags)
            + ("opening_hours" in tags) + 0.5 * (len(tags) > 8))


def proxy_rating(tags: dict) -> float:
    return round(min(4.8, 3.6 + 0.3 * importance(tags)), 1)


def parse_hours(value: str | None, default: tuple[float, float]) -> tuple[float, float]:
    m = re.search(r"(\d{1,2}):(\d{2})\s*-\s*(\d{1,2}):(\d{2})", value or "")
    if not m:
        return default
    o, c = int(m[1]) + int(m[2]) / 60, int(m[3]) + int(m[4]) / 60
    if c <= o:  # past-midnight closing, e.g. 18:00-02:00
        c = 24.0
    return round(o, 2), round(c, 2)


def _geocode(city: str) -> tuple[float, float, float, float]:
    r = requests.get(NOMINATIM, params={"q": city, "format": "json", "limit": 1}, headers=HEADERS, timeout=30)
    r.raise_for_status()
    hits = r.json()
    if not hits:
        raise ValueError(f"Could not find '{city}' on OpenStreetMap")
    s, n, w, e = map(float, hits[0]["boundingbox"])
    if (n - s) > 0.3 or (e - w) > 0.3:  # large area: use a window around the centre
        lat, lon = float(hits[0]["lat"]), float(hits[0]["lon"])
        s, n, w, e = lat - 0.12, lat + 0.12, lon - 0.12, lon + 0.12
    return s, w, n, e


def _post_query(query: str) -> list[dict]:
    """POST one Overpass query; try each mirror, retry on busy errors, never surface raw HTML."""
    last = "no response"
    for url in OVERPASS_URLS:
        for attempt in range(2):
            try:
                r = requests.post(url, data={"data": query}, headers=HEADERS, timeout=60)
                if r.status_code == 200:
                    return r.json().get("elements", [])
                last = f"{url.split('/')[2]} returned HTTP {r.status_code}"
                if r.status_code in (429, 502, 503, 504):  # busy: wait and retry
                    time.sleep(2 * (attempt + 1))
                    continue
                break  # 4xx/500: this mirror will not help, try the next
            except (requests.RequestException, ValueError) as exc:  # network error or non-JSON body
                last = f"{url.split('/')[2]}: {type(exc).__name__}"
                break
    raise RuntimeError(last)


def _overpass(bbox: tuple[float, float, float, float]) -> tuple[list[dict], int]:
    """Run the small queries one by one. Returns (elements, number_of_failed_queries)."""
    b = ",".join(f"{v:.5f}" for v in bbox)
    elements, failed, last = [], 0, ""
    for group in QUERY_GROUPS:
        q = "[out:json][timeout:30];(" + "".join(g.replace("{b}", b) for g in group) + ");out center tags 250;"
        try:
            elements += _post_query(q)
        except RuntimeError as exc:
            failed, last = failed + 1, str(exc)
        time.sleep(0.5)  # be polite to the free public servers
    if failed == len(QUERY_GROUPS):
        raise RuntimeError(f"OpenStreetMap servers are busy or unreachable ({last}). "
                           "Wait a minute and try again, or use the sample dataset.")
    return elements, failed


def elements_to_pois(city: str, elements: list[dict]) -> pd.DataFrame:
    rows, seen = [], set()
    for el in elements:
        tags = el.get("tags", {})
        name, cat = tags.get("name"), classify(tags)
        lat = el.get("lat") or el.get("center", {}).get("lat")
        lon = el.get("lon") or el.get("center", {}).get("lon")
        if not (name and cat and lat and lon) or name.lower() in seen:
            continue
        seen.add(name.lower())
        cost, dur, o, c = DEFAULTS[cat]
        o, c = parse_hours(tags.get("opening_hours"), (o, c))
        rows.append({"city": city, "name": name, "category": cat, "lat": float(lat), "lon": float(lon),
                     "cost": cost, "duration": dur, "rating": proxy_rating(tags), "open": o, "close": c,
                     "description": f"OpenStreetMap: {cat} place" + (f" ({tags['cuisine']})" if "cuisine" in tags else ""),
                     "_imp": importance(tags)})
    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError(f"No usable places found for '{city}'")
    df = (df.sort_values("_imp", ascending=False).groupby("category", group_keys=False)
            .head(PER_CATEGORY).drop(columns="_imp").sort_values(["category", "name"]).reset_index(drop=True))
    if len(df) < MIN_PLACES:
        raise ValueError(f"Only {len(df)} usable places found for '{city}'; try a bigger or better-mapped city")
    return df


def get_city_pois(city: str, refresh: bool = False) -> pd.DataFrame:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache = CACHE_DIR / (re.sub(r"[^a-z0-9]+", "_", city.lower()).strip("_") + ".csv")
    if cache.exists() and not refresh:
        return pd.read_csv(cache)
    elements, failed = _overpass(_geocode(city))
    df = elements_to_pois(city, elements)
    if failed == 0:  # do not cache a partial result
        df.to_csv(cache, index=False)
    return df
