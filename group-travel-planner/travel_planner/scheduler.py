"""Turn each day's chosen places into a timed route that respects opening hours."""
from __future__ import annotations
from .data import haversine_km, travel_minutes


def fmt_time(h: float) -> str:
    hh, mm = int(h), int(round((h - int(h)) * 60))
    if mm == 60:
        hh, mm = hh + 1, 0
    return f"{hh:02d}:{mm:02d}"


def schedule_day(pois, idxs, speed_kmph, start_h=9.0, end_h=23.5):
    remaining, t, cur, stops = list(idxs), start_h, None, []
    while remaining:
        best = None
        for i in remaining:
            r = pois.iloc[i]
            travel = 0.0 if cur is None else travel_minutes(haversine_km(cur[0], cur[1], r.lat, r.lon), speed_kmph)
            arrive = t + travel / 60
            start = max(arrive, float(r.open))
            end = start + r.duration / 60
            if end > min(float(r.close), end_h) + 1e-9:
                continue
            key = (start, travel)
            if best is None or key < best[0]:
                best = (key, i, travel, arrive, start, end)
        if best is None:
            return stops, remaining  # whatever is left cannot be fitted
        _, i, travel, arrive, start, end = best
        r = pois.iloc[i]
        stops.append({"idx": i, "name": r["name"], "category": r.category, "lat": float(r.lat), "lon": float(r.lon),
                      "cost": int(r.cost), "travel_min": round(travel), "wait_min": round((start - arrive) * 60),
                      "start": fmt_time(start), "end": fmt_time(end), "description": r.get("description", "")})
        t, cur = end, (r.lat, r.lon)
        remaining.remove(i)
    return stops, []
