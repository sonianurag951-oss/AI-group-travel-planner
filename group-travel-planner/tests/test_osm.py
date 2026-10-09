import pytest
from travel_planner import osm
from travel_planner.models import Member, TripConfig
from travel_planner.planner import plan_trip

CATS = [("tourism", "museum"), ("historic", "fort"), ("natural", "waterfall"), ("leisure", "park"),
        ("amenity", "restaurant"), ("amenity", "bar"), ("natural", "beach"), ("shop", "mall"),
        ("tourism", "theme_park"), ("amenity", "cafe")]


def fake_elements():
    els = []
    for i, (k, v) in enumerate(CATS * 2):
        tags = {k: v, "name": f"Place {i}"}
        if i % 3 == 0:
            tags["wikidata"] = "Q1"
        els.append({"type": "node", "id": i, "lat": 15.5 + i * 0.002, "lon": 73.8 + i * 0.002, "tags": tags})
    els.append({"type": "way", "id": 99, "center": {"lat": 15.6, "lon": 73.9},
                "tags": {"historic": "castle", "name": "Centre Castle", "opening_hours": "Mo-Su 08:30-17:00"}})
    els.append({"type": "node", "id": 100, "lat": 1, "lon": 1, "tags": {"highway": "bus_stop", "name": "Ignored"}})
    return els


def test_classify_and_hours():
    assert osm.classify({"amenity": "nightclub"}) == "nightlife"
    assert osm.classify({"amenity": "place_of_worship"}) is None
    assert osm.classify({"amenity": "place_of_worship", "wikidata": "Q1"}) == "culture"
    assert osm.parse_hours("Mo-Su 08:30-17:00", (9, 18)) == (8.5, 17.0)
    assert osm.parse_hours("closed on sundays", (9, 18)) == (9, 18)
    assert osm.parse_hours("18:00-02:00", (9, 18))[1] == 24.0


def test_elements_to_pois_schema_and_dedupe():
    df = osm.elements_to_pois("Testville", fake_elements() + fake_elements())
    assert {"city", "name", "category", "lat", "lon", "cost", "duration", "rating", "open", "close"} <= set(df.columns)
    assert df.name.is_unique and "Ignored" not in set(df.name)
    assert df[df.name == "Centre Castle"].open.iloc[0] == 8.5
    assert df.rating.between(3.6, 4.8).all()


def test_too_few_places_rejected():
    with pytest.raises(ValueError):
        osm.elements_to_pois("Empty", fake_elements()[:3])


def test_end_to_end_plan_on_osm_style_data(monkeypatch, tmp_path):
    monkeypatch.setattr(osm, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(osm, "_geocode", lambda c: (15.4, 73.7, 15.7, 74.0))
    monkeypatch.setattr(osm, "_overpass", lambda b: (fake_elements(), 0))
    df = osm.get_city_pois("Testville")
    assert (tmp_path / "testville.csv").exists()
    group = [Member("A", {"culture": 9, "food": 8}, 5000), Member("B", {"nature": 9, "nightlife": 7}, 5000)]
    plan = plan_trip(TripConfig("Testville", days=2), group, pois=df, time_limit_s=5)
    assert plan["metrics"]["stops"] > 0 and plan["cost_pp"] <= 5000


class FakeResp:
    def __init__(self, status, payload=None):
        self.status_code, self._p = status, payload

    def json(self):
        if self._p is None:
            raise ValueError("not json")
        return self._p


def test_falls_back_to_next_mirror_when_first_returns_500(monkeypatch):
    calls = []

    def fake_post(url, **kw):
        calls.append(url)
        return FakeResp(500) if len(calls) == 1 else FakeResp(200, {"elements": fake_elements()})

    monkeypatch.setattr(osm.requests, "post", fake_post)
    monkeypatch.setattr(osm.time, "sleep", lambda s: None)
    assert osm._post_query("q")
    assert calls[0] != calls[1]


def test_all_mirrors_down_gives_clean_message(monkeypatch):
    monkeypatch.setattr(osm.requests, "post", lambda url, **kw: FakeResp(500))
    monkeypatch.setattr(osm.time, "sleep", lambda s: None)
    with pytest.raises(RuntimeError) as err:
        osm._overpass((1, 2, 3, 4))
    assert "busy or unreachable" in str(err.value) and "<html" not in str(err.value).lower()
