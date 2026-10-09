import numpy as np
import pytest
from dataclasses import replace
from travel_planner.data import haversine_km, load_pois
from travel_planner.evaluation import gini, member_satisfaction, baseline_popular
from travel_planner.llm import keyword_parse, validate_profile
from travel_planner.models import Member, TripConfig
from travel_planner.optimizer import budget_cap, day_capacity_minutes, solve_selection, STOP_OVERHEAD_MIN
from travel_planner.planner import plan_trip
from travel_planner.utility import utility_matrix

GROUP = [
    Member("A", {"adventure": 10, "nightlife": 9, "culture": 2, "food": 5, "nature": 5, "shopping": 2, "relaxation": 3}, 8000, "packed"),
    Member("B", {"culture": 10, "shopping": 9, "food": 7, "nature": 4, "adventure": 1, "nightlife": 2, "relaxation": 5}, 5000, "balanced"),
    Member("C", {"food": 10, "relaxation": 9, "nightlife": 6, "culture": 4, "nature": 5, "adventure": 2, "shopping": 5}, 7000, "relaxed"),
]


def test_haversine_known_distance():
    assert haversine_km(28.6139, 77.2090, 19.0760, 72.8777) == pytest.approx(1153, abs=15)  # Delhi-Mumbai


@pytest.mark.parametrize("city", ["Goa", "Jaipur", "Manali"])
@pytest.mark.parametrize("strategy", ["utilitarian", "balanced", "egalitarian"])
def test_plan_respects_hard_constraints(city, strategy):
    cfg = TripConfig(city, days=3, strategy=strategy)
    plan = plan_trip(cfg, GROUP, time_limit_s=5)
    assert plan["cost_pp"] <= budget_cap(GROUP) == 5000
    pois = plan["pois"]
    names = [s["name"] for d in plan["days"] for s in d]
    assert len(names) == len(set(names)), "a place was visited twice"
    cap = day_capacity_minutes(GROUP)
    for stops in plan["days"]:
        assert sum(int(pois.duration[s["idx"]]) + STOP_OVERHEAD_MIN for s in stops) <= cap
        assert sum(s["category"] == "nightlife" for s in stops) <= 1
        for s in stops:  # opening hours honoured
            h = lambda t: int(t[:2]) + int(t[3:]) / 60
            assert h(s["start"]) >= pois.open[s["idx"]] - 1e-6
            assert h(s["end"]) <= pois.close[s["idx"]] + 1e-6


def test_fairness_weight_never_lowers_worst_off_member():
    cfg = TripConfig("Goa", days=3)
    pois = load_pois("Goa")
    U = utility_matrix(pois, GROUP)
    mins = []
    for s in ["utilitarian", "balanced", "egalitarian"]:
        sol = solve_selection(pois, U, GROUP, replace(cfg, strategy=s), time_limit_s=10)
        assert sol["status"] == "OPTIMAL"
        idx = list(sol["assignment"])
        mins.append(U[:, idx].sum(axis=1).min())
    assert mins[0] <= mins[1] + 1e-9 <= mins[2] + 2e-9


def test_satisfaction_bounds_and_gini():
    U = utility_matrix(load_pois("Goa"), GROUP)
    sat = member_satisfaction(U, [0, 1, 2])
    assert ((0 <= sat) & (sat <= 1)).all()
    assert gini([1, 1, 1]) == pytest.approx(0)
    assert gini([0, 0, 1]) > 0.5


def test_baseline_ignores_taste_but_respects_budget():
    pois = load_pois("Jaipur")
    picks = baseline_popular(pois, GROUP, 3, 6)
    assert sum(int(pois.cost[i]) for i in picks) <= budget_cap(GROUP)


def test_empty_group_rejected():
    with pytest.raises(ValueError):
        plan_trip(TripConfig("Goa"), [])


def test_llm_output_is_sanitised():
    out = validate_profile({"prefs": {"food": 99, "nature": -4, "bogus": 3}, "pace": "warp", "budget": 1e9})
    assert out["prefs"]["food"] == 10 and out["prefs"]["nature"] == 0 and "bogus" not in out["prefs"]
    assert out["pace"] == "balanced" and out["budget"] == 200000


def test_keyword_parser_handles_negation_and_budget():
    out = keyword_parse("Love trekking and waterfalls, no nightlife please. Budget ₹4,500, relaxed pace")
    assert out["prefs"]["nature"] >= 8 and out["prefs"]["nightlife"] == 1
    assert out["budget"] == 4500 and out["pace"] == "relaxed"
