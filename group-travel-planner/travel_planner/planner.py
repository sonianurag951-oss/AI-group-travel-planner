from __future__ import annotations
from dataclasses import replace
import pandas as pd
from .data import CITY_SPEED_KMPH, load_pois
from .evaluation import baseline_popular, summarize
from .models import STRATEGY_WEIGHT, Member, TripConfig
from .optimizer import budget_cap, solve_selection
from .scheduler import schedule_day
from .utility import utility_matrix


def _why(cat: str, members: list[Member]) -> str:
    fans = [m.name for m in members if m.pref(cat) >= 0.7]
    lukewarm = [m.name for m in members if m.pref(cat) <= 0.3]
    parts = []
    if fans:
        parts.append("Loved by " + ", ".join(fans))
    if lukewarm:
        parts.append("compromise for " + ", ".join(lukewarm))
    return "; ".join(parts) or "Balanced pick for the whole group"


def plan_trip(config: TripConfig, members: list[Member], pois: pd.DataFrame | None = None,
              time_limit_s: float = 8.0) -> dict:
    if config.strategy not in STRATEGY_WEIGHT:
        raise ValueError(f"Unknown strategy {config.strategy}")
    if not members:
        raise ValueError("Add at least one group member")
    pois = load_pois(config.city) if pois is None else pois
    U = utility_matrix(pois, members)
    speed = CITY_SPEED_KMPH.get(config.city, 25)

    banned, day_plans, sol = set(), [], {}
    for _ in range(4):  # repair loop: re-solve if a stop cannot be fitted into opening hours
        sol = solve_selection(pois, U, members, config, frozenset(banned), time_limit_s)
        by_day: dict[int, list[int]] = {}
        for i, d in sol["assignment"].items():
            by_day.setdefault(d, []).append(i)
        day_plans, dropped = [], set()
        for d in range(config.days):
            stops, left = schedule_day(pois, by_day.get(d, []), speed)
            day_plans.append(stops)
            dropped.update(left)
        if not dropped:
            break
        banned |= dropped

    for stops in day_plans:
        for s in stops:
            s["why"] = _why(s["category"], members)
    selected = [s["idx"] for stops in day_plans for s in stops]
    base_sel = baseline_popular(pois, members, config.days, len(selected))
    return {
        "config": config, "members": members, "pois": pois, "U": U, "days": day_plans,
        "selected": selected, "metrics": summarize(U, selected),
        "baseline_selected": base_sel, "baseline": summarize(U, base_sel),
        "solver": f"{sol.get('method')} ({sol.get('status')})",
        "cost_pp": sum(s["cost"] for st in day_plans for s in st), "budget_cap": budget_cap(members),
    }


def compare_strategies(config: TripConfig, members: list[Member], pois: pd.DataFrame | None = None,
                       time_limit_s: float = 5.0) -> pd.DataFrame:
    rows = []
    for s in STRATEGY_WEIGHT:
        p = plan_trip(replace(config, strategy=s), members, pois=pois, time_limit_s=time_limit_s)
        m = p["metrics"]
        rows.append({"strategy": s, "stops": m["stops"], "avg satisfaction": round(m["avg"], 3),
                     "least happy member": round(m["min"], 3), "inequality (Gini)": round(m["gini"], 3),
                     "cost / person (INR)": p["cost_pp"]})
    b = p["baseline"]
    rows.append({"strategy": "popular-only baseline", "stops": b["stops"], "avg satisfaction": round(b["avg"], 3),
                 "least happy member": round(b["min"], 3), "inequality (Gini)": round(b["gini"], 3),
                 "cost / person (INR)": sum(int(p["pois"].cost[i]) for i in p["baseline_selected"])})
    return pd.DataFrame(rows)
