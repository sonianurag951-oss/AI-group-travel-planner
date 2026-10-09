"""Fairness-aware place selection and day assignment (CP-SAT, greedy fallback)."""
from __future__ import annotations
import numpy as np
import pandas as pd
from .data import haversine_km
from .models import CATEGORY_DAY_LIMIT, PACE_HOURS, STRATEGY_WEIGHT, Member, TripConfig

SCALE = 1000
STOP_OVERHEAD_MIN = 20  # rough travel/buffer allowance per stop


def day_capacity_minutes(members: list[Member]) -> int:
    return int(round(np.mean([PACE_HOURS[m.pace] for m in members]) * 60))


def budget_cap(members: list[Member]) -> int:
    return int(min(m.budget for m in members))  # plan must fit the tightest wallet


def _far_pairs(pois: pd.DataFrame, allowed: list[int], max_km: float):
    lat, lon = pois.lat.values, pois.lon.values
    return [(i, j) for a, i in enumerate(allowed) for j in allowed[a + 1:]
            if haversine_km(lat[i], lon[i], lat[j], lon[j]) > max_km]


def _cat_limit(cat: str, config: TripConfig) -> int:
    return CATEGORY_DAY_LIMIT.get(cat, config.max_per_category_per_day)


def solve_selection(pois, U, members, config, banned=frozenset(), time_limit_s=8.0) -> dict:
    n, days = len(pois), config.days
    cap, budget = day_capacity_minutes(members), budget_cap(members)
    lam = int(round(STRATEGY_WEIGHT[config.strategy] * len(members)))
    cost, dur, cats = pois.cost.values, pois.duration.values, pois.category.values
    allowed = [i for i in range(n) if i not in banned and cost[i] <= budget
               and dur[i] + STOP_OVERHEAD_MIN <= cap]
    far = _far_pairs(pois, allowed, config.max_same_day_km)
    try:
        from ortools.sat.python import cp_model
    except ImportError:
        return _greedy(pois, U, allowed, far, cap, budget, lam, config)

    Ui = np.rint(U * SCALE).astype(int)
    model = cp_model.CpModel()
    x = {(i, d): model.NewBoolVar(f"x{i}_{d}") for i in allowed for d in range(days)}
    for i in allowed:
        model.Add(sum(x[i, d] for d in range(days)) <= 1)
    for d in range(days):
        model.Add(sum(int(dur[i] + STOP_OVERHEAD_MIN) * x[i, d] for i in allowed) <= cap)
        for c in {cats[i] for i in allowed}:
            in_c = [i for i in allowed if cats[i] == c]
            model.Add(sum(x[i, d] for i in in_c) <= _cat_limit(c, config))
        for i, j in far:
            model.Add(x[i, d] + x[j, d] <= 1)
    model.Add(sum(int(cost[i]) * x[i, d] for i in allowed for d in range(days)) <= budget)

    scores = [sum(int(Ui[j, i]) * x[i, d] for i in allowed for d in range(days)) for j in range(len(members))]
    m = model.NewIntVar(0, int(Ui.sum()) + 1, "min_score")
    for s in scores:
        model.Add(m <= s)
    model.Maximize(sum(scores) + lam * m)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit_s
    solver.parameters.num_workers = 1
    solver.parameters.random_seed = 7
    status = solver.Solve(model)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return {"assignment": {}, "method": "cp-sat", "status": solver.StatusName(status)}
    assign = {i: d for (i, d), v in x.items() if solver.Value(v)}
    return {"assignment": assign, "method": "cp-sat", "status": solver.StatusName(status)}


def _greedy(pois, U, allowed, far, cap, budget, lam, config) -> dict:
    cost, dur, cats = pois.cost.values, pois.duration.values, pois.category.values
    far_set = {frozenset(p) for p in far}
    load, spent, chosen = [0] * config.days, 0, {}
    S = np.zeros(U.shape[0])

    def day_ok(i, d):
        if load[d] + dur[i] + STOP_OVERHEAD_MIN > cap:
            return False
        same = [k for k, v in chosen.items() if v == d]
        if sum(cats[k] == cats[i] for k in same) >= _cat_limit(cats[i], config):
            return False
        return not any(frozenset((i, k)) in far_set for k in same)

    while True:
        best = None
        for i in allowed:
            if i in chosen or spent + cost[i] > budget:
                continue
            d = next((d for d in range(config.days) if day_ok(i, d)), None)
            if d is None:
                continue
            new = S + U[:, i]
            gain = new.sum() + lam * new.min() - (S.sum() + lam * S.min())
            if best is None or gain > best[0]:
                best = (gain, i, d)
        if best is None or best[0] <= 0:
            break
        _, i, d = best
        chosen[i], load[d], spent, S = d, load[d] + dur[i] + STOP_OVERHEAD_MIN, spent + cost[i], S + U[:, i]
    return {"assignment": chosen, "method": "greedy", "status": "HEURISTIC"}
