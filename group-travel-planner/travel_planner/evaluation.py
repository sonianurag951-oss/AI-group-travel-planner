from __future__ import annotations
import numpy as np
from .optimizer import STOP_OVERHEAD_MIN, budget_cap, day_capacity_minutes


def member_satisfaction(U: np.ndarray, selected: list[int]) -> np.ndarray:
    """Share of the best possible score for a plan of the same size (1.0 = got their top-k picks)."""
    k = len(selected)
    if k == 0:
        return np.zeros(U.shape[0])
    got = U[:, selected].sum(axis=1)
    best = np.sort(U, axis=1)[:, ::-1][:, :k].sum(axis=1)
    return got / np.maximum(best, 1e-9)


def gini(values) -> float:
    v = np.sort(np.asarray(values, dtype=float))
    if v.sum() == 0:
        return 0.0
    n = len(v)
    return float((2 * np.arange(1, n + 1) - n - 1).dot(v) / (n * v.sum()))


def summarize(U, selected) -> dict:
    sat = member_satisfaction(U, selected)
    return {"per_member": sat.tolist(), "avg": float(sat.mean()), "min": float(sat.min()),
            "gini": gini(sat), "stops": len(selected)}


def baseline_popular(pois, members, days, k: int) -> list[int]:
    """Preference-blind plan: highest-rated places within the same budget and time envelope."""
    budget, cap_total = budget_cap(members), day_capacity_minutes(members) * days
    picks, spent, used = [], 0, 0
    for i in pois.sort_values(["rating", "name"], ascending=[False, True]).index:
        c, t = int(pois.cost[i]), int(pois.duration[i]) + STOP_OVERHEAD_MIN
        if spent + c <= budget and used + t <= cap_total:
            picks.append(int(i)); spent += c; used += t
        if len(picks) == k:
            break
    return picks
