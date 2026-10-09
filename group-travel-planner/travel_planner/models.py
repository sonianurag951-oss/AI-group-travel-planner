from __future__ import annotations
from dataclasses import dataclass, field

CATEGORIES = ["nature", "culture", "food", "adventure", "nightlife", "shopping", "relaxation"]
PACE_HOURS = {"relaxed": 5.0, "balanced": 7.0, "packed": 9.0}
# fairness weight: objective = sum(member scores) + weight * n_members * min(member score)
STRATEGY_WEIGHT = {"utilitarian": 0.0, "balanced": 1.0, "egalitarian": 4.0}
CATEGORY_DAY_LIMIT = {"nightlife": 1}


@dataclass
class Member:
    name: str
    prefs: dict = field(default_factory=dict)  # category -> 0..10
    budget: int = 8000                         # INR per person for paid activities, whole trip
    pace: str = "balanced"

    def pref(self, category: str) -> float:
        return max(0.0, min(10.0, float(self.prefs.get(category, 5)))) / 10.0


@dataclass
class TripConfig:
    city: str
    days: int = 3
    strategy: str = "balanced"
    max_same_day_km: float = 40.0
    max_per_category_per_day: int = 2
