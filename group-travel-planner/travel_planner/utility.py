from __future__ import annotations
import numpy as np
import pandas as pd
from .models import CATEGORIES, Member


def utility_matrix(pois: pd.DataFrame, members: list[Member]) -> np.ndarray:
    """U[j, i] in [0, 1]: how much member j enjoys place i (taste x quality)."""
    U = np.zeros((len(members), len(pois)))
    for j, m in enumerate(members):
        for i, row in enumerate(pois.itertuples()):
            U[j, i] = m.pref(row.category) * (row.rating / 5.0)
    return U


def preference_table(members: list[Member]) -> pd.DataFrame:
    return pd.DataFrame({m.name: {c: m.prefs.get(c, 5) for c in CATEGORIES} for m in members}).T


def disagreement(members: list[Member]) -> pd.Series:
    """Std-dev of preferences per category: where the group is most divided."""
    return preference_table(members).std(ddof=0).sort_values(ascending=False)
