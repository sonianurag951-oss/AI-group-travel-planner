"""Optional LLM layer (Claude API). Every function has an offline fallback, so the app always works."""
from __future__ import annotations
import json
import os
import re
import requests
from .models import CATEGORIES, PACE_HOURS

API_URL = "https://api.anthropic.com/v1/messages"
MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-5-5")

KEYWORDS = {
    "nature": ["nature", "hike", "hiking", "trek", "waterfall", "mountain", "scenic", "outdoor", "view"],
    "culture": ["culture", "history", "heritage", "museum", "fort", "temple", "architecture", "palace"],
    "food": ["food", "foodie", "eat", "eating", "restaurant", "street food", "cuisine", "cafe"],
    "adventure": ["adventure", "adrenaline", "rafting", "paragliding", "scuba", "thrill", "sports"],
    "nightlife": ["nightlife", "party", "club", "bar", "pub", "music", "late night"],
    "shopping": ["shopping", "market", "souvenir", "bazaar", "buy"],
    "relaxation": ["relax", "relaxing", "chill", "beach", "spa", "calm", "peaceful", "slow"],
}
NEGATIONS = ("no ", "not ", "avoid", "hate", "dislike", "don't", "dont", "skip", "boring")


def api_key_available(api_key: str | None = None) -> bool:
    return bool(api_key or os.getenv("ANTHROPIC_API_KEY"))


def _call(system: str, user: str, api_key: str | None = None, max_tokens: int = 900) -> str:
    key = api_key or os.getenv("ANTHROPIC_API_KEY")
    resp = requests.post(
        API_URL, timeout=30,
        headers={"x-api-key": key, "anthropic-version": "2023-06-01", "content-type": "application/json"},
        json={"model": MODEL, "max_tokens": max_tokens, "system": system,
              "messages": [{"role": "user", "content": user}]},
    )
    resp.raise_for_status()
    return "".join(b.get("text", "") for b in resp.json()["content"] if b.get("type") == "text")


def validate_profile(raw: dict) -> dict:
    """Clamp and sanitise LLM output; never trust it blindly."""
    prefs = {c: int(max(0, min(10, round(float(raw.get("prefs", {}).get(c, 5)))))) for c in CATEGORIES}
    pace = raw.get("pace") if raw.get("pace") in PACE_HOURS else "balanced"
    budget = raw.get("budget")
    budget = int(max(0, min(200000, float(budget)))) if isinstance(budget, (int, float)) else None
    return {"prefs": prefs, "pace": pace, "budget": budget}


def keyword_parse(text: str) -> dict:
    t = text.lower()
    prefs = {c: 5 for c in CATEGORIES}
    for cat, words in KEYWORDS.items():
        for w in words:
            for m in re.finditer(re.escape(w), t):
                before = t[max(0, m.start() - 20):m.start()]
                prefs[cat] = 1 if any(n in before for n in NEGATIONS) else min(10, max(prefs[cat], 5) + 3)
    pace = "relaxed" if re.search(r"relax|slow|lazy|easy", t) else "packed" if re.search(r"packed|busy|see everything|full day", t) else "balanced"
    budget = None
    m = re.search(r"(?:₹|rs\.?|inr)\s*([\d,]+)|([\d,]{4,})\s*(?:rs|inr|rupees)?", t)
    if m:
        budget = int((m.group(1) or m.group(2)).replace(",", ""))
    elif re.search(r"budget|cheap|low cost|tight", t):
        budget = 3000
    return validate_profile({"prefs": prefs, "pace": pace, "budget": budget})


def parse_preferences(text: str, api_key: str | None = None) -> dict:
    """Free text -> {prefs, pace, budget, source}. Uses Claude when a key exists, else keyword rules."""
    if api_key_available(api_key):
        system = ("Extract travel preferences. Reply with ONLY a JSON object: "
                  '{"prefs": {"nature":0-10,"culture":0-10,"food":0-10,"adventure":0-10,"nightlife":0-10,'
                  '"shopping":0-10,"relaxation":0-10}, "pace": "relaxed|balanced|packed", "budget": INR number or null}. '
                  "Use 5 for anything not mentioned.")
        try:
            out = _call(system, text, api_key, 300)
            data = json.loads(re.search(r"\{.*\}", out, re.S).group(0))
            return {**validate_profile(data), "source": "claude"}
        except Exception:
            pass
    return {**keyword_parse(text), "source": "keyword rules"}


def fallback_summary(plan: dict) -> str:
    cfg, m = plan["config"], plan["metrics"]
    lines = [f"A {cfg.days}-day {cfg.city} trip for {len(plan['members'])} friends, planned with the "
             f"'{cfg.strategy}' strategy: {m['stops']} stops, about INR {plan['cost_pp']:,} per person.",
             f"The least-happy member still gets {m['min']:.0%} of their best possible plan, "
             f"versus {plan['baseline']['min']:.0%} in a popularity-only plan."]
    for d, stops in enumerate(plan["days"], 1):
        if stops:
            lines.append(f"Day {d}: " + ", ".join(s["name"] for s in stops) + ".")
    return "\n\n".join(lines)


def narrate_plan(plan: dict, api_key: str | None = None) -> str:
    if not api_key_available(api_key):
        return fallback_summary(plan)
    facts = {"city": plan["config"].city, "members": [m.name for m in plan["members"]],
             "days": [[{k: s[k] for k in ("start", "name", "category", "cost", "why")} for s in d] for d in plan["days"]],
             "satisfaction": dict(zip([m.name for m in plan["members"]], [round(v, 2) for v in plan["metrics"]["per_member"]]))}
    system = ("You write friendly, concise trip summaries for a group. Use ONLY the places given; never invent venues "
              "or prices. Mention which compromises were made and who benefits on which day. Max 180 words.")
    try:
        return _call(system, json.dumps(facts), api_key, 500)
    except Exception:
        return fallback_summary(plan)
