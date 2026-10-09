import json
from dataclasses import replace

import pandas as pd
import pydeck as pdk
import streamlit as st

from travel_planner.data import list_cities
from travel_planner.llm import narrate_plan, parse_preferences
from travel_planner.osm import get_city_pois
from travel_planner.models import CATEGORIES, PACE_HOURS, STRATEGY_WEIGHT, Member, TripConfig
from travel_planner.planner import compare_strategies, plan_trip
from travel_planner.utility import disagreement, preference_table

st.set_page_config(page_title="AI Group Travel Planner", page_icon="🧭", layout="wide")

DEMO = [
    ("Aarav", dict(adventure=10, nightlife=9, nature=6, culture=2, food=6, shopping=2, relaxation=3), 8000, "packed"),
    ("Priya", dict(culture=10, shopping=9, food=7, nature=4, adventure=1, nightlife=2, relaxation=5), 6000, "balanced"),
    ("Rohan", dict(food=10, relaxation=9, nightlife=6, culture=4, nature=5, adventure=2, shopping=5), 7000, "relaxed"),
    ("Meera", dict(nature=10, relaxation=8, culture=6, food=5, adventure=4, nightlife=1, shopping=3), 7000, "balanced"),
]
COLORS = [[230, 57, 70], [29, 120, 200], [42, 157, 90], [244, 162, 40], [130, 80, 200]]


def init_state():
    if "uids" in st.session_state:
        return
    st.session_state.uids, st.session_state.next_uid = [], 0
    for name, prefs, budget, pace in DEMO:
        add_member(name, prefs, budget, pace)


def add_member(name=None, prefs=None, budget=8000, pace="balanced"):
    uid = st.session_state.next_uid
    st.session_state.next_uid += 1
    st.session_state.uids.append(uid)
    st.session_state[f"{uid}_name"] = name or f"Friend {uid + 1}"
    st.session_state[f"{uid}_budget"] = budget
    st.session_state[f"{uid}_pace"] = pace
    for c in CATEGORIES:
        st.session_state[f"{uid}_{c}"] = int((prefs or {}).get(c, 5))


def remove_member(uid):
    st.session_state.uids.remove(uid)


def apply_parse(uid):
    res = parse_preferences(st.session_state.get(f"{uid}_text", ""), st.session_state.get("api_key") or None)
    for c in CATEGORIES:
        st.session_state[f"{uid}_{c}"] = res["prefs"][c]
    st.session_state[f"{uid}_pace"] = res["pace"]
    if res["budget"]:
        st.session_state[f"{uid}_budget"] = res["budget"]
    st.session_state[f"{uid}_parsed_by"] = res["source"]


def collect_members():
    return [Member(st.session_state[f"{u}_name"], {c: st.session_state[f"{u}_{c}"] for c in CATEGORIES},
                   int(st.session_state[f"{u}_budget"]), st.session_state[f"{u}_pace"]) for u in st.session_state.uids]


init_state()

with st.sidebar:
    st.title("🧭 Group Travel Planner")
    st.caption("Fairness-aware itineraries for friends who want different things.")
    source = st.radio("Place data", ["Sample dataset (offline)", "Live OpenStreetMap (any city)"],
                      help="OpenStreetMap has no ratings or prices, so those are estimated proxies.")
    if source.startswith("Sample"):
        city = st.selectbox("Destination", list_cities())
    else:
        city = st.text_input("City or town", "Udaipur").strip()
        refresh = st.checkbox("Refresh from OpenStreetMap (ignore cache)")
    days = st.slider("Days", 1, 5, 3)
    strategy = st.radio("Fairness strategy", list(STRATEGY_WEIGHT), index=1,
                        help="utilitarian = maximise total happiness; egalitarian = protect the least-happy member")
    max_km = st.slider("Max distance between stops on one day (km)", 10, 80, 40)
    st.text_input("Anthropic API key (optional)", type="password", key="api_key",
                  help="Enables Claude for free-text preference parsing and the trip summary. Works offline without it.")
    go = st.button("Generate plan", type="primary", width="stretch")

members = collect_members()
if go:
    try:
        with st.spinner("Optimising for the whole group..."):
            pois = None if source.startswith("Sample") else get_city_pois(city, refresh)
            st.session_state.plan = plan_trip(TripConfig(city, days, strategy, float(max_km)), members, pois=pois)
            st.session_state.summary = None
            st.session_state.compare = None
    except (ValueError, RuntimeError) as e:
        st.error(str(e))
    except Exception as e:  # network or API failure
        st.error(f"Could not load place data: {e}")

tab_group, tab_plan, tab_fair, tab_export = st.tabs(["👥 Group", "🗺️ Itinerary", "⚖️ Fairness", "📤 Export"])

with tab_group:
    st.subheader("Who's travelling?")
    for uid in list(st.session_state.uids):
        with st.expander(st.session_state[f"{uid}_name"], expanded=False):
            c1, c2, c3, c4 = st.columns([2, 2, 2, 1])
            c1.text_input("Name", key=f"{uid}_name")
            c2.number_input("Activity budget (INR, whole trip)", 0, 200000, step=500, key=f"{uid}_budget")
            c3.selectbox("Pace", list(PACE_HOURS), key=f"{uid}_pace")
            c4.button("Remove", key=f"rm{uid}", on_click=remove_member, args=(uid,))
            cols = st.columns(4)
            for k, c in enumerate(CATEGORIES):
                cols[k % 4].slider(c.title(), 0, 10, key=f"{uid}_{c}")
            st.text_area("Or describe preferences in your own words", key=f"{uid}_text",
                         placeholder="e.g. I love food walks and heritage sites, hate clubs, budget ₹5000, relaxed pace")
            st.button("Parse description", key=f"parse{uid}", on_click=apply_parse, args=(uid,))
            if st.session_state.get(f"{uid}_parsed_by"):
                st.caption(f"Parsed with: {st.session_state[f'{uid}_parsed_by']}")
    st.button("➕ Add member", on_click=add_member)
    if len(members) > 1:
        st.markdown("**Where the group disagrees most** (std-dev of preferences)")
        st.bar_chart(disagreement(members))
        st.dataframe(preference_table(members), width="stretch")

plan = st.session_state.get("plan")
with tab_plan:
    if not plan:
        st.info("Set up the group, then press **Generate plan** in the sidebar.")
    else:
        m, b = plan["metrics"], plan["baseline"]
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Stops", m["stops"])
        k2.metric("Cost / person", f"₹{plan['cost_pp']:,}", f"cap ₹{plan['budget_cap']:,}", delta_color="off")
        k3.metric("Least-happy member", f"{m['min']:.0%}", f"{(m['min'] - b['min']) * 100:+.0f} pts vs popular-only")
        k4.metric("Average satisfaction", f"{m['avg']:.0%}", f"{(m['avg'] - b['avg']) * 100:+.0f} pts vs popular-only")
        st.caption(f"Solver: {plan['solver']}")
        for d, stops in enumerate(plan["days"], 1):
            st.subheader(f"Day {d}")
            if not stops:
                st.write("Free day.")
            for s in stops:
                if s["wait_min"] >= 30:
                    st.caption(f"…free time ({s['wait_min']} min) until it opens")
                cost = "free" if s["cost"] == 0 else f"₹{s['cost']}"
                st.markdown(f"**{s['start']}–{s['end']} · {s['name']}** · {s['category']} · {cost}  \n"
                            f"{s['description']}  \n*{s['why']}*" + (f" · {s['travel_min']} min travel" if s["travel_min"] else ""))
        rows, paths = [], []
        for d, stops in enumerate(plan["days"]):
            col = COLORS[d % len(COLORS)]
            for s in stops:
                rows.append({"name": s["name"], "day": d + 1, "lat": s["lat"], "lon": s["lon"], "color": col})
            if len(stops) > 1:
                paths.append({"path": [[s["lon"], s["lat"]] for s in stops], "color": col})
        if rows:
            df = pd.DataFrame(rows)
            st.pydeck_chart(pdk.Deck(
                layers=[pdk.Layer("PathLayer", paths, get_path="path", get_color="color", width_min_pixels=3),
                        pdk.Layer("ScatterplotLayer", df, get_position="[lon, lat]", get_fill_color="color",
                                  get_radius=250, radius_min_pixels=7)],
                initial_view_state=pdk.ViewState(latitude=df.lat.mean(), longitude=df.lon.mean(), zoom=9),
                tooltip={"text": "{name} (Day {day})"}))
        if st.button("✨ Write trip summary"):
            st.session_state.summary = narrate_plan(plan, st.session_state.get("api_key") or None)
        if st.session_state.get("summary"):
            st.info(st.session_state.summary)

with tab_fair:
    if not plan:
        st.info("Generate a plan to see fairness analytics.")
    else:
        names = [mm.name for mm in plan["members"]]
        chart = pd.DataFrame({"Optimised plan": plan["metrics"]["per_member"],
                              "Popular-only baseline": plan["baseline"]["per_member"]}, index=names)
        st.markdown("**Satisfaction per member** (1.0 = they got their personal top picks)")
        st.bar_chart(chart)
        st.caption("Baseline = highest-rated places within the same budget and time, ignoring everyone's taste.")
        if st.button("Compare all strategies"):
            with st.spinner("Solving three variants..."):
              st.session_state.compare = compare_strategies(plan["config"], plan["members"], pois=plan["pois"])
        if st.session_state.get("compare") is not None:
            st.dataframe(st.session_state.compare, width="stretch", hide_index=True)
            st.caption("Trade-off: stronger fairness weights usually cost a little average happiness "
                       "to lift the least-happy member and reduce inequality.")

with tab_export:
    if not plan:
        st.info("Generate a plan first.")
    else:
        md = [f"# {plan['config'].days}-day {plan['config'].city} trip\n"]
        for d, stops in enumerate(plan["days"], 1):
            md.append(f"## Day {d}")
            md += [f"- {s['start']}–{s['end']} **{s['name']}** ({s['category']}, ₹{s['cost']}) — {s['why']}" for s in stops]
            md.append("")
        payload = {"city": plan["config"].city, "strategy": plan["config"].strategy, "cost_per_person": plan["cost_pp"],
                   "metrics": plan["metrics"], "days": [[{k: v for k, v in s.items() if k != "idx"} for s in d] for d in plan["days"]]}
        st.download_button("Download itinerary (Markdown)", "\n".join(md), "itinerary.md")
        st.download_button("Download plan (JSON)", json.dumps(payload, indent=2), "plan.json")
        st.code("\n".join(md), language="markdown")
