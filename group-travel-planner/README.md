# AI-Powered Group Travel Planner

Friends rarely want the same trip. This app takes each member's tastes, budget and pace, then builds a day-by-day itinerary that is **optimised for the whole group, not just the average person**. Fairness is a tunable part of the objective, and every plan is measured against a naive baseline.

## What makes it more than a prompt wrapper
| Layer | What it does |
|---|---|
| **Preference model** | Each member scores 7 interest categories (0-10); utility = taste x place rating. |
| **Fairness-aware optimiser** | Google OR-Tools CP-SAT picks places and assigns them to days. Objective = total satisfaction + weight x worst-off member's satisfaction (max-min fairness). Strategies: utilitarian / balanced / egalitarian. |
| **Hard constraints** | Tightest member budget, daily time capacity from group pace, max distance between same-day stops, max 1 nightlife/day, max 2 of any category/day. |
| **Route scheduler** | Orders each day's stops by earliest feasible start, honouring opening hours and travel time (haversine x road factor). A repair loop re-solves if a stop cannot fit. |
| **Evaluation** | Per-member satisfaction, average, minimum, Gini inequality, compared with a popularity-only baseline under the same budget and time. |
| **LLM layer (optional)** | Claude parses free-text preferences into validated JSON and writes the trip summary using only the planned places. Offline keyword fallback means the app works without any API key. |

## Architecture
```
Member prefs --> utility matrix U[member, place]
                      |
 pois.csv --> CP-SAT selection + day assignment (budget, time, distance, category limits,
                      |                           max-min fairness term)
                      v
              per-day route scheduler (opening hours, travel time)  <-- repair loop
                      |
          metrics vs baseline  -->  Streamlit UI (itinerary, map, fairness, export)
```

## Run it
```bash
pip install -r requirements.txt
python -m pytest -q          # 20 tests
streamlit run app.py
```
Optional: paste an Anthropic API key in the sidebar (or set `ANTHROPIC_API_KEY`) for AI preference parsing and summaries. Override the model with `ANTHROPIC_MODEL`.

## Example result (4 friends, Goa, 3 days, demo group)
| Strategy | Avg satisfaction | Least-happy member | Gini |
|---|---|---|---|
| utilitarian | 0.829 | 0.690 | 0.057 |
| balanced | 0.803 | 0.719 | 0.055 |
| egalitarian | 0.794 | 0.743 | 0.028 |
| popular-only baseline | 0.776 | 0.655 | 0.058 |

Reading: pushing fairness costs about 3.5 points of average happiness and lifts the least-happy member by about 5 points while halving inequality. Results vary by city and group; Manali's egalitarian and utilitarian plans were identical for this group.

## Real data: any city via OpenStreetMap
In the sidebar choose **Live OpenStreetMap (any city)** and type a city. `travel_planner/osm.py` geocodes it (Nominatim), pulls attractions, food, nightlife, parks and markets (Overpass API), classifies them into the 7 categories, keeps the 10 most notable per category, and caches the result in `data/osm_cache/`. Free, no API key.

OpenStreetMap has no reviews or prices, so: **rating** is an importance proxy (Wikipedia/Wikidata/website/hours tags), **cost** is a typical INR default per category, and **hours** are parsed from `opening_hours` when simple, else category defaults. For production-grade ratings, swap the loader for the Google Places API (same DataFrame schema, one function to replace).

## Deploy on Streamlit Community Cloud
1. Push this folder to a public GitHub repo (CI in `.github/workflows/ci.yml` runs the tests).
2. Go to share.streamlit.io, click **Create app**, pick the repo, branch `main`, main file `app.py`.
3. (Optional) Add `ANTHROPIC_API_KEY` under **Advanced settings > Secrets** to enable AI parsing and summaries for everyone using your demo. Skip this if you do not want to pay for public usage; the app works without it.
4. Copy the live URL into the top of this README and add 2-3 screenshots in `docs/` (itinerary, map, fairness tab).

**Live demo:** _add your Streamlit URL here_

## Honest limitations
- Live OpenStreetMap mode was unit-tested with mocked responses; verify it against the real API on your machine.
- The sample dataset is a hand-curated sample of 42 places in 3 cities (Goa, Jaipur, Manali). Prices, hours and travel speeds are approximations; verify before real travel.
- Satisfaction is a model score from stated preferences, not measured happiness. A user study would be the next validation step.
- Travel time is straight-line distance x a road factor, not live routing.

## Extend it
1. Replace `data/pois.csv` with Google Places or OpenStreetMap Overpass data.
2. Real routing through OpenRouteService or Google Distance Matrix.
3. Live group voting (shared sessions) and re-planning from votes.
4. Per-member budgets with cost-splitting instead of the tightest-wallet cap.
5. Learn preference weights from past trip ratings.

## Resume bullets
- Built a group travel planner that models conflicting preferences and solves itinerary selection with OR-Tools CP-SAT under budget, time, distance and opening-hour constraints, using a max-min fairness objective.
- Cut inequality (Gini) by about 50% and raised the least-happy member's satisfaction by about 5 points against a popularity baseline, with a tunable fairness/efficiency trade-off.
- Integrated an LLM for free-text preference extraction with schema validation and offline fallback; shipped a Streamlit app with map, analytics, export and a 20-test pytest suite.

## Interview questions this project prepares you for
- Why max-min fairness instead of averaging? What is the cost of it (price of fairness)?
- Why CP-SAT / integer programming rather than greedy or an LLM-only plan? What is your greedy fallback's weakness?
- How do you stop the LLM from hallucinating places or breaking your schema?
- How would you validate that satisfaction scores reflect real people?
- How would you scale to 10,000 places or 20 people?
