# SupplyAI-RL — Explainable Supply Chain Optimization

[![CI](https://github.com/KJSK-Koushik/Supply-AI-RL-Explainable-supply-chain-Optimisation/actions/workflows/ci.yml/badge.svg)](https://github.com/KJSK-Koushik/Supply-AI-RL-Explainable-supply-chain-Optimisation/actions/workflows/ci.yml)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)

A system that helps a shop decide **how much stock to order** and **which supplier to order from**. An AI agent (reinforcement learning) makes the decision. A language model (LLM) explains the decision in plain English and invents crisis situations to test it.

> **Money:** all amounts are shown in **rupees (₹)**. The sales data comes from a UK shop, so the simulator works in pounds inside. We convert at **1 GBP = ₹105** when showing numbers. The rate is set in `configs/display.yaml`. Change it there and everything follows.

---

## The problem in one paragraph

Every day, a shop must decide for each product: how much to order, and from whom. Order too little and you lose the sale. Order too much and your money sits in a warehouse. Pick the wrong supplier and the goods arrive late. Old-style rules like "when stock falls below 20, order 100" work until demand changes — and then they fail without warning. An AI can adapt, but nobody trusts a decision they cannot understand. This project builds the AI **and** the explanation.

---

## What we built

| Part | What it does |
|---|---|
| **Simulator** | A virtual shop. 10 real products, 3 suppliers, 180 days. Customers arrive, stock sells, orders arrive late or short. Tracks profit properly. |
| **RL agent** | An AI trained with MaskablePPO. Each day it picks a quantity and a supplier for every product. |
| **Classical policies** | The 4 textbook methods used in real industry, each tuned properly so the comparison is fair. |
| **LLM explainer** | Turns each decision into 2–3 plain sentences. Every number it writes is checked against the real facts. |
| **LLM crisis generator** | Invents realistic disruptions (demand spikes, supplier failures) to stress-test both the AI and the classical methods. |
| **Curriculum training** | Trains the AI on those crises, then tests it on crises it never saw. |
| **Dashboard** | A screen where you can run a day, see the numbers, and ask for an explanation. |

---

## Headline numbers

This is not a classification project, so there is no "accuracy" in the usual sense — there is no correct answer to be right about. The measure is **profit**. But the project does have real percentages, and here is each one with what it means:

| Number | Value | What it means |
|---|---|---|
| Relative profit | **90.8%** | The AI earns 90.8% of what the best classical method earns, on 30 unseen test runs |
| Fill rate | **95.1%** | Share of customer demand the AI served (best classical: 97.9%) |
| Explanation accuracy — delivered | **100%** | Explanations shown to the user that were correct and complete (30 real decisions) |
| Explanation accuracy — raw model | **83%** | The same test on the LLM's output *before* our safety check |
| Weekday pattern stability | *r* = 0.87 | The weekly demand pattern is the same across two separate years |

**The gap between 83% and 100% is the important finding.** The LLM made up a number in 5 of 30 explanations. Our check caught all 5. The user never saw them.

---

## Main result: the classical method still wins

Tested on 30 unseen scenarios. Same customers and same delivery delays for every method. Profit is per 180-day run.

| Method | Profit | Fill rate | Ordering fees |
|---|---|---|---|
| forecast + safety stock (best classical) | **₹87.8 lakh** | 97.9% | ₹2.9 lakh |
| newsvendor | ₹86.2 lakh | 98.0% | ₹3.4 lakh |
| **AI agent, tuned, 2M steps** | **₹79.7 lakh** | 95.1% | ₹5.8 lakh |
| AI agent, 2nd config, 2M steps | ₹77.0 lakh | 93.6% | ₹7.1 lakh |
| EOQ + reorder point | ₹75.9 lakh | 96.8% | ₹2.8 lakh |
| AI agent, untuned, 1M steps | ₹74.8 lakh | 94.3% | ₹7.4 lakh |
| (s,S) | ₹73.9 lakh | 94.9% | ₹3.1 lakh |
| PPO without action masking | ₹23.7 lakh | 94.8% | ₹14.7 lakh |

**The best classical method beats the AI by ₹8.1 lakh per run.** This is a real difference, not luck: a paired t-test gives *t* = −6.45 with 30 samples.

**We know exactly why the AI loses.** It spreads orders across all three suppliers and pays **twice** the ordering fees (₹5.8 lakh vs ₹2.9 lakh). In one traced run it placed 719 orders; the classical method placed 323.

**Why this negative result is still worth something.** We tuned the classical methods properly instead of leaving them weak. Tuning lifted the best one from ₹69.7 lakh to ₹89.2 lakh. If we had skipped that step, the AI would "win" — and the result would be worthless. Most student projects beat a weak baseline. This one did not, and says so.

### What each improvement was worth

| Step | Profit | Gap to classical |
|---|---|---|
| PPO, no masking | ₹23.7 lakh | −₹64.1 lakh |
| + action masking | ₹74.8 lakh | −₹13.0 lakh |
| + hyperparameter tuning, 2M steps | **₹79.7 lakh** | **−₹8.1 lakh** |

Action masking (blocking illegal choices before the AI decides) was worth about 3×. Tuning then closed 38% of what was left.

---

## Second result: the AI is fragile under crisis

We tested every method on 6 crisis scenarios the AI had never seen (generated by the LLM). Each scenario combines things like a viral demand spike, a supplier going offline, and a cost jump.

| Method | Calm | Crisis | Drop |
|---|---|---|---|
| forecast + safety stock | ₹87.4 lakh | ₹53.7 lakh | −38.6% |
| **AI, tuned (no crisis training)** | ₹80.4 lakh | **₹50.7 lakh** | **−37.0%** |
| AI, untuned (control) | ₹73.8 lakh | ₹38.8 lakh | −47.5% |
| AI, trained on crises | ₹66.0 lakh | ₹38.9 lakh | −41.1% |

**Training the AI on crises did not help.** The crisis-trained AI earns the same under crisis as the control (₹38.9 lakh vs ₹38.8 lakh) but gave up ₹7.7 lakh of normal profit to get there. Its smaller percentage drop is misleading — it drops less because it has less to lose. The paired test is not significant (*t* = +1.69).

The useful finding: **ordinary hyperparameter tuning made the AI more robust than crisis training did.** The tuned AI never saw a crisis in training, yet it handles crises best of all the AI versions.

---

## Phases

| Phase | Description | State |
|---|---|---|
| 0 | Setup, dependencies, configs | done |
| 1 | Data pipeline and calibration | done |
| 2 | Supply chain simulator | done |
| 3 | Classical policies, tuned | done |
| 4 | RL agent, swept and trained to 2M steps | done |
| 5 | Fair evaluation | done |
| 6 | LLM explainer, with safety check | done |
| 7 | LLM crisis generator | done |
| 8 | Dashboard | done — `streamlit run app/dashboard.py` |
| 9 | Curriculum training | done — did not improve robustness |
| 10 | Final report | done — [reports/final_report.md](reports/final_report.md) |

**133 tests pass.** CI runs on every push.

---

## Data

**UCI Online Retail II** — 1,067,371 real sales records from a UK gift shop, December 2009 to December 2011.

**The AI never sees this data directly.** We measure five things from it — average demand per product, how much it varies, the weekday pattern, the month pattern, and how often a day has zero sales — and build the simulator from those measurements.

**We picked 10 products on purpose** to cover three kinds of demand: steady, moderate, and bursty. If all ten behaved the same, one simple rule would handle everything and there would be nothing for an AI to learn.

**A second dataset was tested and rejected.** The Kaggle "Retail Store Inventory Forecasting" dataset is newer (2022–2024) but has no weekday pattern and no seasonal pattern — it is synthetic noise. Details in [reports/dataset_selection.md](reports/dataset_selection.md). *Newer but fake is worse than older but real.*

**Is the data too old?** We split the two years into two separate halves and measured the demand pattern in each. The weekday pattern matches strongly (*r* = 0.87). The busiest month is November in both years. The middle of the year differs. The simulator relies most on the weekly rhythm and the seasonal peak, and both held. Figure 08 shows this. What the data cannot tell us is anything about 2026 itself — that is a stated limitation.

Supplier details (lead times, reliability, fees) are **invented**, because no public dataset has them. The report says so.

---

## Explaining a decision

```bash
python scripts/demo_explainer.py --days 2
```

A real example:

```
--- day 44 (Wednesday, February) [nvidia/nemotron-3-ultra-550b-a55b:free] ---
Product 21212 (Pack Of 72 Retrospot Cake Cases) holds 2,438 units in stock with
nothing on order, providing 14.5 days of cover at the typical daily demand of
169 units. Recent demand has averaged 89 units over the last seven days, below
its usual level, and there was no unmet demand yesterday. No order was placed
today.
```

**The rule:** the LLM is never asked *why* a decision was made. The AI's real reason is a neural network — any "because" the LLM writes would be a made-up story. So the LLM only gets facts the simulator computed (stock, orders, demand) and puts them into sentences. Then we check every number in its output against those facts. If it invented one, we throw the output away and use a fixed template instead.

**We measured this on 30 real decisions.** The LLM invented a number 5 times. The check caught all 5. The user saw correct, complete explanations 100% of the time. The check is essential, not decoration.

Free LLM models get rate-limited often, so the code retries and falls back through three models. If there is no API key or every model refuses, the template takes over. The demo cannot break.

---

## Stress testing

```bash
python scripts/demo_scenarios.py
python scripts/demo_scenarios.py --offline    # no API key needed
```

The LLM was asked for "a realistic 2026 retail supply chain crisis". It wrote:

```
Demand 3x on products [0, 2, 5, 7], days 45-65     viral social media trend
Supplier 2 offline, days 50-63                     cyberattack on mid-tier supplier
Supplier 0 takes 6 extra days, days 55-72          port congestion at major hub
Costs 1.6x, days 60-84                             raw material shortage
```

These overlap — the supplier goes down *during* the demand spike. A person writing tests by hand rarely thinks of that combination.

**Nothing the LLM writes is trusted.** Every value is checked: the scenario type must be one we know, supplier and product numbers must exist, values are clamped to safe ranges, and scenarios are moved to fit inside the 180-day run. We fed it six deliberately broken inputs; three survived, none as written.

---

## Setup

The Python environment lives at **`D:\venvs\supplyai`**, outside the project folder. Windows limits paths to 260 characters, and the project path plus a deep package path breaks that limit.

```bash
python -m venv D:/venvs/supplyai
D:/venvs/supplyai/Scripts/python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
D:/venvs/supplyai/Scripts/python -m pip install -r requirements.txt
```

Check it works:

```bash
D:/venvs/supplyai/Scripts/python -m pytest tests/ -q
```

Machines without the dashboard packages (CI, Kaggle) should run `pytest -q -m "not local_env"`.

The LLM parts need an OpenRouter key in `.env`. Copy `.env.example` to `.env` and paste the key. Without a key, the explainer uses the template and everything still runs.

---

## Layout

```
configs/     all settings (data, simulator, suppliers, LLM, display currency)
src/data/    loading, cleaning, calibration, stability check
src/env/     the simulator
src/agents/  RL training, classical policies, curriculum
src/eval/    evaluation, comparison, robustness, explainer evaluation, figures
src/llm/     explainer and crisis generator
app/         Streamlit dashboard
tests/       133 tests
reports/     final report, dataset selection, presentation deck
results/     every result file and figure; all regenerate from these
```

---

## Regenerate everything

```bash
python -m src.eval.compare            # main comparison
python -m src.eval.robustness         # calm vs crisis
python -m src.eval.explainer_eval     # explanation accuracy
python -m src.data.stability          # is the pattern stable?
python -m src.eval.result_figures     # figures 05, 06
python -m src.eval.trace_figures      # figure 07
```
