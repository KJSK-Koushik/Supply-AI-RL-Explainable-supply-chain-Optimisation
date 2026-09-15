# SupplyAI-RL — Explainable Supply Chain Optimization

[![CI](https://github.com/KJSK-Koushik/Supply-AI-RL-Explainable-supply-chain-Optimisation/actions/workflows/ci.yml/badge.svg)](https://github.com/KJSK-Koushik/Supply-AI-RL-Explainable-supply-chain-Optimisation/actions/workflows/ci.yml)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)

A decision-support system in which a **reinforcement learning agent** makes
inventory replenishment and supplier-selection decisions, and a **large
language model** explains those decisions in plain business language and
generates disruption scenarios to stress-test them.

## The problem

A retailer must decide daily, for each product: *how much to order* and *from
which supplier*. Order too little and you lose sales; order too much and cash
sits in a warehouse; pick the wrong supplier and goods arrive too late either
way. Conventional systems use fixed rules such as "when stock falls below 20,
order 100" — which work until demand or supplier conditions change, and then
fail silently.

An RL agent adapts, but its reasoning is opaque to the managers who must act on
it. This project closes that gap.

## Approach

1. **Simulator** — a Gymnasium environment for 10 products x 3 suppliers,
   calibrated from real retail data, modelling demand, lead times, partial
   deliveries, supplier outages, backlog and cost.
2. **RL agent** — MaskablePPO learns reorder quantity and supplier choice.
3. **Baselines** — tuned (s,S), EOQ + reorder point, newsvendor and
   forecast + safety stock, evaluated on identical random seeds.
4. **LLM explainer** — receives *computed* decision facts and phrases them in
   business English. It is never asked why a decision was made, only to express
   facts already derived, and generated text is checked against those facts.
5. **LLM scenario generator** — emits schema-validated JSON disruption configs
   (demand spikes, supplier delays, outages) that the simulator executes.
6. **Curriculum loop** — LLM-generated scenarios are fed back as RL training
   data, and robustness is compared against an agent trained without them.
7. **Dashboard** — Streamlit app showing inventory, actions, cost, reward,
   explanations and the RL-vs-baseline comparison.

## Data

Calibrated from **UCI Online Retail II** (1,067,371 real transactions,
2009–2011). A second candidate dataset was screened and rejected for
containing no learnable structure — see
[reports/dataset_selection.md](reports/dataset_selection.md).

Supplier attributes are synthetic and declared as such; no public dataset
contains lead times or fill rates.

## Setup

The virtual environment lives at **`D:\venvs\supplyai`**, deliberately outside
the project folder. Windows on this machine has `LongPathsEnabled = 0`, so the
260-character path limit applies. This project's own path is 104 characters
before any package is added, and torch ships a license file nested ~150
characters deep, which overflows the limit and aborts the install. A short
venv path avoids it without needing an admin registry change.

```bash
python -m venv D:/venvs/supplyai
D:/venvs/supplyai/Scripts/python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
D:/venvs/supplyai/Scripts/python -m pip install -r requirements.txt
```

Verify the install:

```bash
D:/venvs/supplyai/Scripts/python -m pytest tests/test_phase0_env.py -q
```

Run the data pipeline:

```bash
D:/venvs/supplyai/Scripts/python -m src.data.run_pipeline
```

This writes `data/processed/daily_demand.csv`,
`data/processed/demand_stats.json` and four EDA figures to `results/figures/`.

An `OPENROUTER_API_KEY` in `.env` is required only from Phase 6 onward; copy
`.env.example` to `.env` when you reach it. The LLM modules fall back to a
deterministic template explainer when no key is present, so nothing breaks
without one.

## Layout

```
configs/     data · env · suppliers · llm  (all behaviour is config-driven)
src/data/    loading, cleaning, calibration, dataset audit, EDA
src/env/     Gymnasium simulator                     [Phase 2]
src/agents/  PPO training and baseline policies      [Phase 3-4]
src/eval/    fixed-seed evaluation harness           [Phase 5]
src/llm/     explainer, scenario generator, fallback [Phase 6-7]
app/         Streamlit dashboard                     [Phase 8]
tests/       per-phase checks
reports/     write-ups and figures for submission
```

## Status

| Phase | Description | State |
|---|---|---|
| 0 | Scaffold, dependencies, configs | done |
| 1 | Data pipeline and calibration | done |
| 2 | Supply chain simulator | done |
| 3 | Baseline policies, tuned | done |
| 4 | RL agent | done - swept, best config trained to 2M steps |
| 5 | Evaluation harness | done - see results below |
| 6 | LLM explainer | done - grounded, with template fallback |
| 7 | LLM scenario generator | done - validated and clamped |
| 8 | Dashboard | done - `streamlit run app/dashboard.py` |
| 9 | LLM-curriculum training | done - did not improve robustness |
| 10 | Report pack | done - [reports/final_report.md](reports/final_report.md) |

133 tests pass. Run them with:

```bash
D:/venvs/supplyai/Scripts/python -m pytest tests/ -q
```

Machines without the dashboard packages installed — CI, Kaggle — should
deselect the checks that need them: `pytest -q -m "not local_env"`.

## Headline numbers

There is no accuracy metric in an RL profit-optimisation problem -- there is no
correct answer to be right about -- so the honest percentages are these, each
with what it actually measures:

| Metric | Value | What it measures |
|---|---|---|
| Relative profit | **90.8%** | tuned RL agent's profit as a share of the best classical policy's, 30 held-out seeds |
| Fill rate | **95.1%** | share of customer demand the agent served (classical best: 97.9%) |
| Explanation accuracy, delivered | **100%** | explanations shown to the user that were grounded and complete (30 decisions) |
| Explanation accuracy, raw model | **83%** | the same test on the LLM's output before the grounding guard |
| Weekday pattern stability | *r* = 0.87 | demand shape agreement across two independent years |

The gap between the two explanation accuracies is the point: the model
invented a number in 5 of 30 outputs, the guard caught every one, and the
user saw none of them.

## Explaining a decision

Phase 6 turns one ordering decision into business English:

```bash
python scripts/demo_explainer.py --days 2
```

```
--- day 44 (Wednesday, February) [nvidia/nemotron-3-ultra-550b-a55b:free] ---
Product 21212 (Pack Of 72 Retrospot Cake Cases) holds 2,438 units in stock with
nothing on order, providing 14.5 days of cover at the typical daily demand of
169 units. Recent demand has averaged 89 units over the last seven days, below
its usual level, and there was no unmet demand yesterday. No order was placed
today.
```

The model is never asked *why* a decision was made. It receives facts the
simulator already computed and phrases them. The agent's real reason is a
policy network; any "because" a language model supplies is a plausible story,
not the cause. Every number in the output is checked against the supplied
facts, and output containing anything else is discarded in favour of a
deterministic template.

Measured on 30 real decisions (`python -m src.eval.explainer_eval`): the raw
model output was grounded 83% of the time -- five of thirty contained a number
the simulator never produced -- and the guard caught all five. Delivered
accuracy, grounded and complete, was 100%. The guard is load-bearing, not
decoration.

Free-tier models are rate-limited constantly, so the client retries with
backoff and falls through a list of models. With no API key, or when every
model refuses, the template takes over and says so -- the demo cannot break.

## Stress testing

Phase 7 asks an LLM for a crisis, refuses to trust any of it, and measures the
damage:

```bash
python scripts/demo_scenarios.py
python scripts/demo_scenarios.py --offline    # hand-written crises, no API needed
```

A generated example -- four disruptions that deliberately overlap:

```
Demand for products [0, 2, 5, 7] is 3.0x normal over days 45-65   viral social media trend
Supplier 2 is offline for days 50-63                              cyberattack on mid-tier supplier
Supplier 0 takes 6 extra days over days 55-72                     port congestion at major hub
Purchase costs are 1.60x over days 60-84                          raw material shortage
```

Nothing the model emits is executed. Every field is parsed, type checked,
range clamped and bounds checked against the real product and supplier counts;
anything that does not fit is dropped, and windows are repositioned to land
inside the episode. Given six hostile inputs -- an invented scenario type,
supplier 9 on a three-supplier world, a 999-day duration, a cost multiplier of
"lots" -- three survive, none as written.

**The RL agent is markedly less robust than the classical policy.** Under the
same disruption, across five seeds:

| Policy | Calm | Disrupted | Change |
|---|---|---|---|
| forecast + safety stock | 84,039 | 72,935 | -13% |
| MaskablePPO agent | 70,206 | 50,454 | **-28%** |

The agent degrades more than twice as badly on conditions it never trained on.
That is the gap Phase 9's curriculum is meant to close.

## Curriculum training (Phase 9): a negative result

Phase 7 showed the agent was brittle, so Phase 9 trained one on LLM-generated
crises -- a pool of disruption sets split in two, training on one half, scored
on the other. `curriculum_1m` matches `masked_1m` in hyperparameters, seed and
step count, so the curriculum is the only variable.

Scored on 10 calm seeds and 6 held-out disruption sets:

| Agent | Calm | Disrupted | Drop |
|---|---|---|---|
| forecast + safety stock | 83,282 | 51,175 | -38.6% |
| **kaggle_full01** (tuned, no curriculum) | **76,567** | **48,244** | **-37.0%** |
| masked_1m (control) | 70,279 | 36,925 | -47.5% |
| **curriculum_1m** | 62,904 | 37,071 | -41.1% |

**The curriculum did not work.** Its smaller percentage drop is the trap this
evaluation was built to expose: the drop is smaller because there is less to
lose. Absolute profit under disruption is 37,071 against the control's 36,925 --
a 0.4% difference, well inside noise -- bought by giving up 7,375 of calm
performance. The paired test on absolute drops gives t = +1.69, not significant.

The more useful finding is next to it: **plain hyperparameter tuning improved
robustness more than curriculum training did.** `kaggle_full01` never saw a
disruption during training and yet has the highest disrupted profit of any
agent and the smallest drop.

One caveat stated rather than buried: both agents got 1M steps, but the
curriculum agent faced a harder and more varied distribution, so part of the
shortfall may be undertraining rather than the method failing. Equal-steps is
the controlled comparison; equal-convergence would be a different experiment.

The drops here are larger than the -13%/-28% quoted under stress testing above
because these are the six held-out LLM sets, which are harsher than the three
hand-written ones.

## Current results

Scored on 30 held-out seeds that neither the agent nor the baselines were
tuned on. Profit is per 180-day episode, in GBP.

The two 2M-step rows come from the Kaggle sweep log rather than from
`results/comparison.json`, which still holds the pre-sweep run: the sweep's
output files have not been retrieved from Kaggle yet, so the figures below are
also from the earlier run. Every number here is quoted from that log verbatim.

| Policy | Profit | Fill rate | Ordering cost |
|---|---|---|---|
| forecast + safety stock (best classical) | 83,624 | 97.9% | 2,729 |
| newsvendor | 82,049 | 98.0% | 3,275 |
| **MaskablePPO agent, tuned, 2M steps** | **75,926** | **95.1%** | **5,522** |
| MaskablePPO agent, 2M steps, runner-up config | 73,331 | 93.6% | 6,774 |
| EOQ + reorder point | 72,302 | 96.8% | 2,626 |
| MaskablePPO agent, untuned, 1M steps | 71,227 | 94.3% | 7,068 |
| (s,S) | 70,365 | 94.9% | 2,905 |
| PPO without action masking, 1M steps | 22,566 | 94.8% | 13,988 |

**The tuned classical policy still beats the RL agent** by 7,698 per episode
(paired t = -6.45, n = 30, significant). The comparison is paired because both
policies meet identical customers on each seed.

A hyperparameter sweep of 8 configurations closed 38% of the gap, from -12,397
to -7,698, and moved the agent above two of the four classical policies. The
winning configuration used the *lowest* learning rate offered (5e-5) and the
larger network, confirming the diagnosis behind the search: the earlier runs
were oscillating because they learned too fast, not because they lacked
capacity.

The agents also carry zero overflow loss, where every classical policy wastes
some -- the agent learned the shared warehouse constraint properly. Its
remaining weakness is specific: it still fragments orders across suppliers,
paying twice the baseline's ordering fees.

This is reported as a result rather than buried. The comparison is fair by
construction — same environment, same seeds, same action granularity, and the
baselines were grid-search tuned rather than left at textbook defaults, which
lifted the best of them from 66,395 to 84,920 during tuning. Beating a weak
baseline would have proved nothing.

Where the remaining gap sits: stockouts (95.1% fill against 97.9%) and
fragmented sourcing (5,522 in ordering fees against 2,729).

Action masking is the largest single lever found so far, worth roughly 3x on
its own. A sweep of 8 configurations over 6M steps is running to test whether
the rest of the gap can be closed.

Regenerate the figures behind these numbers:

```bash
D:/venvs/supplyai/Scripts/python -m src.eval.result_figures
```
