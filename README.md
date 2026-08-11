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
| 0 | Scaffold, dependencies, configs | done - 10/10 tests pass |
| 1 | Data pipeline and calibration | done - 19/19 tests pass |
| 2 | Supply chain simulator | pending |
| 3 | Baseline policies | pending |
| 4 | RL agent | pending |
| 5 | Evaluation harness | pending |
| 6 | LLM explainer | pending |
| 7 | LLM scenario generator | pending |
| 8 | Dashboard | pending |
| 9 | LLM-curriculum training | pending |
| 10 | Report pack | pending |
