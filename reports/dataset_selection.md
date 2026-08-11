# Dataset Selection and Screening

Two candidate datasets were evaluated as the calibration source for the supply
chain simulator. One was rejected. This document records the tests and the
result; `python -m src.data.audit_datasets` reproduces every number.

## Why calibration, not training

A reinforcement learning agent learns by taking an action and observing the
consequence. Historical retail data records only the actions that were actually
taken, so it cannot answer "what would have happened had we ordered 500 units
instead of 200?". A static table therefore cannot train an RL policy.

The dataset's role is to describe what realistic demand *looks like* — its
level, volatility, weekday rhythm, and seasonal shape. Those parameters
configure a simulator, which then generates unlimited consistent days for the
agent to practise on. The final portion of the real series is held out and
replayed unchanged as an evaluation episode, giving an honest test set.

This is the standard construction in the inventory-RL literature (OR-Gym,
RL4CO). Its validity depends entirely on the calibration source containing real
structure — which is what the screening below tests.

## Candidate A — Kaggle *Retail Store Inventory Forecasting* — REJECTED

73,100 rows; 5 stores x 20 products x 731 days (2022-01-01 to 2024-01-01); 15
columns including Price, Discount, Holiday/Promotion, Weather and Seasonality.
Schema-wise an excellent fit — it is one of the few public retail datasets
carrying inventory levels alongside sales.

Correlation of each stated demand driver with `Units Sold`:

| Driver | Correlation | Expected |
|---|---|---|
| Price | +0.0011 | strongly negative |
| Discount | +0.0026 | positive |
| Competitor Pricing | +0.0013 | positive |
| Holiday/Promotion | −0.0004 | positive |
| **Demand Forecast** | **+0.9969** | moderate |

Every genuine driver is indistinguishable from zero. Additional checks:

- **No weekday effect.** Daily means span 135.09–137.31 units across all seven
  days — a spread of 1.6%. Real retail weekday variation is tens of percent.
- **No seasonal effect.** Monthly means span 134.03–139.44 units, a spread of
  4.0%, with no coherent shape.
- **The `Seasonality` column is not tied to the calendar.** January rows are
  labelled Autumn (1,574), Spring (1,565), Summer (1,562) and Winter (1,599) —
  an almost perfectly uniform split. The label is randomly assigned.
- **No promotion lift.** Promotion days average 136.42 units against 136.51 on
  ordinary days; promotions marginally *reduce* sales.
- **Products are interchangeable.** All 20 product means fall within
  133.5–139.1 units. The dataset contains no fast movers and no slow movers.
- **Target leakage.** `Demand Forecast` correlates 0.997 with `Units Sold`; it
  is the target with noise added. Any model consuming it scores near-perfectly
  while learning nothing.

**Conclusion.** The file is uniformly distributed random values under
retail-sounding column names. Calibrating from it would produce ten identical
products, inert promotions and no seasonality — an environment in which a fixed
reorder rule is already optimal and the RL comparison could show no benefit.
Rejected as the calibration source; retained in `data/raw/` solely as the
evidence base for this section.

## Candidate B — UCI Online Retail II — ACCEPTED

1,067,371 transaction lines from a UK-based online gift retailer,
2009-12-01 to 2011-12-09; 8 columns; 604 trading days across 739 calendar days.

The same tests show genuine structure:

**Weekday pattern** (total units, 0 = Monday):

| Mon | Tue | Wed | Thu | Fri | Sat | Sun |
|---|---|---|---|---|---|---|
| 2.06M | 2.17M | 2.05M | 2.38M | 1.70M | **5,119** | 1.05M |

Saturday is effectively zero — the retailer does not despatch on Saturdays.
This is a real operational constraint visible in the data, and the pipeline
treats Saturdays as closed rather than as zero-demand days so that the fitted
means are not biased downwards.

**Seasonal pattern** — a Christmas ramp, November at 1.9x January:

| Jan | Sep | Oct | Nov | Dec |
|---|---|---|---|---|
| 0.78M | 1.16M | 1.25M | **1.48M** | 1.28M |

**Products genuinely differ:**

| StockCode | Description | Mean/day | CV | Zero-days | Price |
|---|---|---|---|---|---|
| 85099B | Jumbo Bag Red Retrospot | 133.1 | 1.35 | 20.3% | £1.95 |
| 85123A | White Hanging Heart T-Light Holder | 130.1 | 1.59 | 18.4% | £2.95 |
| 84077 | World War 2 Gliders Asstd | 149.0 | 2.92 | 36.4% | £0.29 |
| 17003 | Brocade Ring Purse | 96.7 | 5.63 | 59.3% | £0.29 |

Distinct volumes, volatilities, intermittency and price points. This variation
is what makes the RL comparison meaningful: a single reorder point tuned for a
steady mover performs poorly on a bursty one, and vice versa.

**Screening summary produced by the pipeline** (`data/processed/dataset_audit.json`):

| Metric | Kaggle | UCI |
|---|---|---|
| Weekday spread | 1.62% | **68.86%** |
| Monthly spread | 3.97% | **84.62%** |
| Cross-product spread | 4.14% | 110,090x |
| Season label consistency | 0.2575 (0.25 = random) | n/a |
| Target leakage | Demand Forecast @ 0.9969 | none |
| Verdict | REJECTED | ACCEPTED |

**Known limitations, handled explicitly:**

- Wholesale mega-orders dominate some series. StockCode 23843 has 99.9%
  zero-days with a CV of 27.2 — one bulk order is its entire history. Products
  whose largest single day exceeds 35% of total volume are excluded
  (`cleaning.max_single_day_share`).
- Cancellation invoices (prefix `C`), negative quantities, zero prices and
  service lines (`POST`, `DOT`, `M`, `BANK CHARGES`, …) are removed before
  aggregation; counts are logged to `data/processed/data_audit.json`.
- Remaining daily spikes are winsorised at the 99th percentile before fitting
  variance, so one outlier day does not inflate the simulated volatility.

## Pipeline result

`python -m src.data.run_pipeline` reduces 1,067,371 raw transaction lines to
1,036,960 genuine sales rows across 4,722 products and 604 trading days,
producing a 2,993,748-row daily panel. Product screening then discards:

| Rejected for | Count |
|---|---|
| Too low volume (< 15 units/day) | 3,135 |
| Dominated by a single wholesale order (> 35% of volume on one day) | 949 |
| Too intermittent (> 55% zero-days) | 24 |

leaving 181 eligible, shortlisted to the 60 highest-volume, from which 10 are
selected to span three demand regimes:

| StockCode | Description | Mean/day | CV | Zero-days | Regime |
|---|---|---|---|---|---|
| 21212 | Pack of 72 Retrospot Cake Cases | 168.6 | 1.13 | 7.9% | steady |
| 85123A | White Hanging Heart T-Light Holder | 160.9 | 1.02 | 6.6% | steady |
| 85099B | Jumbo Bag Red Retrospot | 145.0 | 1.11 | 9.6% | steady |
| 84991 | 60 Teatime Fairy Cake Cases | 92.6 | 1.20 | 9.8% | steady |
| 84879 | Assorted Colour Bird Ornament | 120.0 | 1.29 | 10.7% | moderate |
| 21977 | Pack of 60 Pink Paisley Cake Cases | 90.7 | 1.44 | 11.7% | moderate |
| 21213 | Pack of 72 Skull Cake Cases | 65.0 | 1.65 | 17.3% | moderate |
| 84077 | World War 2 Gliders Asstd | 163.9 | 2.13 | 29.2% | bursty |
| 22197 | Small Popcorn Holder | 100.2 | 2.09 | 14.3% | bursty |
| 15036 | Assorted Colours Silk Fan | 68.3 | 2.34 | 34.5% | bursty |

Calibrated seasonal factors fed to the simulator (1.0 = an average day):

- **Day of week:** Mon 0.93, Tue 1.22, Wed 1.18, Thu 1.23, Fri 0.93, Sun 0.51
  (Saturday excluded as a non-trading day)
- **Month:** Jan 0.69 rising to Nov 1.75 — a 2.5x seasonal swing

Regime labels are derived from the same winsorised CV reported in
`demand_stats.json`, so a label can never contradict the figure beside it.

## What is real and what is synthetic

| Parameter | Source |
|---|---|
| Mean and variance of daily demand, per product | **Real** — UCI |
| Day-of-week demand factors | **Real** — UCI |
| Monthly / seasonal factors | **Real** — UCI |
| Demand burstiness (overdispersion) | **Real** — UCI |
| Unit selling price | **Real** — UCI median transaction price |
| Unit cost | Derived — price x (1 − assumed 45% gross margin) |
| Supplier lead times, fill rates, MOQ, outages | **Synthetic** — `configs/suppliers.yaml` |
| Holding cost, stockout penalty, ordering cost | **Synthetic** — standard literature ranges |

No public retail dataset contains supplier lead times or fill rates; that
information is commercially confidential. Supplier attributes are therefore
hand-authored to span the classic cost / speed / reliability trade-off and are
declared synthetic wherever they appear.

## Citation

Chen, D. (2019). *Online Retail II* [Dataset]. UCI Machine Learning Repository.
https://doi.org/10.24432/C5CG6D
