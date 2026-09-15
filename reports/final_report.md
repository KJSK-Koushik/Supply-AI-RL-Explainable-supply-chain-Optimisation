# SupplyAI-RL: Explainable Supply Chain Optimization

**An LLM–RL decision-support system for retail inventory**
Course: 22AIE450 Reinforcement Learning — Final Report

---

## Abstract

A retailer must decide daily, for every product, how much to order and from which supplier. This project builds a complete decision-support system for that problem: a supply chain simulator calibrated from 1,067,371 real retail transactions, a reinforcement learning agent (MaskablePPO) that chooses order quantity and supplier jointly, four properly tuned classical inventory policies as competitors, a language-model layer that explains every decision in business English and generates disruption scenarios for stress testing, and a dashboard framed as a shadow-mode deployment.

The headline finding is honest rather than triumphant. On 30 held-out scenarios the tuned RL agent earns £75,926 per 180-day episode — **90.8% of the best classical policy's £83,624** — and beats two of the four classical methods, but the gap of £7,698 is statistically significant (paired *t* = −6.45). We know precisely why: the agent fragments orders across suppliers, paying twice the ordering fees. Action masking was worth roughly 3×; hyperparameter tuning closed 38% of the remaining gap; training on generated crises did not improve robustness, and the sweep-tuned agent turned out to be the most robust one anyway. The explainer produces grounded, complete explanations — and the evaluation shows the grounding check is not decoration: without it, a material fraction of explanations would contain a fabricated number.

Every comparison is fair by construction — same environment, same seeds, same action set, baselines tuned rather than left at textbook defaults — which is what makes the negative results meaningful.

---

## 1. Introduction and motivation

Inventory decisions carry two asymmetric errors. Order too much and capital sits in a warehouse, storage is paid for, and seasonal stock is marked down or written off. Order too little and the sale is lost permanently, because the customer buys elsewhere. Add supplier choice — cheap but slow, fast but expensive, reliable-looking but flaky — and the daily decision becomes a genuine sequential problem: today's order is next week's stock, and every product shares one warehouse.

Classical policies ((s,S), EOQ with reorder point, newsvendor, forecast plus safety stock) treat each product in isolation and assume demand is well behaved. Real demand is not: in the data used here, daily demand variance is roughly 216 times the mean, where a Poisson model assumes they are equal.

Reinforcement learning can in principle learn what a formula cannot express — a joint ordering and sourcing policy that responds to season, volatility and supplier state. But a buyer will not act on a recommendation they cannot interpret. The project therefore pairs an RL decision-maker with an LLM that phrases each decision in business language, under a strict rule: the model states facts the simulator computed and never speculates about causes.

### Objectives

1. Build a supply chain simulation calibrated from real data.
2. Train an RL agent to choose reorder quantity and supplier.
3. Use an LLM to explain each decision in business language.
4. Use an LLM to generate demand-spike and supplier-delay scenarios.
5. Compare the RL agent against traditional inventory policies.

All five are delivered. Objective 5 says *compare*, not *win*; the comparison came back negative and is reported as such.

---

## 2. Problem definition

| | |
|---|---|
| Products | 10 real SKUs from the UCI Online Retail II dataset |
| Suppliers | 3, synthetic, spanning the cost/speed/reliability trade-off |
| Horizon | 180 simulated days per episode, 14-day unscored warm-up |
| Warehouse | one shared 20,000-unit store; overflow is refused at the door and written off |
| Decision | each day, per product: one of 7 quantity buckets × 3 suppliers = 21 options |
| Objective | maximise profit = revenue − purchase − holding − ordering fees − lost-sale penalty − overflow loss |

The three suppliers, from `configs/suppliers.yaml`:

| Supplier | Unit cost | Lead time | Ships | Min order | Order fee | Outage risk |
|---|---|---|---|---|---|---|
| EconoSource | baseline | 5–8 days | 90% | 100 | £20 | low |
| RapidTrade | +35% | 1–3 days | 98% | 20 | £45 | very low |
| MidWay Supply | +15% | 3–5 days | 80% | 50 | £30 | highest |

No column dominates. The cheap supplier ties up a week of demand in transit; the fast one erodes margin; the middle option ships only 80% of what is ordered and fails most often. The right answer changes with stock cover, season and volatility — which is exactly what a fixed reorder rule cannot express.

The action space is 21¹⁰ ≈ 1.7 × 10¹³ combinations per day, coupled in time and across products. That is the textbook definition of a sequential decision problem under uncertainty.

---

## 3. Data

### 3.1 Source and preparation

**UCI Online Retail II** — 1,067,371 transaction lines from a UK online gift retailer, December 2009 to December 2011. Cleaning removes cancellations, returns, negative quantities and missing product codes. Transactions are aggregated to a daily panel per product. Saturdays are dropped entirely rather than zero-filled, because the retailer never traded on Saturdays and zero-filling would teach a false weekly collapse.

Ten SKUs were selected deliberately to span three demand regimes, so that no single fixed rule can serve all of them:

| Regime | Products | Coefficient of variation |
|---|---|---|
| steady | 21212, 85123A, 85099B, 84991 | 1.02 – 1.20 |
| moderate | 84879, 21977, 21213 | 1.29 – 1.65 |
| bursty | 84077, 22197, 15036 | 2.09 – 2.34 |

Calibration uses only data before **1 June 2011**; later data is held back.

### 3.2 A second dataset was screened and rejected

The Kaggle "Retail Store Inventory Forecasting" dataset (2022–2024) was evaluated first. It showed no weekday effect and no seasonality — flat noise with no learnable structure. It was rejected with the evidence recorded in `reports/dataset_selection.md`. *Recent but synthetic is worse than old but real.*

### 3.3 How the data is used

The agent never sees the transaction data. Five statistical properties are extracted — mean and variance of daily demand per product, weekday factors, month factors, and the zero-day fraction — and the simulator reproduces them. Demand is generated by a Gamma-Poisson (negative binomial) process, which produces the quiet weeks and sudden spikes a plain Poisson cannot.

### 3.4 Is the demand shape stable over time?

Faculty raised the age of the data. The simulator does not use the data's levels (prices and volumes are configuration); it uses its *shape*. To test whether that shape is a property of the period or of retail, the two years were split into disjoint windows and the same factors fitted independently on each (`src/data/stability.py`, figure 08):

| Property | Year 1 vs Year 2 |
|---|---|
| Weekday factor | *r* = 0.87 — Tue–Thu busy, Sunday quiet, both years |
| Month factor | *r* = 0.51 (0.64 excluding December, which is partly a window artefact) |
| Peak month | November in both years |
| Variance / mean | 224 and 474 — extreme burstiness in both |

The weekly rhythm and the seasonal peak repeat; the mid-year shape does not. This is reported as found. It supports the two features the simulator leans on hardest and cannot support any claim about 2026 itself — which remains a stated limitation (§10).

---

## 4. The simulator

`src/env/supply_chain_env.py` is a Gymnasium environment. Each simulated day runs five steps in a fixed order:

1. **Deliveries arrive.** Stock ordered days ago lands; if the shared warehouse is full, the excess is refused and written off (already paid for).
2. **Customers arrive.** Demand is drawn per product with weekday and month factors applied.
3. **Sell what is held.** Shortfall becomes a lost sale with a penalty, and carries as backlog.
4. **Place today's orders.** Ordering comes *after* demand, because a real buyer places tomorrow's order once today's sales are known.
5. **Count the money.**

Suppliers have triangular lead times, normally distributed fill rates, minimum order quantities, fixed per-order fees and random outages. Disruption scenarios (demand spikes, supplier delays, outages, price shocks) can be injected as validated data and never as executable content.

Economic choices, each of which corrected an earlier error:

- **Holding cost** 0.0015 per unit-day (~55%/year) — higher than the usual 20–30% cost-of-capital figure, deliberately, because this retailer sells seasonal gift lines that carry markdown risk. An earlier setting was 730%/year, which made stock-holding absurdly punitive.
- **Terminal salvage** of 60% on leftover stock — the earlier total write-off was an artefact of the 180-day horizon.
- **One shared warehouse** rather than per-product limits — per-product capacity had made "order the maximum" trivially optimal, leaving no interior optimum to learn.

A demo (`scripts/demo_simulator.py`) confirms the reward function is sane: crude but sensible policies earn ~£26k; never ordering loses £368k; ordering the maximum achieves the best fill rate of any policy and still loses £69k. If reasonable had not beaten flailing, every later result would be meaningless.

---

## 5. Methods

### 5.1 The RL formulation

| | |
|---|---|
| **State** (88 values) | per product: stock, in transit, days of cover, 7-day mean, 7-day volatility, trend, backlog; weekday one-hot; month as sin/cos; per supplier: available, expected lead time, recent fill rate |
| **Action** | MultiDiscrete over 10 products, each a joint choice among 21 (quantity bucket × supplier) |
| **Reward** | that day's profit, scaled by 0.001 and clipped to ±10; zero during warm-up |
| **Episode** | 180 days |

The action is encoded as one *joint* choice per product rather than two separate ones. That is both 2.8× faster (PPO builds one categorical distribution per action dimension) and the better model: how much to order and who to order from are not independent decisions.

### 5.2 Algorithm

**MaskablePPO** (sb3-contrib) — PPO with invalid-action masking. Suppliers in outage and orders exceeding free warehouse space are blocked before the policy chooses. MLP policy, two hidden layers, CPU training.

Masking was the single largest lever found: best training-eval profit rose from 23,717 (plain PPO) to 72,740 (masked), roughly 3×. Without it, the agent spends most of its training discovering that certain actions are pointless.

### 5.3 Classical baselines, tuned

Four textbook policies, each grid-searched on 12 tuning seeds (100–111) disjoint from every evaluation seed:

| Policy | Tuned parameters |
|---|---|
| (s,S) | s = 11 days, S = 16 days, supplier by urgency |
| EOQ + reorder point | ROP = 8 days, safety factor 1.6, supplier by urgency |
| newsvendor | 3-day horizon, cheapest supplier |
| forecast + safety stock | 1-day window, z = 1.28, review every 4 days, cheapest supplier |

Tuning mattered: widening the grid lifted the best policy from 66,395 to 84,920 during tuning. Boundary optima were detected automatically and the grid extended until the optimum was interior. Beating an untuned baseline would have proved nothing.

Every baseline chooses from the same 21 options per product as the agent (`Policy.quantise` snaps to the same buckets), so no policy has a finer control granularity than another.

### 5.4 Evaluation protocol

Three disjoint seed families:

| Seeds | Purpose |
|---|---|
| 100–111 | tuning the classical baselines |
| 900–905 | selecting RL checkpoints during training |
| **500–529** | **reporting — never used for any selection** |

Every reported number comes from the 30 reporting seeds. Every policy faces identical customers and identical delivery delays on each seed, so comparisons are **paired**: the test statistic is computed on per-seed differences, which removes the episode-to-episode variance that dominates the raw standard deviations.

### 5.5 Hyperparameter sweep

Eight configurations, random search over learning rate {5e-5, 1e-4, 3e-4}, entropy {0.001–0.05}, γ {0.995, 0.999}, rollout length {512, 1024} and network {128×128, 256×256}, with successive halving: 250k-step screening, then the two survivors trained to 2M steps on Kaggle (~4.8 hours).

The winner used the **lowest** learning rate offered (5e-5) and the **larger** network. This confirms the diagnosis behind the search: the first two runs were oscillating because they learned too fast, not because they lacked capacity.

---

## 6. The LLM explainer

`src/llm/explainer.py`. The design rule: **the model is never asked why a decision was made.** The agent's real reason is a policy network; any "because" a language model supplies is a plausible story, not the cause, and presenting it as the cause would make the explainability claim false. So the model does presentation, and the numbers stay the simulator's.

The explainer receives a fact dictionary — stock, in-transit, days of cover, recent demand, order placed, supplier, lead time, fill rate — and returns two or three sentences. A post-check extracts every number in the output and verifies it traces to a supplied fact (tolerating thousands separators and rounding). Output containing anything else is discarded and a deterministic template is used instead.

Models are OpenRouter free-tier: `google/gemma-4-31b-it`, `z-ai/glm-5.2`, `nvidia/nemotron-3-ultra-550b-a55b`, tried in that order with retry and backoff, because free endpoints rate-limit constantly. With no key, or when every model refuses, the template takes over and says so.

A real example (nemotron, day 44):

> Product 21212 (Pack Of 72 Retrospot Cake Cases) holds 2,438 units in stock with nothing on order, providing 14.5 days of cover at the typical daily demand of 169 units. Recent demand has averaged 89 units over the last seven days, below its usual level, and there was no unmet demand yesterday. No order was placed today.

### 6.1 Evaluation

`src/eval/explainer_eval.py` scores 30 real decisions from the tuned baseline, spread across six days and all ten products:

- **grounded** — the raw model output contains only numbers the simulator produced
- **complete** — it states the product, stock level, and (when ordered) quantity and supplier
- **explanation accuracy** — both at once
- **delivered accuracy** — what the user actually sees, after the guard replaces ungrounded output with the template

<!-- EXPLAINER_EVAL_RESULTS -->

---

## 7. The LLM scenario generator

`src/llm/scenario_gen.py`. The value of an LLM here is imagination, not authority. Hand-written stress tests contain the failures their author thought of; asked for "a realistic 2026 retail supply chain crisis", the model returned:

| Days | Event | Cause it named |
|---|---|---|
| 45–65 | demand 3× on four products | viral social media trend |
| 50–63 | supplier 2 offline | cyberattack on mid-tier supplier |
| 55–72 | supplier 0 + 6 days | port congestion at major hub |
| 60–84 | costs 1.6× | raw material shortage |

They cascade — the outage lands inside the spike, the delay overlaps both — which is the case a single-disruption test never reaches. That is the only thing taken from the model. Every field is parsed, type-checked, range-clamped and bounds-checked against the real product and supplier counts before reaching the simulator. Given six deliberately hostile inputs — an invented scenario type, supplier 9 on a three-supplier world, product 14 on a ten-product world, a 999-day duration, an 87× multiplier, a cost multiplier of "lots" — three survive, none as written. Out-of-range product indices are dropped rather than wrapped, because 14 mod 10 would silently redirect a crisis onto a product the model never named. Windows are repositioned to land inside the episode; a clamp on start day alone left "day 400 for 30 days" as a stress test that ran for ten days and was silently mostly absent.

### 7.1 Stress-test result

Under the same hand-written disruption, on five seeds:

| Policy | Calm | Disrupted | Change |
|---|---|---|---|
| forecast + safety stock | 84,039 | 72,935 | −13% |
| MaskablePPO, untuned | 70,206 | 50,454 | **−28%** |

The agent degrades more than twice as badly on conditions it never trained on. This motivated Phase 9.

---

## 8. Curriculum training

`src/agents/curriculum.py`. A pool of 16 disruption sets was generated once from 19 LLM-produced scenarios, cached, and split 10/6. The curriculum agent draws a random training set at every episode reset and is scored on the 6 held-out sets it never saw. Hyperparameters, seed and step count match the untuned agent (`masked_1m`) exactly, so the curriculum is the only variable.

The split is the methodology, not a detail: training and testing on the same crises would show a large improvement while measuring memorisation. (The test suite once overwrote the live experiment's pool with a throwaway one; the cache path is now injectable and tests cannot touch the real experiment.)

---

## 9. Results

### 9.1 Main comparison — 30 held-out seeds

| Policy | Profit (£) | ± std | Fill rate | Ordering fees | Stockout cost | Overflow |
|---|---|---|---|---|---|---|
| forecast + safety stock | **83,624** | 4,389 | 97.9% | 2,729 | 5,821 | 1,204 |
| newsvendor | 82,049 | 4,222 | 98.0% | 3,275 | 7,057 | 242 |
| **RL agent, tuned, 2M steps** | **75,926** | 5,205 | 95.1% | 5,522 | 8,726 | **0** |
| RL agent, 2nd config, 2M | 73,331 | 5,163 | 93.6% | 6,774 | 9,998 | 0 |
| EOQ + reorder point | 72,302 | 7,028 | 96.8% | 2,626 | 9,988 | 3,273 |
| RL agent, untuned, 1M | 71,227 | 5,517 | 94.3% | 7,068 | 10,515 | 0 |
| (s,S) | 70,365 | 6,782 | 94.9% | 2,905 | 15,677 | 406 |
| constant order (control) | 50,751 | 12,723 | 92.3% | 3,334 | 27,644 | 1,606 |
| PPO, no masking, 1M | 22,566 | 8,560 | 94.8% | 13,988 | 10,805 | 16,796 |

**The tuned classical policy beats the tuned RL agent by £7,698 per episode** (paired *t* = −6.45, *n* = 30, *p* < 0.001). The agent reaches **90.8%** of the classical policy's profit and beats two of the four classical methods.

Where the gap sits:

- **Fill rate** 95.1% vs 97.9% — the agent runs out more often, costing ~£2,900 in extra stockout penalties.
- **Ordering fees** £5,522 vs £2,729 — the agent fragments orders. In one traced episode it placed 719 orders across all three suppliers against the baseline's 323, almost all with the cheap one (figure 07, panel D).
- **Overflow** — the agents are the *only* policies with zero overflow loss. They learned the shared-warehouse constraint properly; every classical policy wastes some.

### 9.2 What each intervention was worth

| Step | Training-eval profit | Held-out profit | Gap to classical |
|---|---|---|---|
| PPO, no masking | 23,717 | 22,566 | −61,058 |
| + action masking | 72,740 | 71,227 | −12,397 |
| + hyperparameter tuning, 2M steps | 82,105 | 75,926 | **−7,698** |

Masking ≈ 3×; tuning closed 38% of what remained.

### 9.3 Robustness — 6 held-out LLM-generated crises

| Policy | Calm | Disrupted | Drop |
|---|---|---|---|
| forecast + safety stock | 83,282 | 51,175 | −38.6% |
| **RL agent, tuned (no curriculum)** | 76,567 | **48,244** | **−37.0%** |
| RL agent, untuned (control) | 70,279 | 36,925 | −47.5% |
| RL agent, curriculum-trained | 62,904 | 37,071 | −41.1% |

**Curriculum training did not work.** Its smaller percentage drop is the trap the evaluation was built to expose: the drop is smaller because there is less left to lose. Absolute disrupted profit is 37,071 against the control's 36,925 — a 0.4% difference inside noise — bought by giving up 7,375 of calm performance. Paired *t* = +1.69, not significant.

The useful finding sits beside it: **hyperparameter tuning improved robustness more than training on crises did.** The tuned agent never saw a disruption during training and has both the highest disrupted profit of any agent and the smallest drop.

These drops are larger than in §7.1 because the six held-out LLM sets are harsher than the three hand-written ones.

### 9.4 Figures

| | |
|---|---|
| 01–04 | dataset screening, selected products, seasonality factors, demand regimes |
| 05 | policy comparison with error bars |
| 06 | training curves: no masking → masking → tuning, against the classical line |
| 07 | one episode inside the simulator: stock vs demand, warehouse fill, cumulative profit, supplier mix |
| 08 | pattern stability across two independent years |

All regenerate from committed result files (`python -m src.eval.result_figures`, `trace_figures`, `src.data.stability`).

---

## 10. Discussion

**Why the agent loses.** Its state contains nothing about its own ordering behaviour — no days-since-last-order, no running fee total, no last-supplier. It pays double the ordering fees and has no observation that could tell it so. This is an observability gap, not a tuning failure, and it is the most promising avenue left untried: three extra state features would give the agent access to the thing it is demonstrably bad at.

**Why the negative results are worth having.** Each was produced by an experiment designed so that a positive result would have meant something. The baselines were tuned; the seeds were disjoint; the curriculum was evaluated on crises the agent never saw; the percentage drop was reported beside the absolute one so that "less to lose" could not masquerade as robustness. A project that beat a deliberately weak baseline would have reported success and learned nothing.

**What the explainer is and is not.** It is an accurate, checked restatement of computed facts. It is not an account of the policy network's reasoning, and the project does not claim it is. That distinction is the difference between explainability and confabulation.

---

## 11. Limitations

- **Data age.** Calibrated on 2009–2011 UK retail data; currency values are of that period. The demand *shape* was shown stable across the two years available (§3.4); nothing here can speak to 2026 directly, and real recent SKU-level retail data is not public.
- **Synthetic suppliers.** Lead times, fill rates and fees are hand-authored to span the literature's trade-off; no public dataset contains them.
- **Equal-steps comparison in Phase 9.** The curriculum agent faced a harder distribution with the same budget; part of its shortfall may be undertraining. Equal-convergence is a different experiment.
- **Explainer sample size.** 30 decisions, one policy, one seed.
- **Single retailer, ten products.** Generalisation to other categories is untested.

---

## 12. Toward a usable product

The RL agent would not be deployed today: it earns less than a simpler method and is not more robust. The honest product architecture — which the dashboard (`app/dashboard.py`) already implements — is:

```
decision engine   tuned forecast + safety stock
challenger        RL agent, running in shadow, logged and scored but not acting
explanation       LLM layer on whichever decision is shown, grounding-checked
stress testing    LLM scenario generator
interface         recommend-only dashboard
```

When the challenger consistently beats the incumbent on live data, it is promoted. That is how ML systems actually reach operations. Missing for real use: cold-start handling for new products, order-value guardrails, an audit trail, a live data feed, and a confidence signal for when to defer to a human.

---

## 13. Reproducibility

```bash
python -m pytest tests/ -q                    # 133 tests
python -m src.data.run_pipeline               # rebuild calibration from raw data
python -m src.agents.tune_baselines           # grid-search the classical policies
python -m src.agents.train_ppo --name run     # train an agent
python -m src.eval.compare                    # held-out comparison, paired test
python -m src.eval.robustness                 # calm vs disrupted
python -m src.eval.explainer_eval             # explanation accuracy
python -m src.data.stability                  # pattern stability
streamlit run app/dashboard.py
```

CI runs lint, format and the test suite on every push. The Kaggle sweep notebook is in `kaggle/`. Every result file in `results/` is committed and every figure regenerates from them.

---

## References

1. Schulman, J., Wolski, F., Dhariwal, P., Radford, A., Klimov, O. (2017). Proximal Policy Optimization Algorithms. arXiv:1707.06347.
2. Huang, S., Ontañón, S. (2022). A Closer Look at Invalid Action Masking in Policy Gradient Algorithms. FLAIRS-35.
3. Gijsbrechts, J., Boute, R. N., Van Mieghem, J. A., Zhang, D. (2022). Can Deep Reinforcement Learning Improve Inventory Management? *Manufacturing & Service Operations Management*.
4. Boute, R. N., Gijsbrechts, J., van Jaarsveld, W., Vanvuchelen, N. (2022). Deep Reinforcement Learning for Inventory Control: A Roadmap. *European Journal of Operational Research*.
5. Oroojlooyjadid, A., Nazari, M., Snyder, L. V., Takáč, M. (2022). A Deep Q-Network for the Beer Game. *Manufacturing & Service Operations Management*.
6. Chen, D., Sain, S. L., Guo, K. (2012). Data mining for the online retail industry. *Journal of Database Marketing & Customer Strategy Management*.
7. Raffin, A. et al. (2021). Stable-Baselines3: Reliable Reinforcement Learning Implementations. *JMLR* 22(268).
8. UCI Machine Learning Repository. Online Retail II. https://archive.ics.uci.edu/dataset/502
