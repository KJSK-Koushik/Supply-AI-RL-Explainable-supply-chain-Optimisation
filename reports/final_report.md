# SupplyAI-RL: Explainable Supply Chain Optimization

**An AI decision-support system for retail inventory, with plain-English explanations**
Course: 22AIE450 Reinforcement Learning — Final Report

> **About the money in this report.** All amounts are in rupees (₹). The sales data comes from a UK shop, so the simulator works in pounds inside. We convert at 1 GBP = ₹105 when showing numbers. The rate is one line in `configs/display.yaml`.

---

## Abstract

A shop must decide every day, for every product, how much to order and from which supplier. This project builds a complete system for that decision. It has:

- a **simulator** (a virtual shop) built from 1,067,371 real sales records,
- an **AI agent** (reinforcement learning, MaskablePPO) that picks the quantity and the supplier together,
- **four classical methods** from textbooks, each tuned properly so the comparison is fair,
- an **LLM** (language model) that explains every decision in plain English and invents crisis situations for testing,
- a **dashboard** where a manager can run a day and ask for an explanation.

The main result is honest, not flattering. On 30 unseen test runs, the tuned AI earns **₹79.7 lakh** per 180 days — **90.8% of the best classical method's ₹87.8 lakh** — and beats two of the four classical methods. But the gap of ₹8.1 lakh is real and statistically significant. We know why: the AI spreads its orders across suppliers and pays twice the ordering fees.

Along the way we found four things. Action masking (blocking illegal choices before the AI decides) was worth about 3×. Hyperparameter tuning closed 38% of the remaining gap. Training on crises did *not* make the AI more robust — plain tuning did more. And the LLM explainer's safety check is essential: without it, one explanation in six would contain a made-up number.

Every comparison uses the same simulator, the same random customers, and the same choices. That fairness is what makes the negative results mean something.

---

## 1. Introduction

### 1.1 Why this problem is hard

Ordering stock has two ways to go wrong, and they cost different amounts.

- **Order too much:** money sits in the warehouse, storage costs pile up, and seasonal goods get marked down or thrown away.
- **Order too little:** the customer buys elsewhere. The sale is gone for good.

Now add supplier choice. One supplier is cheap but slow. One is fast but expensive. One looks reasonable but ships only 80% of what you ordered and goes offline more often than the others. And every product shares one warehouse, so ordering a lot of product A leaves less room for product B.

Today's order becomes next week's stock. That makes it a **sequential decision problem**: what you do now changes what you can do later.

### 1.2 Why the usual methods struggle

Classical inventory rules — (s,S), EOQ with reorder point, newsvendor, forecast plus safety stock — treat each product on its own and assume demand is well-behaved. Real demand is not. In our data, daily demand varies about 216 times more than a simple statistical model (Poisson) would expect.

### 1.3 Why explanation matters

Reinforcement learning (RL) can learn a joint ordering-and-supplier policy that a formula cannot express. But a buyer will not act on a recommendation they cannot understand. So this project pairs the AI with a language model that puts each decision into plain sentences — under one strict rule: **the LLM only states facts the simulator computed. It never guesses at reasons.**

### 1.4 Objectives

1. Build a supply chain simulation from real data.
2. Train an RL agent to choose order quantity and supplier.
3. Use an LLM to explain each decision in business language.
4. Use an LLM to generate demand-spike and supplier-delay scenarios.
5. Compare the RL agent against traditional inventory policies.

All five are done. Objective 5 says *compare*, not *win*. The comparison came out against the AI, and we report it that way.

---

## 2. The problem, precisely

| | |
|---|---|
| Products | 10 real products from the UCI Online Retail II dataset |
| Suppliers | 3 (invented, because no public dataset includes supplier details) |
| Time | 180 simulated days per run; the first 14 days are a warm-up and are not scored |
| Warehouse | one shared store of 20,000 units; anything over that is refused and written off |
| Decision | each day, for each product: one of 7 quantity levels × 3 suppliers = 21 options |
| Goal | maximise profit = sales − purchase cost − storage cost − ordering fees − lost-sale penalty − overflow loss |

The three suppliers:

| Supplier | Price | Delivery time | Actually ships | Minimum order | Fee per order | Goes offline |
|---|---|---|---|---|---|---|
| EconoSource | base | 5–8 days | 90% | 100 units | ₹2,100 | rarely |
| RapidTrade | +35% | 1–3 days | 98% | 20 units | ₹4,725 | almost never |
| MidWay Supply | +15% | 3–5 days | 80% | 50 units | ₹3,150 | most often |

None of them is best. The cheap one ties up a week of stock in transit. The fast one eats your margin. The middle one is unreliable. The right choice depends on how much stock you have, what season it is, and how unpredictable demand is right now — exactly the things a fixed rule cannot take into account.

With 21 options per product and 10 products, there are about **17 trillion** possible combinations each day.

---

## 3. Data

### 3.1 Source

**UCI Online Retail II** — 1,067,371 sales records from a UK online gift shop, December 2009 to December 2011.

**Cleaning:** cancelled orders, returns, negative quantities and missing product codes are removed. Sales are grouped into one row per product per day. Saturdays are removed entirely because the shop never traded on Saturdays; filling them with zeros would teach the model a false weekly dip.

**Ten products were chosen on purpose** to cover three kinds of demand:

| Type | Products | How much demand varies (CV) |
|---|---|---|
| steady | 21212, 85123A, 85099B, 84991 | 1.02 – 1.20 |
| moderate | 84879, 21977, 21213 | 1.29 – 1.65 |
| bursty | 84077, 22197, 15036 | 2.09 – 2.34 |

If all ten behaved the same, one simple rule would handle them all and there would be nothing for an AI to learn.

Only data before **1 June 2011** is used to build the simulator. Later data is held back.

### 3.2 A newer dataset was tested and rejected

We first tried the Kaggle "Retail Store Inventory Forecasting" dataset (2022–2024). It has no weekday pattern and no seasonal pattern. It is synthetic noise. We rejected it and recorded the evidence in `reports/dataset_selection.md`. **Newer but fake is worse than older but real.**

### 3.3 How the data is used

The AI never sees the sales records. We measure five things from them — average daily demand per product, how much it varies, the weekday pattern, the month pattern, and how often a day has zero sales — and the simulator reproduces those. Demand is generated with a Gamma-Poisson (negative binomial) model, which produces both quiet weeks and sudden spikes the way real retail does.

### 3.4 Is the data too old?

Faculty asked whether 2009–2011 data can say anything about today. The simulator does not use the data's *prices or volumes* — those are settings. It uses the **shape** of demand. So the right question is: is that shape a feature of retail, or of that period?

We split the two years into two separate halves and measured the shape in each (`src/data/stability.py`, figure 08):

| Property | Year 1 vs Year 2 |
|---|---|
| Weekday pattern | *r* = 0.87 — Tuesday to Thursday busy, Sunday quiet, in both years |
| Month pattern | *r* = 0.51 (0.64 if December is excluded — December is partly a data-window artefact) |
| Busiest month | November, in both years |
| Burstiness (variance ÷ mean) | 224 and 474 — extreme in both years |

The weekly rhythm and the November peak repeat. The middle of the year does not. We report it as found. This supports the two features the simulator depends on most. It cannot say anything about 2026 itself — that stays a limitation (section 11).

---

## 4. The simulator

`src/env/supply_chain_env.py` is a Gymnasium environment. Each day runs five steps in this order:

1. **Deliveries arrive.** Stock ordered days ago lands. If the warehouse is full, the extra is refused and the money is lost.
2. **Customers arrive.** Demand is drawn for each product, adjusted for weekday and month.
3. **Sell what you have.** Any shortfall is a lost sale with a penalty, and carries over as backlog.
4. **Place today's orders.** This comes *after* demand, because a real buyer orders once they know how the day went.
5. **Count the money.**

Suppliers have random delivery times, random partial shipments, minimum order quantities, fixed fees per order, and random outages.

Three settings each fixed an earlier mistake:

- **Storage cost** is 0.15% of the item's cost per day (about 55% per year). That is high on purpose, because these are seasonal gift items that lose value. An earlier setting was 730% per year, which made holding any stock absurdly expensive.
- **Leftover stock at the end** is worth 60% of its cost. An earlier version treated it as a total loss, which was only an artefact of stopping at 180 days.
- **One shared warehouse** instead of a limit per product. With per-product limits, "always order the maximum" was the best strategy, and there was nothing to learn.

A demo (`scripts/demo_simulator.py`) confirms the simulator rewards sensible behaviour: simple reasonable rules earn about ₹27 lakh; never ordering loses ₹3.9 crore; always ordering the maximum gets the best service level of any method and still loses ₹73 lakh. If sensible had not beaten silly, nothing built on top would mean anything.

---

## 5. Methods

### 5.1 How the AI sees the problem

| | |
|---|---|
| **What it sees** (88 numbers) | for each product: stock on hand, stock on the way, days of cover, last-7-day average demand, how much demand is varying, trend, backlog; plus the weekday, the time of year, and each supplier's availability, delivery time and recent reliability |
| **What it decides** | for each of the 10 products, one of 21 choices (quantity level × supplier) |
| **What it is rewarded for** | that day's profit, scaled down and clipped; zero during the 14-day warm-up |
| **One run** | 180 days |

The quantity and supplier are chosen *together* as one option, not as two separate choices. That is 2.8× faster to train and it matches how the decision really works: ordering 8× normal demand only makes sense from a supplier who can actually deliver it.

### 5.2 The algorithm

**MaskablePPO** — PPO (Proximal Policy Optimization) with **action masking**. Before the AI chooses, illegal options are blocked: suppliers that are offline, and orders too big for the warehouse. The network is small (two layers of 128 or 256 units) and trains on CPU.

Masking was the single biggest improvement in the project. Without it, the AI's best score was ₹24.9 lakh. With it, ₹76.4 lakh — about 3×. Without masking, the AI wastes most of its training discovering that some choices are pointless.

### 5.3 The classical methods, tuned

Four textbook methods, each tuned by grid search on 12 tuning runs (seeds 100–111) that are never used for evaluation:

| Method | Tuned settings |
|---|---|
| (s,S) | reorder when stock covers under 11 days, order up to 16 days |
| EOQ + reorder point | reorder at 8 days of cover, safety factor 1.6 |
| newsvendor | 3-day horizon, cheapest supplier |
| forecast + safety stock | 1-day forecast window, z = 1.28, review every 4 days, cheapest supplier |

Tuning mattered a lot: widening the search lifted the best method from ₹69.7 lakh to ₹89.2 lakh. If we had left the classical methods at textbook defaults, the AI would have "won" — and it would have meant nothing.

Every classical method picks from the same 21 options per product as the AI, so no method has finer control than another.

### 5.4 How the evaluation stays fair

Three separate groups of random seeds:

| Seeds | Used for |
|---|---|
| 100–111 | tuning the classical methods |
| 900–905 | picking the best checkpoint during AI training |
| **500–529** | **reporting results — never used to choose anything** |

Every number in this report comes from the 30 reporting seeds. Every method faces the same customers and the same delivery delays on each seed. So comparisons are **paired**: we look at the difference on each seed, which removes the run-to-run randomness that would otherwise hide a real effect.

### 5.5 Hyperparameter search

Eight settings were tried, with a short 250,000-step trial each, and the best two were trained fully to 2 million steps on Kaggle (about 4.8 hours).

The winner used the **lowest** learning rate on offer (5e-5) and the **larger** network. That confirmed our diagnosis: the earlier runs were unstable because they learned too fast, not because the network was too small.

---

## 6. The LLM explainer

`src/llm/explainer.py`

**The rule:** the LLM is never asked *why* a decision was made. The AI's real reason is a neural network. Any "because" the LLM wrote would be a plausible story, not the cause — and presenting a story as the cause would make the word "explainable" false.

So the LLM does presentation only. It receives facts the simulator computed — stock, orders on the way, days of cover, recent demand, what was ordered, from whom, that supplier's delivery time and reliability — and writes two or three sentences. Then we **check every number in its output** against those facts. If any number was invented, the output is thrown away and a fixed template is used instead.

Models are free-tier via OpenRouter: `google/gemma-4-31b-it`, `z-ai/glm-5.2`, `nvidia/nemotron-3-ultra-550b-a55b`, tried in that order with retries, because free models rate-limit constantly. With no key, or if every model refuses, the template takes over.

A real example (nemotron, day 44):

> Product 21212 (Pack Of 72 Retrospot Cake Cases) holds 2,438 units in stock with nothing on order, providing 14.5 days of cover at the typical daily demand of 169 units. Recent demand has averaged 89 units over the last seven days, below its usual level, and there was no unmet demand yesterday. No order was placed today.

### 6.1 Measuring the explainer

`src/eval/explainer_eval.py` scored 30 real decisions (6 days × 10 products):

| Measure | Result |
|---|---|
| LLM reachable | 100% of calls (after retries) |
| **Grounded** — raw LLM output used only real numbers | **83%** (25 of 30) |
| Complete — stated product, stock, and order details | 100% |
| **Raw model accuracy** — grounded *and* complete, before our check | **83%** |
| **Delivered accuracy** — what the user actually saw, after our check | **100%** |
| Length | 2.9 sentences, 58 words on average |

**Five of thirty LLM outputs contained a number the simulator never produced.** They read fluently and confidently. A buyer would not have noticed. The check caught all five and replaced them with the template, so every explanation the user saw was correct and complete.

This is the real finding of the explainer work: **the safety check is essential.** Without it, one explanation in six would carry a made-up figure. With it, none do. An explainer that trusted the model would have been quietly unsafe. This one is measured to be safe.

We report both accuracies deliberately. An earlier version of our metric only scored outputs that had already passed the check, and so reported "100%" for a model that made things up 17% of the time. That number was true and useless. These two are the ones that matter.

---

## 7. The LLM crisis generator

`src/llm/scenario_gen.py`

The LLM's value here is imagination. Crisis tests written by hand contain only the failures the author thought of. Asked for "a realistic 2026 retail supply chain crisis", the LLM wrote:

| Days | Event | Cause it gave |
|---|---|---|
| 45–65 | demand 3× on four products | viral social media trend |
| 50–63 | supplier 2 offline | cyberattack on mid-tier supplier |
| 55–72 | supplier 0 delayed by 6 days | port congestion at major hub |
| 60–84 | costs 1.6× | raw material shortage |

These overlap. The supplier goes down *during* the demand spike; the delay overlaps both. That combination is what really hurts, and a single-event test never reaches it.

**Nothing the LLM writes is trusted.** Every field is checked: the scenario type must be one we know; supplier and product numbers must exist; values are clamped to safe ranges; scenarios are moved so they fall inside the 180-day run. We fed it six deliberately broken inputs — an invented scenario type, supplier 9 on a 3-supplier world, product 14 on a 10-product world, a 999-day duration, an 87× demand multiplier, and a cost multiplier of "lots". Three survived, none as written. Out-of-range product numbers are dropped rather than wrapped, because 14 mod 10 would silently move the crisis to product 4.

### 7.1 First stress-test result

Under the same hand-written crisis, on five seeds:

| Method | Calm | Crisis | Change |
|---|---|---|---|
| forecast + safety stock | ₹88.2 lakh | ₹76.6 lakh | −13% |
| AI, untuned | ₹73.7 lakh | ₹53.0 lakh | **−28%** |

The AI fell more than twice as hard. This is what motivated Phase 9.

---

## 8. Curriculum training

`src/agents/curriculum.py`

We generated a pool of 16 crisis sets from 19 LLM-written scenarios, saved it, and split it 10/6. The AI trained on a random set from the 10 at the start of every run, and was tested on the 6 it had never seen. Its settings, seed and training length exactly match the untuned AI, so the crisis training is the only difference.

**The split is the whole method.** Training and testing on the same crises would show a big improvement that only measured memorisation. (Our own test suite once overwrote the saved pool by accident; the cache path is now injectable so tests cannot touch the real experiment.)

---

## 9. Results

### 9.1 Main comparison — 30 unseen runs

| Method | Profit | ± spread | Fill rate | Ordering fees | Stockout cost | Overflow |
|---|---|---|---|---|---|---|
| forecast + safety stock | **₹87.8 lakh** | ₹4.6 lakh | 97.9% | ₹2.9 lakh | ₹6.1 lakh | ₹1.3 lakh |
| newsvendor | ₹86.2 lakh | ₹4.4 lakh | 98.0% | ₹3.4 lakh | ₹7.4 lakh | ₹0.3 lakh |
| **AI, tuned, 2M steps** | **₹79.7 lakh** | ₹5.5 lakh | 95.1% | ₹5.8 lakh | ₹9.2 lakh | **₹0** |
| AI, 2nd config, 2M steps | ₹77.0 lakh | ₹5.4 lakh | 93.6% | ₹7.1 lakh | ₹10.5 lakh | ₹0 |
| EOQ + reorder point | ₹75.9 lakh | ₹7.4 lakh | 96.8% | ₹2.8 lakh | ₹10.5 lakh | ₹3.4 lakh |
| AI, untuned, 1M steps | ₹74.8 lakh | ₹5.8 lakh | 94.3% | ₹7.4 lakh | ₹11.0 lakh | ₹0 |
| (s,S) | ₹73.9 lakh | ₹7.1 lakh | 94.9% | ₹3.1 lakh | ₹16.5 lakh | ₹0.4 lakh |
| constant order (control) | ₹53.3 lakh | ₹13.4 lakh | 92.3% | ₹3.5 lakh | ₹29.0 lakh | ₹1.7 lakh |
| PPO, no masking | ₹23.7 lakh | ₹9.0 lakh | 94.8% | ₹14.7 lakh | ₹11.3 lakh | ₹17.6 lakh |

**The best classical method beats the tuned AI by ₹8.1 lakh per run** (paired *t* = −6.45, *n* = 30, *p* < 0.001). The AI reaches **90.8%** of the classical profit and beats two of the four classical methods.

Where the gap comes from:

- **Fill rate** 95.1% vs 97.9% — the AI runs out more often, costing about ₹3 lakh more in lost-sale penalties.
- **Ordering fees** ₹5.8 lakh vs ₹2.9 lakh — the AI spreads orders across suppliers. In one traced run it placed 719 orders; the classical method placed 323, almost all with the cheap supplier (figure 07, panel D).
- **Overflow** — the AI versions are the *only* methods with zero overflow loss. They learned the shared-warehouse rule properly. Every classical method wastes some.

### 9.2 What each step was worth

| Step | Training score | Test score | Gap to classical |
|---|---|---|---|
| PPO, no masking | ₹24.9 lakh | ₹23.7 lakh | −₹64.1 lakh |
| + action masking | ₹76.4 lakh | ₹74.8 lakh | −₹13.0 lakh |
| + tuning, 2M steps | ₹86.2 lakh | ₹79.7 lakh | **−₹8.1 lakh** |

Masking ≈ 3×. Tuning closed 38% of what was left.

### 9.3 Robustness — 6 unseen LLM-generated crises

| Method | Calm | Crisis | Drop |
|---|---|---|---|
| forecast + safety stock | ₹87.4 lakh | ₹53.7 lakh | −38.6% |
| **AI, tuned (no crisis training)** | ₹80.4 lakh | **₹50.7 lakh** | **−37.0%** |
| AI, untuned (control) | ₹73.8 lakh | ₹38.8 lakh | −47.5% |
| AI, trained on crises | ₹66.0 lakh | ₹38.9 lakh | −41.1% |

**Crisis training did not work.** The crisis-trained AI's smaller percentage drop is misleading — it drops less because it has less to lose. Under crisis it earns ₹38.9 lakh against the control's ₹38.8 lakh, a 0.4% difference that is inside the noise, and it gave up ₹7.7 lakh of calm profit to get there. Paired *t* = +1.69, not significant.

The useful result sits beside it: **plain hyperparameter tuning made the AI more robust than crisis training did.** The tuned AI never saw a crisis during training, yet it has the highest crisis profit and the smallest drop of all the AI versions.

The drops here are bigger than in section 7.1 because the six LLM-generated crises are harsher than the three hand-written ones.

### 9.4 Figures

| | |
|---|---|
| 01–04 | dataset screening, chosen products, seasonal factors, demand types |
| 05 | all methods compared, with error bars |
| 06 | training curves: no masking → masking → tuning, against the classical line |
| 07 | one run inside the simulator: stock vs demand, warehouse fill, profit, supplier mix |
| 08 | pattern stability across two separate years |

All regenerate from the saved result files.

---

## 10. Discussion

**Why the AI loses.** Its 88 inputs contain nothing about its own ordering history — no "days since I last ordered this", no running total of fees, no "which supplier did I use last time". It pays double the ordering fees and has no way to see that it is doing so. This is a gap in what the AI can observe, not a tuning problem. Adding three inputs per product is the most promising thing left untried.

**Why the negative results are worth having.** Each one came from an experiment designed so that a positive result would have meant something. The classical methods were tuned. The test seeds were separate. The crisis-trained AI was tested on crises it never saw. The percentage drop was shown next to the absolute profit so "less to lose" could not pass for robustness. A project that beat a deliberately weak baseline would have claimed success and learned nothing.

**What the explainer is, and is not.** It is an accurate, checked restatement of computed facts. It is not an account of the neural network's reasoning, and we do not claim it is. That difference is the difference between explanation and invention.

---

## 11. Limitations

- **Data age.** The data is 2009–2011 UK retail; money values are of that period (converted to rupees at a fixed rate). We showed the demand *shape* is stable across the two years we have (section 3.4). Nothing here can speak to 2026 directly. Real, recent, product-level retail sales data is not public.
- **Invented suppliers.** Delivery times, reliability and fees are hand-set to span the standard trade-off. No public dataset has them.
- **Equal training time in Phase 9.** The crisis-trained AI faced a harder task with the same budget, so part of its weaker result may be under-training rather than the method failing.
- **Explainer sample.** 30 decisions, one method, one seed.
- **One shop, ten products.** Other product categories are untested.

---

## 12. Toward a real product

The AI should not be deployed today. It earns less than a simpler method and is not more robust. The honest way to use what we built — which the dashboard already does — is:

```
decision engine    the tuned classical method (what a buyer would actually run)
challenger         the AI, running alongside, logged and scored but not acting
explanation        the LLM layer, on whichever decision is shown, always checked
stress testing     the LLM crisis generator
interface          a recommend-only dashboard
```

When the challenger beats the engine consistently on live data, it gets promoted. That is how AI systems really reach operations. Still missing for real use: handling brand-new products with no history, order-value limits, an audit trail, a live data feed, and a signal for when the system is unsure and a human should decide.

---

## 13. Reproducing everything

```bash
python -m pytest tests/ -q                    # 133 tests
python -m src.data.run_pipeline               # rebuild the simulator settings from raw data
python -m src.agents.tune_baselines           # tune the classical methods
python -m src.agents.train_ppo --name run     # train an AI
python -m src.eval.compare                    # main comparison
python -m src.eval.robustness                 # calm vs crisis
python -m src.eval.explainer_eval             # explanation accuracy
python -m src.data.stability                  # is the pattern stable?
streamlit run app/dashboard.py
```

CI runs lint and all tests on every push. The Kaggle notebook is in `kaggle/`. Every result file is committed; every figure regenerates from them.

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
