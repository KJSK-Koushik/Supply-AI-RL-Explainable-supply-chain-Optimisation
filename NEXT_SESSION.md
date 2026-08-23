# Where we stopped

Phase 9 (curriculum training) is **paused mid-run**, not finished.

## Resume the training

```bash
D:/venvs/supplyai/Scripts/python -u -m src.agents.train_ppo --name curriculum_1m --curriculum --resume
```

Picks up from `checkpoint_150000_steps.zip`. Roughly **90 minutes** left at the
measured 156 steps/s (the curriculum makes each step ~2x more expensive than a
calm one, which is why this is slower than the 48-minute `masked_1m` run).

## Then get the result

```bash
D:/venvs/supplyai/Scripts/python -m src.eval.robustness
```

Scores every policy on calm episodes and on the 6 held-out disruption sets, and
writes `results/robustness.json`.

## The question being answered

Phase 7 found the agent is brittle: under disruption the tuned classical policy
loses 13% of its profit, the RL agent loses 28%. Phase 9 asks whether training
on generated crises closes that gap.

## What to watch for

At 150,000 steps the curriculum agent was well behind on calm episodes:

| Steps | curriculum | masked_1m |
|---|---|---|
| 100,000 | -6,400 | 48,960 |
| 150,000 | 7,219 | 47,757 |

If it stays far behind, a smaller percentage drop under disruption does **not**
prove robustness -- it is easy to lose little when there is little to lose. The
honest reading in that case is "traded average performance for stability", and
`robustness.py` prints calm profit, disrupted profit and the drop side by side
so this stays visible.

## Do not regenerate the curriculum pool

`results/curriculum_pool.json` is committed and holds the exact crises this run
trained against. Rebuilding it would score the agent on a different holdout than
it learned on. `build_pool()` reloads the cache unless `force=True`.

## Still open, unrelated to Phase 9

- Kaggle sweep result files were never downloaded, so slides 11-13, the figures
  and the README still carry pre-sweep numbers (agent 71,227, gap -12,397). The
  post-sweep numbers are agent 75,926, gap -7,698.
- Phase 8 (Streamlit dashboard) not started. Depends on nothing.
- Phase 10 (report pack) not started.
- Faculty raised the dataset age. Agreed plan: a pattern-stability check
  (~15 min) and a market-regime sweep (~1 hour). Neither started.
