# Running the sweep on Kaggle

Kaggle runs notebooks **headless** via *Save & Run All (Commit)* — unlike Colab,
you can close the browser and shut the laptop while it finishes.

## Steps

1. Go to https://www.kaggle.com/code → **New Notebook**
2. **File → Import Notebook** → upload `supplyai_rl_sweep.ipynb`
3. In the right-hand panel:
   - **Internet: On** (required — the notebook clones the repo and downloads
     the UCI dataset)
   - **Accelerator: None** — this workload is CPU-bound. A GPU does nothing for
     a ~20,000-parameter MLP and the simulator cannot use one at all.
4. Click **Save Version → Save & Run All (Commit)**
5. Close the browser. Come back later; output is under the notebook's *Output*
   tab and can be downloaded.

## What it does

| Cell | Purpose |
|---|---|
| 1 | Clone the repo, install the RL libraries Kaggle lacks |
| 2 | Download UCI Online Retail II, rebuild `demand_stats.json` |
| 3 | Run the test suite — proves the environment behaves identically here |
| 4 | Measure throughput on Kaggle's hardware |
| 5 | Run the sweep (screen many configs, fully train the best) |
| 6 | Copy models and results to the downloadable output folder |

## Why cell 3 matters

Results from two machines are only comparable if the environment is identical.
The test suite checks the simulator invariants and the seeded reproducibility,
so if anything differs on Kaggle it fails loudly before hours are spent
training against a different environment. Any run whose origin machine matters
is recorded in `summary.json`.

## Expected timing

Kaggle provides roughly 4 vCPU against the dev laptop's 12 cores, so expect
about half the throughput — roughly 250–300 steps/s versus ~560 locally. The
sweep in cell 5 is sized to finish inside the ~12 hour session limit at that
speed.

Results are appended to `results/sweep_results.json` as each configuration
completes, so hitting the session limit still leaves usable output rather than
losing everything.

## Local equivalent

The same sweep runs locally, faster, if the laptop can stay awake:

```bash
D:/venvs/supplyai/Scripts/python -m src.agents.sweep --budget 10 --n-envs 8
```
