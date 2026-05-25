# Connect-X-Kaggle-ML

Kaggle ConnectX course project. Current focus: a strong pure-search baseline
that can be submitted directly, followed by ML-assisted search.

## Quick Start

Create and activate the environment by following [ENV_SETUP.md](ENV_SETUP.md).

Run a local benchmark:

```bash
conda activate kaggle
python scripts/evaluate_agents.py --episodes 10
```

By default the benchmark uses a short per-move budget for speed. For a stronger,
submit-like local run:

```bash
python scripts/evaluate_agents.py --episodes 10 --time-limit 1.65
```

Validate the Kaggle submission file:

```bash
python scripts/evaluate_agents.py --episodes 2 --validate-submission submission.py
```

The current submit-ready file is [submission.py](submission.py). It is fully
self-contained and exposes the required `agent(observation, configuration)`
function.
