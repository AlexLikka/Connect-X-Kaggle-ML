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

Run a local head-to-head match between two Kaggle-style submission files:

```bash
python scripts/play_submissions.py submission.py submission_ml.py --games 2 --alternate-first --render
```

This local runner follows the Kaggle ConnectX interface:
- each agent must expose `agent(observation, configuration)`
- the returned action must be a legal column index
- invalid actions or full-column moves lose immediately
- `configuration` includes `rows`, `columns`, `inarow`, and `timeout`

## Phase 2: ML-Enhanced Agent (Value + Policy for Move Ordering)

This phase implements the core idea from MODEL_PLAN Phase 3: combine learned
models with search. Two models are trained:

- **Policy model**: 7-class classifier that predicts which column the search
  would choose. Used for move ordering at the root.
- **Value model**: regression model that predicts the negamax evaluation score.
  Combined with policy and heuristic for root-level move ordering.

The search leaves still use the fast hand-crafted `window_score` evaluation —
the ML models run **only at the root** to order moves, so alpha-beta pruning
reaches deeper tactical lines without slowdown.

### Step 1: Generate self-play training data

```bash
conda activate kaggle
python scripts/generate_selfplay_data.py --games 500 --depth 7 --random-opening-moves 2 --output data/selfplay.npz
```

- `--games`: Number of self-play games
- `--depth`: Search depth for data generation (higher = stronger labels)
- `--random-opening-moves`: Sample 0..N random opening plies each game for broader coverage
- `--output`: Output `.npz` file containing features, scores, and best moves

Generated data now also includes:
- Mirrored board augmentation by default
- Per-column root scores for all legal moves
- Legal-move masks for soft policy training

### Step 2: Train both models

```bash
python scripts/train_model.py --data data/selfplay.npz --epochs 3000 --hidden 128
```

- `--data`: Path to `.npz` from Step 1
- `--epochs`: Training epochs (policy uses cross-entropy, value uses MSE)
- `--hidden`: Hidden layer size for both models (default 128)
- `--lr`: Learning rate with decay (default 0.001)
- `--model-output`: Saved weights (default `models/models.json`)
- `--submission-output`: Generated submission (default `submission_ml.py`)

Training outputs:
- Policy accuracy (how often it predicts the same column as search)
- Value loss (MSE between predicted and actual negamax scores)

### Step 3: Validate and evaluate

```bash
# Validate the ML submission
python scripts/evaluate_agents.py --episodes 2 --validate-submission submission_ml.py

# Evaluate against baseline opponents
python scripts/evaluate_agents.py --episodes 10 --time-limit 1.65 --opponents random negamax
```

### How it works

```
Root position
  |
  +-- Policy model: P(col 0..6 | board)
  +-- Value model:  V(board after each col)
  +-- Heuristic:    fast window_score evaluation
  |
  +-- Combined score = 0.4 * policy + 0.3 * value + 0.3 * heuristic
  |
  +-- Order moves by combined score
  |
  +-- Alpha-beta search (fast heuristic eval at leaves)
```

The 0.4 / 0.3 / 0.3 weights are tuned empirically — the policy model provides
global strategic guidance, the value model estimates long-term position quality,
and the hand-crafted heuristic captures immediate tactical threats.
