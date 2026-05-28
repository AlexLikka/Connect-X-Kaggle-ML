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

## Phase 2: ML-Enhanced Agent

### Step 1: Generate self-play training data

Use the strong search agent to play against itself and collect board positions
with negamax evaluation scores as training targets:

```bash
conda activate kaggle
python scripts/generate_selfplay_data.py --games 500 --depth 5 --output data/selfplay.npz
```

- `--games`: Number of self-play games (more games = more data, ~33 samples/game)
- `--depth`: Search depth for data generation (higher = stronger but slower labels)
- `--output`: Output path for the `.npz` data file

### Step 2: Train the value model

Train a 2-layer neural network on the self-play data. The model predicts
negamax scores from board features and exports a self-contained submission file:

```bash
python scripts/train_model.py --data data/selfplay.npz --epochs 2000 --hidden 256
```

- `--data`: Path to the `.npz` data file from Step 1
- `--epochs`: Training epochs
- `--hidden`: Hidden layer size (larger = more expressive but slower inference)
- `--lr`: Learning rate (default 0.001)
- `--batch-size`: Mini-batch size (default 64)
- `--model-output`: Saved model weights (default `models/model.json`)
- `--submission-output`: Generated submission file (default `submission_ml.py`)

This produces `submission_ml.py` — a fully self-contained agent with the neural
network weights embedded. It uses the ML model as the leaf evaluation function
inside the same alpha-beta search.

### Step 3: Validate and evaluate

```bash
# Validate the ML submission
python scripts/evaluate_agents.py --episodes 2 --validate-submission submission_ml.py

# Evaluate against baseline opponents (use full time budget for submit-like strength)
python scripts/evaluate_agents.py --episodes 10 --agent search --time-limit 1.65 --opponents random negamax
```
