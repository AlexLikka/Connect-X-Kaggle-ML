# ConnectX Experiment Notes

Current date: 2026-06-10

## Main finding

The most useful direction is not deeper `depth=7` hard-label data by itself.
The strongest signal found in this round is:

1. Fix distilled value scaling in `scripts/train_model.py`.
2. Keep the original f902rich-style afterstate ML move-ordering formula.
3. Distill from `rl/checkpoints/mixed_c128_b8_iter1.pt`.
4. Increase the light MLP student capacity from hidden 128 to hidden 256.

Best current ML candidates:

- `submission_ml_distill_iter1_vfix_h256.py`
- `submission_ml_distill_d7_from_mixed_vfix_h256.py`

The first is cleaner and beat `submission_ml_f902_rich.py` 5-3 under random
opening evaluation. The second did not improve over it in direct comparison.

## Important bug fixed

`rl.export_policy_samples` writes `value_targets` in `[-1, 1]`, while
`ValueModel.train_step` expects search-score-scale labels and divides labels by
`1e6`.

Before the fix, distilled values were effectively trained as near-zero labels.
`scripts/train_model.py` now scales `value_targets` back to score scale:

```python
scores = value_targets * 1_000_000.0
```

This materially improved the distillation route. The corrected hidden-128
distilled model went from losing badly to `submission.py` in fixed openings to
being competitive in local tests.

## Data observations

`data/selfplay_rich.npz`:

- samples: 25,766
- center label rate: about 0.196
- score std: about 391k

`data/selfplay_rich_d7.npz`:

- samples: 38,154
- center label rate: about 0.177
- score std: about 431k
- more late-game and decisive positions

The `d7` data is not simply a better version of `d6`; it changes the state and
label distribution. Direct hard-label training on d7 did not improve practical
strength.

## Evaluation note

`scripts/play_submissions.py` now supports:

```bash
--random-opening-moves N
```

This matters because deterministic submissions with `--alternate-first` often
repeat the same two games. Random openings give a more useful local signal.

## Experiments run

### Distill iter1 with value scaling fixed, hidden 128

Output:

- `submission_ml_distill_iter1_vfix.py`

Fixed opening:

- vs `submission.py`: 2-2
- vs `submission_ml_f902_rich.py`: 2-2
- vs old `submission_ml_distill_iter1.py`: 2-2

Random opening, seed 10, `--random-opening-moves 2`, 8 games:

- vs `submission.py`: 3-5

### Distill iter1 with value scaling fixed, hidden 256

Output:

- `submission_ml_distill_iter1_vfix_h256.py`

Random opening, seed 10, `--random-opening-moves 2`, 8 games:

- vs `submission.py`: 3-5
- vs `submission_ml_f902_rich.py`: 5-3

This is the best clean candidate from this round.

### d7-from-mixed distill with value scaling fixed, hidden 128

Output:

- `submission_ml_distill_d7_from_mixed_vfix.py`

Random opening, seed 10, `--random-opening-moves 2`, 8 games:

- vs `submission.py`: 3-4-1
- vs `submission_ml_f902_rich.py`: 2-2 in fixed-opening test

Promising but not clearly better than the d6 distill route.

### d7-from-mixed distill with value scaling fixed, hidden 256

Output:

- `submission_ml_distill_d7_from_mixed_vfix_h256.py`

Random opening, seed 10, `--random-opening-moves 2`, 8 games:

- vs `submission.py`: 3-5
- vs `submission_ml_f902_rich.py`: 3-5
- vs `submission_ml_distill_iter1_vfix_h256.py`: 3-5

Not better than `submission_ml_distill_iter1_vfix_h256.py`.

### d6+d7 hard-label combined data

Output:

- data: `data/selfplay_rich_d6d7.npz`
- submission: `submission_ml_f902_rich_d6d7.py`

Random opening, seed 10, `--random-opening-moves 2`, 8 games:

- vs `submission.py`: 3-5

Fixed opening:

- vs `submission.py`: 0-4
- vs `submission_ml_f902_rich.py`: 2-2
- vs `submission_ml_distill_iter1_vfix.py`: 2-2

More hard-label data did not solve the gap.

### Root-policy ordering variant

A root-policy ordering variant was tested on the hidden-256 distill model.

Result:

- vs original afterstate-policy hidden-256 variant: 2-6
- vs `submission_ml_f902_rich.py`: 3-5

Conclusion: keep the original f902rich afterstate-policy ordering behavior.

### Fusion-weight variants

Variants around the default `policy=0.4, value=0.3, heuristic=0.3` were tested:

- `0.6 / 0.2 / 0.2`
- `0.5 / 0.3 / 0.2`
- `0.5 / 0.1 / 0.4`
- `0.3 / 0.5 / 0.2`

All lost 0-4 to `submission.py` in fixed-opening tests. Keep the default
f902rich fusion weights.

## Recommended next step

Use `submission_ml_distill_iter1_vfix_h256.py` as the current best ML-assisted
candidate and compare it on Kaggle if submission budget allows.

For further local work, the highest-value next experiments are:

1. Train 2-3 more hidden-256 distill seeds from `rl/data/distill_targets_iter1.npz`.
2. Pick by random-opening evaluation against `submission.py`,
   `submission_ml_f902_rich.py`, and the current hidden-256 candidate.
3. Do not spend more time on pure d7 hard-label data unless the student or target
   design changes.
