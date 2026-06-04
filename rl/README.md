# ConnectX 的 AlphaZero 风格训练路线

这个目录是一条新的研究分支，和当前已经取得最高分的 `f902rich`
提交路线并行存在。当前稳定基线仍然是 `scripts/train_model.py` 生成的
轻量提交文件；`rl/` 目录用于训练更强的离线 policy-value 模型，再考虑
蒸馏回轻量提交 agent。

## 环境

继续使用现有的 `kaggle` conda 环境：

```bash
conda activate kaggle
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

在本机 CPU 上可以做小规模验证。正式训练建议放到有 GPU 的机器上跑，
命令不需要改，默认会自动使用 CUDA。

## 总体思路

这条路线不是纯强化学习从零开始，而是：

1. 先用现在已有的 `data/selfplay_rich.npz` 做监督预训练。
2. 用预训练好的神经网络指导 MCTS 自博弈。
3. 用 MCTS 产生的新数据继续训练网络。
4. 用 arena 和当前 `submission_ml_best.py` / `submission_ml_f902_rich.py` 对战。
5. 如果离线模型变强，再把它蒸馏成轻量提交模型。

也就是说，我们保留当前强搜索路线，同时尝试更高上限的 AlphaZero /
Expert Iteration 方法。

## 第一阶段：监督预训练

直接使用已有数据，不需要重新生成：

```bash
python -m rl.train_supervised \
  --data data/selfplay_rich.npz \
  --output rl/checkpoints/supervised_c96_b6.pt \
  --epochs 80 \
  --channels 96 \
  --blocks 6 \
  --batch-size 512 \
  --policy-targets hard
```

如果在 GPU 上训练，可以直接用更大的模型：

```bash
python -m rl.train_supervised \
  --data data/selfplay_rich.npz \
  --output rl/checkpoints/supervised_c128_b8.pt \
  --epochs 120 \
  --channels 128 \
  --blocks 8 \
  --batch-size 1024 \
  --policy-targets hard
```

默认推荐 `--policy-targets hard`。它使用 `moves` 里的最佳列标签，和我们当前
表现最好的 f902rich 思路一致。`--policy-targets soft --policy-temperature 200`
可以作为消融实验，但不是主线。

## 第二阶段：MCTS 自博弈

这一步会比较慢，建议在 GPU 上跑：

```bash
python -m rl.self_play \
  --checkpoint rl/checkpoints/supervised_c96_b6.pt \
  --output rl/data/selfplay_az_c96_b6_s160.npz \
  --games 200 \
  --simulations 160
```

更强但更慢的版本：

```bash
python -m rl.self_play \
  --checkpoint rl/checkpoints/supervised_c128_b8.pt \
  --output rl/data/selfplay_az_c128_b8_s320.npz \
  --games 500 \
  --simulations 320
```

参数含义：

- `--checkpoint`: 用哪个神经网络做 MCTS 引导。
- `--games`: 自博弈局数。
- `--simulations`: 每一步 MCTS 模拟次数，越大越强也越慢。
- `--output`: 保存自博弈数据的位置。

## 第三阶段：用自博弈数据继续训练

```bash
python -m rl.train_alphazero \
  --init-checkpoint rl/checkpoints/supervised_c96_b6.pt \
  --data rl/data/selfplay_az_c96_b6_s160.npz \
  --output rl/checkpoints/az_c96_b6_iter1.pt \
  --epochs 30 \
  --batch-size 512
```

可以反复迭代：

```bash
python -m rl.self_play \
  --checkpoint rl/checkpoints/az_c96_b6_iter1.pt \
  --output rl/data/selfplay_az_iter2.npz \
  --games 200 \
  --simulations 160

python -m rl.train_alphazero \
  --init-checkpoint rl/checkpoints/az_c96_b6_iter1.pt \
  --data rl/data/selfplay_az_iter2.npz \
  --output rl/checkpoints/az_c96_b6_iter2.pt \
  --epochs 30
```

## 第四阶段：和当前 best 对战

完整 MCTS agent 默认太慢，不适合直接提交 Kaggle，但适合用来判断离线模型
有没有变强：

```bash
python -m rl.arena \
  --checkpoint rl/checkpoints/az_c96_b6_iter1.pt \
  --opponent submission_ml_best.py \
  --episodes 4 \
  --simulations 80
```

如果这个模型能稳定击败或打平当前 best，说明它有蒸馏价值。

## 第五阶段：导出蒸馏目标

当某个 checkpoint 明显更强时，可以把它在现有 rich 数据上的预测导出来，
后续用来训练一个更轻量的提交模型：

```bash
python -m rl.export_policy_samples \
  --checkpoint rl/checkpoints/az_c96_b6_iter1.pt \
  --data data/selfplay_rich.npz \
  --output rl/data/distill_targets_iter1.npz
```

## 实用建议

- 不要立刻用 `rl/` 取代当前 `f902rich`，它现在是研究分支。
- Kaggle 提交端大概率还是要轻量化，完整 PyTorch + MCTS 通常太慢。
- 如果训练时间有限，优先跑 `supervised_c128_b8.pt` 加一轮 MCTS 自博弈。
- 每次迭代后都要和 `submission_ml_best.py`、`submission_ml_f902_rich.py` 比赛。
- 真正有用的最终产物，通常是「强离线模型蒸馏出来的小模型」，而不是直接提交大模型。
