# ConnectX 初步模型方案

目标：做一个能体现机器学习课程内容、同时在 Kaggle ConnectX 中有竞争力的 agent。核心方向建议是“搜索 + 学习评估函数/策略”的混合系统，而不是只写手工规则或纯搜索。

## 1. 竞赛与约束理解

ConnectX 是一个可配置的连子棋环境。本仓库的默认配置是标准 7 列、6 行、4 连胜。Kaggle 提交通常要求 `submission.py` 中的 agent 函数自包含，运行时不能依赖本地额外文件。

快速开始 notebook 里也提示：提交 agent 应尽量封装完整；可用依赖以 Python 标准库、`numpy`、`scipy`、`gym`、CPU PyTorch 等为主。为了稳妥，我们最终提交优先做成标准库 + `numpy` 可运行。

## 2. 推荐路线

第一阶段：强基线

- 实现合法落子、胜负检测、立即胜利、必须阻挡、中心列偏好等基础策略。
- 实现 bitboard 或数组版 alpha-beta/negamax 搜索，带 transposition table、move ordering、迭代加深和时间控制。
- 建立本地评估脚本，固定对手包括 random、默认 negamax、我们自己的历史版本。

第二阶段：机器学习评估函数

- 用强搜索自博弈生成局面数据：输入为当前棋盘和当前玩家，标签为搜索后的最佳动作、胜负结果或 value 分数。
- 训练轻量模型：优先从 `logistic regression / linear value model / small MLP` 开始。
- 特征可以包括原始棋盘平面、当前玩家平面、每个方向的 2 连/3 连/威胁数量、中心控制、可立即赢/输等。
- 训练目标建议同时做 policy 和 value：policy 学最佳列，value 学局面胜率/搜索分数。

第三阶段：搜索与 ML 融合

- 用 policy 模型做 move ordering，让 alpha-beta 更快搜到好分支。
- 用 value 模型替换或增强叶子节点评估函数。
- 时间不足时直接用 policy 选择；时间充足时用迭代加深搜索。

第四阶段：提交压缩

- 如果模型很小，把权重直接写进 `submission.py`。
- 推理只依赖 `numpy` 或纯 Python list 运算。
- 提交前跑自对弈验证：`env.run([agent, agent])` 必须正常结束。

## 3. 为什么不建议纯强化学习起步

纯 RL 从零训练 ConnectX 容易样本效率低，短期课程项目里可能很难超过强搜索 baseline。更稳的做法是先用搜索生成高质量数据，再做 imitation learning / value learning。这样既有机器学习内容，也能快速形成可提交的强 agent。

## 4. 我建议的近期执行顺序

1. 搭建 `agents/`、`scripts/`、`notebooks/` 目录结构。
2. 写 `agents/baseline.py`：规则型 agent + 搜索型 agent。
3. 写 `scripts/evaluate_agents.py`：批量评估胜率、先后手表现、平均耗时。
4. 写 `scripts/generate_selfplay_data.py`：用搜索生成训练集。
5. 训练第一个轻量 value/policy 模型，并把它蒸馏到可提交 agent。

## 5. 评估指标

- 对 random 的胜率应接近 100%。
- 对默认 negamax 应明显优于随机，并尽快达到稳定压制。
- 自己不同版本之间做 round-robin，避免只对某个对手过拟合。
- 记录每步平均耗时和最大耗时，避免 Kaggle 超时。

## 6. 讨论点

- 课程更看重哪类 ML 内容：监督学习、强化学习、深度学习，还是可解释特征工程？
- 最终报告是否需要 ablation study：例如“纯搜索 vs ML 叶子评估 vs ML move ordering”。
- 是否允许提交中使用 PyTorch。如果不确定，最终用 `numpy` 推理更稳。
