# ConnectX 环境安装说明

本文档记录本项目在本机验证通过的 Kaggle ConnectX 开发环境安装方式。推荐组员统一创建名为 `kaggle` 的 conda 环境。

## 1. 创建 Conda 环境

```bash
conda create -y -n kaggle python=3.8 pip
conda activate kaggle
```

选择 Python 3.8 是为了兼容本仓库中的旧版 `kaggle-environments-0.1.4` 源码，同时保留后续安装训练工具的空间。

## 2. 安装基础依赖

先安装 `jsonschema`，因为 `kaggle-environments-0.1.4/setup.py` 在生成包元数据时会提前 import 它。

```bash
python -m pip install jsonschema numpy scipy gym jupyter ipykernel
```

## 3. 安装本地 Kaggle Environments

在仓库根目录执行：

```bash
python -m pip install -e ./kaggle-environments-0.1.4
```

本项目使用 editable 安装，方便我们后续调试本地环境源码。如果只想固定安装，也可以去掉 `-e`。

## 4. 注册 Jupyter Kernel

```bash
python -m ipykernel install --user --name kaggle --display-name "Python (kaggle)"
```

之后打开 `connectx-getting-started.ipynb` 时，选择 kernel：`Python (kaggle)`。

## 5. 验证安装

```bash
python -c "from kaggle_environments import make, evaluate, version; print('kaggle_environments', version); env=make('connectx', debug=True); print(env.name, env.version); print(env.render(mode='ansi')); print('rewards', evaluate('connectx', ['random', 'negamax'], num_episodes=3)); print('ok')"
```

本机验证输出要点：

```text
kaggle_environments 0.1.4
connectx 1.0.0
rewards [[0, 1], [0, 1], [0, 1]]
ok
```

## 6. 常见问题

如果安装本地包时报错 `ModuleNotFoundError: No module named 'jsonschema'`，说明跳过了第 2 步。先执行：

```bash
python -m pip install jsonschema
```

然后重新执行：

```bash
python -m pip install -e ./kaggle-environments-0.1.4
```

## 7. 当前环境关键版本

本机已验证的关键包版本：

```text
Python 3.8.20
kaggle-environments 0.1.4
jsonschema 4.23.0
numpy 1.24.4
scipy 1.10.1
gym 0.26.2
jupyter 1.1.1
ipykernel 6.29.5
```
