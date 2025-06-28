# DQN强化学习项目

基于PyTorch的DQN（Deep Q-Network）强化学习实现，支持多种游戏环境，包含完整的训练、推理、收敛分析和可视化功能。

## 📋 目录

- [项目概述](#项目概述)
- [快速开始](#快速开始)
- [项目结构](#项目结构)
- [使用方法](#使用方法)
- [收敛分析功能](#收敛分析功能)
- [折线图生成功能](#折线图生成功能)
- [参数说明](#参数说明)
- [常见问题](#常见问题)

## 🎯 项目概述

本项目是一个完整的DQN强化学习实现，支持以下功能：

- **多环境支持**: Maze迷宫游戏和CartPole平衡游戏
- **命令行接口**: 统一的命令行参数控制
- **收敛分析**: 实时监控训练过程和收敛状态
- **可视化工具**: 生成训练过程折线图和统计报告
- **模型管理**: 自动保存和加载训练模型，只保留最优和最后模型
- **推理测试**: 支持模型推理和性能评估

## 🚀 快速开始

### 环境准备

#### 系统要求
- Windows 10/11, macOS, Linux
- Python 3.8+
- CUDA 11.6 (可选，用于GPU加速)

#### 安装uv

**Windows**
```powershell
# 方法1: 使用PowerShell安装（推荐）
irm https://astral.sh/uv/install.ps1 | iex

# 方法2: 使用pip安装
pip install uv

# 方法3: 使用winget安装
winget install astral-sh.uv
```

**macOS/Linux**
```bash
# 方法1: 使用官方安装脚本（推荐）
curl -LsSf https://astral.sh/uv/install.sh | sh

# 方法2: 使用pip安装
pip install uv

# 方法3: 使用Homebrew (macOS)
brew install uv
```

**验证安装**
```bash
uv --version
```

#### 安装项目

**使用uv（推荐）**
```bash
# 1. 克隆项目
git clone <your-repo-url>
cd DQN

# 2. 创建虚拟环境并安装依赖
uv sync

# 3. 激活虚拟环境
uv shell

# 4. 验证安装
python -c "import torch; print(torch.__version__)"
python -c "import gymnasium; print(gymnasium.__version__)"
```

**使用pip（备用方案）**
```bash
# 1. 克隆项目
git clone <your-repo-url>
cd DQN

# 2. 创建虚拟环境
python -m venv .venv

# 3. 激活虚拟环境
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# 4. 安装依赖
pip install -r requirements.txt

# 5. 验证安装
python -c "import torch; print(torch.__version__)"
python -c "import gymnasium; print(gymnasium.__version__)"
```

### 快速体验

#### 使用交互式启动脚本
```bash
uv run python start.py
```

#### 使用命令行

**使用uv（推荐）**
```bash
# 训练CartPole（短时间示例）
uv run python main.py --game cartpole --mode train --episodes 50

# 推理测试（自动选择最优模型）
uv run python main.py --game cartpole --mode inference --episodes 5

# 查看结果
uv run python main.py --game cartpole --mode list-models
```

**使用pip（需要先激活虚拟环境）**
```bash
python main.py --game cartpole --mode train --episodes 50
python main.py --game cartpole --mode inference --episodes 5
python main.py --game cartpole --mode list-models
```

## 📁 项目结构

```
DQN/
├── main.py                    # 主程序入口，支持命令行参数
├── start.py                   # 交互式启动脚本
├── pyproject.toml             # 项目配置和依赖定义
├── requirements.txt           # pip依赖文件（备用）
├── uv.lock                    # 依赖锁定文件
├── uv.toml                    # uv配置文件
├── README.md                  # 项目说明文档
├── .gitignore                 # Git忽略文件配置
├── .uv/                       # uv配置目录
│   └── config.toml           # uv配置文件
│
├── framework/                 # 框架工具目录
│   ├── __init__.py           # 框架包初始化
│   ├── convergence_analysis.py    # 收敛分析工具
│   ├── plot_convergence.py        # 图表生成工具
│   ├── convergence_analysis/      # 收敛分析数据目录
│   └── _img/                     # 项目图片资源
│       ├── maze.png              # 迷宫游戏截图
│       ├── cost.png              # 损失曲线图
│       └── target.jpg            # 目标图片
│
├── games/                     # 游戏模块目录
│   ├── Maze/                  # 迷宫游戏模块
│   │   ├── MazeEnv.py        # 迷宫环境定义
│   │   ├── MazeAgent.py      # 迷宫DQN智能体
│   └── CartPole/             # CartPole游戏模块
│       └── CartPole.py       # CartPole DQN智能体
│
├── models/                    # 全局模型存储目录（只保留最优和最后模型）
│   ├── cartpole_dqn_best.pth         # CartPole全局最优模型
│   ├── cartpole_dqn_final.pth        # CartPole全局最后模型
│   ├── maze_dqn_best.pth              # Maze全局最优模型
│   └── maze_dqn_final.pth             # Maze全局最后模型
│
├── outputs/                   # 输出文件目录（运行时生成）
│   ├── maze/                 # 迷宫游戏输出
│   │   ├── convergence_analysis/  # 收敛分析数据
│   │   ├── plots/            # 图表文件
│   │   ├── reports/          # 报告文件
│   │   ├── logs/             # 日志文件
│   │   └── process_models/   # 过程模型文件
│   └── cartpole/             # CartPole输出
│       ├── convergence_analysis/
│       ├── plots/
│       ├── reports/
│       ├── logs/
│       └── process_models/   # 过程模型文件
│
└── .venv/                    # 虚拟环境目录（uv自动生成）
```

### 核心文件说明

#### 主程序文件
- **main.py**: 项目主入口，提供统一的命令行接口和启动函数
- **start.py**: 交互式启动脚本，提供友好的用户界面

#### 框架工具
- **framework/**: 框架工具目录
  - convergence_analysis.py: 收敛分析工具
  - plot_convergence.py: 图表生成工具

#### 游戏模块
- **games/**: 游戏模块目录
  - **Maze/**: 迷宫游戏模块
    - MazeEnv.py: 迷宫环境定义
    - MazeAgent.py: DQN智能体实现
  - **CartPole/**: CartPole游戏模块
    - CartPole.py: CartPole智能体实现

## 🛠️ UV 依赖管理

本项目使用 [uv](https://github.com/astral-sh/uv) 进行依赖管理和虚拟环境构建。

### 常用uv命令

```bash
uv sync                # 安装/同步依赖
uv add 包名            # 添加新依赖
uv remove 包名         # 移除依赖
uv lock --upgrade      # 升级所有依赖
uv run python ...      # 在虚拟环境中运行脚本
uv shell               # 激活虚拟环境
```

> 项目依赖已全部声明在 pyproject.toml，锁定文件为 uv.lock。

## 🔧 使用方法

### 常用命令

#### 训练命令
```bash
# 使用uv（推荐）
uv run python main.py --game cartpole --mode train --episodes 200

# 使用pip（需要先激活虚拟环境）
python main.py --game cartpole --mode train --episodes 200
```

#### 推理命令
```bash
# 使用uv（推荐）
uv run python main.py --game cartpole --mode inference --episodes 10

# 使用pip（需要先激活虚拟环境）
python main.py --game cartpole --mode inference --episodes 10
```

#### 查看命令
```bash
# 使用uv（推荐）
uv run python main.py --game cartpole --mode list-models

# 使用pip（需要先激活虚拟环境）
python main.py --game cartpole --mode list-models
```

## 📊 收敛分析功能

### 功能概述

项目集成了完整的训练过程收敛分析功能，帮助您监控和分析训练过程中的收敛情况，判断模型是否已经收敛，并提供智能的训练建议。

### 主要功能特性

#### 1. 实时收敛监控
- **自动数据收集**: 训练过程中自动收集每个episode的数据
- **实时指标计算**: 计算奖励、损失、成功率等关键指标
- **定期分析报告**: 每50个episode（CartPole）或每5个episode（Maze）生成分析报告

#### 2. 多维度收敛指标
- **奖励统计**: 平均奖励、标准差、最高/最低奖励、奖励稳定性
- **收敛判断**: 收敛比率、奖励趋势、自动收敛状态判断
- **损失分析**: 平均损失、损失趋势、损失稳定性
- **成功率统计**: 总体成功率、最近成功率

#### 3. 智能训练建议
- **收敛状态判断**: 自动判断模型是否已收敛
- **问题诊断**: 识别训练中的问题（如奖励下降、波动过大等）
- **优化建议**: 提供具体的超参数调整建议

### 使用方法

#### 训练时自动分析
**使用uv（推荐）**
```bash
# CartPole环境
uv run python main.py --game cartpole --mode train --episodes 100

# Maze环境
uv run python main.py --game maze --mode train --episodes 50
```

**使用pip（需要先激活虚拟环境）**
```bash
# CartPole环境
python main.py --game cartpole --mode train --episodes 100

# Maze环境
python main.py --game maze --mode train --episodes 50
```

训练过程中会自动：
- 收集每个episode的训练数据
- 计算收敛指标
- 定期生成分析报告
- 保存分析数据到JSON文件

### 收敛判断标准

#### 自动收敛判断
系统会自动判断模型是否收敛，判断标准：

1. **奖励稳定性**: 最近N个episode的奖励标准差 < 0.1 × 平均奖励
2. **奖励趋势**: 最近奖励的线性回归斜率绝对值 < 0.01
3. **综合判断**: 同时满足稳定性和趋势条件

### 输出文件说明

#### 1. 收敛报告文件
- **格式**: `.txt` 文本文件
- **内容**: 详细的收敛分析报告，包含所有指标和训练建议
- **示例**: `convergence_report_episode_50.txt`

#### 2. 分析数据文件
- **格式**: `.json` JSON文件
- **内容**: 原始训练数据和计算出的收敛指标
- **示例**: `convergence_data_episode_50.json`

## 📈 折线图生成功能

### 功能概述

折线图生成工具用于将DQN训练过程中收集的收敛数据直接生成可视化折线图，支持文本格式和图形化两种输出方式。

### 功能特点

#### 1. 双重输出格式
- **文本格式图表**: 无需额外依赖，直接生成可读的文本报告
- **图形化折线图**: 需要matplotlib库，生成高质量的PNG图片

#### 2. 多种图表类型
- **奖励折线图**: 显示训练过程中奖励值的变化趋势
- **损失折线图**: 显示训练过程中损失值的变化趋势
- **成功率折线图**: 显示训练过程中成功率的变化趋势
- **Epsilon折线图**: 显示探索率衰减曲线
- **综合折线图**: 四合一综合分析图表

### 使用方法

#### 基本使用
```bash
# 使用uv（推荐）
uv run python framework/plot_convergence.py

# 使用pip（需要先激活虚拟环境）
python framework/plot_convergence.py
```

## 📋 参数说明

### 主程序参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--game` | str | 必需 | 游戏类型：`maze` 或 `cartpole` |
| `--mode` | str | 必需 | 运行模式：`train`, `inference`, `list-models`, `list-outputs` |
| `--episodes` | int | 100 | 训练或推理的episode数量 |
| `--model` | str | - | 推理时使用的模型文件名（可选，默认自动选择最优模型） |
| `--output-dir` | str | `outputs/` | 输出文件目录 |
| `--model-dir` | str | `models/` | 模型文件目录 |
| `--save-interval` | int | 50 | 模型保存间隔（episode数） |
| `--render` | bool | True | 训练时显示可视化动画窗口（默认启用） |

### 游戏特定参数

#### Maze游戏
- **环境**: 自定义迷宫环境
- **状态空间**: 迷宫位置坐标
- **动作空间**: 4个方向（上、下、左、右）
- **奖励**: 到达目标+100，撞墙-1，每步-0.1

#### CartPole游戏
- **环境**: Gymnasium CartPole-v1
- **状态空间**: 4维连续状态
- **动作空间**: 2个动作（左、右）
- **奖励**: 每步+1，失败时结束

## ❓ 常见问题

### 环境问题

**Q: 如何激活uv虚拟环境？**
A: 使用 `uv shell` 命令激活虚拟环境，或使用 `uv run python` 直接在虚拟环境中运行脚本。

**Q: 如何激活pip虚拟环境？**
A: Windows使用 `.venv\Scripts\activate`，macOS/Linux使用 `source .venv/bin/activate`。

**Q: 安装依赖失败怎么办？**
A: 确保已正确安装uv，然后运行 `uv sync` 重新安装依赖。如果使用pip，运行 `pip install -r requirements.txt`。

**Q: uv命令找不到怎么办？**
A: 可以按照README中的安装说明重新安装uv，或使用pip作为备用方案。

### 训练问题

**Q: 训练过程中没有收敛怎么办？**
A: 可以尝试调整学习率、增加训练轮次、调整网络结构等。查看收敛分析报告获取具体建议。

**Q: 如何判断模型是否训练完成？**
A: 查看收敛分析报告，当奖励趋于稳定且趋势平缓时，模型通常已经收敛。

### 推理问题

**Q: 推理时找不到模型文件？**
A: 请先完成一次训练，模型会自动保存在 `models/` 目录下。可用 `uv run python main.py --game maze --mode list-models` 查看所有模型文件。

**Q: 推理结果不理想？**
A: 确保使用的是训练完成的模型，可以尝试重新训练或调整超参数。

### 输出问题

**Q: 输出文件在哪里？**
A: 默认在 `outputs/游戏名/` 目录下，使用 `uv run python main.py --game maze --mode list-outputs` 或 `python main.py --game maze --mode list-outputs` 查看。

**Q: 如何生成图表？**
A: 使用 `uv run python framework/plot_convergence.py` 或 `python framework/plot_convergence.py` 生成训练过程图表。

## 📚 参考资料

- [PyTorch官方文档](https://pytorch.org/docs/)
- [Gymnasium文档](https://gymnasium.farama.org/)
- [DQN论文](https://arxiv.org/abs/1312.5602)
- [UV文档](https://docs.astral.sh/uv/)

## 🤝 贡献

欢迎提交Issue和Pull Request来改进项目！

## �� 许可证

本项目采用MIT许可证。