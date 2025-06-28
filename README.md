# DQN强化学习项目

基于PyTorch的DQN（Deep Q-Network）强化学习实现，支持多种游戏环境，包含完整的训练、推理、收敛分析和可视化功能。

## 📋 目录

- [项目概述](#项目概述)
- [快速开始](#快速开始)
- [项目结构](#项目结构)
- [UV依赖管理](#uv依赖管理使用说明)
- [使用方法](#使用方法)
- [收敛分析功能](#收敛分析功能)
- [折线图生成功能](#折线图生成功能)
- [模型导出和推理](#模型导出和推理)
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
- Python 3.7+
- CUDA 11.6 (可选，用于GPU加速)

#### 安装uv

**Windows**
```bash
pip install uv
# 或使用官方安装脚本
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

**macOS/Linux**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
# 或使用 pip
pip install uv
```

#### 安装项目

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
python -c "import gym; print(gym.__version__)"
```

### 快速体验

#### 使用交互式启动脚本
```bash
# 启动交互式界面
uv run python start.py
```

#### 使用命令行

```bash
# 训练迷宫游戏（短时间示例）
uv run python main.py --game maze --mode train --episodes 50

# 训练CartPole（短时间示例）
uv run python main.py --game cartpole --mode train --episodes 50

# 推理测试
uv run python main.py --game maze --mode inference --model maze_dqn_final.pth --episodes 5

# 查看结果
uv run python main.py --game maze --mode list-models
uv run python main.py --game maze --mode list-outputs

# 运行完整示例
uv run python framework/examples.py
```

## 📁 项目结构

```
DQN/
├── main.py                    # 主程序入口，支持命令行参数
├── start.py                   # 交互式启动脚本
├── pyproject.toml             # 项目配置和依赖定义
├── uv.lock                    # 依赖锁定文件
├── uv.toml                    # uv配置文件
├── training_counter.json      # 训练计数器
├── README.md                  # 项目说明文档
├── .gitignore                 # Git忽略文件配置
├── .uv/                       # uv配置目录
│   └── config.toml           # uv配置文件
│
├── framework/                 # 框架工具目录
│   ├── __init__.py           # 框架包初始化
│   ├── convergence_analysis.py    # 收敛分析工具
│   ├── plot_convergence.py        # 图表生成工具
│   ├── examples.py                # 使用示例脚本
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
│   │   └── models/           # 迷宫模型存储目录
│   │
│   └── CartPole/             # CartPole游戏模块
│       └── CartPole.py       # CartPole DQN智能体
│
├── models/                    # 全局模型存储目录
│   ├── maze/                 # 迷宫模型
│   └── cartpole/             # CartPole模型
│
├── outputs/                   # 输出文件目录（运行时生成）
│   ├── maze/                 # 迷宫游戏输出
│   │   ├── convergence_analysis/  # 收敛分析数据
│   │   ├── plots/            # 图表文件
│   │   ├── reports/          # 报告文件
│   │   └── logs/             # 日志文件
│   └── cartpole/             # CartPole输出
│       ├── convergence_analysis/
│       ├── plots/
│       ├── reports/
│       └── logs/
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
  - examples.py: 使用示例脚本

#### 游戏模块
- **games/**: 游戏模块目录
  - **Maze/**: 迷宫游戏模块
    - MazeEnv.py: 迷宫环境定义
    - MazeAgent.py: DQN智能体实现
  - **CartPole/**: CartPole游戏模块
    - CartPole.py: CartPole智能体实现

## 🛠️ UV 依赖管理与环境构建

本项目推荐使用 [uv](https://github.com/astral-sh/uv) 进行依赖管理和虚拟环境构建，极大提升安装速度和依赖一致性。

### 安装uv

**Windows**
```powershell
irm https://astral.sh/uv/install.ps1 | iex
```
**macOS/Linux**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 安装依赖与环境构建

```bash
# 1. 克隆项目
git clone <your-repo-url>
cd DQN

# 2. 安装依赖并自动创建虚拟环境
uv sync

# 3. 激活虚拟环境
uv shell

# 4. 运行主程序
uv run python main.py --game maze --mode train
```

### 常用uv命令

```bash
uv sync                # 安装/同步依赖
uv add 包名            # 添加新依赖
uv remove 包名         # 移除依赖
uv lock --upgrade      # 升级所有依赖
uv run python ...      # 在虚拟环境中运行脚本
uv shell               # 激活虚拟环境
```

> 项目依赖已全部声明在 pyproject.toml，锁定文件为 uv.lock，无需 requirements.txt。

## 🔧 使用方法

### 常用命令

#### 训练命令
```bash
# 基本训练
python main.py --game maze --mode train

# 自定义参数训练
python main.py --game maze --mode train --episodes 200 --save-interval 25

# 自定义输出目录
python main.py --game maze --mode train --output-dir my_outputs --model-dir my_models
```

#### 推理命令
```bash
# 基本推理
python main.py --game maze --mode inference --model maze_dqn_final.pth

# 自定义推理参数
python main.py --game maze --mode inference --model maze_dqn_final.pth --episodes 10
```

#### 查看命令
```bash
# 查看帮助
python main.py --help

# 查看模型列表
python main.py --game maze --mode list-models

# 查看输出文件
python main.py --game maze --mode list-outputs
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

#### 4. 数据管理
- **自动保存**: 训练过程中自动保存分析数据
- **报告生成**: 生成详细的文本分析报告
- **数据加载**: 支持加载已有数据进行后续分析

### 使用方法

#### 训练时自动分析
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

#### 独立使用分析工具
```python
from convergence_analysis import ConvergenceAnalyzer

# 创建分析器
analyzer = ConvergenceAnalyzer(window_size=50)

# 添加训练数据
analyzer.add_episode_data(
    episode=1,
    reward=25.5,
    loss=0.123,
    steps=200,
    success=True,
    epsilon=0.8
)

# 生成分析报告
report = analyzer.generate_convergence_report()
print(report)

# 保存分析数据
analyzer.save_analysis_data("my_analysis.json")
```

### 收敛判断标准

#### 自动收敛判断
系统会自动判断模型是否收敛，判断标准：

1. **奖励稳定性**: 最近N个episode的奖励标准差 < 0.1 × 平均奖励
2. **奖励趋势**: 最近奖励的线性回归斜率绝对值 < 0.01
3. **综合判断**: 同时满足稳定性和趋势条件

#### 手动判断建议
- **已收敛**: 奖励趋于稳定，趋势平缓，可以考虑停止训练
- **未收敛**: 奖励仍在波动或呈下降趋势，建议继续训练

### 输出文件说明

#### 1. 收敛报告文件
- **格式**: `.txt` 文本文件
- **内容**: 详细的收敛分析报告，包含所有指标和训练建议
- **示例**: `convergence_report_episode_50.txt`

#### 2. 分析数据文件
- **格式**: `.json` JSON文件
- **内容**: 原始训练数据和计算出的收敛指标
- **示例**: `convergence_data_episode_50.json`

### 训练建议

#### 基于收敛分析的建议

**模型已收敛**
- ✅ 可以考虑停止训练
- ✅ 保存最终模型进行推理测试
- ✅ 可以尝试调整超参数进行进一步优化

**模型未收敛**
- ⚠️ 建议继续训练
- ⚠️ 如果奖励呈下降趋势，可能需要调整学习率
- ⚠️ 如果奖励波动较大，可能需要增加训练轮次
- ⚠️ 考虑调整网络结构或超参数

#### 超参数调整建议

**学习率调整**
- 如果损失下降缓慢：适当增加学习率
- 如果损失波动很大：适当减小学习率

**探索策略调整**
- 如果成功率低：增加epsilon衰减时间
- 如果收敛慢：调整epsilon初始值

**网络结构调整**
- 如果性能不佳：增加隐藏层节点数
- 如果过拟合：减少网络复杂度

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
- **综合统计报告**: 详细的数值分析和收敛判断

#### 3. 智能分析功能
- 移动平均计算
- 趋势分析
- 收敛判断
- 训练建议

### 使用方法

#### 基本使用
```bash
# 使用默认测试数据生成折线图
python plot_convergence.py

# 为指定数据文件生成折线图
python plot_convergence.py your_data.json

# 指定输出目录
python plot_convergence.py your_data.json output_plots

# 显示图形化图表（需要matplotlib）
python plot_convergence.py your_data.json output_plots show
```

#### 数据文件格式
折线图工具需要JSON格式的收敛数据文件，包含以下字段：

```json
{
    "episode_rewards": [20.5, 25.3, 30.1, ...],      // 每个episode的奖励值
    "episode_losses": [0.5, 0.4, 0.3, ...],          // 每个episode的损失值（可选）
    "episode_successes": [0, 1, 1, 0, ...],          // 每个episode是否成功（可选）
    "episode_epsilons": [0.9, 0.8, 0.7, ...],        // 每个episode的epsilon值（可选）
    "convergence_metrics": {                          // 收敛指标（可选）
        "reward_mean": 45.2,
        "reward_std": 8.5,
        "reward_stability": 3.2,
        "convergence_ratio": 1.05,
        "reward_trend": 0.008,
        "is_converged": true
    }
}
```