# DQN强化学习项目

基于PyTorch的DQN（Deep Q-Network）强化学习实现，支持多种游戏环境，包含完整的训练、推理、收敛分析和可视化功能。

## 📋 目录

- [项目概述](#项目概述)
- [快速开始](#快速开始)
- [项目结构](#项目结构)
- [使用方法](#使用方法)
- [功能特性](#功能特性)
- [参数说明](#参数说明)
- [常见问题](#常见问题)

## 🎯 项目概述

本项目是一个完整的DQN强化学习实现，支持以下功能：

- **多环境支持**: Maze迷宫游戏和CartPole平衡游戏
- **命令行接口**: 统一的命令行参数控制
- **收敛分析**: 实时监控训练过程和收敛状态
- **可视化工具**: 生成训练过程折线图和统计报告
- **模型管理**: 自动保存和加载训练模型，只保留最优模型
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
irm https://astral.sh/uv/install.ps1 | iex
```

**macOS/Linux**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**验证安装**
```bash
uv --version
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
```

### 快速体验

```bash
# 训练CartPole（短时间示例）
uv run python main.py --game cartpole --mode train --episodes 50

# 推理测试（自动选择最优模型）
uv run python main.py --game cartpole --mode inference --episodes 5

# 查看结果
uv run python main.py --game cartpole --mode list-models
```

## 📁 项目结构

```
DQN/
├── main.py                    # 主程序入口，支持命令行参数
├── pyproject.toml             # 项目配置和依赖定义
├── requirements.txt           # pip依赖文件（备用）
├── uv.lock                    # 依赖锁定文件
├── README.md                  # 项目说明文档
├── .gitignore                 # Git忽略文件配置
│
├── framework/                 # 框架工具目录
│   ├── __init__.py           # 框架包初始化
│   ├── BaseAgent.py          # 基础智能体类
│   ├── convergence_analysis.py    # 收敛分析工具
│   └── plot_convergence.py        # 图表生成工具
│
├── games/                     # 游戏模块目录
│   ├── Maze/                  # 迷宫游戏模块
│   │   ├── MazeEnv.py        # 迷宫环境定义
│   │   └── MazeAgent.py      # 迷宫DQN智能体
│   └── CartPole/             # CartPole游戏模块
│       └── CartPoleAgent.py  # CartPole DQN智能体
│
├── docs/                      # 文档目录
│   └── _img/                 # 项目图片资源
│       ├── maze.png          # 迷宫游戏截图
│       ├── cost.png          # 损失曲线图
│       └── target.jpg        # 目标图片
│
├── models/                    # 全局模型存储目录（只保留最优模型）
│   ├── CartPole_dqn_best.pth # CartPole全局最优模型
│   └── Maze_dqn_best.pth     # Maze全局最优模型
│
├── outputs/                   # 输出文件目录（运行时生成）
│   ├── Maze/                 # 迷宫游戏输出
│   │   ├── 1/               # 第1次训练输出
│   │   │   ├── convergence_analysis/  # 收敛分析数据
│   │   │   ├── reports/     # 报告文件
│   │   │   └── process_models/   # 过程模型文件
│   │   └── ...              # 更多训练输出
│   └── CartPole/            # CartPole输出
│       └── ...              # 训练输出结构同上
│
└── .venv/                    # 虚拟环境目录（uv自动生成）
```

## 🛠️ 使用方法

### 常用命令

#### 训练
```bash
uv run python main.py --game cartpole --mode train --episodes 200
uv run python main.py --game maze --mode train --episodes 100
```

#### 推理
```bash
uv run python main.py --game cartpole --mode inference --episodes 10
uv run python main.py --game maze --mode inference --episodes 5
```

#### 查看
```bash
uv run python main.py --game cartpole --mode list-models
uv run python main.py --game cartpole --mode list-outputs
```

## ✨ 功能特性

### 1. 收敛分析
- **实时监控**: 自动收集训练数据，计算收敛指标
- **智能判断**: 自动判断模型是否收敛
- **详细报告**: 生成包含指标和建议的分析报告

### 2. 可视化工具
- **文本图表**: 无需额外依赖的文本格式图表
- **图形化图表**: 高质量的PNG图片（需要matplotlib）
- **多种类型**: 奖励、损失、成功率、综合图表

### 3. 模型管理
- **自动保存**: 训练过程中自动保存最优模型
- **全局最优**: 跨训练会话的全局最优模型管理
- **智能清理**: 自动清理多余模型文件

### 4. 推理测试
- **自动选择**: 自动选择最优模型进行推理
- **性能评估**: 详细的推理统计和成功率分析
- **可视化**: 推理过程可视化展示

## 📋 参数说明

### 主程序参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--game` | str | 必需 | 游戏类型：`maze` 或 `cartpole` |
| `--mode` | str | 必需 | 运行模式：`train`, `inference`, `list-models`, `list-outputs` |
| `--episodes` | int | 400 | 训练或推理的episode数量 |
| `--model` | str | - | 推理时使用的模型文件名（可选，默认自动选择最优模型） |
| `--output-dir` | str | `outputs` | 输出文件目录 |
| `--model-dir` | str | `models` | 模型文件目录 |
| `--save-interval` | int | 50 | 模型保存间隔（episode数） |
| `--render` | bool | True | 训练时显示可视化动画窗口 |

### 游戏环境

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

**Q: 如何激活虚拟环境？**
A: 使用 `uv shell` 命令激活虚拟环境，或使用 `uv run python` 直接在虚拟环境中运行脚本。

**Q: 安装依赖失败怎么办？**
A: 确保已正确安装uv，然后运行 `uv sync` 重新安装依赖。

### 训练问题

**Q: 训练过程中没有收敛怎么办？**
A: 可以尝试调整学习率、增加训练轮次、调整网络结构等。查看收敛分析报告获取具体建议。

**Q: 如何判断模型是否训练完成？**
A: 查看收敛分析报告，当奖励趋于稳定且趋势平缓时，模型通常已经收敛。

### 推理问题

**Q: 推理时找不到模型文件？**
A: 请先完成一次训练，模型会自动保存在 `models/` 目录下。

**Q: 推理结果不理想？**
A: 确保使用的是训练完成的模型，可以尝试重新训练或调整超参数。

## 📚 参考资料

- [PyTorch官方文档](https://pytorch.org/docs/)
- [Gymnasium文档](https://gymnasium.farama.org/)
- [DQN论文](https://arxiv.org/abs/1312.5602)
- [UV文档](https://docs.astral.sh/uv/)

## 🤝 贡献

欢迎提交Issue和Pull Request来改进项目！

## �� 许可证

本项目采用MIT许可证。