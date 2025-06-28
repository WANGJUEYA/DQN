#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试模型推理功能
"""

import os
import sys
import torch
import numpy as np

def test_model_save_load():
    """测试模型保存和加载功能"""
    print("=" * 50)
    print("测试模型保存和加载功能")
    
    # 检查CartPole模型
    cartpole_model_path = "CartPole/models/cartpole_dqn_final.pth"
    if os.path.exists(cartpole_model_path):
        print(f"✓ CartPole模型存在: {cartpole_model_path}")
        
        # 尝试加载模型
        try:
            checkpoint = torch.load(cartpole_model_path, map_location='cpu')
            print("✓ CartPole模型可以正常加载")
            print(f"  包含的键: {list(checkpoint.keys())}")
            
            if 'hyperparameters' in checkpoint:
                hyperparams = checkpoint['hyperparameters']
                print(f"  状态数: {hyperparams.get('N_STATES', 'N/A')}")
                print(f"  动作数: {hyperparams.get('N_ACTIONS', 'N/A')}")
        except Exception as e:
            print(f"✗ CartPole模型加载失败: {e}")
    else:
        print(f"✗ CartPole模型不存在: {cartpole_model_path}")
    
    # 检查Maze模型
    maze_model_path = "Maze/models/maze_dqn_final.pth"
    if os.path.exists(maze_model_path):
        print(f"✓ Maze模型存在: {maze_model_path}")
        
        # 尝试加载模型
        try:
            checkpoint = torch.load(maze_model_path, map_location='cpu')
            print("✓ Maze模型可以正常加载")
            print(f"  包含的键: {list(checkpoint.keys())}")
            
            if 'hyperparameters' in checkpoint:
                hyperparams = checkpoint['hyperparameters']
                print(f"  状态数: {hyperparams.get('N_STATES', 'N/A')}")
                print(f"  动作数: {hyperparams.get('N_ACTIONS', 'N/A')}")
        except Exception as e:
            print(f"✗ Maze模型加载失败: {e}")
    else:
        print(f"✗ Maze模型不存在: {maze_model_path}")

def test_inference_tool():
    """测试推理工具"""
    print("\n" + "=" * 50)
    print("测试推理工具")
    
    # 检查推理工具是否存在
    if os.path.exists("model_inference.py"):
        print("✓ 推理工具存在: model_inference.py")
        
        # 检查是否可以导入
        try:
            from model_inference import ModelInference, DQNNet
            print("✓ 推理工具可以正常导入")
        except ImportError as e:
            print(f"✗ 推理工具导入失败: {e}")
    else:
        print("✗ 推理工具不存在: model_inference.py")

def test_environment_imports():
    """测试环境导入"""
    print("\n" + "=" * 50)
    print("测试环境导入")
    
    # 测试gym导入
    try:
        import gym
        print("✓ gym库可以正常导入")
    except ImportError as e:
        print(f"✗ gym库导入失败: {e}")
    
    # 测试Maze环境导入
    try:
        from Maze.MazeEnv import MazeEnv, DEFAULT_MAZE
        print("✓ Maze环境可以正常导入")
    except ImportError as e:
        print(f"✗ Maze环境导入失败: {e}")

def test_torch_functionality():
    """测试PyTorch功能"""
    print("\n" + "=" * 50)
    print("测试PyTorch功能")
    
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        
        print("✓ PyTorch库可以正常导入")
        print(f"  PyTorch版本: {torch.__version__}")
        print(f"  CUDA可用: {torch.cuda.is_available()}")
        
        # 测试简单的网络创建
        net = nn.Linear(4, 2)
        x = torch.randn(1, 4)
        y = net(x)
        print("✓ 简单网络可以正常前向传播")
        
    except ImportError as e:
        print(f"✗ PyTorch库导入失败: {e}")
    except Exception as e:
        print(f"✗ PyTorch功能测试失败: {e}")

def main():
    """主函数"""
    print("DQN模型推理功能测试")
    print("=" * 50)
    
    # 运行各项测试
    test_torch_functionality()
    test_environment_imports()
    test_model_save_load()
    test_inference_tool()
    
    print("\n" + "=" * 50)
    print("测试完成！")
    print("\n使用建议:")
    print("1. 如果所有测试都通过，可以直接使用推理功能")
    print("2. 如果有失败的测试，请先解决依赖问题")
    print("3. 运行 'python CartPole/CartPole.py' 训练CartPole模型")
    print("4. 运行 'python Maze/MazeAgent.py' 训练Maze模型")
    print("5. 使用 'python model_inference.py <模型路径> <环境类型>' 进行推理")

if __name__ == "__main__":
    main() 