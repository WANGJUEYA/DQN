#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DQN项目主程序
支持多种游戏环境的训练和推理
"""

import argparse
import os
import sys
import torch
from pathlib import Path

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from games.Maze.MazeAgent import MazeAgent
from games.CartPole.CartPoleAgent import CartPoleAgent
# from games.MountainCar.MountainCarAgent import MountainCarAgent

def normalize_input(input_str, choices, input_type):
    """
    标准化输入，支持完整名称和首字母简写
    
    Args:
        input_str (str): 用户输入
        choices (list): 可选值列表
        input_type (str): 输入类型描述
    
    Returns:
        str: 标准化后的值
    """
    if input_str in choices:
        return input_str
    
    # 尝试首字母匹配
    for choice in choices:
        if choice.startswith(input_str.lower()):
            return choice
    
    # 尝试首字母组合匹配（如 'cp' 匹配 'cartpole'）
    input_lower = input_str.lower()
    for choice in choices:
        # 提取首字母
        initials = ''.join(word[0] for word in choice.split())
        if initials == input_lower:
            return choice
    
    # 特殊简写映射
    special_mappings = {
        'lm': 'list-models',
        'lo': 'list-outputs',
        't': 'train',
        'i': 'inference',
        'm': 'maze',
        'cp': 'cartpole'
    }
    
    if input_lower in special_mappings:
        return special_mappings[input_lower]
    
    # 如果都不匹配，抛出错误
    raise ValueError(f"无效的{input_type}: '{input_str}'。可选值: {', '.join(choices)}")


def start_training(agent_class, episodes=1000, save_interval=50, output_dir="outputs", model_dir="models", render=False):
    """
    启动模型训练
    
    Args:
        agent_class (class): 游戏代理类
        episodes (int): 训练episode数量
        save_interval (int): 保存间隔
        output_dir (str): 输出目录
        model_dir (str): 模型目录
        render (bool): 是否在训练时显示可视化动画窗口
    """
    print(f"🚀 启动 {agent_class.__name__} 训练...")
    agent = agent_class(output_dir, model_dir, training_mode=True)
    agent.train(episodes, save_interval, render=render)
    print(f"✅ 训练完成！")


def start_inference(agent_class, model_name=None, episodes=5, output_dir="outputs", model_dir="models"):
    """
    启动模型推理
    
    Args:
        agent_class (class): 游戏代理类
        model_name (str): 模型文件名，如果为None则使用最优模型
        episodes (int): 推理episode数量
        output_dir (str): 输出目录
        model_dir (str): 模型目录
    """
    print(f"🔍 启动 {agent_class.__name__} 推理...")
    agent = agent_class(output_dir, model_dir, training_mode=False)
    agent.inference(model_name, episodes)
    print(f"✅ 推理完成！")


def list_models(agent_class, output_dir="outputs", model_dir="models"):
    """列出指定游戏的模型"""
    agent = agent_class(output_dir, model_dir, training_mode=False)
    agent.list_models()


def list_outputs(agent_class, output_dir="outputs", model_dir="models"):
    """列出指定游戏的输出"""
    agent = agent_class(output_dir, model_dir, training_mode=False)
    agent.list_outputs()


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="DQN项目主程序")
    parser.add_argument("--game", "-g", required=True, 
                       help="游戏类型 (maze/m 或 cartpole/cp)")
    parser.add_argument("--mode", "-m", required=True,
                       help="运行模式 (train/t, inference/i, list-models/lm, list-outputs/lo)")
    parser.add_argument("--episodes", "-e", type=int, default=400,
                       help="训练或推理的episode数量 (默认: 400)")
    parser.add_argument("--model", "-M", type=str,
                       help="模型文件名")
    parser.add_argument("--output-dir", "-o", type=str, default="outputs",
                       help="输出目录 (默认: outputs)")
    parser.add_argument("--model-dir", "-d", type=str, default="models",
                       help="模型目录 (默认: models)")
    parser.add_argument("--save-interval", "-s", type=int, default=50,
                       help="保存间隔 (默认: 50)")
    parser.add_argument("--render", action="store_true", default=True, help="训练时显示可视化动画窗口（默认启用）")
    
    args = parser.parse_args()
    
    # 标准化游戏类型输入
    game_choices = ["maze", "cartpole"]
    try:
        game_name = normalize_input(args.game, game_choices, "游戏类型")
    except ValueError as e:
        print(f"错误: {e}")
        print("支持的游戏类型:")
        print("  maze (m) - 迷宫游戏")
        print("  cartpole (cp) - 倒立摆游戏")
        return
    
    # 标准化模式输入
    mode_choices = ["train", "inference", "list-models", "list-outputs"]
    try:
        mode = normalize_input(args.mode, mode_choices, "运行模式")
    except ValueError as e:
        print(f"错误: {e}")
        print("支持的运行模式:")
        print("  train (t) - 训练模型")
        print("  inference (i) - 推理模型")
        print("  list-models (lm) - 列出模型")
        print("  list-outputs (lo) - 列出输出")
        return
    
    # 根据模式执行相应操作
    agent_class = MazeAgent if game_name == "maze" else CartPoleAgent
    if mode == "train":
        start_training(
            agent_class=agent_class,
            episodes=args.episodes,
            save_interval=args.save_interval,
            output_dir=args.output_dir,
            model_dir=args.model_dir,
            render=args.render
        )
    elif mode == "inference":
        start_inference(
            agent_class=agent_class,
            model_name=args.model,  # 可以为None，此时使用最优模型
            episodes=args.episodes,
            output_dir=args.output_dir,
            model_dir=args.model_dir
        )
    elif mode == "list-models":
        list_models(agent_class, args.output_dir, args.model_dir)
    elif mode == "list-outputs":
        list_outputs(agent_class, args.output_dir, args.model_dir)


if __name__ == "__main__":
    main() 