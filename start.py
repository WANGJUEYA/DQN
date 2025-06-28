#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DQN项目启动脚本
提供交互式选择功能
"""

import sys
import os
from pathlib import Path

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from main import start_training, start_inference, DQNProject


def print_banner():
    """打印项目横幅"""
    print("=" * 60)
    print("🤖 DQN强化学习项目")
    print("=" * 60)
    print("支持的游戏: Maze迷宫 | CartPole平衡")
    print("=" * 60)


def get_user_choice():
    """获取用户选择"""
    print("\n请选择操作:")
    print("1. 训练模型")
    print("2. 推理测试")
    print("3. 查看模型列表")
    print("4. 查看输出文件")
    print("5. 退出")
    
    while True:
        try:
            choice = input("\n请输入选择 (1-5): ").strip()
            if choice in ['1', '2', '3', '4', '5']:
                return choice
            else:
                print("❌ 无效选择，请输入 1-5")
        except KeyboardInterrupt:
            print("\n\n👋 再见！")
            sys.exit(0)


def get_game_choice():
    """获取游戏选择"""
    print("\n请选择游戏:")
    print("1. Maze迷宫")
    print("2. CartPole平衡")
    
    while True:
        try:
            choice = input("请输入选择 (1-2): ").strip()
            if choice == '1':
                return 'maze'
            elif choice == '2':
                return 'cartpole'
            else:
                print("❌ 无效选择，请输入 1 或 2")
        except KeyboardInterrupt:
            print("\n\n👋 再见！")
            sys.exit(0)


def get_episodes():
    """获取episode数量"""
    while True:
        try:
            episodes = input("请输入episode数量 (默认100): ").strip()
            if episodes == '':
                return 100
            episodes = int(episodes)
            if episodes > 0:
                return episodes
            else:
                print("❌ episode数量必须大于0")
        except ValueError:
            print("❌ 请输入有效的数字")
        except KeyboardInterrupt:
            print("\n\n👋 再见！")
            sys.exit(0)


def get_model_name():
    """获取模型文件名"""
    while True:
        try:
            model_name = input("请输入模型文件名 (例如: maze_dqn_final.pth): ").strip()
            if model_name:
                return model_name
            else:
                print("❌ 请输入模型文件名")
        except KeyboardInterrupt:
            print("\n\n👋 再见！")
            sys.exit(0)


def main():
    """主函数"""
    print_banner()
    
    while True:
        choice = get_user_choice()
        
        if choice == '5':
            print("👋 再见！")
            break
            
        game = get_game_choice()
        
        if choice == '1':  # 训练
            print(f"\n🚀 开始训练 {game} 模型...")
            episodes = get_episodes()
            try:
                start_training(game, episodes=episodes)
                print(f"✅ {game} 模型训练完成！")
            except Exception as e:
                print(f"❌ 训练失败: {e}")
                
        elif choice == '2':  # 推理
            print(f"\n🔍 开始推理 {game} 模型...")
            model_name = get_model_name()
            episodes = get_episodes()
            try:
                start_inference(game, model_name, episodes=episodes)
                print(f"✅ {game} 模型推理完成！")
            except Exception as e:
                print(f"❌ 推理失败: {e}")
                
        elif choice == '3':  # 查看模型
            try:
                project = DQNProject(game)
                project.list_models()
            except Exception as e:
                print(f"❌ 查看模型失败: {e}")
                
        elif choice == '4':  # 查看输出
            try:
                project = DQNProject(game)
                project.list_outputs()
            except Exception as e:
                print(f"❌ 查看输出失败: {e}")
        
        # 询问是否继续
        try:
            continue_choice = input("\n是否继续其他操作? (y/n): ").strip().lower()
            if continue_choice not in ['y', 'yes', '是']:
                print("👋 再见！")
                break
        except KeyboardInterrupt:
            print("\n\n👋 再见！")
            break


if __name__ == "__main__":
    main() 