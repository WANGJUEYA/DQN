#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DQN项目使用示例
展示各种功能的使用方法
"""

import os
import sys
import subprocess
from pathlib import Path

def run_command(cmd, description):
    """运行命令并显示结果"""
    print(f"\n{'='*60}")
    print(f"执行: {description}")
    print(f"命令: {cmd}")
    print('='*60)
    
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, encoding='utf-8')
        if result.stdout:
            print("输出:")
            print(result.stdout)
        if result.stderr:
            print("错误:")
            print(result.stderr)
        return result.returncode == 0
    except Exception as e:
        print(f"执行失败: {e}")
        return False

def main():
    """主函数 - 展示各种使用示例"""
    print("DQN项目使用示例")
    print("本脚本将展示项目的各种功能使用方法")
    
    # 检查Python环境
    print(f"\n当前Python版本: {sys.version}")
    print(f"当前工作目录: {os.getcwd()}")
    
    # 示例1: 查看帮助信息
    run_command("python main.py --help", "查看主程序帮助信息")
    
    # 示例2: 列出可用模型（如果存在）
    run_command("python main.py --game maze --mode list-models", "列出迷宫游戏可用模型")
    run_command("python main.py --game cartpole --mode list-models", "列出CartPole可用模型")
    
    # 示例3: 列出输出文件（如果存在）
    run_command("python main.py --game maze --mode list-outputs", "列出迷宫游戏输出文件")
    run_command("python main.py --game cartpole --mode list-outputs", "列出CartPole输出文件")
    
    # 示例4: 短时间训练示例（仅用于演示）
    print("\n" + "="*60)
    print("训练示例（短时间，仅用于演示）")
    print("="*60)
    
    # 迷宫游戏短时间训练
    print("\n开始迷宫游戏短时间训练...")
    success = run_command(
        "python main.py --game maze --mode train --episodes 10 --save-interval 5",
        "迷宫游戏短时间训练（10个episode）"
    )
    
    if success:
        print("\n迷宫游戏训练完成！")
        # 查看生成的输出
        run_command("python main.py --game maze --mode list-models", "查看生成的模型")
        run_command("python main.py --game maze --mode list-outputs", "查看生成的输出文件")
    
    # CartPole短时间训练
    print("\n开始CartPole短时间训练...")
    success = run_command(
        "python main.py --game cartpole --mode train --episodes 10 --save-interval 5",
        "CartPole短时间训练（10个episode）"
    )
    
    if success:
        print("\nCartPole训练完成！")
        # 查看生成的输出
        run_command("python main.py --game cartpole --mode list-models", "查看生成的模型")
        run_command("python main.py --game cartpole --mode list-outputs", "查看生成的输出文件")
    
    # 示例5: 推理示例（如果有模型）
    print("\n" + "="*60)
    print("推理示例")
    print("="*60)
    
    # 检查是否有模型文件
    maze_model = Path("models/maze/maze_dqn_final.pth")
    cartpole_model = Path("models/cartpole/cartpole_dqn_final.pth")
    
    if maze_model.exists():
        run_command(
            f"python main.py --game maze --mode inference --model {maze_model.name} --episodes 3",
            "迷宫游戏推理测试"
        )
    else:
        print("迷宫游戏模型不存在，跳过推理测试")
    
    if cartpole_model.exists():
        run_command(
            f"python main.py --game cartpole --mode inference --model {cartpole_model.name} --episodes 3",
            "CartPole推理测试"
        )
    else:
        print("CartPole模型不存在，跳过推理测试")
    
    # 示例6: 自定义参数训练示例
    print("\n" + "="*60)
    print("自定义参数训练示例")
    print("="*60)
    
    print("\n使用自定义输出目录训练...")
    run_command(
        "python main.py --game maze --mode train --episodes 5 --output-dir custom_outputs --model-dir custom_models",
        "使用自定义目录训练迷宫游戏"
    )
    
    # 查看自定义输出
    if Path("custom_outputs").exists():
        run_command("python main.py --game maze --mode list-outputs", "查看自定义输出文件")
    
    print("\n" + "="*60)
    print("示例执行完成！")
    print("="*60)
    print("\n更多使用方法请参考:")
    print("- README.md: 项目说明和使用指南")
    print("- docs/目录: 详细文档")
    print("- python main.py --help: 命令行帮助")

if __name__ == "__main__":
    main() 