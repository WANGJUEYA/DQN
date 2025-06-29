#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DQN训练过程收敛数据折线图生成工具
支持文本格式和图形化折线图
"""

import json
import os
import math
import datetime
from typing import List, Dict, Optional
import numpy as np

# 尝试导入matplotlib
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
    print("✓ matplotlib已安装，支持图形化折线图")
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("⚠ matplotlib未安装，仅支持文本格式图表")

def load_convergence_data(data_file: str) -> Optional[Dict]:
    """加载收敛数据"""
    try:
        with open(data_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"成功加载数据文件: {data_file}")
        return data
    except Exception as e:
        print(f"加载数据文件失败: {e}")
        return None

def get_training_info(data: Dict) -> Dict:
    """获取训练信息"""
    training_info = {
        "environment": data.get("environment", "未知环境"),
        "start_time": data.get("start_time", "未知"),
        "end_time": data.get("end_time", "未知"),
        "total_episodes": len(data.get("episode_rewards", [])),
        "training_duration": "未知",
        "model_name": data.get("model_name", "未知模型"),
        "hyperparameters": data.get("hyperparameters", {})
    }
    
    # 计算训练时长
    if training_info["start_time"] != "未知" and training_info["end_time"] != "未知":
        try:
            start_time = datetime.datetime.fromisoformat(training_info["start_time"])
            end_time = datetime.datetime.fromisoformat(training_info["end_time"])
            duration = end_time - start_time
            training_info["training_duration"] = str(duration).split('.')[0]  # 去掉微秒
        except:
            training_info["training_duration"] = "未知"
    
    return training_info

def create_text_plots(data: Dict, save_dir: str):
    """创建文本格式的折线图"""
    rewards = data.get('episode_rewards', [])
    losses = data.get('episode_losses', [])
    successes = data.get('episode_successes', [])
    
    episodes = list(range(1, len(rewards) + 1))
    
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)
    
    # 1. 奖励折线图
    if rewards:
        create_reward_text_plot(episodes, rewards, save_dir)
    
    # 2. 损失折线图
    if losses:
        create_loss_text_plot(episodes, losses, save_dir)
    
    # 3. 成功率折线图
    if successes:
        create_success_text_plot(episodes, successes, save_dir)
    
    # 4. 综合报告
    create_comprehensive_report(episodes, rewards, losses, successes, save_dir, data)

def create_reward_text_plot(episodes: List[int], rewards: List[float], save_dir: str):
    """创建奖励文本折线图"""
    if not rewards:
        print("没有奖励数据可供绘制")
        return
        
    avg_reward = sum(rewards) / len(rewards)
    max_reward = max(rewards)
    min_reward = min(rewards)
    
    chart_lines = []
    chart_lines.append("=" * 60)
    chart_lines.append("DQN训练过程 - Episode奖励变化")
    chart_lines.append("=" * 60)
    chart_lines.append(f"总Episode数: {len(rewards)}")
    chart_lines.append(f"平均奖励: {avg_reward:.2f}")
    chart_lines.append(f"最高奖励: {max_reward:.2f}")
    chart_lines.append(f"最低奖励: {min_reward:.2f}")
    chart_lines.append("")
    
    # 分段统计 - 根据数据量调整分段数
    if len(rewards) >= 5:
        segments = min(5, len(rewards) // 2)  # 确保每段至少有2个数据点
        segment_size = max(1, len(rewards) // segments)
        chart_lines.append("分段统计:")
        for i in range(segments):
            start_idx = i * segment_size
            end_idx = (i + 1) * segment_size if i < segments - 1 else len(rewards)
            segment_rewards = rewards[start_idx:end_idx]
            if segment_rewards:  # 确保段不为空
                avg_segment = sum(segment_rewards) / len(segment_rewards)
                chart_lines.append(f"第{i+1}段 (Episode {start_idx+1}-{end_idx}): 平均奖励 {avg_segment:.2f}")
    else:
        chart_lines.append("数据量较少，跳过分段统计")
    
    chart_lines.append("=" * 60)
    
    # 保存到文件
    save_path = os.path.join(save_dir, 'rewards_plot.txt')
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(chart_lines))
    
    print(f"奖励折线图已保存到: {save_path}")
    print('\n'.join(chart_lines))

def create_loss_text_plot(episodes: List[int], losses: List[float], save_dir: str):
    """创建损失文本折线图"""
    if not losses:
        print("没有损失数据可供绘制")
        return
        
    avg_loss = sum(losses) / len(losses)
    max_loss = max(losses)
    min_loss = min(losses)
    
    chart_lines = []
    chart_lines.append("=" * 60)
    chart_lines.append("DQN训练过程 - Episode损失变化")
    chart_lines.append("=" * 60)
    chart_lines.append(f"总Episode数: {len(losses)}")
    chart_lines.append(f"平均损失: {avg_loss:.4f}")
    chart_lines.append(f"最高损失: {max_loss:.4f}")
    chart_lines.append(f"最低损失: {min_loss:.4f}")
    chart_lines.append("")
    
    # 分段统计 - 根据数据量调整分段数
    if len(losses) >= 5:
        segments = min(5, len(losses) // 2)  # 确保每段至少有2个数据点
        segment_size = max(1, len(losses) // segments)
        chart_lines.append("分段统计:")
        for i in range(segments):
            start_idx = i * segment_size
            end_idx = (i + 1) * segment_size if i < segments - 1 else len(losses)
            segment_losses = losses[start_idx:end_idx]
            if segment_losses:  # 确保段不为空
                avg_segment = sum(segment_losses) / len(segment_losses)
                chart_lines.append(f"第{i+1}段 (Episode {start_idx+1}-{end_idx}): 平均损失 {avg_segment:.4f}")
    else:
        chart_lines.append("数据量较少，跳过分段统计")
    
    chart_lines.append("=" * 60)
    
    # 保存到文件
    save_path = os.path.join(save_dir, 'losses_plot.txt')
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(chart_lines))
    
    print(f"损失折线图已保存到: {save_path}")

def create_success_text_plot(episodes: List[int], successes: List[float], save_dir: str):
    """创建成功率文本折线图"""
    if not successes:
        print("没有成功率数据可供绘制")
        return
        
    overall_success_rate = sum(successes) / len(successes)
    
    chart_lines = []
    chart_lines.append("=" * 60)
    chart_lines.append("DQN训练过程 - Episode成功率变化")
    chart_lines.append("=" * 60)
    chart_lines.append(f"总Episode数: {len(successes)}")
    chart_lines.append(f"总体成功率: {overall_success_rate:.2%}")
    chart_lines.append("")
    
    # 分段统计 - 根据数据量调整分段数
    if len(successes) >= 5:
        segments = min(5, len(successes) // 2)  # 确保每段至少有2个数据点
        segment_size = max(1, len(successes) // segments)
        chart_lines.append("分段统计:")
        for i in range(segments):
            start_idx = i * segment_size
            end_idx = (i + 1) * segment_size if i < segments - 1 else len(successes)
            segment_successes = successes[start_idx:end_idx]
            if segment_successes:  # 确保段不为空
                avg_segment = sum(segment_successes) / len(segment_successes)
                chart_lines.append(f"第{i+1}段 (Episode {start_idx+1}-{end_idx}): 成功率 {avg_segment:.2%}")
    else:
        chart_lines.append("数据量较少，跳过分段统计")
    
    chart_lines.append("=" * 60)
    
    # 保存到文件
    save_path = os.path.join(save_dir, 'success_rates_plot.txt')
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(chart_lines))
    
    print(f"成功率折线图已保存到: {save_path}")

def create_comprehensive_report(episodes: List[int], rewards: List[float], 
                              losses: List[float], successes: List[float], save_dir: str, data: Dict):
    """创建综合统计报告"""
    training_info = get_training_info(data)
    
    chart_lines = []
    chart_lines.append("=" * 60)
    chart_lines.append("DQN训练过程综合分析报告")
    chart_lines.append("=" * 60)
    chart_lines.append(f"总Episode数: {len(episodes)}")
    chart_lines.append("")
    
    # 训练信息
    chart_lines.append("【训练信息】")
    chart_lines.append(f"训练环境: {training_info['environment']}")
    chart_lines.append(f"模型名称: {training_info['model_name']}")
    chart_lines.append(f"开始时间: {training_info['start_time']}")
    chart_lines.append(f"结束时间: {training_info['end_time']}")
    chart_lines.append(f"训练时长: {training_info['training_duration']}")
    
    # 超参数信息
    if training_info['hyperparameters']:
        chart_lines.append("")
        chart_lines.append("【超参数】")
        for key, value in training_info['hyperparameters'].items():
            chart_lines.append(f"{key}: {value}")
    
    chart_lines.append("")
    
    # 奖励统计
    if rewards:
        avg_reward = sum(rewards) / len(rewards)
        max_reward = max(rewards)
        min_reward = min(rewards)
        recent_avg = sum(rewards[-20:]) / min(20, len(rewards))
        
        chart_lines.append("【奖励统计】")
        chart_lines.append(f"平均奖励: {avg_reward:.2f}")
        chart_lines.append(f"最高奖励: {max_reward:.2f}")
        chart_lines.append(f"最低奖励: {min_reward:.2f}")
        chart_lines.append(f"最近20个episode平均奖励: {recent_avg:.2f}")
        chart_lines.append("")
    
    # 损失统计
    if losses:
        avg_loss = sum(losses) / len(losses)
        max_loss = max(losses)
        min_loss = min(losses)
        recent_avg = sum(losses[-20:]) / min(20, len(losses))
        
        chart_lines.append("【损失统计】")
        chart_lines.append(f"平均损失: {avg_loss:.4f}")
        chart_lines.append(f"最高损失: {max_loss:.4f}")
        chart_lines.append(f"最低损失: {min_loss:.4f}")
        chart_lines.append(f"最近20个episode平均损失: {recent_avg:.4f}")
        chart_lines.append("")
    
    # 成功率统计
    if successes:
        overall_success_rate = sum(successes) / len(successes)
        recent_success_rate = sum(successes[-20:]) / min(20, len(successes))
        
        chart_lines.append("【成功率统计】")
        chart_lines.append(f"总体成功率: {overall_success_rate:.2%}")
        chart_lines.append(f"最近20个episode成功率: {recent_success_rate:.2%}")
        chart_lines.append("")
    
    # 收敛分析
    if rewards and len(rewards) >= 20:
        chart_lines.append("【收敛分析】")
        
        # 计算收敛指标
        recent_rewards = rewards[-20:]
        recent_std = calculate_std(recent_rewards)
        recent_trend = calculate_trend(recent_rewards)
        
        chart_lines.append(f"最近20个episode奖励标准差: {recent_std:.2f}")
        chart_lines.append(f"最近20个episode奖励趋势: {recent_trend:.4f}")
        
        # 收敛判断
        if recent_std < avg_reward * 0.1 and abs(recent_trend) < 0.01:
            chart_lines.append("收敛状态: 已收敛 ✓")
            chart_lines.append("建议: 可以考虑停止训练")
        else:
            chart_lines.append("收敛状态: 未收敛 ⚠")
            if recent_trend < 0:
                chart_lines.append("建议: 奖励呈下降趋势，可能需要调整学习率")
            if recent_std > avg_reward * 0.1:
                chart_lines.append("建议: 奖励波动较大，可能需要增加训练轮次")
        chart_lines.append("")
    
    chart_lines.append("=" * 60)
    
    # 保存到文件
    save_path = os.path.join(save_dir, 'comprehensive_report.txt')
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(chart_lines))
    
    print(f"综合报告已保存到: {save_path}")

def create_graphical_plots(data: Dict, save_dir: str, show_plots: bool = True):
    """创建图形化折线图"""
    if not MATPLOTLIB_AVAILABLE:
        print("matplotlib未安装，跳过图形化图表生成")
        return
    
    rewards = data.get('episode_rewards', [])
    losses = data.get('episode_losses', [])
    successes = data.get('episode_successes', [])
    epsilons = data.get('episode_epsilons', [])
    
    episodes = list(range(1, len(rewards) + 1)) if rewards else []
    training_info = get_training_info(data)
    
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)
    
    # 1. 奖励折线图
    if rewards:
        create_reward_graphical_plot(episodes, rewards, save_dir, show_plots, training_info)
    else:
        create_empty_reward_plot(save_dir, show_plots, training_info)
    
    # 2. 损失折线图
    if losses:
        create_loss_graphical_plot(episodes, losses, save_dir, show_plots, training_info)
    else:
        create_empty_loss_plot(save_dir, show_plots, training_info)
    
    # 3. 成功率折线图
    if successes:
        create_success_graphical_plot(episodes, successes, save_dir, show_plots, training_info)
    else:
        create_empty_success_plot(save_dir, show_plots, training_info)
    
    # 4. Epsilon折线图
    if epsilons:
        create_epsilon_graphical_plot(episodes, epsilons, save_dir, show_plots, training_info)
    else:
        create_empty_epsilon_plot(save_dir, show_plots, training_info)
    
    # 5. 综合折线图
    create_comprehensive_graphical_plot(episodes, rewards, losses, successes, epsilons, save_dir, show_plots, training_info)

def add_training_info_to_plot(training_info: Dict):
    """在图表中添加训练信息"""
    info_text = f"环境: {training_info['environment']}\n"
    info_text += f"模型: {training_info['model_name']}\n"
    info_text += f"开始: {training_info['start_time']}\n"
    info_text += f"结束: {training_info['end_time']}\n"
    info_text += f"时长: {training_info['training_duration']}\n"
    info_text += f"Episode: {training_info['total_episodes']}"
    
    # 添加超参数信息
    if training_info['hyperparameters']:
        info_text += "\n\n超参数:"
        for key, value in list(training_info['hyperparameters'].items())[:3]:  # 只显示前3个
            info_text += f"\n{key}: {value}"
    
    plt.figtext(0.02, 0.02, info_text, fontsize=8, 
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8),
                verticalalignment='bottom')

def create_reward_graphical_plot(episodes: List[int], rewards: List[float], save_dir: str, show_plots: bool, training_info: Dict):
    """创建奖励图形化折线图"""
    plt.figure(figsize=(12, 8))
    
    # 原始奖励数据
    plt.plot(episodes, rewards, alpha=0.6, color='lightblue', label='原始奖励', linewidth=1)
    
    # 移动平均线
    if len(rewards) >= 10:
        window_size = min(20, len(rewards) // 5)
        moving_avg = calculate_moving_average(rewards, window_size)
        plt.plot(episodes, moving_avg, color='blue', label=f'移动平均({window_size})', linewidth=2)
    
    plt.title('DQN训练过程 - Episode奖励变化', fontsize=16, fontweight='bold')
    plt.xlabel('Episode', fontsize=12)
    plt.ylabel('奖励值', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 添加统计信息
    avg_reward = sum(rewards) / len(rewards)
    max_reward = max(rewards)
    min_reward = min(rewards)
    plt.text(0.02, 0.98, f'平均奖励: {avg_reward:.2f}\n最高奖励: {max_reward:.2f}\n最低奖励: {min_reward:.2f}', 
            transform=plt.gca().transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # 添加训练信息
    add_training_info_to_plot(training_info)
    
    plt.tight_layout()
    
    # 保存图片
    save_path = os.path.join(save_dir, 'rewards_plot.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"奖励折线图(图形)已保存到: {save_path}")
    
    if show_plots:
        plt.show()
    else:
        plt.close()

def create_loss_graphical_plot(episodes: List[int], losses: List[float], save_dir: str, show_plots: bool, training_info: Dict):
    """创建损失图形化折线图"""
    plt.figure(figsize=(12, 8))
    
    # 原始损失数据
    plt.plot(episodes, losses, alpha=0.6, color='lightcoral', label='原始损失', linewidth=1)
    
    # 移动平均线
    if len(losses) >= 10:
        window_size = min(20, len(losses) // 5)
        moving_avg = calculate_moving_average(losses, window_size)
        plt.plot(episodes, moving_avg, color='red', label=f'移动平均({window_size})', linewidth=2)
    
    plt.title('DQN训练过程 - Episode损失变化', fontsize=16, fontweight='bold')
    plt.xlabel('Episode', fontsize=12)
    plt.ylabel('损失值', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 添加统计信息
    avg_loss = sum(losses) / len(losses)
    max_loss = max(losses)
    min_loss = min(losses)
    plt.text(0.02, 0.98, f'平均损失: {avg_loss:.4f}\n最高损失: {max_loss:.4f}\n最低损失: {min_loss:.4f}', 
            transform=plt.gca().transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))
    
    # 添加训练信息
    add_training_info_to_plot(training_info)
    
    plt.tight_layout()
    
    # 保存图片
    save_path = os.path.join(save_dir, 'losses_plot.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"损失折线图(图形)已保存到: {save_path}")
    
    if show_plots:
        plt.show()
    else:
        plt.close()

def create_success_graphical_plot(episodes: List[int], successes: List[float], save_dir: str, show_plots: bool, training_info: Dict):
    """创建成功率图形化折线图"""
    if not MATPLOTLIB_AVAILABLE:
        print("matplotlib未安装，跳过成功率图形化图表生成")
        return
    
    plt.figure(figsize=(10, 5))
    
    if successes and episodes:
        plt.plot(episodes, successes, label="成功(1)/失败(0)", color="orange", alpha=0.5)
        
        # 计算滑动窗口成功率
        window_size = min(20, len(successes))
        success_rates = []
        window_episodes = []
        for i in range(0, len(successes), window_size):
            window_successes = successes[i:i+window_size]
            if len(window_successes) > 0:
                success_rates.append(sum(window_successes) / len(window_successes))
                window_episodes.append(episodes[i+window_size//2] if i+window_size//2 < len(episodes) else episodes[-1])
        if success_rates:
            plt.plot(window_episodes, success_rates, label=f"滑动窗口({window_size})成功率", color="red", linewidth=2)
        
        plt.xlabel("Episode")
        plt.ylabel("成功率")
        plt.title("DQN训练过程 - 成功率变化")
        plt.legend()
        plt.grid(True, alpha=0.3)
    else:
        plt.text(0.5, 0.5, '暂无成功率数据\n无法生成成功率变化图', ha='center', va='center', 
                transform=plt.gca().transAxes, fontsize=16,
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        plt.xlabel("Episode")
        plt.ylabel("成功率")
        plt.title("DQN训练过程 - 成功率变化")
    
    # 添加训练信息
    add_training_info_to_plot(training_info)
    
    plt.tight_layout()
    
    save_path = os.path.join(save_dir, 'success_rates_plot.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"成功率折线图(图形)已保存到: {save_path}")
    
    if show_plots:
        plt.show()
    else:
        plt.close()

def create_epsilon_graphical_plot(episodes: List[int], epsilons: List[float], save_dir: str, show_plots: bool, training_info: Dict):
    """创建Epsilon图形化折线图"""
    plt.figure(figsize=(12, 8))
    
    plt.plot(episodes, epsilons, color='purple', label='Epsilon值', linewidth=2)
    
    # 添加探索/利用分界线
    plt.axhline(y=0.5, color='orange', linestyle='--', alpha=0.7, label='探索/利用分界线')
    plt.axhline(y=0.1, color='red', linestyle='--', alpha=0.7, label='最小探索率')
    
    plt.title('DQN训练过程 - Epsilon衰减曲线', fontsize=16, fontweight='bold')
    plt.xlabel('Episode', fontsize=12)
    plt.ylabel('Epsilon值', fontsize=12)
    plt.ylim(0, 1)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 添加统计信息
    initial_epsilon = epsilons[0] if epsilons else 0
    final_epsilon = epsilons[-1] if epsilons else 0
    plt.text(0.02, 0.98, f'初始Epsilon: {initial_epsilon:.2f}\n最终Epsilon: {final_epsilon:.2f}', 
            transform=plt.gca().transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='plum', alpha=0.8))
    
    # 添加训练信息
    add_training_info_to_plot(training_info)
    
    plt.tight_layout()
    
    # 保存图片
    save_path = os.path.join(save_dir, 'epsilons_plot.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Epsilon折线图(图形)已保存到: {save_path}")
    
    if show_plots:
        plt.show()
    else:
        plt.close()

def create_comprehensive_graphical_plot(episodes: List[int], rewards: List[float], 
                                      losses: List[float], successes: List[float], 
                                      epsilons: List[float], save_dir: str, show_plots: bool, training_info: Dict):
    """创建综合图形化折线图"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('DQN训练过程综合分析', fontsize=18, fontweight='bold')
    
    # 1. 奖励图
    ax1 = axes[0, 0]
    if rewards and episodes:
        ax1.plot(episodes, rewards, alpha=0.6, color='lightblue', linewidth=1)
        if len(rewards) >= 10:
            window_size = min(20, len(rewards) // 5)
            moving_avg = calculate_moving_average(rewards, window_size)
            ax1.plot(episodes, moving_avg, color='blue', linewidth=2)
        ax1.set_title('奖励变化')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('奖励')
        ax1.grid(True, alpha=0.3)
    else:
        ax1.text(0.5, 0.5, '暂无奖励数据', ha='center', va='center', 
                transform=ax1.transAxes, fontsize=14,
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        ax1.set_title('奖励变化')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('奖励')
    
    # 2. 损失图
    ax2 = axes[0, 1]
    if losses and episodes:
        ax2.plot(episodes, losses, alpha=0.6, color='lightcoral', linewidth=1)
        if len(losses) >= 10:
            window_size = min(20, len(losses) // 5)
            moving_avg = calculate_moving_average(losses, window_size)
            ax2.plot(episodes, moving_avg, color='red', linewidth=2)
        ax2.set_title('损失变化')
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('损失')
        ax2.grid(True, alpha=0.3)
    else:
        ax2.text(0.5, 0.5, '暂无损失数据', ha='center', va='center', 
                transform=ax2.transAxes, fontsize=14,
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        ax2.set_title('损失变化')
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('损失')
    
    # 3. 成功率图
    ax3 = axes[1, 0]
    if successes and episodes:
        window_size = min(20, len(successes) // 5)
        success_rates = []
        for i in range(len(successes)):
            start_idx = max(0, i - window_size + 1)
            window_successes = successes[start_idx:i+1]
            if len(window_successes) > 0:
                success_rates.append(sum(window_successes) / len(window_successes))
            else:
                success_rates.append(0)
        ax3.plot(episodes, success_rates, color='green', linewidth=2)
        ax3.axhline(y=0.8, color='orange', linestyle='--', alpha=0.7)
        ax3.set_title('成功率变化')
        ax3.set_xlabel('Episode')
        ax3.set_ylabel('成功率')
        ax3.set_ylim(0, 1)
        ax3.grid(True, alpha=0.3)
    else:
        ax3.text(0.5, 0.5, '暂无成功率数据', ha='center', va='center', 
                transform=ax3.transAxes, fontsize=14,
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        ax3.set_title('成功率变化')
        ax3.set_xlabel('Episode')
        ax3.set_ylabel('成功率')
        ax3.set_ylim(0, 1)
    
    # 4. Epsilon图
    ax4 = axes[1, 1]
    if epsilons and episodes:
        ax4.plot(episodes, epsilons, color='purple', linewidth=2)
        ax4.axhline(y=0.5, color='orange', linestyle='--', alpha=0.7)
        ax4.set_title('Epsilon衰减')
        ax4.set_xlabel('Episode')
        ax4.set_ylabel('Epsilon')
        ax4.set_ylim(0, 1)
        ax4.grid(True, alpha=0.3)
    else:
        ax4.text(0.5, 0.5, '暂无Epsilon数据', ha='center', va='center', 
                transform=ax4.transAxes, fontsize=14,
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        ax4.set_title('Epsilon衰减')
        ax4.set_xlabel('Episode')
        ax4.set_ylabel('Epsilon')
        ax4.set_ylim(0, 1)
    
    # 添加训练信息到综合图表
    info_text = f"环境: {training_info['environment']}\n"
    info_text += f"模型: {training_info['model_name']}\n"
    info_text += f"时长: {training_info['training_duration']}\n"
    info_text += f"Episode: {training_info['total_episodes']}"
    
    fig.text(0.02, 0.02, info_text, fontsize=10, 
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8),
             verticalalignment='bottom')
    
    plt.tight_layout()
    
    # 保存图片
    save_path = os.path.join(save_dir, 'comprehensive_plot.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"综合折线图(图形)已保存到: {save_path}")
    
    if show_plots:
        plt.show()
    else:
        plt.close()

# 辅助计算函数
def calculate_moving_average(data: List[float], window_size: int) -> List[float]:
    """计算移动平均"""
    if len(data) < window_size:
        return data
    
    moving_avg = []
    for i in range(len(data)):
        start_idx = max(0, i - window_size + 1)
        window_data = data[start_idx:i+1]
        moving_avg.append(sum(window_data) / len(window_data))
    
    return moving_avg

def calculate_std(data: List[float]) -> float:
    """计算标准差"""
    if len(data) < 2:
        return 0.0
    mean = sum(data) / len(data)
    variance = sum((x - mean) ** 2 for x in data) / (len(data) - 1)
    return math.sqrt(variance)

def calculate_trend(data: List[float]) -> float:
    """计算数据趋势"""
    if len(data) < 2:
        return 0.0
    
    x = np.arange(len(data))
    slope, _ = np.polyfit(x, data, 1)
    return slope

def plot_convergence_data(data_file: str, save_dir: str = "plots", show_plots: bool = True):
    """绘制收敛数据图表"""
    data = load_convergence_data(data_file)
    if data is None:
        print(f"无法加载数据文件: {data_file}")
        return
    
    create_text_plots(data, save_dir)
    create_graphical_plots(data, save_dir, show_plots)

def create_empty_reward_plot(save_dir: str, show_plots: bool, training_info: Dict):
    """创建空的奖励图表"""
    plt.figure(figsize=(12, 8))
    plt.text(0.5, 0.5, '暂无奖励数据\n无法生成奖励变化图', ha='center', va='center', 
            transform=plt.gca().transAxes, fontsize=16, 
            bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    plt.title('DQN训练过程 - Episode奖励变化', fontsize=16, fontweight='bold')
    plt.xlabel('Episode', fontsize=12)
    plt.ylabel('奖励值', fontsize=12)
    
    # 添加训练信息
    add_training_info_to_plot(training_info)
    
    plt.tight_layout()
    
    save_path = os.path.join(save_dir, 'rewards_plot.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"空奖励折线图已保存到: {save_path}")
    
    if show_plots:
        plt.show()
    else:
        plt.close()

def create_empty_loss_plot(save_dir: str, show_plots: bool, training_info: Dict):
    """创建空的损失图表"""
    plt.figure(figsize=(12, 8))
    plt.text(0.5, 0.5, '暂无损失数据\n无法生成损失变化图', ha='center', va='center', 
            transform=plt.gca().transAxes, fontsize=16, 
            bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    plt.title('DQN训练过程 - Episode损失变化', fontsize=16, fontweight='bold')
    plt.xlabel('Episode', fontsize=12)
    plt.ylabel('损失值', fontsize=12)
    
    # 添加训练信息
    add_training_info_to_plot(training_info)
    
    plt.tight_layout()
    
    save_path = os.path.join(save_dir, 'losses_plot.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"空损失折线图已保存到: {save_path}")
    
    if show_plots:
        plt.show()
    else:
        plt.close()

def create_empty_success_plot(save_dir: str, show_plots: bool, training_info: Dict):
    """创建空的成功率图表"""
    plt.figure(figsize=(10, 5))
    plt.text(0.5, 0.5, '暂无成功率数据\n无法生成成功率变化图', ha='center', va='center', 
            transform=plt.gca().transAxes, fontsize=16, 
            bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    plt.title('DQN训练过程 - 成功率变化', fontsize=16, fontweight='bold')
    plt.xlabel('Episode', fontsize=12)
    plt.ylabel('成功率', fontsize=12)
    
    # 添加训练信息
    add_training_info_to_plot(training_info)
    
    plt.tight_layout()
    
    save_path = os.path.join(save_dir, 'success_rates_plot.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"空成功率折线图已保存到: {save_path}")
    
    if show_plots:
        plt.show()
    else:
        plt.close()

def create_empty_epsilon_plot(save_dir: str, show_plots: bool, training_info: Dict):
    """创建空的Epsilon图表"""
    plt.figure(figsize=(12, 8))
    plt.text(0.5, 0.5, '暂无Epsilon数据\n无法生成Epsilon衰减图', ha='center', va='center', 
            transform=plt.gca().transAxes, fontsize=16, 
            bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    plt.title('DQN训练过程 - Epsilon衰减曲线', fontsize=16, fontweight='bold')
    plt.xlabel('Episode', fontsize=12)
    plt.ylabel('Epsilon值', fontsize=12)
    
    # 添加训练信息
    add_training_info_to_plot(training_info)
    
    plt.tight_layout()
    
    save_path = os.path.join(save_dir, 'epsilons_plot.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"空Epsilon折线图已保存到: {save_path}")
    
    if show_plots:
        plt.show()
    else:
        plt.close()

# 使用示例
if __name__ == "__main__":
    import sys
    
    # 检查命令行参数
    if len(sys.argv) > 1:
        data_file = sys.argv[1]
        save_dir = sys.argv[2] if len(sys.argv) > 2 else "plots"
        show_plots = len(sys.argv) > 3 and sys.argv[3].lower() == 'show'
        
        print(f"正在为数据文件 {data_file} 生成折线图...")
        plot_convergence_data(data_file, save_dir, show_plots)
    else:
        # 默认使用测试数据
        print("使用默认测试数据生成折线图...")
        
        # 创建测试数据
        test_data = {
            "episode_rewards": [20 + i * 0.5 + (hash(str(i)) % 100 - 50) / 25 for i in range(100)],
            "episode_losses": [0.5 * math.exp(-i / 25) + (hash(str(i)) % 100) / 2000 for i in range(100)],
            "episode_successes": [1.0 if (hash(str(i)) % 100) / 100 < min(0.1 + i * 0.008, 0.95) else 0.0 for i in range(100)],
            "episode_epsilons": [max(0.1, 0.9 * math.exp(-i / 30)) for i in range(100)],
            "environment": "CartPole-v1",
            "model_name": "DQN_CartPole",
            "start_time": "2024-01-15T10:30:00",
            "end_time": "2024-01-15T11:45:00",
            "hyperparameters": {
                "learning_rate": 0.001,
                "epsilon": 0.9,
                "epsilon_decay": 0.995,
                "batch_size": 32,
                "memory_size": 10000
            }
        }
        
        # 保存测试数据
        with open("test_convergence_data.json", 'w', encoding='utf-8') as f:
            json.dump(test_data, f, ensure_ascii=False, indent=2)
        
        # 生成折线图
        plot_convergence_data("test_convergence_data.json", "test_plots", False)
        
        print("测试完成！") 