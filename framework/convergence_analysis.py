#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DQN训练过程收敛分析工具
提供多种收敛指标和可视化功能
"""

import numpy as np
from collections import deque
import os
import json
from typing import List, Dict, Tuple, Optional, Union

# 尝试导入pandas
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    print("警告: pandas未安装，某些分析功能将不可用")

# 尝试导入matplotlib
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    MATPLOTLIB_AVAILABLE = True
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("警告: matplotlib未安装，可视化功能将不可用")

class ConvergenceAnalyzer:
    """训练过程收敛分析器"""
    
    def __init__(self, window_size: int = 100, smoothing_factor: float = 0.9):
        """
        初始化收敛分析器
        
        Args:
            window_size: 滑动窗口大小，用于计算移动平均
            smoothing_factor: 指数平滑因子
        """
        self.window_size = window_size
        self.smoothing_factor = smoothing_factor
        
        # 存储训练数据
        self.episode_rewards = []
        self.episode_losses = []
        self.episode_steps = []
        self.episode_successes = []
        self.epsilon_values = []
        
        # 收敛指标
        self.convergence_metrics = {}
        
    def add_episode_data(self, episode: int, reward: float, loss=None, 
                        steps=None, success=None, epsilon=None):
        """
        添加一个episode的训练数据
        
        Args:
            episode: episode编号
            reward: 总奖励
            loss: 平均损失
            steps: 步数
            success: 是否成功
            epsilon: epsilon值
        """
        self.episode_rewards.append(reward)
        if loss is not None:
            self.episode_losses.append(loss)
        if steps is not None:
            self.episode_steps.append(steps)
        if success is not None:
            self.episode_successes.append(1.0 if success else 0.0)
        if epsilon is not None:
            self.epsilon_values.append(epsilon)
    
    def calculate_moving_average(self, data: List[float], window_size=None):
        """计算移动平均"""
        if window_size is None:
            window_size = self.window_size
        
        if len(data) < window_size:
            return data
        
        moving_avg = []
        for i in range(len(data)):
            start_idx = max(0, i - window_size + 1)
            window_data = data[start_idx:i+1]
            moving_avg.append(np.mean(window_data))
        
        return moving_avg
    
    def calculate_exponential_smoothing(self, data: List[float], alpha=None):
        """计算指数平滑"""
        if alpha is None:
            alpha = self.smoothing_factor
        
        if not data:
            return []
        
        smoothed = [data[0]]
        for i in range(1, len(data)):
            smoothed.append(alpha * data[i] + (1 - alpha) * smoothed[i-1])
        
        return smoothed
    
    def calculate_convergence_metrics(self):
        """计算收敛指标"""
        if len(self.episode_rewards) < 10:
            return {}
        
        metrics = {}
        
        # 1. 奖励收敛指标
        rewards = np.array(self.episode_rewards)
        metrics['reward_mean'] = np.mean(rewards)
        metrics['reward_std'] = np.std(rewards)
        metrics['reward_max'] = np.max(rewards)
        metrics['reward_min'] = np.min(rewards)
        
        # 计算奖励稳定性（最近N个episode的标准差）
        recent_rewards = rewards[-self.window_size:] if len(rewards) >= self.window_size else rewards
        metrics['reward_stability'] = np.std(recent_rewards)
        
        # 2. 收敛判断指标
        if len(rewards) >= self.window_size:
            # 计算最近窗口的平均奖励与整体平均奖励的比值
            recent_mean = np.mean(recent_rewards)
            overall_mean = np.mean(rewards)
            metrics['convergence_ratio'] = recent_mean / overall_mean if overall_mean > 0 else 0
            
            # 计算奖励趋势（线性回归斜率）
            x = np.arange(len(recent_rewards))
            slope, _ = np.polyfit(x, recent_rewards, 1)
            metrics['reward_trend'] = slope
        
        # 3. 损失收敛指标（如果有损失数据）
        if self.episode_losses:
            losses = np.array(self.episode_losses)
            metrics['loss_mean'] = np.mean(losses)
            metrics['loss_std'] = np.std(losses)
            metrics['loss_trend'] = np.polyfit(np.arange(len(losses)), losses, 1)[0]
        
        # 4. 成功率指标（如果有成功数据）
        if self.episode_successes:
            successes = np.array(self.episode_successes)
            metrics['success_rate'] = np.mean(successes)
            metrics['success_rate_recent'] = np.mean(successes[-self.window_size:]) if len(successes) >= self.window_size else np.mean(successes)
        
        # 5. 收敛状态判断
        metrics['is_converged'] = self._check_convergence(metrics)
        
        self.convergence_metrics = metrics
        return metrics
    
    def _check_convergence(self, metrics):
        """检查是否收敛"""
        # 收敛条件：
        # 1. 奖励稳定性高（标准差小）
        # 2. 最近奖励趋势平缓
        # 3. 收敛比率接近1
        
        if 'reward_stability' not in metrics or 'reward_trend' not in metrics:
            return False
        
        stability_threshold = 0.1 * metrics.get('reward_mean', 1.0)  # 稳定性阈值
        trend_threshold = 0.01  # 趋势阈值
        
        is_stable = metrics['reward_stability'] < stability_threshold
        is_flat_trend = abs(metrics['reward_trend']) < trend_threshold
        
        return bool(is_stable and is_flat_trend)  # 确保返回Python原生布尔类型
    
    def plot_convergence_analysis(self, save_path=None, show_plot=True, total_seconds=None):
        """绘制收敛分析图表"""
        if not MATPLOTLIB_AVAILABLE:
            print("matplotlib未安装，无法生成图表")
            return
        
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            
            metrics = self.calculate_convergence_metrics()
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle('DQN训练过程收敛分析', fontsize=16, fontweight='bold')
            has_rewards = len(self.episode_rewards) > 0
            has_losses = len(self.episode_losses) > 0
            has_successes = len(self.episode_successes) > 0
            if has_rewards:
                episodes = range(1, len(self.episode_rewards) + 1)
            else:
                episodes = []
            # 1. 奖励曲线
            ax1 = axes[0, 0]
            if has_rewards:
                ax1.plot(episodes, self.episode_rewards, alpha=0.6, label='原始奖励', color='lightblue')
                if len(self.episode_rewards) >= self.window_size:
                    moving_avg = self.calculate_moving_average(self.episode_rewards)
                    ax1.plot(episodes, moving_avg, label=f'移动平均({self.window_size})', linewidth=2, color='blue')
                smoothed = self.calculate_exponential_smoothing(self.episode_rewards)
                ax1.plot(episodes, smoothed, label='指数平滑', linewidth=2, color='red')
                ax1.set_title('Episode奖励变化')
                ax1.set_xlabel('Episode')
                ax1.set_ylabel('奖励')
                ax1.legend()
                ax1.grid(True, alpha=0.3)
            else:
                ax1.text(0.5, 0.5, '暂无奖励数据', ha='center', va='center', transform=ax1.transAxes, fontsize=14)
                ax1.set_title('Episode奖励变化')
                ax1.set_xlabel('Episode')
                ax1.set_ylabel('奖励')
            # 2. 损失曲线
            ax2 = axes[0, 1]
            if has_losses:
                ax2.plot(episodes, self.episode_losses, alpha=0.6, label='原始损失', color='lightcoral')
                smoothed_loss = self.calculate_exponential_smoothing(self.episode_losses)
                ax2.plot(episodes, smoothed_loss, label='指数平滑', linewidth=2, color='red')
                ax2.set_title('Episode损失变化')
                ax2.set_xlabel('Episode')
                ax2.set_ylabel('损失')
                ax2.legend()
                ax2.grid(True, alpha=0.3)
            else:
                ax2.text(0.5, 0.5, '暂无损失数据', ha='center', va='center', transform=ax2.transAxes, fontsize=14)
                ax2.set_title('Episode损失变化')
                ax2.set_xlabel('Episode')
                ax2.set_ylabel('损失')
            # 3. 成功率曲线
            ax3 = axes[1, 0]
            if has_successes:
                success_rates = []
                for i in range(len(self.episode_successes)):
                    start_idx = max(0, i - self.window_size + 1)
                    window_successes = self.episode_successes[start_idx:i+1]
                    success_rates.append(np.mean(window_successes))
                ax3.plot(episodes, success_rates, label=f'成功率({self.window_size}窗口)', linewidth=2, color='green')
                ax3.set_title('Episode成功率变化')
                ax3.set_xlabel('Episode')
                ax3.set_ylabel('成功率')
                ax3.legend()
                ax3.grid(True, alpha=0.3)
            else:
                ax3.text(0.5, 0.5, '暂无成功率数据', ha='center', va='center', transform=ax3.transAxes, fontsize=14)
                ax3.set_title('Episode成功率变化')
                ax3.set_xlabel('Episode')
                ax3.set_ylabel('成功率')
            # 4. 收敛指标统计
            ax4 = axes[1, 1]
            # 始终显示表格，即使没有数据
            metric_names = ['平均奖励', '奖励标准差', '奖励稳定性', '收敛比率', '奖励趋势', '收敛状态']
            
            if metrics and has_rewards:
                metric_values = [
                    f"{metrics.get('reward_mean', 0):.2f}",
                    f"{metrics.get('reward_std', 0):.2f}",
                    f"{metrics.get('reward_stability', 0):.2f}",
                    f"{metrics.get('convergence_ratio', 0):.2f}",
                    f"{metrics.get('reward_trend', 0):.4f}",
                    "已收敛" if metrics.get('is_converged', False) else "未收敛"
                ]
            else:
                metric_values = ["无法分析", "无法分析", "无法分析", "无法分析", "无法分析", "无法分析"]
            
            # 总耗时
            if total_seconds is not None:
                if total_seconds < 60:
                    time_str = f"{total_seconds:.1f}秒"
                else:
                    m = int(total_seconds // 60)
                    s = int(total_seconds % 60)
                    time_str = f"{m}分{s}秒"
                metric_names.append('总耗时')
                metric_values.append(time_str)
            else:
                metric_names.append('总耗时')
                metric_values.append("未知")
            
            table_data = [[name, value] for name, value in zip(metric_names, metric_values)]
            table = ax4.table(cellText=table_data, colLabels=['指标', '值'], 
                            cellLoc='center', loc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1, 2)
            
            for i in range(len(metric_names)):
                if metric_names[i] == '收敛状态':
                    if metric_values[i] == "已收敛":
                        color = 'lightgreen'
                    elif metric_values[i] == "未收敛":
                        color = 'lightcoral'
                    else:
                        color = 'lightgray'
                elif metric_names[i] == '总耗时':
                    color = 'khaki'
                elif metric_values[i] == "无法分析" or metric_values[i] == "未知":
                    color = 'lightgray'
                else:
                    color = 'lightblue'
                table[(i+1, 0)].set_facecolor(color)
                table[(i+1, 1)].set_facecolor(color)
            
            ax4.set_title('收敛指标统计')
            ax4.axis('off')
            plt.tight_layout()
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"图表已保存到: {save_path}")
            if show_plot:
                plt.show()
            plt.close()
        except Exception as e:
            print(f"生成图表时出错: {e}")
            import traceback
            traceback.print_exc()
    
    def plot_reward_distribution(self, save_path=None, show_plot=True):
        """绘制奖励分布图"""
        if not MATPLOTLIB_AVAILABLE:
            print("matplotlib未安装，无法生成图表")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle('奖励分布分析', fontsize=16, fontweight='bold')
        
        if self.episode_rewards:
            rewards = np.array(self.episode_rewards)
            
            # 1. 直方图
            ax1.hist(rewards, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
            ax1.axvline(np.mean(rewards), color='red', linestyle='--', label=f'平均值: {np.mean(rewards):.2f}')
            ax1.axvline(np.median(rewards), color='green', linestyle='--', label=f'中位数: {np.median(rewards):.2f}')
            ax1.set_title('奖励分布直方图')
            ax1.set_xlabel('奖励值')
            ax1.set_ylabel('频次')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # 2. 箱线图
            ax2.boxplot(rewards, patch_artist=True, boxprops=dict(facecolor='lightblue'))
            ax2.set_title('奖励箱线图')
            ax2.set_ylabel('奖励值')
            ax2.grid(True, alpha=0.3)
            
            # 添加统计信息
            stats_text = f"""
            统计信息:
            样本数: {len(rewards)}
            平均值: {np.mean(rewards):.2f}
            标准差: {np.std(rewards):.2f}
            最小值: {np.min(rewards):.2f}
            最大值: {np.max(rewards):.2f}
            中位数: {np.median(rewards):.2f}
            """
            ax2.text(0.02, 0.98, stats_text, transform=ax2.transAxes, 
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        else:
            # 没有数据时显示提示信息
            ax1.text(0.5, 0.5, '暂无奖励数据\n无法生成分布图', ha='center', va='center', 
                    transform=ax1.transAxes, fontsize=14, 
                    bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
            ax1.set_title('奖励分布直方图')
            ax1.set_xlabel('奖励值')
            ax1.set_ylabel('频次')
            
            ax2.text(0.5, 0.5, '暂无奖励数据\n无法生成箱线图', ha='center', va='center', 
                    transform=ax2.transAxes, fontsize=14,
                    bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
            ax2.set_title('奖励箱线图')
            ax2.set_ylabel('奖励值')
        
        plt.tight_layout()
        
        # 保存图表
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"奖励分布图已保存到: {save_path}")
        
        # 显示图表
        if show_plot:
            plt.show()
        
        plt.close()
    
    def save_analysis_data(self, filepath):
        """保存分析数据到JSON文件"""
        # 转换numpy类型为Python原生类型
        def convert_numpy_types(obj):
            # 处理numpy标量类型
            if hasattr(obj, 'item'):  # numpy标量类型都有item()方法
                return obj.item()
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_numpy_types(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            elif isinstance(obj, tuple):
                return tuple(convert_numpy_types(item) for item in obj)
            else:
                return obj
        
        data = {
            'episode_rewards': self.episode_rewards,
            'episode_losses': self.episode_losses,
            'episode_steps': self.episode_steps,
            'episode_successes': self.episode_successes,
            'epsilon_values': self.epsilon_values,
            'convergence_metrics': self.convergence_metrics,
            'window_size': self.window_size,
            'smoothing_factor': self.smoothing_factor
        }
        
        # 转换所有numpy类型
        data = convert_numpy_types(data)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        print(f"分析数据已保存到: {filepath}")
    
    def load_analysis_data(self, filepath):
        """从JSON文件加载分析数据"""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.episode_rewards = data.get('episode_rewards', [])
        self.episode_losses = data.get('episode_losses', [])
        self.episode_steps = data.get('episode_steps', [])
        self.episode_successes = data.get('episode_successes', [])
        self.epsilon_values = data.get('epsilon_values', [])
        self.convergence_metrics = data.get('convergence_metrics', {})
        self.window_size = data.get('window_size', 100)
        self.smoothing_factor = data.get('smoothing_factor', 0.9)
        
        print(f"分析数据已从 {filepath} 加载")
    
    def generate_convergence_report(self, save_path=None, total_seconds=None):
        """生成收敛分析报告"""
        if not self.episode_rewards:
            return "没有数据可供分析"
        
        metrics = self.calculate_convergence_metrics()
        
        report = []
        report.append("=" * 60)
        report.append("DQN训练过程收敛分析报告")
        report.append("=" * 60)
        report.append("")
        
        # 基本信息
        report.append("1. 训练基本信息")
        report.append("-" * 30)
        report.append(f"总Episode数: {len(self.episode_rewards)}")
        report.append(f"滑动窗口大小: {self.window_size}")
        report.append(f"平滑因子: {self.smoothing_factor}")
        # 添加训练时间信息
        if total_seconds is not None:
            if total_seconds < 60:
                time_str = f"{total_seconds:.1f}秒"
            else:
                m = int(total_seconds // 60)
                s = int(total_seconds % 60)
                time_str = f"{m}分{s}秒"
            report.append(f"训练耗时: {time_str}")
        else:
            report.append("训练耗时: 未知")
        report.append("")
        
        # 奖励统计
        report.append("2. 奖励统计")
        report.append("-" * 30)
        report.append(f"平均奖励: {metrics.get('reward_mean', 0):.2f}")
        report.append(f"奖励标准差: {metrics.get('reward_std', 0):.2f}")
        report.append(f"最高奖励: {metrics.get('reward_max', 0):.2f}")
        report.append(f"最低奖励: {metrics.get('reward_min', 0):.2f}")
        report.append(f"奖励稳定性: {metrics.get('reward_stability', 0):.2f}")
        report.append("")
        
        # 收敛分析
        report.append("3. 收敛分析")
        report.append("-" * 30)
        report.append(f"收敛比率: {metrics.get('convergence_ratio', 0):.2f}")
        report.append(f"奖励趋势: {metrics.get('reward_trend', 0):.4f}")
        report.append(f"收敛状态: {'已收敛' if metrics.get('is_converged', False) else '未收敛'}")
        report.append("")
        
        # 损失统计（如果有）
        if self.episode_losses:
            report.append("4. 损失统计")
            report.append("-" * 30)
            report.append(f"平均损失: {metrics.get('loss_mean', 0):.4f}")
            report.append(f"损失标准差: {metrics.get('loss_std', 0):.4f}")
            report.append(f"损失趋势: {metrics.get('loss_trend', 0):.6f}")
            report.append("")
        
        # 成功率统计（如果有）
        if self.episode_successes:
            report.append("5. 成功率统计")
            report.append("-" * 30)
            report.append(f"总体成功率: {metrics.get('success_rate', 0):.2%}")
            report.append(f"最近成功率: {metrics.get('success_rate_recent', 0):.2%}")
            report.append("")
        
        # 收敛建议
        report.append("6. 收敛建议")
        report.append("-" * 30)
        if metrics.get('is_converged', False):
            report.append("✓ 模型已收敛，可以考虑停止训练或降低学习率")
        else:
            report.append("✗ 模型未收敛，建议继续训练或调整超参数")
            if metrics.get('reward_trend', 0) < 0:
                report.append("  - 奖励趋势为负，可能需要调整学习率或网络结构")
            if metrics.get('reward_stability', 0) > 0.1 * metrics.get('reward_mean', 1.0):
                report.append("  - 奖励不稳定，可能需要增加经验回放缓冲区大小")
        
        report.append("")
        report.append("=" * 60)
        
        report_text = "\n".join(report)
        
        # 保存报告
        if save_path:
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(report_text)
            print(f"收敛报告已保存到: {save_path}")
        
        return report_text

def create_convergence_monitor(env_name="cartpole"):
    """创建收敛监控器"""
    return ConvergenceAnalyzer()

# 使用示例
if __name__ == "__main__":
    # 创建收敛分析器
    analyzer = ConvergenceAnalyzer(window_size=50, smoothing_factor=0.9)
    
    # 模拟训练数据
    np.random.seed(42)
    episodes = 200
    
    # 模拟奖励数据（逐渐收敛）
    base_reward = 20
    for i in range(episodes):
        # 模拟奖励逐渐增加并趋于稳定
        progress = min(i / 100, 1.0)
        noise = np.random.normal(0, 2)
        reward = base_reward + progress * 30 + noise
        reward = max(0, reward)  # 确保奖励非负
        
        # 模拟损失逐渐下降
        loss = 0.5 * np.exp(-i / 50) + np.random.normal(0, 0.05)
        loss = max(0, loss)
        
        # 模拟成功率逐渐提高
        success_rate = min(0.1 + progress * 0.8, 0.95)
        success = np.random.random() < success_rate
        
        # 模拟epsilon衰减
        epsilon = max(0.1, 0.9 * np.exp(-i / 30))
        
        analyzer.add_episode_data(i+1, reward, loss, steps=100, success=success, epsilon=epsilon)
    
    # 生成分析
    print("生成收敛分析...")
    analyzer.plot_convergence_analysis(save_path="convergence_analysis.png")
    analyzer.plot_reward_distribution(save_path="reward_distribution.png")
    
    # 生成报告
    report = analyzer.generate_convergence_report(save_path="convergence_report.txt")
    print(report)
    
    # 保存数据
    analyzer.save_analysis_data("convergence_data.json") 