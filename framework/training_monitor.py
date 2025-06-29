#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
训练监控模块
用于监控训练过程中的资源使用、性能和稳定性
"""

import time
import psutil
import threading
import warnings
from typing import Optional, Callable
from dataclasses import dataclass
from pathlib import Path

@dataclass
class TrainingMetrics:
    """训练指标数据类"""
    episode: int
    reward: float
    loss: float
    steps: int
    episode_time: float
    memory_usage_mb: float
    cpu_usage_percent: float
    gpu_usage_percent: Optional[float] = None

class TrainingMonitor:
    """训练监控器"""
    
    def __init__(self, output_dir: str = "outputs", log_interval: int = 10):
        self.output_dir = Path(output_dir)
        self.log_interval = log_interval
        self.metrics_history = []
        self.start_time = None
        self.monitoring = False
        self.monitor_thread = None
        
        # 创建监控输出目录
        self.monitor_dir = self.output_dir / "training_monitor"
        self.monitor_dir.mkdir(parents=True, exist_ok=True)
        
        # 性能警告阈值
        self.memory_warning_threshold = 80.0  # 内存使用率警告阈值(%)
        self.cpu_warning_threshold = 90.0     # CPU使用率警告阈值(%)
        self.episode_timeout = 300            # Episode超时时间(秒)
        
    def start_monitoring(self):
        """开始监控"""
        self.start_time = time.time()
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_resources, daemon=True)
        self.monitor_thread.start()
        print("🔍 训练监控已启动")
        
    def stop_monitoring(self):
        """停止监控"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        print("🔍 训练监控已停止")
        
    def _monitor_resources(self):
        """资源监控线程"""
        while self.monitoring:
            try:
                # 获取系统资源使用情况
                memory_percent = psutil.virtual_memory().percent
                cpu_percent = psutil.cpu_percent(interval=1)
                
                # 检查GPU使用情况（如果可用）
                gpu_percent = None
                try:
                    import torch
                    if torch.cuda.is_available():
                        gpu_percent = torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated() * 100
                except:
                    pass
                
                # 检查资源使用是否超过阈值
                if memory_percent > self.memory_warning_threshold:
                    print(f"⚠️ 内存使用率过高: {memory_percent:.1f}%")
                    
                if cpu_percent > self.cpu_warning_threshold:
                    print(f"⚠️ CPU使用率过高: {cpu_percent:.1f}%")
                    
                time.sleep(self.log_interval)
                
            except Exception as e:
                print(f"⚠️ 资源监控错误: {e}")
                time.sleep(self.log_interval)
                
    def add_episode_metrics(self, metrics: TrainingMetrics):
        """添加episode指标"""
        self.metrics_history.append(metrics)
        
        # 检查episode时间是否超时
        if metrics.episode_time > self.episode_timeout:
            print(f"⚠️ Episode {metrics.episode} 耗时过长: {metrics.episode_time:.1f}s")
            
        # 检查内存使用情况
        if metrics.memory_usage_mb > 1000:  # 超过1GB
            print(f"⚠️ Episode {metrics.episode} 内存使用过高: {metrics.memory_usage_mb:.1f}MB")
            
    def get_current_memory_usage(self) -> float:
        """获取当前内存使用量(MB)"""
        return psutil.Process().memory_info().rss / 1024 / 1024
        
    def get_current_cpu_usage(self) -> float:
        """获取当前CPU使用率(%)"""
        return psutil.cpu_percent()
        
    def get_current_gpu_usage(self) -> Optional[float]:
        """获取当前GPU使用率(%)"""
        try:
            import torch
            if torch.cuda.is_available():
                return torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated() * 100
        except:
            pass
        return None
        
    def generate_monitoring_report(self, filename: str = "training_monitor_report.txt"):
        """生成监控报告"""
        if not self.metrics_history:
            print("⚠️ 没有监控数据可生成报告")
            return
            
        report_path = self.monitor_dir / filename
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("训练监控报告\n")
            f.write("=" * 50 + "\n\n")
            
            # 总体统计
            total_episodes = len(self.metrics_history)
            total_time = time.time() - self.start_time if self.start_time else 0
            
            f.write(f"总训练episodes: {total_episodes}\n")
            f.write(f"总训练时间: {total_time:.1f}秒\n")
            f.write(f"平均episode时间: {total_time/total_episodes:.1f}秒\n\n")
            
            # 资源使用统计
            memory_usage = [m.memory_usage_mb for m in self.metrics_history]
            cpu_usage = [m.cpu_usage_percent for m in self.metrics_history]
            
            f.write("资源使用统计:\n")
            f.write(f"  内存使用 - 平均: {sum(memory_usage)/len(memory_usage):.1f}MB, "
                   f"最大: {max(memory_usage):.1f}MB\n")
            f.write(f"  CPU使用 - 平均: {sum(cpu_usage)/len(cpu_usage):.1f}%, "
                   f"最大: {max(cpu_usage):.1f}%\n\n")
            
            # 性能警告统计
            timeout_episodes = [m for m in self.metrics_history if m.episode_time > self.episode_timeout]
            high_memory_episodes = [m for m in self.metrics_history if m.memory_usage_mb > 1000]
            
            f.write("性能警告统计:\n")
            f.write(f"  超时episodes: {len(timeout_episodes)}\n")
            f.write(f"  高内存使用episodes: {len(high_memory_episodes)}\n\n")
            
            # 详细episode数据
            f.write("详细episode数据:\n")
            f.write("Episode | 奖励 | 损失 | 步数 | 时间(s) | 内存(MB) | CPU(%)\n")
            f.write("-" * 70 + "\n")
            
            for metrics in self.metrics_history[-20:]:  # 只显示最后20个episodes
                f.write(f"{metrics.episode:7d} | {metrics.reward:6.1f} | {metrics.loss:6.4f} | "
                       f"{metrics.steps:4d} | {metrics.episode_time:7.1f} | "
                       f"{metrics.memory_usage_mb:8.1f} | {metrics.cpu_usage_percent:5.1f}\n")
        
        print(f"📊 监控报告已生成: {report_path}")
        
    def cleanup(self):
        """清理资源"""
        self.stop_monitoring()
        
        # 清理过期的监控数据
        if len(self.metrics_history) > 1000:  # 保留最近1000个episodes的数据
            self.metrics_history = self.metrics_history[-1000:]

def create_training_monitor(output_dir: str = "outputs") -> TrainingMonitor:
    """创建训练监控器实例"""
    return TrainingMonitor(output_dir)

# 全局监控器实例
_global_monitor: Optional[TrainingMonitor] = None

def get_global_monitor() -> TrainingMonitor:
    """获取全局监控器实例"""
    global _global_monitor
    if _global_monitor is None:
        _global_monitor = create_training_monitor()
    return _global_monitor

def start_global_monitoring():
    """启动全局监控"""
    monitor = get_global_monitor()
    monitor.start_monitoring()

def stop_global_monitoring():
    """停止全局监控"""
    global _global_monitor
    if _global_monitor:
        _global_monitor.stop_monitoring()
        _global_monitor.generate_monitoring_report()
        _global_monitor.cleanup()
        _global_monitor = None 