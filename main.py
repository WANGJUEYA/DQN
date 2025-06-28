#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DQN项目主程序
支持多种游戏环境的训练和推理
"""

import argparse
import os
import sys
import json
import torch
from pathlib import Path

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from games.Maze.MazeAgent import MazeAgent
from games.CartPole.CartPole import DQN
from framework.convergence_analysis import ConvergenceAnalyzer
from framework.plot_convergence import plot_convergence_data


def start_training(game_type, episodes=1000, save_interval=50, output_dir="outputs", model_dir="models", render=False):
    """
    启动模型训练
    
    Args:
        game_type (str): 游戏类型 ('maze' 或 'cartpole')
        episodes (int): 训练episode数量
        save_interval (int): 保存间隔
        output_dir (str): 输出目录
        model_dir (str): 模型目录
        render (bool): 是否在训练时显示可视化动画窗口
    """
    print(f"🚀 启动 {game_type} 模型训练...")
    project = DQNProject(game_type, output_dir, model_dir)
    project.train(episodes, save_interval, render=render)
    print(f"✅ {game_type} 模型训练完成！")


def start_inference(game_type, model_name=None, episodes=5, output_dir="outputs", model_dir="models"):
    """
    启动模型推理
    
    Args:
        game_type (str): 游戏类型 ('maze' 或 'cartpole')
        model_name (str): 模型文件名，如果为None则使用最优模型
        episodes (int): 推理episode数量
        output_dir (str): 输出目录
        model_dir (str): 模型目录
    """
    print(f"🔍 启动 {game_type} 模型推理...")
    project = DQNProject(game_type, output_dir, model_dir)
    project.inference(model_name, episodes)
    print(f"✅ {game_type} 模型推理完成！")


class DQNProject:
    """DQN项目主控制器"""
    
    def __init__(self, game_type, output_dir="outputs", model_dir="models"):
        """
        初始化项目控制器
        
        Args:
            game_type (str): 游戏类型 ('maze' 或 'cartpole')
            output_dir (str): 输出目录
            model_dir (str): 模型目录
        """
        self.game_type = game_type.lower()
        self.output_dir = Path(output_dir)
        self.model_dir = Path(model_dir)
        
        # 获取训练编号（count），并创建输出目录结构
        self.training_number = self._get_training_number_and_create_output_dir()
        
        # 初始化游戏特定的组件
        self._init_game_components()
        
        # 初始化收敛分析器
        self.convergence_analyzer = ConvergenceAnalyzer()
        
    def _get_training_number_and_create_output_dir(self):
        """获取当前训练编号，并创建outputs/{gameName}/{count}目录"""
        game_output_root = self.output_dir / self.game_type
        game_output_root.mkdir(parents=True, exist_ok=True)
        # 查找所有数字子目录
        max_count = 0
        for d in game_output_root.iterdir():
            if d.is_dir() and d.name.isdigit():
                max_count = max(max_count, int(d.name))
        count = max_count + 1
        # 创建本次输出目录
        self.game_output_dir = game_output_root / str(count)
        self.game_output_dir.mkdir(parents=True, exist_ok=True)
        # 只创建reports目录，其他目录在需要时创建
        (self.game_output_dir / "reports").mkdir(exist_ok=True)
        return count
        
    def _init_game_components(self):
        """初始化游戏特定的组件"""
        if self.game_type == "maze":
            from games.Maze.MazeEnv import DEFAULT_MAZE, MazeEnv
            self.env = MazeEnv(DEFAULT_MAZE)
            self.agent = MazeAgent()
            self.game_name = "Maze"
        elif self.game_type == "cartpole":
            import gymnasium as gym
            # CartPole环境需要指定渲染模式
            self.env = gym.make('CartPole-v1', render_mode='human').unwrapped
            self.agent = DQN()
            self.game_name = "CartPole"
        else:
            raise ValueError(f"不支持的游戏类型: {self.game_type}")
            
    def _get_training_number(self):
        """获取当前训练次数并递增"""
        # 从models目录下的模型文件名中获取当前游戏的最大训练次数
        max_training_number = 0
        
        if self.model_dir.exists():
            # 查找所有当前游戏的训练模型文件
            pattern = f"{self.game_type}_dqn_training_*_*.pth"
            training_models = list(self.model_dir.glob(pattern))
            
            for model_file in training_models:
                # 从文件名中提取训练次数
                # 格式: {game_type}_dqn_training_{number}_{type}.pth
                parts = model_file.stem.split('_')
                if len(parts) >= 4 and parts[0] == self.game_type and parts[1] == "dqn" and parts[2] == "training":
                    try:
                        training_number = int(parts[3])
                        max_training_number = max(max_training_number, training_number)
                    except ValueError:
                        continue
        
        # 返回最大训练次数加一
        return max_training_number + 1
        
    def _cleanup_old_models(self, current_training_number):
        """清理旧的模型文件，将过程模型移动到outputs目录，只在models目录保留最优和最后模型"""
        if self.model_dir.exists():
            # 获取所有模型文件
            model_files = list(self.model_dir.glob("*.pth"))
            
            # 保留在models目录的文件模式
            keep_in_models_patterns = [
                f"{self.game_type}_dqn_best.pth",  # 全局最优模型
                f"{self.game_type}_dqn_final.pth",  # 全局最后模型
                f"{self.game_type}_dqn_training_{current_training_number}_best.pth",  # 当前训练最优
                f"{self.game_type}_dqn_training_{current_training_number}_final.pth"  # 当前训练最后
            ]
            
            # 创建outputs目录用于存储过程模型
            process_models_dir = self.game_output_dir / "process_models"
            process_models_dir.mkdir(exist_ok=True)
            
            # 处理每个模型文件
            for model_file in model_files:
                should_keep_in_models = False
                for pattern in keep_in_models_patterns:
                    if model_file.name == pattern:
                        should_keep_in_models = True
                        break
                
                if not should_keep_in_models:
                    try:
                        # 将过程模型移动到outputs目录
                        target_path = process_models_dir / model_file.name
                        import shutil
                        shutil.move(str(model_file), str(target_path))
                        print(f"移动过程模型到: {target_path}")
                    except Exception as e:
                        print(f"移动模型文件失败 {model_file.name}: {e}")
        
    def train(self, episodes, save_interval=50, model_name=None, render=False):
        """
        训练模型
        
        Args:
            episodes (int): 训练episode数量
            save_interval (int): 保存间隔
            model_name (str): 模型名称
            render (bool): 是否在训练时显示可视化动画窗口
        """
        print(f"开始训练 {self.game_name} 模型...")
        print(f"训练参数: episodes={episodes}, save_interval={save_interval}")
        
        # 获取训练次数
        training_number = self._get_training_number()
        print(f"这是第 {training_number} 次训练")
        
        # 清理旧模型
        self._cleanup_old_models(training_number)
        
        # 训练状态跟踪
        best_reward = float('-inf')
        best_episode = 0
        
        # 训练循环
        for episode in range(episodes):
            # 重置环境
            reset_result = self.env.reset()
            if isinstance(reset_result, tuple):
                state = reset_result[0]  # 新版本gymnasium返回(state, info)
            else:
                state = reset_result  # 旧版本直接返回state
                
            total_reward = 0
            total_loss = 0
            steps = 0
            success = False
            
            while True:
                if render:
                    self.env.render()
                # 选择动作
                action = self.agent.choose_action(state)
                step_result = self.env.step(action)
                
                # 处理step返回值
                if len(step_result) == 5:  # 新版本gymnasium: (next_state, reward, terminated, truncated, info)
                    next_state, reward, terminated, truncated, info = step_result
                    done = terminated or truncated
                else:  # 旧版本: (next_state, reward, done, info)
                    next_state, reward, done, info = step_result
                
                # 存储经验
                self.agent.store_transition(state, action, reward, next_state)
                
                # 学习
                if self.agent.point > 32:
                    loss = self.agent.learn()
                    total_loss += loss
                
                state = next_state
                total_reward += reward
                steps += 1
                
                # 每100步输出一次进度
                if steps % 100 == 0:
                    current_loss = total_loss / max(steps, 1)
                    print(f"  Episode {episode + 1}, Step {steps}: Reward={total_reward:.2f}, Loss={current_loss:.4f}")
                
                if done:
                    success = self._is_success(episode, steps, total_reward)
                    break
            
            # 添加到收敛分析器
            avg_loss = total_loss / max(steps, 1)
            self.convergence_analyzer.add_episode_data(
                episode=episode,
                reward=total_reward,
                loss=avg_loss,
                steps=steps,
                success=success,
                epsilon=0.9  # 这里可以根据episode调整
            )
            
            # 每个回合都输出详细信息
            print(f"Episode {episode + 1}/{episodes}: "
                  f"Reward={total_reward:.2f}, "
                  f"Steps={steps}, "
                  f"AvgLoss={avg_loss:.4f}, "
                  f"Success={success}")
            
            # 检查是否是最优模型
            if total_reward > best_reward:
                best_reward = total_reward
                best_episode = episode + 1
                # 保存最优模型
                best_model_name = f"{self.game_type}_dqn_training_{training_number}_best.pth"
                best_model_path = self.model_dir / best_model_name
                self.agent.save_model(str(best_model_path))
                # 在模型文件中添加奖励信息
                checkpoint = torch.load(best_model_path, map_location='cpu', weights_only=False)
                checkpoint['best_reward'] = best_reward
                checkpoint['best_episode'] = best_episode
                checkpoint['training_number'] = training_number
                torch.save(checkpoint, best_model_path)
                print(f"  🎉 发现新的最优模型！Episode {episode + 1}, Reward: {total_reward:.2f}")
            
            # 定期保存
            if (episode + 1) % save_interval == 0:
                self._save_checkpoint(episode + 1, training_number)
                self._generate_reports(episode + 1)
                print(f"  💾 保存检查点: Episode {episode + 1}")
        
        # 最终保存
        final_model_name = f"{self.game_type}_dqn_training_{training_number}_final.pth"
        final_model_path = self.model_dir / final_model_name
        self.agent.save_model(str(final_model_path))
        
        # 更新全局最优和最后模型
        global_best_model_path = self.model_dir / f"{self.game_type}_dqn_best.pth"
        global_final_model_path = self.model_dir / f"{self.game_type}_dqn_final.pth"
        
        # 复制当前训练的最优模型到全局最优（如果更好）
        if best_reward > self._get_global_best_reward():
            import shutil
            shutil.copy2(best_model_path, global_best_model_path)
            # 在全局最优模型中添加奖励信息
            checkpoint = torch.load(global_best_model_path, map_location='cpu', weights_only=False)
            checkpoint['best_reward'] = best_reward
            checkpoint['best_episode'] = best_episode
            checkpoint['training_number'] = training_number
            torch.save(checkpoint, global_best_model_path)
            print(f"  🌟 更新全局最优模型！训练 {training_number}, Episode {best_episode}, Reward: {best_reward:.2f}")
        
        # 复制当前训练的最后模型到全局最后
        import shutil
        shutil.copy2(final_model_path, global_final_model_path)
        # 在全局最后模型中添加训练信息
        checkpoint = torch.load(global_final_model_path, map_location='cpu', weights_only=False)
        checkpoint['training_number'] = training_number
        checkpoint['final_episode'] = episodes
        torch.save(checkpoint, global_final_model_path)
        print(f"  📁 更新全局最后模型！训练 {training_number}")
        
        # 生成最终报告
        self._generate_reports(episodes)
        
        print(f"\n🎯 训练完成！")
        print(f"最优模型: {best_model_name} (Episode {best_episode}, Reward: {best_reward:.2f})")
        print(f"最后模型: {final_model_name}")
        print(f"全局最优: {self.game_type}_dqn_best.pth")
        print(f"全局最后: {self.game_type}_dqn_final.pth")
        
    def _get_global_best_reward(self):
        """获取全局最优模型的奖励值"""
        global_best_model_path = self.model_dir / f"{self.game_type}_dqn_best.pth"
        if global_best_model_path.exists():
            try:
                checkpoint = torch.load(global_best_model_path, map_location='cpu', weights_only=False)
                if 'best_reward' in checkpoint:
                    return checkpoint['best_reward']
            except:
                pass
        return float('-inf')
        
    def _is_success(self, episode, steps, reward):
        """判断是否成功"""
        if self.game_type == "maze":
            return reward > 0  # 迷宫游戏有奖励就算成功
        elif self.game_type == "cartpole":
            return steps >= 195  # CartPole成功标准
        return False
        
    def _save_checkpoint(self, episode, training_number):
        """保存检查点数据"""
        # 确保convergence_analysis目录存在
        convergence_dir = self.game_output_dir / "convergence_analysis"
        convergence_dir.mkdir(exist_ok=True)
        
        self.convergence_analyzer.save_analysis_data(
            str(convergence_dir / f"convergence_data_episode_{episode}.json")
        )
        
    def _generate_reports(self, episode):
        """生成报告和图表"""
        # 生成收敛报告
        report = self.convergence_analyzer.generate_convergence_report()
        report_path = self.game_output_dir / "reports" / f"convergence_report_episode_{episode}.txt"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
            
        # 生成图表（不弹出窗口，只保存文件）
        data_file = str(self.game_output_dir / "convergence_analysis" / f"convergence_data_episode_{episode}.json")
        if os.path.exists(data_file):
            # 确保plots目录存在
            plots_dir = self.game_output_dir / "plots"
            plots_dir.mkdir(exist_ok=True)
            
            plot_convergence_data(
                data_file=data_file,
                save_dir=str(plots_dir),
                show_plots=False  # 不弹出图表窗口
            )
        
    def inference(self, model_name=None, episodes=5):
        """
        模型推理
        
        Args:
            model_name (str): 模型文件名，如果为None则使用最优模型
            episodes (int): 推理episode数量
        """
        print(f"开始推理 {self.game_name} 模型...")
        
        # 如果未指定模型，自动选择最优模型
        if model_name is None:
            model_name = self._get_best_model()
            if model_name is None:
                print(f"未找到 {self.game_name} 的最优模型，请先训练模型或指定模型文件")
                return
            print(f"自动选择最优模型: {model_name}")
        
        model_path = self.model_dir / model_name
        if not model_path.exists():
            print(f"模型文件不存在: {model_path}")
            return
            
        # 加载模型
        self.agent.load_model(str(model_path))
        
        # 推理循环
        total_rewards = []
        total_steps = []
        success_count = 0
        
        for episode in range(episodes):
            # 重置环境
            reset_result = self.env.reset()
            if isinstance(reset_result, tuple):
                state = reset_result[0]  # 新版本gymnasium返回(state, info)
            else:
                state = reset_result  # 旧版本直接返回state
                
            total_reward = 0
            steps = 0
            
            while True:
                # 渲染环境（显示动画）
                self.env.render()
                
                # 处理pygame事件，防止窗口假死
                try:
                    import pygame
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            self.env.close()
                            print("用户关闭了推理窗口")
                            return
                except ImportError:
                    pass  # 如果不是pygame环境，忽略
                
                # 使用训练好的模型进行预测
                action = self.agent.predict_action(state, epsilon=0.0)
                step_result = self.env.step(action)
                
                # 处理step返回值
                if len(step_result) == 5:  # 新版本gymnasium: (next_state, reward, terminated, truncated, info)
                    next_state, reward, terminated, truncated, info = step_result
                    done = terminated or truncated
                else:  # 旧版本: (next_state, reward, done, info)
                    next_state, reward, done, info = step_result
                
                state = next_state
                total_reward += reward
                steps += 1
                
                # 添加适当的延时，让动画更平滑
                try:
                    import time
                    time.sleep(0.05)  # 50ms延时
                except ImportError:
                    pass
                
                if done:
                    success = self._is_success(episode, steps, total_reward)
                    if success:
                        success_count += 1
                    break
            
            total_rewards.append(total_reward)
            total_steps.append(steps)
            
            print(f"Episode {episode + 1}: Reward={total_reward:.2f}, Steps={steps}, Success={success}")
        
        # 关闭环境
        self.env.close()
        
        # 打印统计信息
        avg_reward = sum(total_rewards) / len(total_rewards)
        avg_steps = sum(total_steps) / len(total_steps)
        success_rate = success_count / episodes
        
        print(f"\n推理统计:")
        print(f"平均奖励: {avg_reward:.2f}")
        print(f"平均步数: {avg_steps:.2f}")
        print(f"成功率: {success_rate:.2%}")
        
    def list_models(self):
        """列出可用的模型"""
        print(f"可用的 {self.game_name} 模型:")
        if self.model_dir.exists():
            models = list(self.model_dir.glob("*.pth"))
            if models:
                # 按文件名排序
                models.sort(key=lambda x: x.name)
                
                # 分类显示
                print("  全局模型:")
                global_models = [m for m in models if not m.name.startswith(f"{self.game_type}_dqn_training_")]
                for model in global_models:
                    try:
                        checkpoint = torch.load(model, map_location='cpu', weights_only=False)
                        if 'best_reward' in checkpoint:
                            print(f"    - {model.name} (最优奖励: {checkpoint['best_reward']:.2f}, 训练: {checkpoint.get('training_number', 'N/A')})")
                        elif 'training_number' in checkpoint:
                            print(f"    - {model.name} (训练: {checkpoint['training_number']}, Episode: {checkpoint.get('final_episode', 'N/A')})")
                        else:
                            print(f"    - {model.name}")
                    except:
                        print(f"    - {model.name}")
                
                print("  训练历史模型:")
                training_models = [m for m in models if m.name.startswith(f"{self.game_type}_dqn_training_")]
                for model in training_models:
                    try:
                        checkpoint = torch.load(model, map_location='cpu', weights_only=False)
                        if 'best_reward' in checkpoint:
                            print(f"    - {model.name} (最优奖励: {checkpoint['best_reward']:.2f}, Episode: {checkpoint.get('best_episode', 'N/A')})")
                        else:
                            print(f"    - {model.name}")
                    except:
                        print(f"    - {model.name}")
            else:
                print("  没有找到模型文件")
        else:
            print("  模型目录不存在")
            
    def list_outputs(self):
        """列出输出文件"""
        print(f"{self.game_name} 输出文件:")
        
        # 收敛分析
        convergence_dir = self.game_output_dir / "convergence_analysis"
        if convergence_dir.exists():
            files = list(convergence_dir.glob("*.json"))
            if files:
                print("  收敛分析数据:")
                for file in files:
                    print(f"    - {file.name}")
            else:
                print("  收敛分析数据: 无")
        else:
            print("  收敛分析数据: 目录不存在")
                    
        # 图表
        plots_dir = self.game_output_dir / "plots"
        if plots_dir.exists():
            files = list(plots_dir.glob("*.png"))
            if files:
                print("  图表文件:")
                for file in files:
                    print(f"    - {file.name}")
            else:
                print("  图表文件: 无")
        else:
            print("  图表文件: 目录不存在")
                    
        # 报告
        reports_dir = self.game_output_dir / "reports"
        if reports_dir.exists():
            files = list(reports_dir.glob("*.txt"))
            if files:
                print("  报告文件:")
                for file in files:
                    print(f"    - {file.name}")
            else:
                print("  报告文件: 无")
        else:
            print("  报告文件: 目录不存在")
        
        # 过程模型
        process_models_dir = self.game_output_dir / "process_models"
        if process_models_dir.exists():
            files = list(process_models_dir.glob("*.pth"))
            if files:
                print("  过程模型:")
                for file in files:
                    print(f"    - {file.name}")
            else:
                print("  过程模型: 无")
        else:
            print("  过程模型: 目录不存在")

    def _get_best_model(self):
        """获取最优模型文件名"""
        if not self.model_dir.exists():
            return None
            
        # 首先尝试查找全局最优模型
        best_model_path = self.model_dir / f"{self.game_type}_dqn_best.pth"
        if best_model_path.exists():
            return best_model_path.name
            
        # 如果没有全局最优模型，查找训练历史中的最优模型
        pattern = f"{self.game_type}_dqn_training_*_best.pth"
        training_best_models = list(self.model_dir.glob(pattern))
        
        if training_best_models:
            # 找到训练次数最大的最优模型
            max_training_number = 0
            best_model = None
            
            for model_path in training_best_models:
                parts = model_path.stem.split('_')
                if len(parts) >= 4:
                    try:
                        training_number = int(parts[3])
                        if training_number > max_training_number:
                            max_training_number = training_number
                            best_model = model_path
                    except ValueError:
                        continue
            
            if best_model:
                return best_model.name
        
        # 如果都没有，尝试查找最后模型
        final_model_path = self.model_dir / f"{self.game_type}_dqn_final.pth"
        if final_model_path.exists():
            return final_model_path.name
            
        return None


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
    
    # 如果都不匹配，抛出错误
    raise ValueError(f"无效的{input_type}: '{input_str}'。可选值: {', '.join(choices)}")


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
        game_type = normalize_input(args.game, game_choices, "游戏类型")
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
    if mode == "train":
        start_training(
            game_type=game_type,
            episodes=args.episodes,
            save_interval=args.save_interval,
            output_dir=args.output_dir,
            model_dir=args.model_dir,
            render=args.render
        )
    elif mode == "inference":
        start_inference(
            game_type=game_type,
            model_name=args.model,  # 可以为None，此时使用最优模型
            episodes=args.episodes,
            output_dir=args.output_dir,
            model_dir=args.model_dir
        )
    elif mode == "list-models":
        project = DQNProject(game_type, args.output_dir, args.model_dir)
        project.list_models()
    elif mode == "list-outputs":
        project = DQNProject(game_type, args.output_dir, args.model_dir)
        project.list_outputs()


if __name__ == "__main__":
    main() 