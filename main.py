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


def start_inference(game_type, model_name, episodes=5, output_dir="outputs", model_dir="models"):
    """
    启动模型推理
    
    Args:
        game_type (str): 游戏类型 ('maze' 或 'cartpole')
        model_name (str): 模型文件名
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
        
        # 创建目录结构
        self._create_directories()
        
        # 初始化游戏特定的组件
        self._init_game_components()
        
        # 加载训练计数器
        self.training_counter = self._load_training_counter()
        
    def _create_directories(self):
        """创建必要的目录结构"""
        # 游戏特定的输出目录
        self.game_output_dir = self.output_dir / self.game_type
        self.game_output_dir.mkdir(parents=True, exist_ok=True)
        
        # 游戏特定的模型目录
        self.game_model_dir = self.model_dir / self.game_type
        self.game_model_dir.mkdir(parents=True, exist_ok=True)
        
        # 子目录
        (self.game_output_dir / "convergence_analysis").mkdir(exist_ok=True)
        (self.game_output_dir / "plots").mkdir(exist_ok=True)
        (self.game_output_dir / "logs").mkdir(exist_ok=True)
        (self.game_output_dir / "reports").mkdir(exist_ok=True)
        
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
            
        # 初始化收敛分析器
        self.convergence_analyzer = ConvergenceAnalyzer()
        
    def _load_training_counter(self):
        """加载训练计数器"""
        counter_file = Path("training_counter.json")
        if counter_file.exists():
            with open(counter_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        else:
            return {"maze": 0, "cartpole": 0}
            
    def _save_training_counter(self):
        """保存训练计数器"""
        counter_file = Path("training_counter.json")
        with open(counter_file, 'w', encoding='utf-8') as f:
            json.dump(self.training_counter, f, ensure_ascii=False, indent=2)
            
    def _get_training_number(self):
        """获取当前训练次数并递增"""
        self.training_counter[self.game_type] += 1
        self._save_training_counter()
        return self.training_counter[self.game_type]
        
    def _cleanup_old_models(self, current_training_number):
        """清理旧的模型文件，只保留最优模型和最后模型"""
        if self.game_model_dir.exists():
            # 获取所有模型文件
            model_files = list(self.game_model_dir.glob("*.pth"))
            
            # 保留的文件模式
            keep_patterns = [
                f"{self.game_type}_dqn_best.pth",  # 最优模型
                f"{self.game_type}_dqn_final.pth",  # 最后模型
                f"{self.game_type}_dqn_training_{current_training_number}_best.pth",  # 当前训练最优
                f"{self.game_type}_dqn_training_{current_training_number}_final.pth"  # 当前训练最后
            ]
            
            # 删除不需要的文件
            for model_file in model_files:
                should_keep = False
                for pattern in keep_patterns:
                    if model_file.name == pattern:
                        should_keep = True
                        break
                
                if not should_keep:
                    try:
                        model_file.unlink()
                        print(f"删除旧模型文件: {model_file.name}")
                    except Exception as e:
                        print(f"删除文件失败 {model_file.name}: {e}")
        
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
                best_model_path = self.game_model_dir / best_model_name
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
        final_model_path = self.game_model_dir / final_model_name
        self.agent.save_model(str(final_model_path))
        
        # 更新全局最优和最后模型
        global_best_model_path = self.game_model_dir / f"{self.game_type}_dqn_best.pth"
        global_final_model_path = self.game_model_dir / f"{self.game_type}_dqn_final.pth"
        
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
        global_best_model_path = self.game_model_dir / f"{self.game_type}_dqn_best.pth"
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
        """保存检查点"""
        # 保存收敛数据
        self.convergence_analyzer.save_analysis_data(
            str(self.game_output_dir / "convergence_analysis" / f"convergence_data_episode_{episode}.json")
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
            plot_convergence_data(
                data_file=data_file,
                save_dir=str(self.game_output_dir / "plots"),
                show_plots=False  # 不弹出图表窗口
            )
        
    def inference(self, model_name, episodes=5):
        """
        模型推理
        
        Args:
            model_name (str): 模型文件名
            episodes (int): 推理episode数量
        """
        print(f"开始推理 {self.game_name} 模型...")
        
        model_path = self.game_model_dir / model_name
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
        if self.game_model_dir.exists():
            models = list(self.game_model_dir.glob("*.pth"))
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
                    
        # 图表
        plots_dir = self.game_output_dir / "plots"
        if plots_dir.exists():
            files = list(plots_dir.glob("*.png"))
            if files:
                print("  图表文件:")
                for file in files:
                    print(f"    - {file.name}")
                    
        # 报告
        reports_dir = self.game_output_dir / "reports"
        if reports_dir.exists():
            files = list(reports_dir.glob("*.txt"))
            if files:
                print("  报告文件:")
                for file in files:
                    print(f"    - {file.name}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="DQN项目主程序")
    parser.add_argument("--game", "-g", required=True, 
                       choices=["maze", "cartpole"], 
                       help="游戏类型 (maze/cartpole)")
    parser.add_argument("--mode", "-m", required=True,
                       choices=["train", "inference", "list-models", "list-outputs"],
                       help="运行模式")
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
    parser.add_argument("--render", action="store_true", help="训练时显示可视化动画窗口")
    
    args = parser.parse_args()
    
    # 根据模式执行相应操作
    if args.mode == "train":
        start_training(
            game_type=args.game,
            episodes=args.episodes,
            save_interval=args.save_interval,
            output_dir=args.output_dir,
            model_dir=args.model_dir,
            render=args.render
        )
    elif args.mode == "inference":
        if not args.model:
            print("推理模式需要指定模型文件 (--model)")
            return
        start_inference(
            game_type=args.game,
            model_name=args.model,
            episodes=args.episodes,
            output_dir=args.output_dir,
            model_dir=args.model_dir
        )
    elif args.mode == "list-models":
        project = DQNProject(args.game, args.output_dir, args.model_dir)
        project.list_models()
    elif args.mode == "list-outputs":
        project = DQNProject(args.game, args.output_dir, args.model_dir)
        project.list_outputs()


if __name__ == "__main__":
    main() 