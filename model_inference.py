#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DQN模型推理工具
支持CartPole和Maze环境的模型推理
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gym
from Maze.MazeEnv import MazeEnv, DEFAULT_MAZE

class DQNNet(nn.Module):
    """通用的DQN网络结构"""
    
    def __init__(self, n_states, n_actions, hidden_size=100):
        super(DQNNet, self).__init__()
        self.fc1 = nn.Linear(n_states, hidden_size)
        self.fc2 = nn.Linear(hidden_size, n_actions)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)

class ModelInference:
    """模型推理类"""
    
    def __init__(self, model_path, env_type="cartpole"):
        """
        初始化推理器
        
        Args:
            model_path: 模型文件路径
            env_type: 环境类型 ("cartpole" 或 "maze")
        """
        self.model_path = model_path
        self.env_type = env_type
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 加载模型
        self.load_model()
        
        # 创建环境
        self.create_environment()
        
    def load_model(self):
        """加载训练好的模型"""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"模型文件不存在: {self.model_path}")
        
        # 加载checkpoint
        checkpoint = torch.load(self.model_path, map_location=self.device)
        
        # 获取超参数
        hyperparams = checkpoint.get('hyperparameters', {})
        self.n_states = hyperparams.get('N_STATES', 4)  # 默认CartPole状态数
        self.n_actions = hyperparams.get('N_ACTIONS', 2)  # 默认CartPole动作数
        
        # 创建网络
        self.net = DQNNet(self.n_states, self.n_actions).to(self.device)
        
        # 加载模型参数
        if 'evaluate_net_state_dict' in checkpoint:
            self.net.load_state_dict(checkpoint['evaluate_net_state_dict'])
        else:
            # 兼容旧版本模型
            self.net.load_state_dict(checkpoint)
        
        # 设置为评估模式
        self.net.eval()
        
        print(f"模型已加载: {self.model_path}")
        print(f"环境类型: {self.env_type}")
        print(f"状态数: {self.n_states}, 动作数: {self.n_actions}")
        print(f"设备: {self.device}")
    
    def create_environment(self):
        """创建环境"""
        if self.env_type.lower() == "cartpole":
            self.env = gym.make('CartPole-v0')
            self.env = self.env.unwrapped
        elif self.env_type.lower() == "maze":
            self.env = MazeEnv(DEFAULT_MAZE)
        else:
            raise ValueError(f"不支持的环境类型: {self.env_type}")
    
    def predict_action(self, state, epsilon=0.0):
        """
        预测动作
        
        Args:
            state: 当前状态
            epsilon: 探索概率（0表示完全贪婪）
        
        Returns:
            预测的动作
        """
        if np.random.uniform() < epsilon:
            return np.random.randint(0, self.n_actions)
        
        # 转换为tensor
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        # 前向传播
        with torch.no_grad():
            q_values = self.net(state_tensor)
            action = torch.argmax(q_values, dim=1).item()
        
        return action
    
    def run_episode(self, render=True, max_steps=1000):
        """
        运行一个episode
        
        Args:
            render: 是否渲染环境
            max_steps: 最大步数
        
        Returns:
            episode_reward, steps, success
        """
        state = self.env.reset()
        episode_reward = 0
        steps = 0
        
        while steps < max_steps:
            if render:
                self.env.render()
            
            # 预测动作
            action = self.predict_action(state, epsilon=0.0)
            
            # 执行动作
            next_state, reward, done, _ = self.env.step(action)
            
            episode_reward += reward
            steps += 1
            state = next_state
            
            if done:
                break
        
        # 判断是否成功（根据环境类型）
        success = False
        if self.env_type.lower() == "cartpole":
            success = steps >= 195  # CartPole通常认为195步以上为成功
        elif self.env_type.lower() == "maze":
            success = reward > 0  # Maze环境正奖励表示成功
        
        return episode_reward, steps, success
    
    def run_multiple_episodes(self, num_episodes=10, render=True):
        """
        运行多个episode
        
        Args:
            num_episodes: episode数量
            render: 是否渲染环境
        
        Returns:
            统计结果
        """
        print(f"\n开始运行 {num_episodes} 个episode...")
        print("=" * 50)
        
        rewards = []
        steps_list = []
        successes = []
        
        for episode in range(num_episodes):
            print(f"\nEpisode {episode + 1}/{num_episodes}")
            
            reward, steps, success = self.run_episode(render=render)
            
            rewards.append(reward)
            steps_list.append(steps)
            successes.append(success)
            
            status = "成功" if success else "失败"
            print(f"  奖励: {reward:.2f}, 步数: {steps}, 状态: {status}")
        
        # 计算统计信息
        avg_reward = np.mean(rewards)
        avg_steps = np.mean(steps_list)
        success_rate = np.mean(successes) * 100
        
        print("\n" + "=" * 50)
        print("推理结果统计:")
        print(f"平均奖励: {avg_reward:.2f}")
        print(f"平均步数: {avg_steps:.1f}")
        print(f"成功率: {success_rate:.1f}% ({sum(successes)}/{num_episodes})")
        print(f"最高奖励: {max(rewards):.2f}")
        print(f"最低奖励: {min(rewards):.2f}")
        print(f"最长步数: {max(steps_list)}")
        print(f"最短步数: {min(steps_list)}")
        
        return {
            'rewards': rewards,
            'steps': steps_list,
            'successes': successes,
            'avg_reward': avg_reward,
            'avg_steps': avg_steps,
            'success_rate': success_rate
        }
    
    def close(self):
        """关闭环境"""
        self.env.close()

def main():
    """主函数"""
    if len(sys.argv) < 2:
        print("使用方法:")
        print("python model_inference.py <模型路径> [环境类型] [episode数量]")
        print("\n示例:")
        print("python model_inference.py models/cartpole_dqn_final.pth cartpole 5")
        print("python model_inference.py models/maze_dqn_final.pth maze 10")
        return
    
    model_path = sys.argv[1]
    env_type = sys.argv[2] if len(sys.argv) > 2 else "cartpole"
    num_episodes = int(sys.argv[3]) if len(sys.argv) > 3 else 5
    
    try:
        # 创建推理器
        inference = ModelInference(model_path, env_type)
        
        # 运行推理
        results = inference.run_multiple_episodes(num_episodes, render=True)
        
        # 关闭环境
        inference.close()
        
    except Exception as e:
        print(f"推理过程中出现错误: {e}")
        return

if __name__ == "__main__":
    main() 