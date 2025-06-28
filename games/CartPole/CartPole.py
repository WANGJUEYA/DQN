import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import os
import json
import sys

# 添加收敛分析导入
sys.path.append('..')
from framework.convergence_analysis import ConvergenceAnalyzer

# gymnasium=0.29.0 https://blog.csdn.net/qq_43674552/article/details/127344366

# Hyper Parameters 超参数
EPOCH = 400  # 400个episode循环
BATCH_SIZE = 32  # 样本数量
LR = 0.01  # learning rate | 学习率
EPSILON = 0.9  # greedy policy
GAMMA = 0.9  # reward discount
TARGET_REPLACE_ITER = 100  # target update frequency | 目标网络更新频率
MEMORY_CAPACITY = 2000  # 记忆库容量
env = gym.make('CartPole-v1')  # 使用gymnasium库中的环境：CartPole，且打开封装
env = env.unwrapped  # 打开环境封装
N_ACTIONS = env.action_space.n  # 杆子动作个数 (2个)
N_STATES = env.observation_space.shape[0]  # 杆子状态个数 (4个)

# 模型保存路径
MODEL_SAVE_DIR = "models"
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

# 收敛分析保存路径
CONVERGENCE_SAVE_DIR = "convergence_analysis"
os.makedirs(CONVERGENCE_SAVE_DIR, exist_ok=True)

"""
torch.nn是专门为神经网络设计的模块化接口。nn构建于Autograd之上，可以用来定义和运行神经网络。
nn.Module是nn中十分重要的类，包含网络各层的定义及forward方法。
定义网络：
    需要继承nn.Module类，并实现forward方法。
    一般把网络中具有可学习参数的层放在构造函数__init__()中。
    只要在nn.Module的子类中定义了forward函数，backward函数就会被自动实现(利用Autograd)。
"""


# 定义Net类 (定义网络)
class Net(nn.Module):

    def __init__(self):  # 定义Net的一系列属性
        # nn.Module的子类函数必须在构造函数中执行父类的构造函数
        super(Net, self).__init__()  # 等价与nn.Module.__init__()
        self.fc1 = nn.Linear(N_STATES, 20)  # 设置第一个全连接层(输入层到隐藏层): 状态数个神经元到20个神经元
        self.fc1.weight.data.normal_(0, 0.1)  # 权重初始化 (均值为0，方差为0.1的正态分布)
        self.fc2 = nn.Linear(20, N_ACTIONS)  # 设置第二个全连接层(隐藏层到输出层): 20个神经元到动作数个神经元
        self.fc2.weight.data.normal_(0, 0.1)  # 权重初始化 (均值为0，方差为0.1的正态分布)

    def forward(self, x):  # 定义forward函数 (x为状态)
        x = F.relu(self.fc1(x))  # 连接输入层到隐藏层，且使用激励函数ReLU来处理经过隐藏层后的值
        return self.fc2(x)  # 连接隐藏层到输出层，获得最终的输出值 (即动作值)


# 定义DQN类 (定义两个网络)
class DQN(object):

    def __init__(self):  # 定义DQN的一系列属性
        self.target_net, self.evaluate_net = Net(), Net()  # 利用Net创建两个神经网络: 评估网络和目标网络
        self.memory = np.zeros((MEMORY_CAPACITY, N_STATES * 2 + 2))  # 初始化记忆库，一行代表一个transition
        self.loss_Function = nn.MSELoss()  # 使用均方损失函数 (loss(xi, yi)=(xi-yi)^2)
        self.optimizer = torch.optim.Adam(self.evaluate_net.parameters(), lr=LR)  # 使用Adam优化器 (输入为评估网络的参数和学习率)
        self.point = 0  # for storing memory
        self.learn_step = 0  # for target updating
        
        # 初始化收敛分析器
        self.convergence_analyzer = ConvergenceAnalyzer(window_size=50)

    def choose_action(self, s):  # 定义动作选择函数 (s为状态)
        # 确保状态是正确的格式
        if isinstance(s, tuple):
            s = s[0]  # 如果是元组，取第一个元素（状态）
        s = torch.unsqueeze(torch.FloatTensor(s), 0)  # 将s转换成32-bit floating point形式，并在dim=0增加维数为1的维度
        if np.random.uniform() < EPSILON:  # epsilon-greedy 生成一个在[0, 1)内的随机数，如果小于EPSILON，选择最优动作
            return torch.max(self.evaluate_net.forward(s), 1)[1].data.numpy()[0]  # 通过对评估网络输入状态s，前向传播获得动作值
        else:  # 随机选择动作
            return np.random.randint(0, N_ACTIONS)  # 这里action随机等于0或1 (N_ACTIONS = 2)

    def store_transition(self, s, a, r, s_):  # 定义记忆存储函数 (这里输入为一个transition)
        # 确保状态是正确的格式
        if isinstance(s, tuple):
            s = s[0]
        if isinstance(s_, tuple):
            s_ = s_[0]
        self.memory[self.point % MEMORY_CAPACITY, :] = np.hstack((s, [a, r], s_))  # 如果记忆库满了，便覆盖旧的数据
        self.point += 1  # memory_counter自加1

    def sample_batch_data(self, batch_size):  # 抽取记忆库中的批数据
        perm_idx = np.random.choice(len(self.memory), batch_size)
        return self.memory[perm_idx]

    def learn(self) -> float:  # 定义学习函数(记忆库已满后便开始学习)
        # 目标网络参数更新
        if self.learn_step % TARGET_REPLACE_ITER == 0:  # 一开始触发，然后每100步触发
            self.target_net.load_state_dict(self.evaluate_net.state_dict())  # 将评估网络的参数赋给目标网络
        self.learn_step += 1  # 学习步数自加1

        # 抽取32个索引对应的32个transition，存入batch_memory
        batch_memory = self.sample_batch_data(BATCH_SIZE)
        # 将32个s抽出，转为32-bit floating point形式，并存储到batch_state中，batch_state为32行4列
        batch_state = torch.FloatTensor(batch_memory[:, :N_STATES])
        # 将32个a抽出，转为64-bit integer (signed)形式，并存储到batch_action中 (LongTensor类型方便后面torch.gather的使用)，batch_action为32行1列
        batch_action = torch.LongTensor(batch_memory[:, N_STATES: N_STATES + 1].astype(int))
        # 将32个r抽出，转为32-bit floating point形式，并存储到batch_reward中，batch_reward为32行1列
        batch_reward = torch.FloatTensor(batch_memory[:, N_STATES + 1: N_STATES + 2])
        # 将32个s_抽出，转为32-bit floating point形式，并存储到batch_next_state中，batch_next_state为32行4列
        batch_next_state = torch.FloatTensor(batch_memory[:, -N_STATES:])

        # 获取32个transition的评估值和目标值，并利用损失函数和优化器进行评估网络参数更新
        q_eval = self.evaluate_net(batch_state).gather(1, batch_action)
        # eval_net(b_s)通过评估网络输出32行每个b_s对应的一系列动作值，然后.gather(1, b_a)代表对每行对应索引b_a的Q值提取进行聚合
        q_next = self.target_net(batch_next_state).detach()  # target network
        # q_next不进行反向传递误差，所以detach；q_next表示通过目标网络输出32行每个b_s_对应的一系列动作值
        q_target = batch_reward + GAMMA * q_next.max(1)[0].view(BATCH_SIZE, 1)
        # q_next.max(1)[0]表示只返回每一行的最大值，不返回索引(长度为32的一维张量)；.view()表示把前面所得到的一维张量变成(BATCH_SIZE, 1)的形状；最终通过公式得到目标值
        loss = self.loss_Function(q_eval, q_target)

        # 输入32个评估值和32个目标值，使用均方损失函数
        self.optimizer.zero_grad()  # 清空上一步的残余更新参数值
        loss.backward()  # 误差反向传播, 计算参数更新值
        self.optimizer.step()  # 更新评估网络的所有参数

        return loss.data.numpy()  # 返回损失函数数值

    def predict_action(self, s, epsilon=0.0):  # 推理预测函数，用于模型推理
        """推理预测函数，epsilon=0表示完全贪婪策略"""
        # 确保状态是正确的格式
        if isinstance(s, tuple):
            s = s[0]  # 如果是元组，取第一个元素（状态）
        s = torch.unsqueeze(torch.FloatTensor(s), 0)
        if np.random.uniform() < epsilon:  # 如果epsilon>0，仍然有探索
            return np.random.randint(0, N_ACTIONS)
        else:
            return torch.max(self.evaluate_net.forward(s), 1)[1].data.numpy()[0]

    def save_model(self, model_path="cartpole_dqn.pt"):
        """保存模型"""
        # 如果传入的是完整路径，直接使用；否则使用相对路径
        if os.path.isabs(model_path) or "/" in model_path or "\\" in model_path:
            full_path = model_path
        else:
            full_path = os.path.join(MODEL_SAVE_DIR, model_path)
            
        # 确保目录存在
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        
        torch.save({
            'evaluate_net_state_dict': self.evaluate_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'hyperparameters': {
                'N_STATES': N_STATES,
                'N_ACTIONS': N_ACTIONS,
                'LR': LR,
                'GAMMA': GAMMA,
                'EPSILON': EPSILON,
                'MEMORY_CAPACITY': MEMORY_CAPACITY,
                'BATCH_SIZE': BATCH_SIZE,
                'TARGET_REPLACE_ITER': TARGET_REPLACE_ITER
            }
        }, full_path)
        print(f"模型已保存到: {full_path}")

    def load_model(self, model_path="cartpole_dqn.pt"):
        """加载模型"""
        # 如果传入的是完整路径，直接使用；否则使用相对路径
        if os.path.isabs(model_path) or "/" in model_path or "\\" in model_path:
            full_path = model_path
        else:
            full_path = os.path.join(MODEL_SAVE_DIR, model_path)
            
        if os.path.exists(full_path):
            checkpoint = torch.load(full_path)
            self.evaluate_net.load_state_dict(checkpoint['evaluate_net_state_dict'])
            self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print(f"模型已从 {full_path} 加载")
            return True
        else:
            print(f"模型文件 {full_path} 不存在")
            return False

    def add_episode_data_to_analyzer(self, episode: int, reward: float, avg_loss: float, 
                                   steps: int, success: bool, epsilon: float):
        """添加episode数据到收敛分析器"""
        self.convergence_analyzer.add_episode_data(
            episode=episode,
            reward=reward,
            loss=avg_loss,
            steps=steps,
            success=success,
            epsilon=epsilon
        )

    def save_convergence_analysis(self, episode: int):
        """保存收敛分析结果"""
        # 每50个episode保存一次分析
        if episode % 50 == 0:
            # 生成收敛报告
            report = self.convergence_analyzer.generate_convergence_report()
            
            # 保存报告到文件
            with open(f"{CONVERGENCE_SAVE_DIR}/convergence_report_episode_{episode}.txt", 'w', encoding='utf-8') as f:
                f.write(report)
            
            # 保存分析数据
            self.convergence_analyzer.save_analysis_data(
                f"{CONVERGENCE_SAVE_DIR}/convergence_data_episode_{episode}.json"
            )
            
            # 打印当前收敛状态
            print(f"\n=== Episode {episode} 收敛分析 ===")
            print(report)

def inference_demo(model_name="cartpole_dqn.pt", episodes=5):
    """推理演示函数"""
    print("=" * 50)
    print("开始推理演示...")
    
    # 创建新的DQN实例用于推理
    dqn_inference = DQN()
    
    # 加载训练好的模型
    if not dqn_inference.load_model(model_name):
        print("无法加载模型，请先训练模型")
        return
    
    # 设置推理模式
    dqn_inference.evaluate_net.eval()
    dqn_inference.target_net.eval()
    
    total_rewards = []
    
    for episode in range(episodes):
        state = env.reset()
        episode_reward = 0
        step = 0
        
        print(f"\nEpisode {episode + 1}:")
        
        while True:
            env.render()  # 显示动画
            
            # 使用推理模式选择动作（epsilon=0，完全贪婪）
            action = dqn_inference.predict_action(state, epsilon=0.0)
            
            # 执行动作
            next_state, reward, done, _ = env.step(action)
            
            episode_reward += reward
            step += 1
            
            if done:
                print(f"  步数: {step}, 总奖励: {episode_reward:.2f}")
                total_rewards.append(episode_reward)
                break
                
            state = next_state
    
    env.close()
    
    avg_reward = np.mean(total_rewards)
    print(f"\n推理结果:")
    print(f"平均奖励: {avg_reward:.2f}")
    print(f"最高奖励: {max(total_rewards):.2f}")
    print(f"最低奖励: {min(total_rewards):.2f}")


if __name__ == "__main__":
    import sys
    
    # 检查命令行参数
    if len(sys.argv) > 1 and sys.argv[1] == "inference":
        # 推理模式
        model_name = sys.argv[2] if len(sys.argv) > 2 else "cartpole_dqn.pt"
        episodes = int(sys.argv[3]) if len(sys.argv) > 3 else 5
        inference_demo(model_name, episodes)
    else:
        # 训练模式
        dqn = DQN()

        writer = SummaryWriter("run/MemoryCapacity_100_CustomReward/")
        writer.add_graph(dqn.evaluate_net, torch.randn(1, N_STATES))

        global_step = 0  # 绘图横坐标
        for i in range(EPOCH):  # episode循环
            s = env.reset()  # 重置环境
            running_loss = 0  # 损失函数值
            cumulated_reward = 0  # 初始化该循环对应的episode的总奖励
            step = 0
            loss_count = 0  # 用于计算平均损失

            while True:
                global_step += 1
                env.render()  # 显示实验动画
                a = dqn.choose_action(s)  # 输入该步对应的状态s，选择动作
                s_, r, done, _ = env.step(a)  # 执行动作，获得反馈

                # 修改奖励 (不修改也可以，修改奖励只是为了更快地得到训练好的摆杆)
                x, x_dot, theta, theta_dot = s_
                r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
                r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
                r = r1 + r2

                dqn.store_transition(s, a, r, s_)  # 存储样本

                cumulated_reward += r  # 逐步加上一个episode内每个step的reward
                if dqn.point > MEMORY_CAPACITY:  # 如果累计的transition数量超过了记忆库的固定容量2000
                    # 开始学习 (抽取记忆，即32个transition，并对评估网络参数进行更新，并在开始学习后每隔100次将评估网络的参数赋给目标网络)
                    loss = dqn.learn()
                    running_loss += loss
                    loss_count += 1
                    if done or step > 2000:
                        avg_loss = running_loss / loss_count if loss_count > 0 else 0
                        
                        # 判断是否成功（CartPole通常认为195步以上为成功）
                        success = step >= 195
                        
                        # 计算当前epsilon值
                        current_epsilon = max(0.1, EPSILON * np.exp(-i / 100))
                        
                        # 添加数据到收敛分析器
                        dqn.add_episode_data_to_analyzer(
                            episode=i+1,
                            reward=cumulated_reward,
                            avg_loss=avg_loss,
                            steps=step,
                            success=success,
                            epsilon=current_epsilon
                        )
                        
                        print("FAILEpisode: %d| Step: %d| Loss:  %.4f, Reward: %.2f" % (
                            i, step, avg_loss, cumulated_reward))
                        writer.add_scalar("training/Loss", avg_loss, global_step)
                        writer.add_scalar("training/Reward", cumulated_reward, global_step)
                        
                        # 保存收敛分析
                        dqn.save_convergence_analysis(i+1)
                        break
                else:
                    print("\rCollecting experience: %d / %d..." % (dqn.point, MEMORY_CAPACITY), end='')

                if done:
                    break
                if step % 100 == 99:
                    print("Episode: %d| Step: %d| Loss:  %.4f, Reward: %.2f" % (
                        i, step, running_loss / step, cumulated_reward))
                step += 1
                s = s_
            
            # 每50个episode保存一次模型
            if (i + 1) % 50 == 0:
                dqn.save_model(f"cartpole_dqn_episode_{i+1}.pt")
        
        # 训练完成后保存最终模型
        dqn.save_model("cartpole_dqn_final.pt")
        
        # 生成最终收敛分析
        print("\n生成最终收敛分析...")
        final_report = dqn.convergence_analyzer.generate_convergence_report()
        
        # 保存最终报告
        with open(f"{CONVERGENCE_SAVE_DIR}/final_convergence_report.txt", 'w', encoding='utf-8') as f:
            f.write(final_report)
        
        # 保存最终分析数据
        dqn.convergence_analyzer.save_analysis_data(
            f"{CONVERGENCE_SAVE_DIR}/final_convergence_data.json"
        )
        
        print(final_report)
        
        env.close()
        print("\n训练完成！模型和收敛分析已保存。")
        print("使用方法:")
        print("python CartPole.py inference                    # 使用默认模型进行推理")
        print("python CartPole.py inference model_name.pt     # 使用指定模型进行推理")
        print("python CartPole.py inference model_name.pt 10  # 使用指定模型进行10次推理")
