import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from framework.BaseAgent import BaseAgent

# Hyper Parameters 超参数
BATCH_SIZE = 32  # 样本数量
LR = 0.01  # learning rate | 学习率
EPSILON = 0.9  # greedy policy
GAMMA = 0.9  # reward discount
TARGET_REPLACE_ITER = 100  # target update frequency | 目标网络更新频率
MEMORY_CAPACITY = 2000  # 记忆库容量
N_ACTIONS = 2  # 杆子动作个数 (2个)
N_STATES = 4  # 杆子状态个数 (4个)

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

class CartPoleAgent(BaseAgent):

    def _init_env_and_agent(self):
        self.env = gym.make('CartPole-v1', render_mode='human')
        self.agent = self
        self._max_steps = 500
        self._game_name = "CartPole"
        self.target_net, self.evaluate_net = Net(), Net()
        self.memory = np.zeros((MEMORY_CAPACITY, N_STATES * 2 + 2))
        self.loss_Function = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.evaluate_net.parameters(), lr=LR)
        self.point = 0
        self.learn_step = 0

    def _is_success(self, episode, steps, reward):
        return steps >= 500

    def _get_max_steps(self):
        return self._max_steps

    @property
    def game_name(self):
        return self._game_name

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
