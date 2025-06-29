import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from framework.BaseAgent import BaseAgent
from games.Maze.MazeEnv import DEFAULT_MAZE, MazeEnv

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

    def __init__(self, agent: BaseAgent):  # 定义Net的一系列属性
        # nn.Module的子类函数必须在构造函数中执行父类的构造函数
        super(Net, self).__init__()  # 等价与nn.Module.__init__()
        self.fc1 = nn.Linear(agent.num_states, 100)  # 设置第一个全连接层(输入层到隐藏层): 状态数个神经元到20个神经元
        self.fc1.weight.data.normal_(0, 0.1)  # 权重初始化 (均值为0，方差为0.1的正态分布)
        self.fc2 = nn.Linear(100, agent.num_actions)  # 设置第二个全连接层(隐藏层到输出层): 20个神经元到动作数个神经元
        self.fc2.weight.data.normal_(0, 0.1)  # 权重初始化 (均值为0，方差为0.1的正态分布)

    def forward(self, x):  # 定义forward函数 (x为状态)
        x = F.relu(self.fc1(x))  # 连接输入层到隐藏层，且使用激励函数ReLU来处理经过隐藏层后的值
        return self.fc2(x)  # 连接隐藏层到输出层，获得最终的输出值 (即动作值)


class MazeAgent(BaseAgent):

    def _init_env_and_agent(self):
        self.env = MazeEnv(DEFAULT_MAZE)
        self.num_actions = self.env.action_space.n  # 老鼠的行为空间
        self.num_states = self.env.observation_space  # 老鼠的状态空间 当前位置
        self.target_net, self.evaluate_net = Net(self), Net(self)
        self.memory = np.zeros((self.memory_capacity, self.num_states * 2 + 2))
        self.loss_Function = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.evaluate_net.parameters(), lr=self.lr)
        self.point = 0
        self.learn_step = 0

    @property
    def game_name(self):
        return "Maze"
