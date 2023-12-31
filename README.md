## DQN_base

DQN algorithm by Pytorch - some simple demo

## 简介

它是一个简单的迷宫类游戏，如下图这样
![](_img/maze.png)

DQN训练过程的损失曲线，如下图这样
![](_img/cost.png)

## 系统环境

+ windows + CUDA=11.6
+ miniconda3
+ python=[3.7.3](https://www.python.org/downloads/)
    - conda create -n dqn python=3.7.3
+ pytorch=[1.7.1](https://pytorch.org/get-started/previous-versions/)
    - conda install pytorch==1.7.1 cudatoolkit=11.0 -c pytorch
    - pip install "torch-1.7.1+cpu-cp37-cp37m-win_amd64.whl" # https://download.pytorch.org/whl/torch/
+ pip install gym[classic_control] # gym=0.26.2
+ pip install pygame==2.1.0 tensorboard tensorboardX

## 参考连接

+ https://blog.csdn.net/ningmengzhihe/article/details/130147972