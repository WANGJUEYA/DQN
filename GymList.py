import gym

if __name__ == "__main__":
    # 获取所有环境的列表
    all_envs = gym.envs.registry.all()

    # 打印环境的名称
    for env in all_envs:
        print(env)
