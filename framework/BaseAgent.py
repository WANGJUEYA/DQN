from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, SupportsFloat, TypeVar
from typing import Optional

import numpy as np
import torch

from framework.convergence_analysis import ConvergenceAnalyzer

ObsType = TypeVar("ObsType")
ActType = TypeVar("ActType")


class BaseAgent(ABC):
    """基础智能体类，封装训练、推理、模型管理、分析等通用逻辑"""

    def __init__(self, output_dir="outputs", model_dir="models", training_mode=False):
        # 环境变量
        self.output_dir = Path(output_dir)
        self.model_dir = Path(model_dir)
        self.convergence_analyzer = ConvergenceAnalyzer()
        # 智能体超参数
        self.batch_size = 32  # 样本数量
        self.lr = 0.01  # learning rate | 学习率
        self.epsilon = 0.9  # greedy policy 随机策略
        self.gamma = 0.9  # reward discount
        self.target_replace_iter = 100  # target update frequency | 目标网络更新频率
        self.memory_capacity = 2000  # 记忆容量

        self._init_env_and_agent()
        self.training_number = None
        if training_mode:
            self.training_number = self._get_training_number_and_create_output_dir()
        else:
            # 非训练模式下，找到最新的训练编号和输出目录
            game_output_root = self.output_dir / self.game_name
            if game_output_root.exists():
                all_dirs = [d for d in game_output_root.iterdir() if d.is_dir() and d.name.isdigit()]
                if all_dirs:
                    latest = max(all_dirs, key=lambda d: int(d.name))
                    self.game_output_dir = latest
                else:
                    self.game_output_dir = game_output_root / "1"
            else:
                self.game_output_dir = game_output_root / "1"

    @property
    @abstractmethod
    def game_name(self) -> str:
        pass

    @abstractmethod
    def _init_env_and_agent(self):
        """初始化环境和智能体自身，需设置self.env, self._game_name"""
        pass

    def choose_action(self, s):  # 定义动作选择函数 (s为状态)
        # 确保状态是正确的格式
        if isinstance(s, tuple):
            s = s[0]  # 如果是元组，取第一个元素（状态）
        s = torch.unsqueeze(torch.FloatTensor(s), 0)  # 将s转换成32-bit floating point形式，并在dim=0增加维数为1的维度
        if np.random.uniform() < self.epsilon:  # epsilon-greedy 生成一个在[0, 1)内的随机数，如果小于EPSILON，选择最优动作
            return torch.max(self.evaluate_net.forward(s), 1)[1].data.numpy()[0]  # 通过对评估网络输入状态s，前向传播获得动作值
        else:  # 随机选择动作
            return np.random.randint(0, self.num_actions)  # 这里action随机

    def execute_step(self, action) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        return self.env.step(action)

    def store_transition(self, s, a, r, s_):  # 定义记忆存储函数 (这里输入为一个transition)
        # 确保状态是正确的格式
        if isinstance(s, tuple):
            s = s[0]
        if isinstance(s_, tuple):
            s_ = s_[0]
        self.memory[self.point % self.memory_capacity, :] = np.hstack((s, [a, r], s_))  # 如果记忆库满了，便覆盖旧的数据
        self.point += 1  # memory_counter自加1

    def sample_batch_data(self, batch_size):  # 抽取记忆库中的批数据
        perm_idx = np.random.choice(len(self.memory), batch_size)
        return self.memory[perm_idx]

    def learn(self) -> float:  # 定义学习函数(记忆库已满后便开始学习)
        # 目标网络参数更新
        if self.learn_step % self.target_replace_iter == 0:  # 一开始触发，然后每100步触发
            self.target_net.load_state_dict(self.evaluate_net.state_dict())  # 将评估网络的参数赋给目标网络
        self.learn_step += 1  # 学习步数自加1

        # 抽取32个索引对应的32个transition，存入batch_memory
        batch_memory = self.sample_batch_data(self.batch_size)
        # 将32个s抽出，转为32-bit floating point形式，并存储到batch_state中，batch_state为32行4列
        batch_state = torch.FloatTensor(batch_memory[:, :self.num_states])
        # 将32个a抽出，转为64-bit integer (signed)形式，并存储到batch_action中 (LongTensor类型方便后面torch.gather的使用)，batch_action为32行1列
        batch_action = torch.LongTensor(batch_memory[:, self.num_states: self.num_states + 1].astype(int))
        # 将32个r抽出，转为32-bit floating point形式，并存储到batch_reward中，batch_reward为32行1列
        batch_reward = torch.FloatTensor(batch_memory[:, self.num_states + 1: self.num_states + 2])
        # 将32个s_抽出，转为32-bit floating point形式，并存储到batch_next_state中，batch_next_state为32行4列
        batch_next_state = torch.FloatTensor(batch_memory[:, -self.num_states:])

        # 获取32个transition的评估值和目标值，并利用损失函数和优化器进行评估网络参数更新
        q_eval = self.evaluate_net(batch_state).gather(1, batch_action)
        # eval_net(b_s)通过评估网络输出32行每个b_s对应的一系列动作值，然后.gather(1, b_a)代表对每行对应索引b_a的Q值提取进行聚合
        q_next = self.target_net(batch_next_state).detach()  # target network
        # q_next不进行反向传递误差，所以detach；q_next表示通过目标网络输出32行每个b_s_对应的一系列动作值
        q_target = batch_reward + self.gamma * q_next.max(1)[0].view(self.batch_size, 1)
        # q_next.max(1)[0]表示只返回每一行的最大值，不返回索引(长度为32的一维张量)；.view()表示把前面所得到的一维张量变成(BATCH_SIZE, 1)的形状；最终通过公式得到目标值
        loss = self.loss_Function(q_eval, q_target)

        # 输入32个评估值和32个目标值，使用均方损失函数
        self.optimizer.zero_grad()  # 清空上一步的残余更新参数值
        loss.backward()  # 误差反向传播, 计算参数更新值
        self.optimizer.step()  # 更新评估网络的所有参数

        return loss.data.numpy()  # 返回损失函数数值

    def _get_training_number_and_create_output_dir(self) -> int:
        game_output_root = self.output_dir / self.game_name
        game_output_root.mkdir(parents=True, exist_ok=True)
        max_count = 0
        for d in game_output_root.iterdir():
            if d.is_dir() and d.name.isdigit():
                max_count = max(max_count, int(d.name))
        count = max_count + 1
        self.game_output_dir = game_output_root / str(count)
        self.game_output_dir.mkdir(parents=True, exist_ok=True)
        (self.game_output_dir / "reports").mkdir(exist_ok=True)
        return count

    def _get_training_number(self) -> int:
        max_training_number = 0
        if self.model_dir.exists():
            pattern = f"{self.game_name}_dqn_training_*_*.pt"
            training_models = list(self.model_dir.glob(pattern))
            for model_file in training_models:
                parts = model_file.stem.split('_')
                if len(parts) >= 4 and parts[0] == self.game_name and parts[1] == "dqn" and parts[2] == "training":
                    try:
                        training_number = int(parts[3])
                        max_training_number = max(max_training_number, training_number)
                    except ValueError:
                        continue
        return max_training_number + 1

    def train(self, episodes: int, save_interval: int = 50, model_name: Optional[str] = None, render: bool = False):
        import time
        print(f"开始训练 {self.game_name} 模型...")
        print(f"训练参数: episodes={episodes}, save_interval={save_interval}")
        if self.training_number is None:
            self.training_number = self._get_training_number_and_create_output_dir()
        training_number = self.training_number
        print(f"这是第 {training_number} 次训练")
        best_reward = float('-inf')
        best_episode = 0
        best_model_path = None  # 初始化best_model_path

        train_start_time = time.time()  # 记录训练开始时间
        try:
            for episode in range(episodes + self.memory_capacity):
                state = self.env.reset()  # 重置环境
                total_loss = 0  # 损失函数值
                total_reward = 0  # 初始化该循环对应的episode的总奖励
                steps = 0  # 本回合的总步数
                success = False  # 本回合是否成功

                while True:
                    if render:
                        self.env.render()

                    action = self.choose_action(state)  # 输入该步对应的状态s，选择动作
                    observation, reward, terminated, truncated, info = self.execute_step(action)  # 执行动作，获得反馈

                    self.store_transition(state, action, reward, observation)
                    total_reward += reward
                    if self.point > self.memory_capacity:
                        # 开始学习 (抽取记忆，即32个transition，并对评估网络参数进行更新，并在开始学习后每隔100次将评估网络的参数赋给目标网络)
                        loss = self.learn()
                        total_loss += loss
                    else:
                        print("\rCollecting experience: %d / %d..." % (self.point, self.memory_capacity), end='')
                    # 完成或者最大步数跳出循环
                    if terminated or truncated:
                        if terminated:
                            success = True
                        else:
                            print(f"  Episode {episode + 1} 达到停止条件，强制结束")
                        break

                    steps += 1
                    state = observation

                if self.point <= self.memory_capacity:
                    continue

                avg_loss = total_loss / max(steps, 1)

                self.convergence_analyzer.add_episode_data(
                    episode=episode,
                    reward=total_reward,
                    loss=avg_loss,
                    steps=steps,
                    success=success,
                    epsilon=0.9
                )

                if steps % 200 == 0:
                    print(f"Episode {episode + 1}/{episodes}: "
                          f"Reward={total_reward:.2f}, "
                          f"Steps={steps}, "
                          f"AvgLoss={avg_loss:.4f}, "
                          f"Success={success}")

                if total_reward > best_reward:
                    best_reward = total_reward
                    best_episode = episode + 1
                    # 保存最优模型到models目录
                    best_model_name = f"{self.game_name}_dqn_training_{training_number}_best.pt"
                    best_model_path = self.model_dir / best_model_name
                    self.save_model(str(best_model_path))
                    checkpoint = torch.load(best_model_path, map_location='cpu', weights_only=False)
                    checkpoint['best_reward'] = best_reward
                    checkpoint['best_episode'] = best_episode
                    checkpoint['training_number'] = training_number
                    torch.save(checkpoint, best_model_path)
                    print(f"  🎉 发现新的最优模型！Episode {episode + 1}, Reward: {total_reward:.2f}")

                if (episode + 1) % save_interval == 0:
                    self._save_checkpoint(episode, training_number)
                    self._generate_reports(episode, train_start_time)

            final_model_name = f"{self.game_name}_dqn_training_{training_number}_final.pt"
            final_model_path = self.model_dir / final_model_name
            self.save_model(str(final_model_path))
            # 更新全局最优模型
            if best_model_path is not None:  # 确保有最优模型存在
                global_best_model_path = self.model_dir / f"{self.game_name}_dqn_best.pt"
                current_global_best_reward = self._get_global_best_reward()
                if best_reward > current_global_best_reward:
                    import shutil
                    shutil.copy2(str(best_model_path), str(global_best_model_path))
                    print(f"  🏆 更新全局最优模型！Reward: {best_reward:.2f}")
            self._save_checkpoint(episodes - 1, training_number)
            self._generate_reports(episodes - 1, train_start_time)
        except KeyboardInterrupt:
            print(f"\n⚠️ 训练被用户中断")
            # 保存当前进度
            if best_model_path is not None:
                print(f"已保存最优模型: {best_model_path}")
        except Exception as e:
            print(f"❌ 训练过程中出现异常: {e}")
            import traceback
            traceback.print_exc()
        finally:
            print(f"✅ 训练完成！")
            print(f"最优奖励: {best_reward:.2f} (Episode {best_episode})")
            print(f"输出目录: {self.game_output_dir}")

    def _get_global_best_reward(self) -> float:
        global_best_model_path = self.model_dir / f"{self.game_name}_dqn_best.pt"
        if global_best_model_path.exists():
            try:
                checkpoint = torch.load(global_best_model_path, map_location='cpu', weights_only=False)
                return checkpoint.get('best_reward', float('-inf'))
            except:
                return float('-inf')
        return float('-inf')

    def _save_checkpoint(self, episode: int, training_number: int):
        convergence_dir = self.game_output_dir / "convergence_analysis"
        convergence_dir.mkdir(exist_ok=True)

        # 保存收敛分析数据
        convergence_file = convergence_dir / f"convergence_data_{episode}.json"
        self.convergence_analyzer.save_analysis_data(str(convergence_file))

        # 保存模型检查点到outputs目录（不在models目录）
        checkpoint_model_name = f"{self.game_name}_dqn_training_{training_number}_checkpoint_{episode}.pt"
        checkpoint_model_path = self.game_output_dir / "process_models" / checkpoint_model_name
        checkpoint_model_path.parent.mkdir(exist_ok=True)
        self.save_model(str(checkpoint_model_path))

        # 保存检查点元数据
        checkpoint_metadata = {
            'episode': episode,
            'training_number': training_number,
            'game_name': self.game_name,
            'convergence_file': str(convergence_file.relative_to(self.game_output_dir))
        }
        checkpoint_metadata_file = convergence_dir / f"checkpoint_metadata_{episode}.json"
        import json
        with open(checkpoint_metadata_file, 'w', encoding='utf-8') as f:
            json.dump(checkpoint_metadata, f, ensure_ascii=False, indent=2)

        print(f"  💾 保存检查点: Episode {episode + 1}")

    def _generate_reports(self, episode: int, train_start_time=None):
        """生成训练报告"""
        import time
        reports_dir = self.game_output_dir / "reports"
        reports_dir.mkdir(exist_ok=True)

        # 计算训练时间
        total_seconds = None
        if train_start_time is not None:
            total_seconds = time.time() - train_start_time

        # 生成收敛分析报告
        report_path = reports_dir / f"convergence_report_{episode}.txt"
        self.convergence_analyzer.generate_convergence_report(str(report_path), total_seconds=total_seconds)

        # 生成收敛分析图表
        plot_path = reports_dir / f"convergence_plot_{episode}.png"
        print(f"  📊 正在生成图表: {plot_path}")
        try:
            self.convergence_analyzer.plot_convergence_analysis(str(plot_path), show_plot=False,
                                                                total_seconds=total_seconds)
            if plot_path.exists():
                print(f"  ✅ 图表生成成功: {plot_path}")
            else:
                print(f"  ❌ 图表文件未生成: {plot_path}")
        except Exception as e:
            print(f"  ❌ 图表生成失败: {e}")

        print(f"  📊 生成报告: Episode {episode + 1}")

    def inference(self, model_name: Optional[str] = None, episodes: int = 5):
        """模型推理"""
        print(f"开始推理 {self.game_name} 模型...")

        # 确定使用的模型
        if model_name is None:
            best_model = self._get_best_model()
            if best_model is None:
                print("❌ 未找到可用的模型文件")
                return
            model_path = self.model_dir / best_model
        else:
            model_path = self.model_dir / model_name
            if not model_path.exists():
                print(f"❌ 模型文件不存在: {model_path}")
                return

        print(f"使用模型: {model_path.name}")

        # 加载模型
        try:
            self.load_model(str(model_path))
            print("✅ 模型加载成功")
        except Exception as e:
            print(f"❌ 模型加载失败: {e}")
            return

        # 执行推理
        total_rewards = []
        total_steps = []
        success_count = 0

        for episode in range(episodes):
            reset_result = self.env.reset()
            if isinstance(reset_result, tuple):
                state = reset_result[0]
            else:
                state = reset_result

            total_reward = 0
            steps = 0

            while True:
                self.env.render()
                action = self.choose_action(state)  # 推理时不使用随机探索
                step_result = self.env.step(action)

                if len(step_result) == 5:
                    next_state, reward, terminated, truncated, info = step_result
                    done = terminated or truncated
                else:
                    next_state, reward, done, info = step_result

                state = next_state
                total_reward += reward
                steps += 1

                if steps >= max_steps:
                    done = True
                    print(f"  Episode {episode + 1} 达到最大步数限制 ({max_steps})，强制结束")

                if done:
                    success = self._is_success(episode, steps, total_reward)
                    break

            total_rewards.append(total_reward)
            total_steps.append(steps)
            if success:
                success_count += 1

            print(f"Episode {episode + 1}/{episodes}: "
                  f"Reward={total_reward:.2f}, "
                  f"Steps={steps}, "
                  f"Success={success}")

        # 输出推理统计
        avg_reward = sum(total_rewards) / len(total_rewards)
        avg_steps = sum(total_steps) / len(total_steps)
        success_rate = success_count / episodes

        print(f"\n📊 推理统计:")
        print(f"平均奖励: {avg_reward:.2f}")
        print(f"平均步数: {avg_steps:.1f}")
        print(f"成功率: {success_rate:.1%} ({success_count}/{episodes})")
        print(f"✅ 推理完成！")

    def list_models(self):
        """列出可用的模型文件"""
        print(f"📁 {self.game_name} 模型文件列表:")
        print(f"模型目录: {self.model_dir}")

        if not self.model_dir.exists():
            print("❌ 模型目录不存在")
            return

        models = list(self.model_dir.glob("*.pt"))
        if not models:
            print("❌ 未找到模型文件")
            return

        # 分类显示模型
        global_models = [m for m in models if not m.name.startswith(f"{self.game_name}_dqn_training_")]
        training_models = [m for m in models if m.name.startswith(f"{self.game_name}_dqn_training_")]

        if global_models:
            print("\n🌍 全局模型:")
            for model in sorted(global_models):
                size_mb = model.stat().st_size / (1024 * 1024)
                print(f"  📄 {model.name} ({size_mb:.1f} MB)")

        if training_models:
            print("\n🎯 训练模型:")
            # 按训练编号分组
            training_groups = {}
            for model in training_models:
                parts = model.stem.split('_')
                if len(parts) >= 4:
                    training_num = parts[3]
                    if training_num not in training_groups:
                        training_groups[training_num] = []
                    training_groups[training_num].append(model)

            for training_num in sorted(training_groups.keys(), key=int):
                print(f"  训练 #{training_num}:")
                for model in sorted(training_groups[training_num]):
                    size_mb = model.stat().st_size / (1024 * 1024)
                    print(f"    📄 {model.name} ({size_mb:.1f} MB)")

    def list_outputs(self):
        """列出训练输出文件"""
        print(f"📁 {self.game_name} 输出文件列表:")
        print(f"输出目录: {self.output_dir}")

        game_output_root = self.output_dir / self.game_name
        if not game_output_root.exists():
            print("❌ 输出目录不存在")
            return

        training_dirs = [d for d in game_output_root.iterdir() if d.is_dir() and d.name.isdigit()]
        if not training_dirs:
            print("❌ 未找到训练输出")
            return

        print(f"\n🎯 训练输出 (共 {len(training_dirs)} 次):")
        for training_dir in sorted(training_dirs, key=lambda x: int(x.name)):
            print(f"\n  训练 #{training_dir.name}:")
            print(f"    路径: {training_dir}")

            # 检查子目录
            subdirs = ['reports', 'convergence_analysis', 'process_models']
            for subdir in subdirs:
                subdir_path = training_dir / subdir
                if subdir_path.exists():
                    files = list(subdir_path.iterdir())
                    if files:
                        print(f"    📁 {subdir}: {len(files)} 个文件")
                        for file in sorted(files)[:3]:  # 只显示前3个文件
                            print(f"      📄 {file.name}")
                        if len(files) > 3:
                            print(f"      ... 还有 {len(files) - 3} 个文件")

    def _get_best_model(self) -> Optional[str]:
        """获取最优模型文件名"""
        # 首先尝试全局最优模型
        best_model_path = self.model_dir / f"{self.game_name}_dqn_best.pt"
        if best_model_path.exists():
            return best_model_path.name

        # 然后尝试训练过程中的最优模型
        pattern = f"{self.game_name}_dqn_training_*_best.pt"
        best_models = list(self.model_dir.glob(pattern))
        if best_models:
            # 按训练编号排序，取最新的
            best_models.sort(key=lambda x: int(x.stem.split('_')[3]))
            return best_models[-1].name

        # 最后尝试最终模型
        final_model_path = self.model_dir / f"{self.game_name}_dqn_final.pt"
        if final_model_path.exists():
            return final_model_path.name

        return None

    def save_model(self, path: str):
        """保存模型"""
        # 确保模型目录存在
        model_dir = Path(path).parent
        model_dir.mkdir(parents=True, exist_ok=True)

        torch.save({
            'evaluate_net_state_dict': self.evaluate_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'game_name': self.game_name,
        }, path)

    def load_model(self, path: str):
        """加载模型"""
        checkpoint = torch.load(path, map_location='cpu')
        self.evaluate_net.load_state_dict(checkpoint['evaluate_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
