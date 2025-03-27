import argparse
from pathlib import Path
import datetime
import gymnasium as gym
from gymnasium.wrappers import (
    FrameStackObservation,
    TransformObservation,
    GrayscaleObservation,
    ResizeObservation,
    MaxAndSkipObservation,
    RecordVideo,
)
from openai_gym_compatibility import register_gymnasium_envs
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecNormalize
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.type_aliases import ReplayBufferSamples
from typing import Dict, List, Tuple, Union, Optional, Any, Type, cast

# from nes_py.wrappers import JoypadSpace
from joypad_space import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
import torch as th
import torch.nn.functional as F
import numpy as np

register_gymnasium_envs()


class ScoreRewardWrapper(gym.Wrapper):
    def __init__(
        self,
        env,
        score_weight=0.01,
        flag_reward=50,
        timeout_frames=600,
        position_threshold=10,
    ):
        """
        基于游戏得分、旗子状态和超时检测的奖励包装器

        参数:
            env: 环境
            score_weight: 得分奖励的权重系数
            flag_reward: 得到旗子时的额外奖励
            timeout_frames: 马里奥停滞不前的最大帧数，超过后判定为卡住
            position_threshold: 判定为移动的最小距离阈值
        """
        super().__init__(env)
        self.score_weight = score_weight
        self.flag_reward = flag_reward
        self.timeout_frames = timeout_frames
        self.position_threshold = position_threshold
        self.last_score = 0

        # 添加位置和超时检测相关变量
        self.last_x_pos = 0
        self.no_progress_frames = 0
        self.max_x_pos = 0

    def reset(self, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        self.last_score = 0

        # 重置位置和超时检测变量
        self.last_x_pos = 0
        self.no_progress_frames = 0
        self.max_x_pos = 0

        return obs, info

    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)

        # 基础奖励
        modified_reward = float(reward)  # Convert to float to fix type issues

        # 获取当前得分
        current_score = 0
        current_x_pos = 0

        try:
            if isinstance(info, dict):
                # 获取得分
                if "score" in info:
                    current_score = info["score"]

                # 获取当前x坐标
                if "x_pos" in info:
                    current_x_pos = info["x_pos"]
        except (TypeError, KeyError):
            # 如果info不是字典或没有相关键，使用默认值
            current_score = self.last_score
            current_x_pos = self.last_x_pos

        # 计算得分增量
        score_increment = current_score - self.last_score

        # 如果得分增加，提供额外奖励
        if score_increment > 0:
            # 得分奖励与得分增量成正比
            modified_reward += float(self.score_weight * score_increment)

        # 超时检测逻辑
        # 更新最大前进位置
        if current_x_pos > self.max_x_pos:
            self.max_x_pos = current_x_pos
            # 重置停滞帧计数
            self.no_progress_frames = 0
            # 给予前进奖励（可选）
            modified_reward += 0.1  # 小幅奖励前进行为
        else:
            # 如果没有前进超过阈值，增加停滞帧计数
            x_diff = abs(current_x_pos - self.last_x_pos)
            if x_diff < self.position_threshold:
                self.no_progress_frames += 1
            else:
                # 即使没有突破最大距离，但只要有移动也减少计数（避免惩罚探索行为）
                self.no_progress_frames = max(0, self.no_progress_frames - 1)

        # 如果停滞帧数超过阈值且游戏未结束，提前终止
        if self.no_progress_frames >= self.timeout_frames and not done:
            done = True
            truncated = True
            modified_reward -= 10.0  # 因卡住而终止给予负面奖励

        if done:
            # 检查是否得到旗子
            if info.get("flag_get", False):
                modified_reward += float(self.flag_reward)
            else:
                # 如果是因为超时终止，降低惩罚（因为可能只是卡住，不是真正失败）
                if self.no_progress_frames >= self.timeout_frames:
                    modified_reward -= float(self.flag_reward / 2)
                else:
                    modified_reward -= float(self.flag_reward)

        # 更新上一次得分和位置
        self.last_score = current_score
        self.last_x_pos = current_x_pos

        return obs, modified_reward, done, truncated, info


class DoubleDQN(DQN):
    """
    Double Deep Q-Network (Double DQN)

    Paper: https://arxiv.org/abs/1509.06461

    The main difference with regular DQN is that Double DQN uses the online network
    to select actions and the target network to evaluate these actions in the computation
    of the target Q-values.
    """

    def __init__(self, *args, **kwargs):
        # Initialize parent class
        super().__init__(*args, **kwargs)

    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        """
        Modified training method to implement Double DQN.
        """
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update learning rate according to schedule
        self._update_learning_rate(self.policy.optimizer)

        losses = []
        for _ in range(gradient_steps):
            # 确保replay_buffer不为None
            if self.replay_buffer is None:
                # 如果replay_buffer未初始化，跳过训练
                self.logger.record(
                    "train/error", "Replay buffer is None, skipping training step"
                )
                break

            # Sample replay buffer
            replay_data = self.replay_buffer.sample(
                batch_size, env=self._vec_normalize_env
            )

            with th.no_grad():
                # ===== Double DQN modification =====
                # Use current network to select actions (instead of target network)
                next_q_values = self.q_net(replay_data.next_observations)
                next_actions = next_q_values.argmax(dim=1).reshape(-1, 1)

                # Use target network to evaluate the Q-values of these actions
                next_q_values_target = self.q_net_target(replay_data.next_observations)
                next_q_values = th.gather(
                    next_q_values_target, dim=1, index=next_actions
                )
                # ===== End of Double DQN modification =====

                # Avoid potential broadcast issue
                next_q_values = next_q_values.reshape(-1, 1)
                # 1-step TD target
                target_q_values = (
                    replay_data.rewards
                    + (1 - replay_data.dones) * self.gamma * next_q_values
                )

            # Get current Q-values estimates
            current_q_values = self.q_net(replay_data.observations)

            # Retrieve the q-values for the actions from the replay buffer
            current_q_values = th.gather(
                current_q_values, dim=1, index=replay_data.actions.long()
            )

            # 计算损失
            loss = F.smooth_l1_loss(current_q_values, target_q_values)

            losses.append(loss.item())

            # Optimize the policy
            self.policy.optimizer.zero_grad()
            loss.backward()
            # Clip gradient norm
            th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.policy.optimizer.step()

        # Increase update counter
        self._n_updates += gradient_steps

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/loss", np.mean(losses))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Super Mario Bros Training with Stable-Baselines3"
    )
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=2000000,
        help="total timesteps to train (default: 1000000)",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.00025,
        help="learning rate (default: 0.00025)",
    )
    parser.add_argument(
        "--buffer-size",
        type=int,
        default=100000,
        help="size of the replay buffer (default: 200000)",
    )
    parser.add_argument(
        "--learning-starts",
        type=int,
        default=100000,
        help="how many steps before learning starts (default: 100000)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="batch size for training (default: 128)",
    )
    parser.add_argument(
        "--train-freq",
        type=int,
        default=4,
        help="update the model every n steps (default: 4)",
    )
    parser.add_argument(
        "--gradient-steps",
        type=int,
        default=1,
        help="how many gradient steps to do after each rollout (default: 1)",
    )
    parser.add_argument(
        "--target-update-interval",
        type=int,
        default=10000,
        help="update the target network every n steps (default: 10000)",
    )
    parser.add_argument(
        "--exploration-fraction",
        type=float,
        default=0.1,
        help="fraction of entire training period over which exploration rate is reduced (default: 0.1)",
    )
    parser.add_argument(
        "--exploration-initial-eps",
        type=float,
        default=1.0,
        help="initial value of random action probability (default: 1.0)",
    )
    parser.add_argument(
        "--exploration-final-eps",
        type=float,
        default=0.05,
        help="final value of random action probability (default: 0.05)",
    )
    parser.add_argument(
        "--save-freq",
        type=int,
        default=50000,
        help="how often to save the model (default: 50000)",
    )
    parser.add_argument(
        "--eval-freq",
        type=int,
        default=50000,
        help="evaluate the agent every n steps (default: 50000)",
    )
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=5,
        help="number of episodes to evaluate (default: 10)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=12313213,
        help="random seed (default: 42)",
    )
    parser.add_argument(
        "--render", action="store_true", help="render the environment during evaluation"
    )
    parser.add_argument(
        "--load-model",
        type=str,
        default=None,
        help="path to the model to load for continued training",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="device to use for training (default: auto)",
    )
    parser.add_argument(
        "--record-video",
        action="store_true",
        help="record video during evaluation",
    )
    parser.add_argument(
        "--video-freq",
        type=int,
        default=1,
        help="record video every n evaluation episodes (default: 1)",
    )
    parser.add_argument(
        "--video-length",
        type=int,
        default=10000,
        help="maximum length of recorded videos in steps (default: 10000)",
    )
    # 添加超时检测相关参数
    parser.add_argument(
        "--timeout-frames",
        type=int,
        default=600,
        help="maximum number of frames Mario can be stuck before early termination (default: 600)",
    )
    parser.add_argument(
        "--position-threshold",
        type=int,
        default=10,
        help="minimum distance Mario must move to not be considered stuck (default: 10)",
    )
    return parser.parse_args()


def make_env(
    render=False,
    record_video=False,
    video_folder=None,
    seed=0,
    timeout_frames=600,
    position_threshold=10,
):
    """Create the wrapped environment for training or evaluation"""
    # Set appropriate render mode
    if render:
        render_mode = "human"
    elif record_video:
        render_mode = "rgb_array"  # Required for video recording
    else:
        render_mode = None

    env = gym.make(
        "GymV21Environment-v0",
        env_id="SuperMarioBros-1-1-v0",
        render_mode=render_mode,
    )

    env = JoypadSpace(env, SIMPLE_MOVEMENT)

    # 添加得分奖励和超时检测
    env = ScoreRewardWrapper(
        env,
        score_weight=0.02,
        timeout_frames=timeout_frames,
        position_threshold=position_threshold,
    )
    # 归一化
    env = MaxAndSkipObservation(env, skip=4)
    env = ResizeObservation(env, shape=(84, 84))
    env = GrayscaleObservation(env)
    # env = TransformObservation(env, func=lambda x: x / 255.0)
    env = FrameStackObservation(env, stack_size=4)
    # 将 RecordVideo 移到所有预处理包装器之后
    # Record video if specified - 放在最后以确保录制的是最终处理后的画面
    if record_video and video_folder is not None:
        env = RecordVideo(
            env,
            video_folder=video_folder,
            name_prefix="mario-eval",
        )
    env.reset(seed=seed)
    return env


if __name__ == "__main__":
    args = parse_args()

    # Create save directory
    save_dir = Path("checkpoints_sb3") / datetime.datetime.now().strftime(
        "%Y-%m-%dT%H-%M-%S"
    )
    save_dir.mkdir(parents=True)

    # Create video directory if video recording is enabled
    video_dir = None
    if args.record_video:
        video_dir = save_dir / "videos"
        video_dir.mkdir(exist_ok=True)

    # Create environments
    env = DummyVecEnv(
        [
            lambda: make_env(
                render=False,
                seed=args.seed,
                record_video=args.record_video,
                video_folder=str(video_dir) if video_dir else None,
                timeout_frames=args.timeout_frames,
                position_threshold=args.position_threshold,
            )
        ]
    )

    eval_env = DummyVecEnv(
        [
            lambda: make_env(
                render=args.render,
                seed=args.seed,
                timeout_frames=args.timeout_frames,
                position_threshold=args.position_threshold,
            )
        ]
    )
    # eval_env = VecFrameStack(eval_env, n_stack=4)

    # Create callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=args.save_freq,
        save_path=str(save_dir),
        name_prefix="mario_model",
        save_replay_buffer=False,
        save_vecnormalize=True,
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(save_dir / "best_model"),
        log_path=str(save_dir / "eval_logs"),
        eval_freq=args.eval_freq,
        n_eval_episodes=args.eval_episodes,
        deterministic=False,
        render=args.render,
        verbose=1,
    )

    # 检查是否需要加载已保存的模型
    if args.load_model:
        print(f"Loading model from {args.load_model}")
        model = DoubleDQN.load(
            args.load_model,
            env=env,
            device=args.device,
            buffer_size=args.buffer_size,
            learning_starts=args.learning_starts,
            batch_size=args.batch_size,
            train_freq=args.train_freq,
            gradient_steps=args.gradient_steps,
            target_update_interval=args.target_update_interval,
            exploration_fraction=args.exploration_fraction,
            exploration_initial_eps=args.exploration_initial_eps,
            exploration_final_eps=args.exploration_final_eps,
            tensorboard_log=str(save_dir / "tb_logs"),
        )
        # 可以选择更新学习率等参数
        model.learning_rate = args.learning_rate
    else:
        # Initialize the DoubleDQN agent
        model = DoubleDQN(
            "CnnPolicy",
            env,
            learning_rate=args.learning_rate,
            buffer_size=args.buffer_size,
            learning_starts=args.learning_starts,
            batch_size=args.batch_size,
            train_freq=args.train_freq,
            gradient_steps=args.gradient_steps,
            target_update_interval=args.target_update_interval,
            exploration_fraction=args.exploration_fraction,
            exploration_initial_eps=args.exploration_initial_eps,
            exploration_final_eps=args.exploration_final_eps,
            tensorboard_log=str(save_dir / "tb_logs"),
            device=args.device,
            verbose=1,
        )

    # Train the agent
    model.learn(
        total_timesteps=args.total_timesteps,
        callback=[checkpoint_callback, eval_callback],
        log_interval=10,
        reset_num_timesteps=(
            False if args.load_model else True
        ),  # 如果继续训练，不重置时间步
    )

    # Save the final model
    model.save(str(save_dir / "final_model"))
