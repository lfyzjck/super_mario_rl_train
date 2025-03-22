import argparse
from pathlib import Path
import datetime
import gymnasium as gym
import shimmy
from gymnasium.wrappers import (
    FrameStackObservation,
    TransformObservation,
    GrayscaleObservation,
    ResizeObservation,
    MaxAndSkipObservation,
    RecordVideo,
)
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


class ScoreRewardWrapper(gym.Wrapper):
    def __init__(self, env, score_weight=0.01, flag_reward=50):
        """
        基于游戏得分和旗子状态的奖励包装器

        参数:
            env: 环境
            score_weight: 得分奖励的权重系数
            flag_reward: 得到旗子时的额外奖励
        """
        super().__init__(env)
        self.score_weight = score_weight
        self.flag_reward = flag_reward
        self.last_score = 0

    def reset(self, seed=None, options=None):
        obs = self.env.reset(seed=seed, options=options)
        self.last_score = 0
        return obs

    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)

        # 基础奖励
        modified_reward = float(reward)  # Convert to float to fix type issues

        # 获取当前得分
        current_score = 0

        try:
            if isinstance(info, dict):
                # 获取得分
                if "score" in info:
                    current_score = info["score"]
        except (TypeError, KeyError):
            # 如果info不是字典或没有相关键，使用默认值
            current_score = self.last_score

        # 计算得分增量
        score_increment = current_score - self.last_score

        # 如果得分增加，提供额外奖励
        if score_increment > 0:
            # 得分奖励与得分增量成正比
            modified_reward += float(self.score_weight * score_increment)

        if done:
            # 检查是否得到旗子
            if info["flag_get"]:
                modified_reward += float(self.flag_reward)
            else:
                modified_reward -= float(self.flag_reward)

        # 更新上一次得分
        self.last_score = current_score

        return obs, modified_reward, done, truncated, info


class PrioritizedReplayBuffer(ReplayBuffer):
    """
    Prioritized Experience Replay (PER) buffer.

    Preferentially samples transitions with higher TD errors to accelerate learning.

    Paper: https://arxiv.org/abs/1511.05952
    """

    def __init__(
        self,
        buffer_size: int,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        device: Union[th.device, str] = "auto",
        n_envs: int = 1,
        optimize_memory_usage: bool = False,
        handle_timeout_termination: bool = True,
        per_alpha: float = 0.6,
        per_beta: float = 0.4,
        per_beta_increment: float = 0.001,
        per_epsilon: float = 1e-6,
    ):
        """
        Initialize the PrioritizedReplayBuffer.

        Args:
            buffer_size: Max number of transitions to store
            observation_space: Observation space
            action_space: Action space
            device: PyTorch device
            n_envs: Number of parallel environments
            optimize_memory_usage: Enable memory efficient mode
            handle_timeout_termination: Handle timeout termination properly
            per_alpha: How much prioritization to use (0 = no prioritization, 1 = full prioritization)
            per_beta: Importance sampling weight (0 = no correction, 1 = full correction)
            per_beta_increment: Increment of beta parameter per sampling
            per_epsilon: Small constant to ensure non-zero priority
        """
        super().__init__(
            buffer_size,
            observation_space,
            action_space,
            device,
            n_envs=n_envs,
            optimize_memory_usage=optimize_memory_usage,
            handle_timeout_termination=handle_timeout_termination,
        )

        # PER parameters
        self.alpha = per_alpha
        self.beta = per_beta
        self.beta_increment_per_sampling = per_beta_increment
        self.epsilon = per_epsilon

        # Priority storage with same size as replay buffer
        self.priorities = np.zeros((buffer_size,), dtype=np.float32)

        # Initial max priority for new items
        self.max_priority = 1.0

        # Store current indices for updating priorities
        self.current_indices = None

    def add(
        self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        infos: List[Dict[str, Any]],
    ) -> None:
        """
        Add a new transition to the buffer with max priority.
        """
        # Call parent add method to handle the basic buffers
        idx = self.pos
        super().add(obs, next_obs, action, reward, done, infos)

        # New transitions get max priority to ensure they're sampled at least once
        self.priorities[idx] = self.max_priority

    def sample(
        self, batch_size: int, env: Optional[VecNormalize] = None
    ) -> ReplayBufferSamples:
        """
        Sample a batch of transitions with prioritized sampling.

        Returns:
            ReplayBufferSamples: standard samples with additional attributes
        """
        # Calculate sampling probabilities based on priorities
        if self.full:
            priorities = self.priorities
        else:
            priorities = self.priorities[: self.pos]

        # Convert priorities to probabilities with alpha
        probabilities = priorities**self.alpha
        probabilities /= probabilities.sum()

        # Sample indices based on probabilities
        indices = np.random.choice(len(probabilities), size=batch_size, p=probabilities)

        # Update beta value
        self.beta = min(1.0, self.beta + self.beta_increment_per_sampling)

        # Calculate importance sampling weights
        weights = (len(probabilities) * probabilities[indices]) ** (-self.beta)
        weights /= weights.max()  # Normalize weights

        # Store current indices for updating priorities later
        self.current_indices = indices

        # Use parent class's sample method to get samples
        # Type safety: we're explicitly overriding the parent method's signature
        samples = super().sample(batch_size, env)

        # Add weights and indices as custom attributes
        # Using setattr to add attributes at runtime
        weights_tensor = th.tensor(
            weights, dtype=th.float32, device=samples.observations.device
        )
        samples_dict = samples.__dict__
        samples_dict["weights"] = weights_tensor
        samples_dict["indices"] = indices

        return samples

    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray) -> None:
        """
        Update priorities based on TD errors.

        Args:
            indices: Indices of samples to update
            priorities: New priorities (TD errors)
        """
        # Add small epsilon to ensure non-zero probability
        priorities = priorities + self.epsilon

        # Update priorities
        self.priorities[indices] = priorities

        # Update max priority for new items
        self.max_priority = max(self.max_priority, priorities.max())


class DoubleDQN(DQN):
    """
    Double Deep Q-Network (Double DQN) with Prioritized Experience Replay

    Paper: https://arxiv.org/abs/1509.06461

    The main difference with regular DQN is that Double DQN uses the online network
    to select actions and the target network to evaluate these actions in the computation
    of the target Q-values.

    This implementation also includes Prioritized Experience Replay.
    """

    def __init__(self, *args, **kwargs):
        # Initialize parent class - now we rely on the replay_buffer_class and replay_buffer_kwargs
        # DQN will handle the creation of our PrioritizedReplayBuffer
        super().__init__(*args, **kwargs)

    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        """
        Modified training method to support prioritized experience replay.
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

            # Sample replay buffer with priorities
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

            # Compute TD errors for priority updating
            td_errors = (
                th.abs(target_q_values - current_q_values)
                .detach()
                .cpu()
                .numpy()
                .flatten()
            )

            # Check if weights attribute exists (using PrioritizedReplayBuffer)
            if hasattr(replay_data, "__dict__") and "weights" in replay_data.__dict__:
                # Apply importance sampling weights from PER
                weighted_loss = F.smooth_l1_loss(
                    current_q_values, target_q_values, reduction="none"
                )
                weights = replay_data.__dict__["weights"].reshape(-1, 1)
                loss = (weighted_loss * weights).mean()
            else:
                # Fallback to regular loss if not using PER
                loss = F.smooth_l1_loss(current_q_values, target_q_values)

            losses.append(loss.item())

            # Optimize the policy
            self.policy.optimizer.zero_grad()
            loss.backward()
            # Clip gradient norm
            th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.policy.optimizer.step()

            # Update priorities in the replay buffer if it's a PrioritizedReplayBuffer
            if (
                isinstance(self.replay_buffer, PrioritizedReplayBuffer)
                and hasattr(replay_data, "__dict__")
                and "indices" in replay_data.__dict__
            ):
                indices = replay_data.__dict__["indices"]
                self.replay_buffer.update_priorities(indices, td_errors)

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
        "--per-alpha",
        type=float,
        default=0.6,
        help="alpha parameter for prioritized replay buffer (default: 0.6)",
    )
    parser.add_argument(
        "--per-beta",
        type=float,
        default=0.4,
        help="initial beta parameter for prioritized replay buffer (default: 0.4)",
    )
    parser.add_argument(
        "--per-beta-increment",
        type=float,
        default=0.001,
        help="increment for beta parameter per sampling (default: 0.001)",
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
    return parser.parse_args()


def make_env(
    render=False,
    record_video=False,
    video_folder=None,
    seed=0,
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
    # 归一化
    env = MaxAndSkipObservation(env, skip=4)
    env = ResizeObservation(env, shape=(84, 84))
    env = GrayscaleObservation(env)
    # env = TransformObservation(env, func=lambda x: x / 255.0)
    env = FrameStackObservation(env, stack_size=4)

    # 添加得分奖励
    env = ScoreRewardWrapper(env, score_weight=0.02)
    # 将 RecordVideo 移到所有预处理包装器之后
    # Record video if specified - 放在最后以确保录制的是最终处理后的画面
    if record_video and video_folder is not None:
        env = RecordVideo(
            env,
            video_folder=video_folder,
            name_prefix="mario-eval",
        )

    env = Monitor(env)  # Needed for the callbacks

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
            )
        ]
    )
    # env = VecFrameStack(env, n_stack=4)

    eval_env = DummyVecEnv(
        [
            lambda: make_env(
                render=args.render,
                seed=args.seed,
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
        model = DoubleDQN.load(args.load_model, env=env)
        # 可以选择更新学习率等参数
        model.learning_rate = args.learning_rate
    else:
        # Initialize the DoubleDQN agent with prioritized experience replay
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
            # 使用 replay_buffer_class 和 replay_buffer_kwargs 参数来初始化优先经验回放
            replay_buffer_class=PrioritizedReplayBuffer,
            replay_buffer_kwargs={
                "per_alpha": args.per_alpha,
                "per_beta": args.per_beta,
                "per_beta_increment": args.per_beta_increment,
                "per_epsilon": 1e-6,
            },
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
