import argparse
from pathlib import Path
import datetime
import gymnasium as gym
from openai_gym_compatibility import register_gymnasium_envs
from gymnasium.wrappers import (
    FrameStackObservation,
    GrayscaleObservation,
    ResizeObservation,
    MaxAndSkipObservation,
)
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecNormalize
import torch as th
import numpy as np
from joypad_space import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

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


def parse_args():
    parser = argparse.ArgumentParser(
        description="Super Mario Bros Training with PPO (Stable-Baselines3)"
    )
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=2000000,
        help="total timesteps to train (default: 2000000)",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.0003,
        help="learning rate (default: 0.0003)",
    )
    parser.add_argument(
        "--n-steps",
        type=int,
        default=2048,
        help="number of steps to run for each environment per update (default: 2048)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="minibatch size (default: 64)",
    )
    parser.add_argument(
        "--n-epochs",
        type=int,
        default=10,
        help="number of epoch when optimizing the surrogate loss (default: 10)",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.99,
        help="discount factor (default: 0.99)",
    )
    parser.add_argument(
        "--gae-lambda",
        type=float,
        default=0.95,
        help="factor for trade-off of bias vs variance for Generalized Advantage Estimator (default: 0.95)",
    )
    parser.add_argument(
        "--clip-range",
        type=float,
        default=0.2,
        help="clipping parameter for PPO (default: 0.2)",
    )
    parser.add_argument(
        "--clip-range-vf",
        type=float,
        default=None,
        help="clipping parameter for value function, set to None for no clipping (default: None)",
    )
    parser.add_argument(
        "--ent-coef",
        type=float,
        default=0.01,
        help="entropy coefficient for the loss calculation (default: 0.01)",
    )
    parser.add_argument(
        "--vf-coef",
        type=float,
        default=0.5,
        help="value function coefficient for the loss calculation (default: 0.5)",
    )
    parser.add_argument(
        "--max-grad-norm",
        type=float,
        default=0.5,
        help="maximum norm for gradient clipping (default: 0.5)",
    )
    parser.add_argument(
        "--use-sde",
        action="store_true",
        help="whether to use generalized State Dependent Exploration",
    )
    parser.add_argument(
        "--sde-sample-freq",
        type=int,
        default=-1,
        help="sample a new noise matrix every n steps when using sde (-1 = only at the beginning of the rollout, default: -1)",
    )
    parser.add_argument(
        "--target-kl",
        type=float,
        default=None,
        help="target KL divergence between updates, stop early if reached (default: None)",
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
        help="number of episodes to evaluate (default: 5)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=12313213,
        help="random seed (default: 12313213)",
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
        default=False,
        help="record video during evaluation",
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
    # PPO特有参数
    parser.add_argument(
        "--num-envs",
        type=int,
        default=4,
        help="number of parallel environments for vectorized training (default: 4)",
    )
    return parser.parse_args()


def make_env(
    idx: int,
    record_video=False,
    video_folder=None,
    seed=0,
    timeout_frames=600,
    position_threshold=10,
    rank=0,
    render_mode: str='rgb_array',
):
    """Create the wrapped environment for training or evaluation"""

    def _init():

        # Record video if specified - 放在最后以确保录制的是最终处理后的画面
        if record_video and video_folder is not None and idx == 0:
            env = gym.make(
                "GymV21Environment-v0",
                env_id="SuperMarioBros-1-1-v0",
                render_mode=render_mode,
            )
            env = gym.wrappers.RecordVideo(
                env,
                video_folder=video_folder,
                name_prefix=f"mario-eval-{rank}",
            )
        else:
            env = gym.make(
                "GymV21Environment-v0",
                env_id="SuperMarioBros-1-1-v0",
                render_mode=render_mode,
            )

        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = JoypadSpace(env, SIMPLE_MOVEMENT)

        # 添加得分奖励和超时检测
        env = ScoreRewardWrapper(
            env,
            score_weight=0.01,
            timeout_frames=timeout_frames,
            position_threshold=position_threshold,
        )
        # 归一化
        env = MaxAndSkipObservation(env, skip=4)
        env = ResizeObservation(env, shape=(84, 84))
        env = GrayscaleObservation(env)
        # env = TransformObservation(env, func=lambda x: x / 255.0)
        env = FrameStackObservation(env, stack_size=4)

        # Monitor env for logging
        env = Monitor(env)
        env.reset(seed=seed + rank)
        return env

    return _init


if __name__ == "__main__":
    args = parse_args()

    # Create save directory
    save_dir = Path("checkpoints_ppo_sb3") / datetime.datetime.now().strftime(
        "%Y-%m-%dT%H-%M-%S"
    )
    save_dir.mkdir(parents=True)

    # Create video directory if video recording is enabled
    video_dir = None
    if args.record_video:
        video_dir = save_dir / "videos"
        video_dir.mkdir(exist_ok=True)
    # Set appropriate render mode
    if args.render:
        render_mode = "human"
    elif args.record_video:
        render_mode = "rgb_array"  # Required for video recording
    else:
        render_mode = "rgb_array"

    # Create vectorized environments for PPO training
    envs = []
    for i in range(args.num_envs):
        envs.append(
            make_env(
                idx=i,
                seed=args.seed,
                record_video=args.record_video,  # Don't record during training
                video_folder=str(video_dir) if video_dir else None,
                timeout_frames=args.timeout_frames,
                position_threshold=args.position_threshold,
                rank=i,
            )
        )

    env = DummyVecEnv(envs)
    # Optional: Add VecNormalize for observation and reward normalization
    env = VecNormalize(
        env,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
        clip_reward=10.0,
        training=True,
        gamma=args.gamma,
        epsilon=1e-8,
    )

    # Create a separate environment for evaluation
    eval_env = DummyVecEnv(
        [
            make_env(
                idx=0,
                seed=args.seed,
                record_video=False,
                timeout_frames=args.timeout_frames,
                position_threshold=args.position_threshold,
                render_mode=render_mode,
            )
        ]
    )

    # Also normalize the evaluation environment (with the same stats as the training env)
    eval_env = VecNormalize(
        eval_env,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
        clip_reward=10.0,
        training=False,  # do not update stats at test time
    )

    # Copy stats from training env
    if isinstance(env, VecNormalize) and isinstance(eval_env, VecNormalize):
        eval_env.obs_rms = env.obs_rms
        eval_env.ret_rms = env.ret_rms

    # Create callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=args.save_freq // args.num_envs,  # Adjust for number of envs
        save_path=str(save_dir),
        name_prefix="mario_model",
        save_vecnormalize=True,
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(save_dir / "best_model"),
        log_path=str(save_dir / "eval_logs"),
        eval_freq=args.eval_freq // args.num_envs,  # Adjust for number of envs
        n_eval_episodes=args.eval_episodes,
        deterministic=True,
        render=args.render,
        verbose=1,
    )

    # 检查是否需要加载已保存的模型
    if args.load_model:
        print(f"Loading model from {args.load_model}")
        model = PPO.load(
            args.load_model,
            env=env,
            device=args.device,
            # PPO specific parameters
            learning_rate=args.learning_rate,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            n_epochs=args.n_epochs,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            clip_range=args.clip_range,
            clip_range_vf=args.clip_range_vf,
            ent_coef=args.ent_coef,
            vf_coef=args.vf_coef,
            max_grad_norm=args.max_grad_norm,
            use_sde=args.use_sde,
            sde_sample_freq=args.sde_sample_freq,
            target_kl=args.target_kl,
            tensorboard_log=str(save_dir / "tb_logs"),
        )
    else:
        # Initialize the PPO agent
        model = PPO(
            "CnnPolicy",
            env,
            policy_kwargs=dict(normalize_images=False),
            # PPO specific parameters
            learning_rate=args.learning_rate,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            n_epochs=args.n_epochs,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            clip_range=args.clip_range,
            clip_range_vf=args.clip_range_vf,
            ent_coef=args.ent_coef,
            vf_coef=args.vf_coef,
            max_grad_norm=args.max_grad_norm,
            use_sde=args.use_sde,
            sde_sample_freq=args.sde_sample_freq,
            target_kl=args.target_kl,
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

    # Save the VecNormalize statistics
    if isinstance(env, VecNormalize):
        env.save(str(save_dir / "vec_normalize.pkl"))
