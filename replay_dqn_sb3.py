import argparse
from pathlib import Path
import gymnasium as gym
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecFrameStack, VecNormalize, DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
import torch as th
import numpy as np
import time

# Import custom modules from training script
from supermario_dqn_sb3 import (
    DoubleDQN,
    JoypadSpace,
    SIMPLE_MOVEMENT,
    ScoreRewardWrapper,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Replay a trained DQN agent for Super Mario Bros"
    )
    parser.add_argument(
        "--algorithm",
        type=str,
        default="dqn",
        help="algorithm to use (default: dqn), options: dqn, ppo",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="path to the trained model checkpoint",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="device to use (default: auto), options: auto, cpu, cuda, mps",
    )
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=5,
        help="number of episodes to play (default: 5)",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        default=False,
        help="render the environment (default: True)",
    )
    parser.add_argument(
        "--record-video",
        action="store_true",
        help="record video of the gameplay",
    )
    parser.add_argument(
        "--video-dir",
        type=str,
        default="./replay_videos",
        help="directory to save recorded videos (default: ./replay_videos)",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        default=True,
        help="use deterministic actions (default: True)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="random seed (default: 0)",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.0,
        help="delay between frames in seconds (default: 0.0)",
    )
    # 添加超时检测相关参数
    parser.add_argument(
        "--timeout-frames",
        type=int,
        default=1000,
        help="maximum number of frames Mario can be stuck before early termination (default: 200)",
    )
    parser.add_argument(
        "--position-threshold",
        type=int,
        default=10,
        help="minimum distance Mario must move to not be considered stuck (default: 10)",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.99,
        help="discount factor for PPO (default: 0.99)",
    )
    return parser.parse_args()


def replay(
    model_path,
    algorithm="dqn",
    num_episodes=5,
    render=True,
    record_video=False,
    video_dir="./replay_videos",
    deterministic=True,
    seed=0,
    delay=0.0,
    timeout_frames=200,
    position_threshold=10,
    gamma=0.99,
):
    """Replay the trained agent"""
    # Create video directory if needed
    if record_video:
        video_folder = Path(video_dir)
        video_folder.mkdir(parents=True, exist_ok=True)
    else:
        video_folder = None

    if render:
        render_mode = "human"
    else:
        render_mode = "rgb_array"

    # Create environment
    if algorithm == "dqn":
        from supermario_dqn_sb3 import make_env as make_env_for_dqn
        env = make_env_for_dqn(
            render=render,
            record_video=record_video,
            video_folder=str(video_folder) if video_folder else None,
            seed=seed,
            timeout_frames=timeout_frames,
            position_threshold=position_threshold,
        )
        model = DoubleDQN.load(model_path, env=env, device=args.device)
        print(f"Loaded DoubleDQN model from {model_path}")
    elif algorithm == "ppo":
        # For PPO, we need to use VecNormalize with a DummyVecEnv
        from supermario_ppo_sb3 import make_env as make_env_for_ppo
        env_init_fn = make_env_for_ppo(
            render_mode=render_mode,
            record_video=record_video,
            video_folder=str(video_folder) if video_folder else None,
            seed=seed,
            timeout_frames=timeout_frames,
            position_threshold=position_threshold,
        )
        env_vec = DummyVecEnv([env_init_fn])
        env = VecNormalize(
            env_vec,
            norm_obs=True,
            norm_reward=True,
            clip_obs=10.0,
            clip_reward=10.0,
            training=False,
            gamma=gamma,
            epsilon=1e-8,
        )
        model = PPO.load(model_path, env=env, device=args.device)
        print(f"Loaded PPO model from {model_path}")

    # Play episodes
    total_rewards = []
    episode_lengths = []

    # Use evaluate_policy function from stable-baselines3
    print(f"\n===== Evaluating Model for {num_episodes} Episodes =====")
    print(f"Algorithm: {algorithm}, Deterministic: {deterministic}")
    
    # Get mean and std for summary, but don't collect detailed rewards
    mean_reward, std_reward = evaluate_policy(
        model=model,
        env=env,
        n_eval_episodes=num_episodes,
        deterministic=deterministic,
        render=render,
        return_episode_rewards=False,
        warn=True
    )
    
    # Initialize with empty lists in case we don't run the following evaluations
    rewards = []
    lengths = []
    
    # If we want to slow down the evaluation for better visualization
    if delay > 0:
        # We can't directly modify evaluate_policy, so we'll evaluate again with our custom callback
        def delay_callback(_locals, _globals):
            time.sleep(delay)
            return None  # Change return value to None instead of True
        
        rewards, lengths = evaluate_policy(
            model=model,
            env=env,
            n_eval_episodes=num_episodes,
            deterministic=deterministic,
            render=render,
            return_episode_rewards=True,
            callback=delay_callback,
            warn=False
        )
    else:
        # Get rewards and lengths if not using delay
        rewards, lengths = evaluate_policy(
            model=model,
            env=env,
            n_eval_episodes=num_episodes,
            deterministic=deterministic,
            render=render,
            return_episode_rewards=True,
            warn=False
        )
    
    # Ensure we have the episode data for the statistics
    total_rewards = rewards
    episode_lengths = lengths

    # Print overall statistics
    print("\n===== Replay Results =====")
    print(f"Episodes played: {num_episodes}")
    print(f"Average reward: {mean_reward:.2f} ± {std_reward:.2f}")
    print(f"Average episode length: {np.mean(episode_lengths):.2f}")
    
    # Calculate success rate only if we have actual reward data
    if isinstance(total_rewards, list) and len(total_rewards) > 0:
        success_rate = sum(1 for r in total_rewards if r > 0) / num_episodes
        print(f"Success rate: {success_rate:.2%}")
    else:
        print("Success rate: N/A (no episode data available)")
    
    # Close the environment
    env.close()


if __name__ == "__main__":
    args = parse_args()

    replay(
        model_path=args.model_path,
        algorithm=args.algorithm,
        num_episodes=args.num_episodes,
        render=args.render,
        record_video=args.record_video,
        video_dir=args.video_dir,
        deterministic=args.deterministic,
        seed=args.seed,
        delay=args.delay,
        timeout_frames=args.timeout_frames,
        position_threshold=args.position_threshold,
        gamma=args.gamma,
    )
