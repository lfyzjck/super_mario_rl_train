import argparse
import os
import numpy as np
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.evaluation import evaluate_policy

from supermario_ppo_sb3 import make_env


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True, 
                        help="Path to the trained model file")
    parser.add_argument("--vecnormalize-path", type=str, required=False, default=None,
                        help="Path to the saved VecNormalize statistics")
    parser.add_argument("--run-name", type=str, required=False, default="eval-super-mario",
                        help="Name for this evaluation run")
    parser.add_argument("--eval-episodes", type=int, required=False, default=3,
                        help="Number of episodes to evaluate")
    parser.add_argument("--device", type=str, required=False, default="auto",
                        help="Device to use for model inference (cpu, cuda, auto)")
    parser.add_argument("--render", action="store_true", default=True,
                        help="Render the environment in human-viewable mode")
    parser.add_argument("--capture-video", action="store_true", default=False, 
                        help="Capture video of the evaluation")
    parser.add_argument("--video-folder", type=str, default="./videos/",
                        help="Folder to save videos in")
    return parser.parse_args()


def evaluate_model(
    model_path,
    vecnormalize_path=None,
    eval_episodes=3,
    device="auto",
    render=True,
    capture_video=False,
    video_folder="./videos/",
    timeout_frames=600,
    position_threshold=10,
):
    """
    Evaluate a trained model with optional rendering and video recording
    
    Args:
        model_path: Path to the trained model
        vecnormalize_path: Path to the saved VecNormalize statistics
        eval_episodes: Number of episodes to evaluate
        device: Device to use for inference
        render: Whether to render in human-viewable mode
        capture_video: Whether to capture video
        video_folder: Folder to save videos in
        timeout_frames: Maximum frames before timeout
        position_threshold: Threshold for detecting if Mario is stuck
    """
    # Create video directory if needed
    if capture_video:
        Path(video_folder).mkdir(parents=True, exist_ok=True)
    
    # Set render mode based on arguments
    render_mode = "rgb_array" if capture_video else ("human" if render else "rgb_array")
    
    # Create evaluation environment
    eval_env = DummyVecEnv(
        [
            make_env(
                idx=0,
                record_video=capture_video,
                video_folder=video_folder,
                render_mode=render_mode,
                timeout_frames=timeout_frames,
                position_threshold=position_threshold,
            )
        ]
    )
    
    # Load VecNormalize stats if provided
    if vecnormalize_path is not None:
        eval_env = VecNormalize.load(vecnormalize_path, eval_env)
        # Don't update the normalization statistics during evaluation
        eval_env.training = False
        eval_env.norm_reward = False  # Optional: don't normalize rewards during eval
    
    # Load the model
    model = PPO.load(model_path, env=eval_env, device=device)
    
    # Use SB3's built-in evaluation function
    mean_reward, std_reward = evaluate_policy(
        model,
        eval_env,
        n_eval_episodes=eval_episodes,
        deterministic=True,
        render=render,
        return_episode_rewards=False,
    )
    
    # Close the environment
    eval_env.close()
    
    # Print evaluation summary
    print("\nEvaluation Summary:")
    print(f"Number of episodes: {eval_episodes}")
    print(f"Mean reward: {mean_reward:.2f}")
    print(f"Std reward: {std_reward:.2f}")
    
    return mean_reward, std_reward


if __name__ == "__main__":
    args = parse_args()
    
    # Warn user if both render and capture_video are enabled
    if args.render and args.capture_video:
        print("Note: When capturing video, render mode will be set to 'rgb_array' instead of 'human'")
        print("Video will be saved to the specified folder but won't be displayed in real-time")
    
    evaluate_model(
        model_path=args.model_path,
        vecnormalize_path=args.vecnormalize_path,
        eval_episodes=args.eval_episodes,
        device=args.device,
        render=args.render,
        capture_video=args.capture_video,
        video_folder=args.video_folder,
    )