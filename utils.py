import random, datetime
from pathlib import Path

import gym
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from gym.wrappers.frame_stack import FrameStack
from gym.wrappers.transform_observation import TransformObservation
# from gym.wrappers.gray_scale_observation import GrayScaleObservation
from nes_py.wrappers import JoypadSpace
from wrappers import (
    ScoreRewardWrapper,
    SkipFrame,
    ResizeObservation,
    GrayScaleObservation,
)


def find_latest_checkpoint(save_dir: Path) -> Path | None:
    """find the latest checkpoint in the save_dir and its subdirectories recursively"""
    checkpoints = list(save_dir.rglob("*.chkpt"))
    if not checkpoints:
        return None
    return sorted(checkpoints)[-1]


def create_env() -> gym.Env:
    """
    Create a Super Mario Bros environment
    """
    env = gym_super_mario_bros.make("SuperMarioBros-1-1-v0", new_step_api=True)

    # env = JoypadSpace(env, [["right"], ["right", "A"]])
    env = JoypadSpace(env, SIMPLE_MOVEMENT)

    # 添加进度奖励
    # env = ProgressRewardWrapper(env)
    # 添加得分奖励
    env = ScoreRewardWrapper(env, score_weight=0.01)


    # 跳帧
    env = SkipFrame(env, skip=4)
    # 灰度化
    env = GrayScaleObservation(env)
    # 调整大小
    env = ResizeObservation(env, shape=84)
    # 归一化
    env = TransformObservation(env, f=lambda x: x / 255.0)
    env = FrameStack(env, num_stack=4)
    env.reset()
    return env
