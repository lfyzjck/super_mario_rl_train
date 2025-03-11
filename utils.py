import random, datetime
from pathlib import Path

import gym
import gym_super_mario_bros
from gym.wrappers import FrameStack, TransformObservation
from nes_py.wrappers import JoypadSpace
from wrappers import SkipFrame, GrayScaleObservation, ResizeObservation


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
    env = gym_super_mario_bros.make('SuperMarioBros-1-1-v0')

    env = JoypadSpace(
        env,
        [['right'],
        ['right', 'A']]
    )

    env = SkipFrame(env, skip=5)
    env = GrayScaleObservation(env)
    env = ResizeObservation(env, shape=84)
    env = TransformObservation(env, f=lambda x: x / 255.)
    env = FrameStack(env, num_stack=4)

    env.reset()

    return env


if __name__ == "__main__":
    print(find_latest_checkpoint(Path('checkpoints')))

