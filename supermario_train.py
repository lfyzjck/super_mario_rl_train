import datetime

# setup parser
import argparse
from pathlib import Path

# Gym is an OpenAI toolkit for RL
import gym

# Super Mario environment for OpenAI Gym
import gym_super_mario_bros
import matplotlib.pyplot as plt
import numpy as np
import torch
from gym.spaces import Box
from gym.wrappers import FrameStack

# NES Emulator for OpenAI Gym
from nes_py.wrappers import JoypadSpace
from PIL import Image
from torch import nn
from torchvision import transforms as T
from tensordict import TensorDict
from torchrl.data import TensorDictReplayBuffer, LazyMemmapStorage
from metrics import MetricLogger
from utils import find_latest_checkpoint, create_env
from agent import Mario
from wrappers import SkipFrame, GrayScaleObservation, ResizeObservation

# %%
# Initialize Super Mario environment
env = create_env()


def parse_args():
    parser = argparse.ArgumentParser(description='Super Mario Bros Training')
    parser.add_argument('--batch-size', type=int, default=64,
                    help='batch size for training (default: 64)')
    parser.add_argument('--render', action='store_true',
                    help='render the game while training')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    use_cuda = torch.cuda.is_available()
    print(f"Using CUDA: {use_cuda}")
    print()

    save_dir = Path("checkpoints") / datetime.datetime.now().strftime(
        "%Y-%m-%dT%H-%M-%S"
    )
    save_dir.mkdir(parents=True)
    checkpoint = find_latest_checkpoint(Path("checkpoints"))
    print(f"Loading checkpoint from {checkpoint}")

    mario = Mario(
        state_dim=(4, 84, 84),
        action_dim=env.action_space.n,
        save_dir=save_dir,
        checkpoint=checkpoint,
        batch_size=args.batch_size
    )

    logger = MetricLogger(save_dir)

    episodes = 2000
    for e in range(episodes):

        state = env.reset()

        # Play the game!
        while True:
            if args.render:
                env.render()
            # env.render()

            # Run agent on the state
            action = mario.act(state)

            # Agent performs action
            next_state, reward, done, info = env.step(action)

            # Remember
            mario.cache(state, next_state, action, reward, done)

            # Learn
            q, loss = mario.learn()

            # Logging
            logger.log_step(reward, loss, q)

            # Update state
            state = next_state

            # Check if end of game
            if done or info["flag_get"]:
                break

        logger.log_episode()

        if e % 20 == 0:
            logger.record(
                episode=e, epsilon=mario.exploration_rate, step=mario.curr_step
            )
