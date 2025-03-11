import random, datetime
from pathlib import Path

import gym
import gym_super_mario_bros
from gym.wrappers import FrameStack, TransformObservation
from nes_py.wrappers import JoypadSpace

from metrics import MetricLogger
from agent import Mario
from wrappers import ResizeObservation, SkipFrame, GrayScaleObservation
from utils import find_latest_checkpoint, create_env


env = create_env()
save_dir = Path('checkpoints') / datetime.datetime.now().strftime('%Y-%m-%dT%H-%M-%S')
save_dir.mkdir(parents=True)

checkpoint = find_latest_checkpoint(Path('checkpoints'))
print(f"Loading checkpoint from {checkpoint}")
mario = Mario(state_dim=(4, 84, 84), action_dim=env.action_space.n, save_dir=save_dir, checkpoint=checkpoint)
mario.exploration_rate = mario.exploration_rate_min

logger = MetricLogger(save_dir)

episodes = 100

for e in range(episodes):

    state = env.reset()

    while True:

        env.render()

        action = mario.act(state)

        next_state, reward, done, info = env.step(action)

        mario.cache(state, next_state, action, reward, done)

        logger.log_step(reward, None, None)

        state = next_state

        if done or info['flag_get']:
            break

    logger.log_episode()

    if e % 20 == 0:
        logger.record(
            episode=e,
            epsilon=mario.exploration_rate,
            step=mario.curr_step
        )
