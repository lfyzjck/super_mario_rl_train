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

import torch


# 创建环境
env = create_env()

# 查找最新的检查点
checkpoint = find_latest_checkpoint(Path("checkpoints"))
if checkpoint is None:
    print("No checkpoint found. Please train a model first.")
    exit(1)

print(f"Loading checkpoint from {checkpoint}")

# 创建保存目录（用于日志）
save_dir = Path("checkpoints") / datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
save_dir.mkdir(parents=True)

# 创建Mario智能体并加载检查点
mario = Mario(
    state_dim=(4, 84, 84),
    action_dim=env.action_space.n,
    save_dir=save_dir,
    checkpoint=checkpoint,
)

# 设置为评估模式（最小探索率）
# 探索率越高越可能会出现随机动作
mario.exploration_rate = mario.exploration_rate_min

# 创建日志记录器
logger = MetricLogger(save_dir)

# 模拟的回合数
episodes = 100

# 开始模拟
for e in range(episodes):
    print(f"Episode {e+1}/{episodes}")

    # 重置环境
    state = env.reset()
    total_reward = 0

    # 单个回合的模拟
    while True:
        # 渲染游戏画面
        env.render()

        # 使用模型选择动作
        with torch.no_grad():  # 不计算梯度，提高性能
            action = mario.act(state)

        # 执行动作
        next_state, reward, done, info = env.step(action)
        total_reward += reward

        # 更新状态
        state = next_state

        # 记录步骤
        logger.log_step(reward, None, None)

        # 检查是否结束回合
        if done or info["flag_get"]:
            break

    # 记录回合结果
    logger.log_episode()
    print(f"Episode {e+1} - Total reward: {total_reward:.2f}")

    # 每20个回合记录一次
    if e % 20 == 0:
        logger.record(episode=e, epsilon=mario.exploration_rate, step=mario.curr_step)

# 关闭环境
env.close()
print("Simulation completed!")
