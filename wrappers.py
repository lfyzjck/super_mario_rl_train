import gym
import numpy as np
import torch
from gym.spaces import Box
from torchvision import transforms as T


"""
struct of info
```json
{
  'coins': 0,
  'flag_get': False,
  'life': 2,
  'score': 0,
  'stage': 1,
  'status': 'small',
  'time': 400,
  'world': 1,
  'x_pos': 40,
  'y_pos': 79
}
```
"""
# %%
class SkipFrame(gym.Wrapper):
    def __init__(self, env, skip):
        """Return only every `skip`-th frame"""
        super().__init__(env)
        self._skip = skip

    def step(self, action):
        """Repeat action, and sum reward"""
        total_reward = 0.0
        done = False
        for i in range(self._skip):
            # Accumulate reward and repeat the same action
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            if done:
                break
        return obs, total_reward, done, info

class GrayScaleObservation(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        obs_shape = self.observation_space.shape[:2]
        self.observation_space = Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)

    def permute_orientation(self, observation):
        # permute [H, W, C] array to [C, H, W] tensor
        observation = np.transpose(observation, (2, 0, 1))
        observation = torch.tensor(observation.copy(), dtype=torch.float)
        return observation

    def observation(self, observation):
        observation = self.permute_orientation(observation)
        transform = T.Grayscale()
        observation = transform(observation)
        return observation


class ResizeObservation(gym.ObservationWrapper):
    def __init__(self, env, shape):
        super().__init__(env)
        if isinstance(shape, int):
            self.shape = (shape, shape)
        else:
            self.shape = tuple(shape)

        obs_shape = self.shape + self.observation_space.shape[2:]
        self.observation_space = Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)

    def observation(self, observation):
        # Convert numpy array to PyTorch tensor if needed
        if isinstance(observation, np.ndarray):
            # Ensure the array is in the format [C, H, W]
            if observation.ndim == 3 and observation.shape[2] <= 3:  # [H, W, C] format
                observation = np.transpose(observation, (2, 0, 1))
            observation = torch.tensor(observation.copy(), dtype=torch.float32)
        
        transforms = T.Compose([T.Resize(self.shape), T.Normalize(0, 255)])
        observation = transforms(observation).squeeze(0)
        return observation


class ProgressRewardWrapper(gym.Wrapper):
    def __init__(self, env, level_length=3000, stuck_threshold=80, stuck_penalty=0.5):
        super().__init__(env)
        self.level_length = level_length
        self.max_x_pos = 0
        
        # 卡住检测参数
        self.stuck_threshold = stuck_threshold  # 卡住的帧数阈值
        self.stuck_penalty = stuck_penalty      # 卡住的惩罚值
        self.stuck_counter = 0                  # 卡住计数器
        self.last_x_pos = 0                     # 上一帧的x位置
        
    def reset(self):
        obs = self.env.reset()
        self.max_x_pos = 0
        self.stuck_counter = 0
        self.last_x_pos = 0
        return obs
        
    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        
        # 基础奖励
        progress_reward = reward
        
        # 获取当前x位置，确保info是字典类型并包含x_pos键
        current_x_pos = 0
        try:
            if isinstance(info, dict) and 'x_pos' in info:
                current_x_pos = info['x_pos']
        except (TypeError, KeyError):
            # 如果info不是字典或没有x_pos键，使用默认值
            current_x_pos = self.last_x_pos
        
        # 新的最大位置奖励
        if current_x_pos > self.max_x_pos:
            progress_reward += 1 * (current_x_pos - self.max_x_pos)
            self.max_x_pos = current_x_pos

        # print(f"progress_reward max_x_pos: {self.max_x_pos}, current_x_pos: {current_x_pos}, progress_reward: {progress_reward}")
        
        # 进度百分比奖励
        if self.level_length > 0:  # 防止除以零
            progress = current_x_pos / self.level_length
            progress_reward += 0.5 * progress
        
        # print(f"progress_reward max_x_pos: {self.max_x_pos}, current_x_pos: {current_x_pos}, progress_reward: {progress_reward}")
        # 卡住惩罚机制
        if abs(current_x_pos - self.last_x_pos) < 1:  # 如果位置几乎没变
            self.stuck_counter += 1
            if self.stuck_counter >= self.stuck_threshold:
                # 应用卡住惩罚
                progress_reward -= self.stuck_penalty
                # 可以根据卡住时间增加惩罚
                additional_penalty = min(0.2, (self.stuck_counter - self.stuck_threshold) * 0.01)
                progress_reward -= additional_penalty
                # print(f"progress_reward max_x_pos: {self.max_x_pos}, current_x_pos: {current_x_pos}, progress_reward: {progress_reward} ")
        else:
            # 重置卡住计数器
            self.stuck_counter = 0
        
        # 更新上一帧位置
        self.last_x_pos = current_x_pos
        
        return obs, progress_reward, done, info


class ScoreRewardWrapper(gym.Wrapper):
    def __init__(self, env, score_weight=0.01, death_penalty=5.0):
        """
        基于游戏得分和生命值的奖励包装器
        
        参数:
            env: 环境
            score_weight: 得分奖励的权重系数
            death_penalty: 死亡惩罚的权重系数
        """
        super().__init__(env)
        self.score_weight = score_weight
        self.death_penalty = death_penalty
        self.last_score = 0
        self.last_life = 2  # 默认初始生命值为2
        
    def reset(self):
        obs = self.env.reset()
        self.last_score = 0
        self.last_life = 2  # 重置生命值
        return obs
        
    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        
        # 基础奖励
        modified_reward = reward
        
        # 获取当前得分和生命值
        current_score = 0
        current_life = 2  # 默认值
        
        try:
            if isinstance(info, dict):
                # 获取得分
                if 'score' in info:
                    current_score = info['score']
                
                # 获取生命值
                if 'life' in info:
                    current_life = info['life']
        except (TypeError, KeyError):
            # 如果info不是字典或没有相关键，使用默认值
            current_score = self.last_score
            current_life = self.last_life
        
        # 计算得分增量
        score_increment = current_score - self.last_score
        
        # 如果得分增加，提供额外奖励
        if score_increment > 0:
            # 得分奖励与得分增量成正比
            modified_reward += self.score_weight * score_increment
        
        # 检测生命值减少（死亡）
        if current_life < self.last_life:
            # 应用死亡惩罚
            modified_reward -= self.death_penalty
            # 可以在日志中记录死亡事件
            # print(f"Mario died! Life reduced from {self.last_life} to {current_life}. Applied penalty: {self.death_penalty}")
        
        # 更新上一次得分和生命值
        self.last_score = current_score
        self.last_life = current_life
        
        return obs, modified_reward, done, info
