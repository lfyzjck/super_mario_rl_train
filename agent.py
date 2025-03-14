import copy
from pathlib import Path
import numpy as np
import torch
from torch import nn
from tensordict import TensorDict
from torchrl.data import TensorDictReplayBuffer, LazyMemmapStorage, LazyTensorStorage
from torchrl.data.replay_buffers import SamplerWithoutReplacement, PrioritizedSampler
from torchrl.data.replay_buffers.samplers import PrioritizedSliceSampler

from neural import MarioNet


class Mario:
    def __init__(
        self, 
        state_dim, 
        action_dim, 
        save_dir, 
        checkpoint: Path | None = None,
        batch_size: int = 64,
        gamma: float = 0.9,
        burnin: float = 1e5,
        learn_every: int = 4,
        sync_every: float = 1e4,
        memory_size: int = 50000,
        per_alpha: float = 0.6,
        per_beta: float = 0.4,
        per_beta_increment: float = 0.001,
        per_eps: float = 1e-6
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.save_dir = save_dir
        # memory size 表示 replay buffer 的大小，默认保存在 cpu 上
        self.memory_size = memory_size

        self.use_cuda = torch.cuda.is_available()
        if torch.backends.mps.is_available():
            self.device = "mps"
            print("MPS is available!")
        elif self.use_cuda:
            self.device = "cuda"
        else:
            self.device = "cpu"

        # Mario's DNN to predict the most optimal action - we implement this in the Learn section
        self.net = MarioNet(self.state_dim, self.action_dim).float()
        if self.device == "cuda":
            self.net = self.net.to(device=self.device)
        elif self.device == "mps":  # Add this condition to handle MPS device
            self.net = self.net.to(device=self.device)

        # set memory storage device
        if self.use_cuda:
            self.memory_storage_device = torch.device("cuda")
        elif self.device == "mps":  # Add MPS condition
            self.memory_storage_device = torch.device("cpu")  # Keep memory on CPU for MPS compatibility
        else:
            self.memory_storage_device = torch.device("cpu")

        # Hyperparameters
        self.exploration_rate = 1
        self.exploration_rate_decay = 0.99999975
        self.exploration_rate_min = 0
        self.curr_step = 0

        self.save_every = 3e4  # no. of experiences between saving Mario Net

        # PER parameters
        self.per_alpha = per_alpha  # how much prioritization to use (0 = no prioritization, 1 = full prioritization)
        self.per_beta = per_beta  # importance sampling weight (0 = no correction, 1 = full correction)
        self.per_beta_increment = per_beta_increment  # increment for beta annealing
        self.per_eps = per_eps  # small constant to prevent zero priority

        # 先设置batch_size，确保在创建buffer前已初始化
        self.batch_size = batch_size
        self.gamma = gamma
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=0.00025)
        self.loss_fn = torch.nn.SmoothL1Loss(reduction='none')  # Changed to 'none' to apply importance weights

        # Initialize prioritized replay buffer
        storage = LazyTensorStorage(self.memory_size, device=self.memory_storage_device)
        sampler = PrioritizedSampler(
            max_capacity=self.memory_size,
            alpha=self.per_alpha,
            beta=self.per_beta,
            eps=self.per_eps
        )
        self.memory = TensorDictReplayBuffer(
            storage=storage,
            sampler=sampler,
            batch_size=self.batch_size
        )
        
        self.burnin = burnin  # min. experiences before training
        self.learn_every = learn_every  # no. of experiences between updates to Q_online
        self.sync_every = sync_every  # no. of experiences between Q_target & Q_online sync

        # Load checkpoint after all attributes have been initialized
        if checkpoint:
            self.load(checkpoint)

    def act(self, state):
        """
        Given a state, choose an epsilon-greedy action and update value of step.  ​
        Inputs:
        state(LazyFrame): A single observation of the current state, dimension is (state_dim)
        Outputs:
        action_idx (int): An integer representing which action Mario will perform
        """
        # EXPLORE
        if np.random.rand() < self.exploration_rate:
            action_idx = np.random.randint(self.action_dim)

        # EXPLOIT
        else:
            state = state.__array__()
            if self.use_cuda or self.device == "mps":
                state = torch.tensor(state).to(device=self.device)
            else:
                state = torch.tensor(state)
            state = state.unsqueeze(0)
            action_values = self.net(state, model="online")
            action_idx = torch.argmax(action_values, dim=1).item()

        # decrease exploration_rate
        self.exploration_rate *= self.exploration_rate_decay
        self.exploration_rate = max(self.exploration_rate_min, self.exploration_rate)

        # increment step
        self.curr_step += 1
        return action_idx

    def cache(self, state, next_state, action, reward, done):
        """
        Store the experience to self.memory (replay buffer)

        Inputs:
        state (LazyFrame),
        next_state (LazyFrame),
        action (int),
        reward (float),
        done(bool))
        """
        state = state.__array__()
        next_state = next_state.__array__()

        if self.use_cuda or self.device == "mps":  # Add MPS condition here
            state = torch.tensor(state).to(device=self.device)
            next_state = torch.tensor(next_state).to(device=self.device)
            action = torch.tensor([action]).to(device=self.device)
            reward = torch.tensor([reward]).to(device=self.device)
            done = torch.tensor([done]).to(device=self.device)
        else:
            state = torch.tensor(state)
            next_state = torch.tensor(next_state)
            action = torch.tensor([action])
            reward = torch.tensor([reward])
            done = torch.tensor([done])

        # 创建要存储的经验数据
        td = TensorDict(
            {
                "state": state,
                "next_state": next_state,
                "action": action,
                "reward": reward,
                "done": done,
            },
            batch_size=[],
        )
        
        # 添加经验到缓冲区
        # PrioritizedSampler会自动为新样本分配最大优先级
        self.memory.add(td)

    def recall(self):
        """
        Retrieve a batch of experiences from memory with importance sampling weights
        """
        # 首先，更新beta值（重要性采样权重）
        self.per_beta = min(1.0, self.per_beta + self.per_beta_increment)
        
        # 从内存中采样，并获取样本索引和权重
        samples = self.memory.sample()
        batch = samples.to(self.device)
        
        # 获取采样索引和权重（如果可用）
        indices = getattr(samples, "_indices", None)
        weights = getattr(samples, "_weights", None)
        
        # 如果没有权重，默认所有权重为1
        if weights is None:
            weights = torch.ones(self.batch_size, device=self.device)
        
        state, next_state, action, reward, done = (
            batch.get(key)
            for key in ("state", "next_state", "action", "reward", "done")
        )
        
        return state, next_state, action.squeeze(), reward.squeeze(), done.squeeze(), indices, weights

    def td_estimate(self, state, action):
        """
        估计当前状态下的Q值
        """
        current_Q = self.net(state, model="online")[
            np.arange(0, self.batch_size), action
        ]  # Q_online(s,a)
        return current_Q

    @torch.no_grad()
    def td_target(self, reward, next_state, done):
        next_state_Q = self.net(next_state, model="online")
        best_action = torch.argmax(next_state_Q, dim=1)
        next_Q = self.net(next_state, model="target")[
            np.arange(0, self.batch_size), best_action
        ]
        return (reward + (1 - done.float()) * self.gamma * next_Q).float()

    def update_Q_online(self, td_estimate, td_target, weights=None):
        """
        使用重要性采样权重更新在线Q网络
        """
        # 计算TD误差
        td_errors = td_estimate - td_target
        
        # 如果提供了权重，则应用权重
        if weights is not None:
            # 应用重要性采样权重
            loss = self.loss_fn(td_estimate, td_target) * weights
            loss = loss.mean()
        else:
            # 没有权重时直接计算损失
            loss = self.loss_fn(td_estimate, td_target).mean()
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item(), td_errors.detach().abs()

    def sync_Q_target(self):
        # 同步参数 online net 到 target net
        self.net.target.load_state_dict(self.net.online.state_dict())

    def save(self):
        save_path = (
            self.save_dir / f"mario_net_{int(self.curr_step // self.save_every)}.chkpt"
        )
        torch.save(
            {
                'model': self.net.state_dict(),
                'exploration_rate': self.exploration_rate,
                'curr_step': self.curr_step,
                'optimizer': self.optimizer.state_dict(),
                'memory_size': len(self.memory),  # Current memory usage
                'max_memory_size': self.memory_size,  # Maximum memory capacity
                'per_alpha': self.per_alpha,  # 保存PER相关参数
                'per_beta': self.per_beta,
                'per_beta_increment': self.per_beta_increment,
                'per_eps': self.per_eps,
            },
            save_path,
        )
        print(f"MarioNet saved to {save_path} at step {self.curr_step}")

    def load(self, load_path: Path):
        """load checkpoint"""
        if not load_path.exists():
            raise ValueError(f"Checkpoint {load_path} does not exist")

        try:
            ckp = torch.load(load_path, map_location=self.device)
            
            # 加载模型参数
            if 'model' in ckp:
                self.net.load_state_dict(ckp['model'])
            else:
                print("Warning: Model state not found in checkpoint")
            
            # 加载探索率
            if 'exploration_rate' in ckp:
                self.exploration_rate = ckp['exploration_rate']
            
            # 加载当前步数
            if 'curr_step' in ckp:
                self.curr_step = ckp['curr_step']
            
            # 加载优化器状态
            if 'optimizer' in ckp and self.optimizer is not None:
                self.optimizer.load_state_dict(ckp['optimizer'])
            
            # 加载内存大小设置
            if 'max_memory_size' in ckp:
                # Only update if different from current setting
                if self.memory_size != ckp['max_memory_size']:
                    print(f"Updating memory size from {self.memory_size} to {ckp['max_memory_size']}")
                    self.memory_size = ckp['max_memory_size']
                    # Recreate memory buffer with loaded size
                    
                    # 加载PER相关参数
                    if 'per_alpha' in ckp:
                        self.per_alpha = ckp['per_alpha']
                    if 'per_beta' in ckp:
                        self.per_beta = ckp['per_beta']
                    if 'per_beta_increment' in ckp:
                        self.per_beta_increment = ckp['per_beta_increment']
                    if 'per_eps' in ckp:
                        self.per_eps = ckp['per_eps']
                    
                    # 使用更新后的参数重新创建优先级回放缓冲区
                    storage = LazyTensorStorage(self.memory_size, device=self.memory_storage_device)
                    sampler = PrioritizedSampler(
                        max_capacity=self.memory_size,
                        alpha=self.per_alpha,
                        beta=self.per_beta,
                        eps=self.per_eps
                    )
                    self.memory = TensorDictReplayBuffer(
                        storage=storage,
                        sampler=sampler,
                        batch_size=self.batch_size
                    )
            
            print(f"Successfully loaded model from {load_path}")
            print(f"Current step: {self.curr_step}, Exploration rate: {self.exploration_rate}")
            
            if 'memory_size' in ckp:
                print(f"Previous memory usage: {ckp['memory_size']}")
            
            # 打印PER参数
            print(f"PER Alpha: {self.per_alpha}, Beta: {self.per_beta}")
            
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            raise

    def learn(self):
        if self.curr_step % self.sync_every == 0:
            self.sync_Q_target()

        if self.curr_step % self.save_every == 0:
            self.save()

        if self.curr_step < self.burnin:
            return None, None

        if self.curr_step % self.learn_every != 0:
            return None, None

        # Sample from memory with importance sampling weights
        state, next_state, action, reward, done, indices, weights = self.recall()

        # Get TD Estimate
        td_est = self.td_estimate(state, action)

        # Get TD Target
        td_tgt = self.td_target(reward, next_state, done)

        # 计算损失并更新网络，同时获取TD误差
        loss, td_errors = self.update_Q_online(td_est, td_tgt, weights)

        # 更新优先级
        if indices is not None:
            # 确保TD误差至少为self.per_eps，防止优先级为0
            priorities = td_errors.clamp(min=self.per_eps).cpu().numpy()
            
            # 更新采样器的优先级
            # 根据TorchRL的实现，PrioritizedSampler可能使用不同的方法更新优先级
            # 尝试几种可能的更新方法
            try:
                if hasattr(self.memory.sampler, 'update_priorities'):
                    self.memory.sampler.update_priorities(indices, priorities)
                elif hasattr(self.memory.sampler, 'update_priority'):
                    for idx, priority in zip(indices, priorities):
                        self.memory.sampler.update_priority(idx, priority)
                # 如果没有这些方法，尝试直接更新priorities属性
                elif hasattr(self.memory.sampler, 'priorities'):
                    self.memory.sampler.priorities[indices] = priorities
            except Exception as e:
                print(f"Warning: Failed to update priorities: {e}")
                # 继续执行而不更新优先级

        return (td_est.mean().item(), loss)
