import copy
from pathlib import Path
import numpy as np
import torch
from torch import nn
from tensordict import TensorDict
from torchrl.data import TensorDictReplayBuffer, LazyMemmapStorage


class MarioNet(nn.Module):
    """mini cnn structure
    input -> (conv2d + relu) x 3 -> flatten -> (dense + relu) x 2 -> output
    """

    def __init__(self, input_dim, output_dim):
        super().__init__()
        c, h, w = input_dim

        if h != 84:
            raise ValueError(f"Expecting input height: 84, got: {h}")
        if w != 84:
            raise ValueError(f"Expecting input width: 84, got: {w}")

        self.online = nn.Sequential(
            nn.Conv2d(in_channels=c, out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim),
        )

        self.target = copy.deepcopy(self.online)

        # Q_target parameters are frozen.
        for p in self.target.parameters():
            p.requires_grad = False

    def forward(self, input, model):
        if model == "online":
            return self.online(input)
        elif model == "target":
            return self.target(input)


class Mario:
    def __init__(
        self, 
        state_dim, 
        action_dim, 
        save_dir, 
        checkpoint: Path | None = None,
        batch_size: int = 32,
        gamma: float = 0.9,
        burnin: float = 1e5,
        learn_every: int = 4,
        sync_every: float = 1e4
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.save_dir = save_dir

        self.use_cuda = torch.cuda.is_available()
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
            print("MPS is available!")
        elif torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"

        # Mario's DNN to predict the most optimal action - we implement this in the Learn section
        self.net = MarioNet(self.state_dim, self.action_dim).float()
        if self.device == "cuda" or self.device == "mps":
            self.net = self.net.to(device=self.device)

        # Hyperparameters
        self.exploration_rate = 1
        self.exploration_rate_decay = 0.99999975
        self.exploration_rate_min = 0
        self.curr_step = 0

        self.save_every = 3e4  # no. of experiences between saving Mario Net
        self.memory = TensorDictReplayBuffer(
            storage=LazyMemmapStorage(100000, device=torch.device("cpu"))
        )
        self.batch_size = batch_size
        self.gamma = gamma
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=0.00025)
        self.loss_fn = torch.nn.SmoothL1Loss()

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
            if self.use_cuda:
                state = torch.tensor(state).cuda()
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

        if self.use_cuda:
            state = torch.tensor(state).cuda(device=self.device)
            next_state = torch.tensor(next_state).cuda(device=self.device)
            action = torch.tensor([action]).cuda(device=self.device)
            reward = torch.tensor([reward]).cuda(device=self.device)
            done = torch.tensor([done]).cuda(device=self.device)
        else:
            state = torch.tensor(state)
            next_state = torch.tensor(next_state)
            action = torch.tensor([action])
            reward = torch.tensor([reward])
            done = torch.tensor([done])

        self.memory.add(
            TensorDict(
                {
                    "state": state,
                    "next_state": next_state,
                    "action": action,
                    "reward": reward,
                    "done": done,
                },
                batch_size=[],
            )
        )

    def recall(self):
        """
        Retrieve a batch of experiences from memory
        """
        batch = self.memory.sample(self.batch_size).to("cpu")
        state, next_state, action, reward, done = (
            batch.get(key)
            for key in ("state", "next_state", "action", "reward", "done")
        )
        return state, next_state, action.squeeze(), reward.squeeze(), done.squeeze()

    def td_estimate(self, state, action):
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

    def update_Q_online(self, td_estimate, td_target):
        loss = self.loss_fn(td_estimate, td_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def sync_Q_target(self):
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
                'memory_size': len(self.memory),  # 只保存大小，不保存整个内存
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
            
            print(f"Successfully loaded model from {load_path}")
            print(f"Current step: {self.curr_step}, Exploration rate: {self.exploration_rate}")
            
            if 'memory_size' in ckp:
                print(f"Previous memory size: {ckp['memory_size']}")
            
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

        # Sample from memory
        state, next_state, action, reward, done = self.recall()

        # Get TD Estimate
        td_est = self.td_estimate(state, action)

        # Get TD Target
        td_tgt = self.td_target(reward, next_state, done)

        # Backpropagate loss through Q_online
        loss = self.update_Q_online(td_est, td_tgt)

        return (td_est.mean().item(), loss)
