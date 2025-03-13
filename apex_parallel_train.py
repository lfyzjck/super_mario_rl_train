import datetime
import contextlib
import argparse
import time
import os
from pathlib import Path
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch.utils.tensorboard.writer import SummaryWriter
import numpy as np
from collections import deque
import random
import sys

from utils import find_latest_checkpoint, create_env
from agent import Mario, MarioNet
from metrics import MetricLogger

# Define a shared replay buffer
class SharedReplayBuffer:
    def __init__(self, capacity, state_dim, device="cpu"):
        """
        Shared replay buffer for all worker processes
        
        Args:
            capacity: Maximum size of the buffer
            state_dim: Dimensions of the state (C, H, W)
            device: Device to store tensors on
        """
        self.capacity = capacity
        self.device = device
        self.lock = mp.Lock()
        
        # 减少初始内存分配，使用更小的初始缓冲区大小
        initial_size = min(10000, capacity)
        print(f"Initializing replay buffer with initial size: {initial_size} (max: {capacity})")
        
        # Use shared memory for the buffer with smaller initial allocation
        self.states = torch.zeros((initial_size, *state_dim), dtype=torch.float32, device=device).share_memory_()
        self.next_states = torch.zeros((initial_size, *state_dim), dtype=torch.float32, device=device).share_memory_()
        self.actions = torch.zeros(initial_size, dtype=torch.long, device=device).share_memory_()
        self.rewards = torch.zeros(initial_size, dtype=torch.float32, device=device).share_memory_()
        self.dones = torch.zeros(initial_size, dtype=torch.bool, device=device).share_memory_()
        self.priorities = torch.ones(initial_size, dtype=torch.float32, device=device).share_memory_()
        
        # 当前缓冲区大小
        self.current_buffer_size = initial_size
        
        # Shared counter for buffer position
        self.pos = mp.Value('i', 0)
        self.size = mp.Value('i', 0)
        
    def _resize_buffer(self, new_size):
        """动态调整缓冲区大小"""
        print(f"Resizing replay buffer from {self.current_buffer_size} to {new_size}")
        
        # 创建新的更大的缓冲区
        new_states = torch.zeros((new_size, *self.states.shape[1:]), dtype=torch.float32, device=self.device).share_memory_()
        new_next_states = torch.zeros((new_size, *self.next_states.shape[1:]), dtype=torch.float32, device=self.device).share_memory_()
        new_actions = torch.zeros(new_size, dtype=torch.long, device=self.device).share_memory_()
        new_rewards = torch.zeros(new_size, dtype=torch.float32, device=self.device).share_memory_()
        new_dones = torch.zeros(new_size, dtype=torch.bool, device=self.device).share_memory_()
        new_priorities = torch.ones(new_size, dtype=torch.float32, device=self.device).share_memory_()
        
        # 复制现有数据
        size = min(self.size.value, self.current_buffer_size)
        if size > 0:
            new_states[:size] = self.states[:size]
            new_next_states[:size] = self.next_states[:size]
            new_actions[:size] = self.actions[:size]
            new_rewards[:size] = self.rewards[:size]
            new_dones[:size] = self.dones[:size]
            new_priorities[:size] = self.priorities[:size]
        
        # 替换旧缓冲区
        self.states = new_states
        self.next_states = new_next_states
        self.actions = new_actions
        self.rewards = new_rewards
        self.dones = new_dones
        self.priorities = new_priorities
        self.current_buffer_size = new_size
        
    def push(self, state, next_state, action, reward, done, priority=None):
        """Add a new experience to the buffer with thread safety"""
        with self.lock:
            idx = self.pos.value
            
            # 检查是否需要扩展缓冲区
            if self.size.value >= self.current_buffer_size and self.current_buffer_size < self.capacity:
                new_size = min(self.capacity, self.current_buffer_size * 2)
                self._resize_buffer(new_size)
            
            # Convert numpy arrays to tensors if needed
            if isinstance(state, np.ndarray):
                state = torch.from_numpy(state).float()
            # 处理 LazyFrames 类型
            elif hasattr(state, '__array__'):  # 检查是否有 __array__ 方法 (LazyFrames 有这个方法)
                state = torch.FloatTensor(np.array(state))
            
            if isinstance(next_state, np.ndarray):
                next_state = torch.from_numpy(next_state).float()
            # 处理 LazyFrames 类型
            elif hasattr(next_state, '__array__'):
                next_state = torch.FloatTensor(np.array(next_state))
            
            # Store transition
            self.states[idx % self.current_buffer_size] = state
            self.next_states[idx % self.current_buffer_size] = next_state
            self.actions[idx % self.current_buffer_size] = action
            self.rewards[idx % self.current_buffer_size] = reward
            self.dones[idx % self.current_buffer_size] = done
            
            # Update priority if provided
            if priority is not None:
                self.priorities[idx % self.current_buffer_size] = priority
            else:
                self.priorities[idx % self.current_buffer_size] = 1.0  # Default priority
            
            # Update position and size
            self.pos.value = (self.pos.value + 1) % self.capacity
            self.size.value = min(self.size.value + 1, self.capacity)
            
            return idx % self.current_buffer_size
            
    def sample(self, batch_size, alpha=0.6, beta=0.4):
        """Sample a batch of experiences with prioritized experience replay"""
        with self.lock:
            size = min(self.size.value, self.current_buffer_size)
            if size == 0:
                return None
                
            # Calculate sampling probabilities
            probs = self.priorities[:size] ** alpha
            probs /= probs.sum()
            
            # Sample indices based on priorities
            indices = torch.multinomial(probs, min(batch_size, size), replacement=True)
            
            # Calculate importance sampling weights
            weights = (size * probs[indices]) ** (-beta)
            weights /= weights.max()
            
            # Return batch
            batch = {
                'states': self.states[indices],
                'next_states': self.next_states[indices],
                'actions': self.actions[indices],
                'rewards': self.rewards[indices],
                'dones': self.dones[indices],
                'weights': weights,
                'indices': indices
            }
            
            return batch
            
    def update_priorities(self, indices, priorities):
        """Update priorities for sampled experiences"""
        with self.lock:
            # 确保索引在有效范围内
            valid_indices = indices[indices < self.current_buffer_size]
            if len(valid_indices) > 0:
                self.priorities[valid_indices] = priorities[:len(valid_indices)]
            
    def __len__(self):
        """Return the current size of the buffer"""
        return min(self.size.value, self.current_buffer_size)

def parse_args():
    parser = argparse.ArgumentParser(description='Ape-X DQN Parallel Super Mario Bros Training')
    parser.add_argument('--batch-size', type=int, default=512,
                    help='batch size for training (default: 512)')
    parser.add_argument('--render', action='store_true',
                    help='render the game while training (only for main process)')
    parser.add_argument('--buffer-size', type=int, default=100000,
                    help='maximum replay buffer size (default: 100000)')
    parser.add_argument('--episodes', type=int, default=2000,
                    help='number of episodes to train (default: 2000)')
    parser.add_argument('--num-actors', type=int, default=4,
                    help='number of actor processes (default: 4)')
    parser.add_argument('--sync-every', type=int, default=1000,
                    help='synchronize model parameters every N steps (default: 1000)')
    parser.add_argument('--save-every', type=int, default=5000,
                    help='save model every N steps (default: 5000)')
    parser.add_argument('--learning-rate', type=float, default=0.00025,
                    help='learning rate (default: 0.00025)')
    parser.add_argument('--alpha', type=float, default=0.6,
                    help='prioritized replay alpha (default: 0.6)')
    parser.add_argument('--beta', type=float, default=0.4,
                    help='prioritized replay beta (default: 0.4)')
    parser.add_argument('--beta-annealing', type=float, default=0.001,
                    help='beta annealing rate (default: 0.001)')
    return parser.parse_args()

def actor_process(rank, args, model_path, shared_model, shared_buffer, counter, lock, done_event):
    """
    Actor process that collects experiences and adds them to the shared replay buffer
    
    Args:
        rank: Process ID
        args: Command line arguments
        model_path: Path to save checkpoints
        shared_model: Shared model parameters
        shared_buffer: Shared replay buffer
        counter: Shared counter for synchronization
        lock: Lock for synchronization
        done_event: Event to signal when training is done
    """
    try:
        torch.manual_seed(args.seed + rank)
        
        # Create environment for this worker
        env = create_env()
        
        # Create local model
        local_model = MarioNet(
            input_dim=(4, 84, 84),
            output_dim=env.action_space.n if hasattr(env.action_space, 'n') else env.action_space.shape[0]
        )
        
        # Initialize with shared model parameters
        local_model.load_state_dict(shared_model.state_dict())
        
        # Set exploration rate based on actor rank (different exploration rates for diversity)
        exploration_rate = 0.4 * (1.0 / (rank + 1))
        
        # Create logger for this actor
        actor_log_dir = model_path / f"actor_{rank}"
        actor_log_dir.mkdir(parents=True, exist_ok=True)
        logger = MetricLogger(actor_log_dir)
        
        # Training loop
        episode = 0
        while not done_event.is_set() and episode < args.episodes:
            state = env.reset()
            episode_reward = 0
            done = False
            
            # Play one episode
            while not done:
                # Only render in the main process if requested
                if rank == 0 and args.render:
                    env.render()
                
                # Select action with epsilon-greedy policy
                if random.random() < exploration_rate:
                    action = env.action_space.sample()
                else:
                    # 处理 LazyFrames 类型
                    if hasattr(state, '__array__'):
                        state_tensor = torch.FloatTensor(np.array(state)).unsqueeze(0)
                    else:
                        state_tensor = torch.FloatTensor(state).unsqueeze(0)
                    
                    with torch.no_grad():
                        q_values = local_model(state_tensor, model="online")
                    action = q_values.argmax().item()
                
                # Take action in environment
                result = env.step(action)
                if len(result) == 5:  # Handle both gym API versions
                    next_state, reward, done, _, info = result
                else:
                    next_state, reward, done, info = result
                
                # Store transition in shared buffer
                shared_buffer.push(state, next_state, action, reward, done)
                
                # Update state
                state = next_state
                episode_reward += reward
                
                # Log step metrics
                logger.log_step(reward, None, None)
                
                # Synchronize with shared model periodically
                with counter.get_lock():
                    counter.value += 1
                    if counter.value % args.sync_every == 0:
                        # Synchronize parameters from shared model to local model
                        local_model.load_state_dict(shared_model.state_dict())
                
                # Check if episode is done
                if done or (isinstance(info, dict) and info.get("flag_get", False)):
                    break
            
            # Log episode results
            logger.log_episode()
            if rank == 0 and episode % 20 == 0:
                logger.record(
                    episode=episode,
                    epsilon=exploration_rate,
                    step=counter.value
                )
            
            # Decay exploration rate
            exploration_rate = max(0.1, exploration_rate * 0.996)
            
            episode += 1
        
        # Clean up
        env.close()
    except Exception as e:
        print(f"Error in actor process {rank}: {e}")
        import traceback
        traceback.print_exc()
        done_event.set()  # 通知其他进程终止

def learner_process(args, model_path, shared_model, shared_buffer, counter, lock, done_event):
    """
    Learner process that updates the shared model based on experiences from the replay buffer
    
    Args:
        args: Command line arguments
        model_path: Path to save checkpoints
        shared_model: Shared model parameters
        shared_buffer: Shared replay buffer
        counter: Shared counter for synchronization
        lock: Lock for synchronization
        done_event: Event to signal when training is done
    """
    try:
        torch.manual_seed(args.seed)
        
        # Create optimizer
        optimizer = torch.optim.Adam(shared_model.parameters(), lr=args.learning_rate)
        
        # Create target network
        target_model = MarioNet(input_dim=(4, 84, 84), output_dim=shared_model.online[-1].out_features)
        target_model.load_state_dict(shared_model.state_dict())
        
        # Create logger
        learner_log_dir = model_path / "learner"
        learner_log_dir.mkdir(parents=True, exist_ok=True)
        logger = MetricLogger(learner_log_dir)
        writer = SummaryWriter(model_path / "tensorboard")
        
        # Training loop
        step = 0
        beta = args.beta  # Initial beta for importance sampling
        
        while not done_event.is_set() and step < args.episodes * 10000:  # Arbitrary large number
            # Wait until buffer has enough samples
            if len(shared_buffer) < args.batch_size:
                time.sleep(0.1)
                continue
            
            # Sample batch from replay buffer
            batch = shared_buffer.sample(args.batch_size, alpha=args.alpha, beta=beta)
            if batch is None:
                continue
            
            # Compute TD error
            states = batch['states']
            next_states = batch['next_states']
            actions = batch['actions']
            rewards = batch['rewards']
            dones = batch['dones']
            weights = batch['weights']
            indices = batch['indices']
            
            # Compute Q values
            q_values = shared_model(states, model="online")
            next_q_values = target_model(next_states, model="online")
            next_q_state_values = shared_model(next_states, model="target")
            
            # Select Q values for chosen actions
            q_value = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
            
            # Double Q-learning: use online network to select actions, target network to evaluate
            next_q_value = next_q_state_values.gather(
                1, next_q_values.argmax(dim=1, keepdim=True)
            ).squeeze(1)
            
            # Compute expected Q values
            expected_q_value = rewards + (1 - dones.float()) * 0.99 * next_q_value
            
            # Compute loss with importance sampling weights
            loss = F.smooth_l1_loss(q_value, expected_q_value.detach(), reduction='none')
            weighted_loss = (loss * weights).mean()
            
            # Update priorities in replay buffer
            priorities = loss.detach().abs() + 1e-6  # Add small constant to avoid zero priority
            shared_buffer.update_priorities(indices, priorities)
            
            # Optimize the model
            optimizer.zero_grad()
            weighted_loss.backward()
            
            # Clip gradients to stabilize training
            for param in shared_model.parameters():
                if param.grad is not None:
                    param.grad.data.clamp_(-1, 1)
            
            optimizer.step()
            
            # Periodically update target network
            if step % 1000 == 0:
                target_model.load_state_dict(shared_model.state_dict())
            
            # Log metrics
            if step % 100 == 0:
                writer.add_scalar('Loss/train', weighted_loss.item(), step)
                writer.add_scalar('Q_value/mean', q_value.mean().item(), step)
                writer.add_scalar('Buffer/size', len(shared_buffer), step)
                writer.add_scalar('Hyperparameters/beta', beta, step)
            
            # Save model periodically
            if step % args.save_every == 0:
                with lock:
                    torch.save(
                        shared_model.state_dict(),
                        model_path / f"mario_net_{step}.chkpt"
                    )
            
            # Anneal beta for importance sampling
            beta = min(1.0, beta + args.beta_annealing)
            
            step += 1
        
        # Final save
        torch.save(
            shared_model.state_dict(),
            model_path / "mario_net_final.chkpt"
        )
        
        writer.close()
    except Exception as e:
        print(f"Error in learner process: {e}")
        import traceback
        traceback.print_exc()
        done_event.set()  # 通知其他进程终止

def main():
    # Parse arguments
    args = parse_args()
    
    # Set up multiprocessing - 安全地设置启动方法
    try:
        # 检查当前启动方法
        if mp.get_start_method() != 'spawn':
            mp.set_start_method('spawn', force=True)
    except RuntimeError:
        # 如果已经设置过启动方法，则忽略错误
        print("Multiprocessing start method already set, using existing method:", mp.get_start_method())
    
    # Set random seed for reproducibility
    args.seed = 123
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Check if CUDA is available
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"Using device: {device}")
    
    # 打印内存使用情况
    try:
        import psutil
        process = psutil.Process(os.getpid())
        print(f"Initial memory usage: {process.memory_info().rss / 1024 / 1024:.2f} MB")
    except ImportError:
        print("psutil not installed, memory usage monitoring disabled")
    
    # Create save directory
    save_dir = Path("checkpoints") / datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    save_dir.mkdir(parents=True)
    
    # Load latest checkpoint if available
    checkpoint = find_latest_checkpoint(Path("checkpoints"))
    print(f"Loading checkpoint from {checkpoint}")
    
    # Create a temporary environment to get action space size
    temp_env = create_env()
    action_dim = temp_env.action_space.n if hasattr(temp_env.action_space, 'n') else temp_env.action_space.shape[0]
    temp_env.close()
    
    # Create shared model
    shared_model = MarioNet(input_dim=(4, 84, 84), output_dim=action_dim)
    
    if checkpoint:
        # Load parameters from checkpoint
        state_dict = torch.load(checkpoint, map_location=device)
        shared_model.load_state_dict(state_dict)
    
    # Share model parameters across processes
    shared_model.share_memory()
    
    # Create shared replay buffer - 修复 device 参数类型问题
    device_str = str(device)  # 将 device 转换为字符串
    shared_buffer = SharedReplayBuffer(
        capacity=args.buffer_size,
        state_dim=(4, 84, 84),
        device=device_str
    )
    
    # Create shared counter and lock
    counter = mp.Value('i', 0)
    lock = mp.Lock()
    
    # Create event to signal when training is done
    done_event = mp.Event()
    
    # Create and start actor processes
    processes = []
    for rank in range(args.num_actors):
        p = mp.Process(
            target=actor_process,
            args=(rank, args, save_dir, shared_model, shared_buffer, counter, lock, done_event)
        )
        p.daemon = True  # 设置为守护进程，这样主进程退出时它们会自动终止
        p.start()
        processes.append(p)
        time.sleep(1)  # Stagger process starts
    
    # Create and start learner process
    learner = mp.Process(
        target=learner_process,
        args=(args, save_dir, shared_model, shared_buffer, counter, lock, done_event)
    )
    learner.daemon = True  # 设置为守护进程
    learner.start()
    processes.append(learner)
    
    try:
        # 添加定期打印内存使用情况
        if 'psutil' in sys.modules:
            while not done_event.is_set():
                time.sleep(30)  # 每30秒打印一次
                print(f"Current memory usage: {process.memory_info().rss / 1024 / 1024:.2f} MB")
                print(f"Replay buffer size: {len(shared_buffer)}")
                
                # 检查所有进程是否还在运行
                all_alive = True
                for i, p in enumerate(processes):
                    if not p.is_alive():
                        print(f"Process {i} is no longer alive")
                        all_alive = False
                
                if not all_alive:
                    print("Some processes have died, terminating training")
                    done_event.set()
        else:
            # 如果没有psutil，只等待进程完成
            for p in processes:
                p.join()
    except KeyboardInterrupt:
        print("Training interrupted")
        done_event.set()
    finally:
        # 确保所有进程都已终止
        done_event.set()
        
        # 等待进程终止
        for p in processes:
            p.join(timeout=5)  # 设置超时，避免无限等待
            
        # 检查是否有进程仍在运行
        for i, p in enumerate(processes):
            if p.is_alive():
                print(f"Process {i} is still running, terminating forcefully")
                p.terminate()
    
    print("Training complete")

if __name__ == "__main__":
    main() 