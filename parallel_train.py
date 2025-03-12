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

from utils import find_latest_checkpoint, create_env
from agent import Mario, MarioNet
from metrics import MetricLogger

def parse_args():
    parser = argparse.ArgumentParser(description='Parallel Super Mario Bros Training')
    parser.add_argument('--batch-size', type=int, default=64,
                    help='batch size for training (default: 64)')
    parser.add_argument('--render', action='store_true',
                    help='render the game while training (only for main process)')
    parser.add_argument('--burnin', type=int, default=100000,
                    help='number of steps to fill replay buffer before learning starts (default: 100000)')
    parser.add_argument('--episodes', type=int, default=2000,
                    help='number of episodes to train (default: 2000)')
    parser.add_argument('--num-processes', type=int, default=4,
                    help='number of parallel training processes (default: 4)')
    parser.add_argument('--sync-every', type=int, default=1000,
                    help='synchronize model parameters every N steps (default: 1000)')
    parser.add_argument('--save-every', type=int, default=5000,
                    help='save model every N steps (default: 5000)')
    return parser.parse_args()

def worker_process(rank, args, model_path, shared_model, counter, lock, done_event):
    """
    Worker process that collects experiences and updates the shared model
    
    Args:
        rank: Process ID
        args: Command line arguments
        model_path: Path to save checkpoints
        shared_model: Shared model parameters
        counter: Shared counter for synchronization
        lock: Lock for synchronization
        done_event: Event to signal when training is done
    """
    torch.manual_seed(args.seed + rank)
    
    # Create environment for this worker
    env = create_env()
    
    # Create local model
    local_model = Mario(
        state_dim=(4, 84, 84),
        action_dim=env.action_space.n if hasattr(env.action_space, 'n') else env.action_space.shape[0],
        save_dir=model_path,
        batch_size=args.batch_size,
        burnin=args.burnin // args.num_processes  # Distribute burnin across processes
    )
    
    # Initialize with shared model parameters
    local_model.net.load_state_dict(shared_model.state_dict())
    
    # Create local optimizer
    optimizer = torch.optim.Adam(local_model.net.online.parameters(), lr=0.00025)
    
    # Create logger for this worker
    worker_log_dir = model_path / f"worker_{rank}"
    worker_log_dir.mkdir(parents=True, exist_ok=True)
    logger = MetricLogger(worker_log_dir)
    
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
            
            # Select action
            action = local_model.act(state)
            
            # Take action in environment
            result = env.step(action)
            if len(result) == 5:  # Handle both gym API versions
                next_state, reward, done, _, info = result
            else:
                next_state, reward, done, info = result
            
            # Store transition in local buffer
            local_model.cache(state, next_state, action, reward, done)
            
            # Update state
            state = next_state
            episode_reward += reward
            
            # Perform local update
            q, loss = local_model.learn()
            
            # Log metrics
            logger.log_step(reward, loss, q)
            
            # Synchronize with shared model periodically
            with counter.get_lock():
                counter.value += 1
                if counter.value % args.sync_every == 0:
                    # Synchronize parameters from shared model to local model
                    local_model.net.load_state_dict(shared_model.state_dict())
                
                # Update shared model with local gradients
                if counter.value % (args.sync_every // 10) == 0:
                    with lock:
                        for param, shared_param in zip(
                            local_model.net.online.parameters(),
                            shared_model.online.parameters()
                        ):
                            if param.grad is not None and shared_param.grad is None:
                                shared_param._grad = param.grad.detach().clone()
                            elif param.grad is not None:
                                shared_param._grad += param.grad.detach().clone()
                
                # Save model periodically (only in main process)
                if rank == 0 and counter.value % args.save_every == 0:
                    with lock:
                        torch.save(
                            shared_model.state_dict(),
                            model_path / f"mario_net_{counter.value}.chkpt"
                        )
            
            # Check if episode is done
            if done or (isinstance(info, dict) and info.get("flag_get", False)):
                break
        
        # Log episode results
        logger.log_episode()
        if rank == 0 and episode % 20 == 0:
            logger.record(
                episode=episode,
                epsilon=local_model.exploration_rate,
                step=counter.value
            )
        
        episode += 1
    
    # Clean up
    env.close()

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
    
    # Create shared counter and lock
    counter = mp.Value('i', 0)
    lock = mp.Lock()
    
    # Create event to signal when training is done
    done_event = mp.Event()
    
    # Create and start worker processes
    processes = []
    for rank in range(args.num_processes):
        p = mp.Process(
            target=worker_process,
            args=(rank, args, save_dir, shared_model, counter, lock, done_event)
        )
        p.start()
        processes.append(p)
        time.sleep(1)  # Stagger process starts
    
    try:
        # Wait for all processes to finish
        for p in processes:
            p.join()
    except KeyboardInterrupt:
        print("Training interrupted")
        done_event.set()
        
        # Wait for processes to terminate
        for p in processes:
            p.join()
    
    print("Training complete")

if __name__ == "__main__":
    main() 