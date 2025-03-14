import datetime
import contextlib

# setup parser
import argparse
from pathlib import Path

# Gym is an OpenAI toolkit for RL
import torch
import torch.profiler as profiler
from torch.utils.tensorboard.writer import SummaryWriter
# NES Emulator for OpenAI Gym
from torchvision import transforms as T
# Replace MetricLogger with TensorBoardLogger
from tensorboard_logger import TensorBoardLogger
from utils import find_latest_checkpoint, create_env
from agent import Mario
from wrappers import SkipFrame, GrayScaleObservation, ResizeObservation

# Initialize Super Mario environment
env = create_env()


@contextlib.contextmanager
def create_pytorch_profiler(log_dir: str = './logs', use_cuda: bool = False):
    writer = SummaryWriter(log_dir)
    activities = [profiler.ProfilerActivity.CPU]
    if use_cuda:
        activities.append(profiler.ProfilerActivity.CUDA)
        
    with profiler.profile(
        activities=activities,
        schedule=profiler.schedule(wait=10, warmup=10, active=80, repeat=1),
        on_trace_ready=profiler.tensorboard_trace_handler(log_dir),
        record_shapes=True,
        profile_memory=False,
        with_stack=True
    ) as prof:
        yield prof
    writer.close()


def parse_args():
    parser = argparse.ArgumentParser(description='Super Mario Bros Training')
    parser.add_argument('--batch-size', type=int, default=64,
                    help='batch size for training (default: 64)')
    parser.add_argument('--render', action='store_true',
                    help='render the game while training')
    parser.add_argument('--burnin', type=int, default=100000,
                    help='number of steps to fill replay buffer before learning starts (default: 100000)')
    parser.add_argument('--episodes', type=int, default=5000,
                    help='number of episodes to train (default: 2000)')
    parser.add_argument('--profile', action='store_true',
                    help='enable PyTorch profiler for performance analysis')
    parser.add_argument('--profile-episodes', type=int, default=5,
                    help='number of episodes to profile (default: 5)')
    parser.add_argument('--log-model', action='store_true',
                    help='log model weights and gradients to TensorBoard')
    parser.add_argument('--log-interval', type=int, default=20,
                    help='interval between logging metrics (default: 20)')
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
        batch_size=args.batch_size,
        burnin=args.burnin
    )

    # Replace MetricLogger with TensorBoardLogger
    logger = TensorBoardLogger(save_dir)
    
    # Log the model architecture if requested
    if args.log_model:
        # Create a sample input tensor for the model graph
        sample_input = torch.zeros((1, 4, 84, 84), dtype=torch.float32)
        if use_cuda:
            sample_input = sample_input.cuda()
        # Add model graph to TensorBoard
        logger.add_graph(mario.net, sample_input)

    print(f"Burnin phase: Agent will explore for {args.burnin} steps before learning starts")
    print(f"Current step: {mario.curr_step}, Learning will start at step: {args.burnin}")
    print(f"Training for {args.episodes} episodes")
    
    episodes = args.episodes
    
    # Training loop
    for e in range(episodes):
        state = env.reset()
        # Enable profiler only for specific episodes if requested
        profiling = args.profile and e < args.profile_episodes
        
        if profiling:
            profile_dir = save_dir / f"profile_episode_{e}"
            profile_dir.mkdir(exist_ok=True)
            prof_context = create_pytorch_profiler(str(profile_dir), use_cuda)
        else:
            # Create a dummy context manager when not profiling
            prof_context = contextlib.nullcontext()
            
        with prof_context as prof:

            # Play the game!
            while True:
                if args.render:
                    env.render()

                # Run agent on the state
                action = mario.act(state)

                # Agent performs action
                next_state, reward, done, info = env.step(action)

                # Remember
                mario.cache(state, next_state, action, reward, done)

                # Learn
                q, loss = mario.learn()

                # Logging - pass the action to track action distribution
                logger.log_step(reward, loss, q, action)

                # Update state
                state = next_state
                
                # Call profiler step if we're profiling
                if profiling and prof is not None:
                    prof.step()

                # Check if end of game
                if done or info['flag_get']:
                    break

        logger.log_episode()

        if e % args.log_interval == 0:
            logger.record(
                episode=e, epsilon=mario.exploration_rate, step=mario.curr_step
            )
            
            # Log model weights and gradients if requested
            if args.log_model:
                logger.log_model_weights(mario.net.online, mario.curr_step)
                # Only log gradients if we're past the burnin phase
                if mario.curr_step > args.burnin:
                    logger.log_model_gradients(mario.net.online, mario.curr_step)
    
    # Close the TensorBoard logger
    logger.close()

