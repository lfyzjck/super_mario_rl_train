import datetime
import contextlib

# setup parser
import argparse
from pathlib import Path

# Gym is an OpenAI toolkit for RL
import torch
import torch.profiler as profiler
from torch.utils.tensorboard import SummaryWriter
# NES Emulator for OpenAI Gym
from torchvision import transforms as T
from metrics import MetricLogger
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
    parser.add_argument('--episodes', type=int, default=2000,
                    help='number of episodes to train (default: 2000)')
    parser.add_argument('--profile', action='store_true',
                    help='enable PyTorch profiler for performance analysis')
    parser.add_argument('--profile-episodes', type=int, default=5,
                    help='number of episodes to profile (default: 5)')
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

    logger = MetricLogger(save_dir)

    print(f"Burnin phase: Agent will explore for {args.burnin} steps before learning starts")
    print(f"Current step: {mario.curr_step}, Learning will start at step: {args.burnin}")
    print(f"Training for {args.episodes} episodes")
    
    episodes = args.episodes
    
    # Training loop
    for e in range(episodes):
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
            state = env.reset()

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

                # Logging
                logger.log_step(reward, loss, q)

                # Update state
                state = next_state
                
                # Call profiler step if we're profiling
                if profiling and prof is not None:
                    prof.step()

                # Check if end of game
                if done or (isinstance(info, dict) and info.get("flag_get", False)):
                    break

        logger.log_episode()

        if e % 20 == 0:
            logger.record(
                episode=e, epsilon=mario.exploration_rate, step=mario.curr_step
            )

