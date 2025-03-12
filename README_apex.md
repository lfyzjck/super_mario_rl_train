# Super Mario Bros Ape-X DQN Parallel Training

This project implements an Ape-X DQN (Distributed Prioritized Experience Replay) architecture for training a reinforcement learning agent to play Super Mario Bros using PyTorch and multiprocessing.

## Features

- **Distributed Architecture**: Multiple actor processes collect experiences while a central learner process updates the model
- **Prioritized Experience Replay**: Experiences are sampled based on their TD-error, focusing learning on the most informative transitions
- **Shared Replay Buffer**: All actors contribute to a shared replay buffer, increasing experience diversity
- **Asynchronous Updates**: The learner process updates the model asynchronously from experience collection
- **Exploration Diversity**: Different actors use different exploration rates to increase state-space coverage

## Requirements

- Python 3.8+
- PyTorch 1.8+
- gym-super-mario-bros
- OpenAI Gym
- numpy
- tensorboard

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd super_mario_rl_train

# Install dependencies
pip install -r requirements.txt
```

## Usage

To start Ape-X DQN parallel training:

```bash
python apex_parallel_train.py --num-actors 4 --episodes 2000 --batch-size 512
```

### Command Line Arguments

- `--num-actors`: Number of actor processes (default: 4)
- `--batch-size`: Batch size for training (default: 512)
- `--render`: Render the game while training (only for main process)
- `--buffer-size`: Replay buffer size (default: 1,000,000)
- `--episodes`: Number of episodes to train (default: 2000)
- `--sync-every`: Synchronize model parameters every N steps (default: 1000)
- `--save-every`: Save model every N steps (default: 5000)
- `--learning-rate`: Learning rate (default: 0.00025)
- `--alpha`: Prioritized replay alpha parameter (default: 0.6)
- `--beta`: Prioritized replay beta parameter (default: 0.4)
- `--beta-annealing`: Beta annealing rate (default: 0.001)

## How It Works

### Architecture

The Ape-X DQN architecture consists of:

1. **Multiple Actor Processes**:
   - Each actor has its own environment instance
   - Actors use different exploration rates to increase diversity
   - Actors collect experiences and add them to the shared replay buffer
   - Actors periodically sync their model parameters with the shared model

2. **Central Learner Process**:
   - Samples batches from the shared replay buffer using prioritized experience replay
   - Updates the shared model parameters
   - Updates experience priorities based on TD errors
   - Periodically saves model checkpoints

3. **Shared Replay Buffer**:
   - Stores experiences from all actors
   - Implements prioritized experience replay
   - Uses shared memory for efficient inter-process communication

### Prioritized Experience Replay

The implementation uses prioritized experience replay with:

- **Priority**: TD-error magnitude determines sampling probability
- **Alpha**: Controls how much prioritization is used (α=0 is uniform sampling)
- **Beta**: Controls importance sampling correction (β=1 fully corrects bias)
- **Beta Annealing**: Beta gradually increases during training to reduce bias

## Performance Considerations

- **CPU Usage**: Each process runs on a separate CPU core, so ensure your system has enough cores
- **Memory Usage**: The shared replay buffer can consume significant memory (adjust `--buffer-size` accordingly)
- **GPU Utilization**: The learner process can benefit from GPU acceleration
- **Process Communication**: Shared memory minimizes overhead between processes

## Monitoring

Training progress can be monitored using TensorBoard:

```bash
tensorboard --logdir=checkpoints
```

The following metrics are tracked:
- Loss values
- Q-value estimates
- Replay buffer size
- Beta parameter value

## Troubleshooting

- If you encounter CUDA out-of-memory errors, try reducing the batch size or using CPU only
- If experiencing high memory usage, reduce the replay buffer size
- If training is unstable, try adjusting the learning rate or prioritization parameters 