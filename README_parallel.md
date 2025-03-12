# Super Mario Bros Parallel Training

This project implements parallel training for a reinforcement learning agent playing Super Mario Bros using PyTorch and multiprocessing.

## Features

- Multi-process training using PyTorch's multiprocessing
- Shared model parameters across processes
- Distributed experience collection
- Synchronized model updates
- Configurable number of parallel workers

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

To start parallel training:

```bash
python parallel_train.py --num-processes 4 --episodes 2000 --batch-size 64
```

### Command Line Arguments

- `--num-processes`: Number of parallel training processes (default: 4)
- `--batch-size`: Batch size for training (default: 64)
- `--render`: Render the game while training (only for main process)
- `--burnin`: Number of steps to fill replay buffer before learning starts (default: 100000)
- `--episodes`: Number of episodes to train (default: 2000)
- `--sync-every`: Synchronize model parameters every N steps (default: 1000)
- `--save-every`: Save model every N steps (default: 5000)

## How It Works

1. **Shared Model**: A central model is shared across all processes using PyTorch's `share_memory()`.

2. **Worker Processes**: Each worker process:
   - Creates its own environment
   - Collects experiences using the shared model
   - Performs local updates
   - Periodically synchronizes with the shared model

3. **Synchronization**: 
   - Workers pull the latest parameters from the shared model periodically
   - Workers push their gradients to the shared model
   - A lock ensures thread-safe updates

4. **Checkpointing**:
   - The main process saves checkpoints at regular intervals

## Performance Considerations

- **CPU Usage**: Each process runs on a separate CPU core, so ensure your system has enough cores.
- **Memory Usage**: Multiple environments and replay buffers increase memory usage.
- **GPU Utilization**: If using CUDA, the shared model can benefit from GPU acceleration.

## Monitoring

Training progress can be monitored using TensorBoard:

```bash
tensorboard --logdir=checkpoints
```

## Troubleshooting

- If you encounter "RuntimeError: CUDA error: device-side assert triggered", try running without CUDA by setting `CUDA_VISIBLE_DEVICES=""`.
- If experiencing high memory usage, reduce the number of processes or the replay buffer size. 