# TensorBoard Monitoring for Super Mario RL Training

This project now includes TensorBoard integration for monitoring and visualizing the training process of the reinforcement learning agent.

## Features

The TensorBoard logger provides the following visualizations:

- **Episode Metrics**: Reward, length, average loss, and average Q-value per episode
- **Training Metrics**: Moving averages of rewards, episode lengths, losses, and Q-values
- **Action Distribution**: Visualization of which actions the agent is taking
- **Model Analysis**: Weights, gradients, and network architecture (optional)
- **Performance Profiling**: Integration with PyTorch profiler

## Usage

### Basic Training with TensorBoard

```bash
python supermario_train.py
```

This will start training with default parameters and TensorBoard logging enabled.

### Advanced Options

```bash
# Enable model weight and gradient logging
python supermario_train.py --log-model

# Change the logging interval (default: every 20 episodes)
python supermario_train.py --log-interval 10

# Enable performance profiling
python supermario_train.py --profile --profile-episodes 3
```

### Viewing TensorBoard Logs

1. Install TensorBoard if you haven't already:
   ```bash
   pip install tensorboard
   ```

2. Start the TensorBoard server:
   ```bash
   tensorboard --logdir=checkpoints/YYYY-MM-DDTHH-MM-SS/tensorboard
   ```
   Replace `YYYY-MM-DDTHH-MM-SS` with the timestamp of your training run.

3. Open your browser and navigate to `http://localhost:6006`

## Visualizations Available

### Scalars
- **Episode/Reward**: Reward per episode
- **Episode/Length**: Episode length
- **Episode/AvgLoss**: Average loss per episode
- **Episode/AvgQValue**: Average Q-value per episode
- **Training/Epsilon**: Exploration rate over time
- **Training/MeanReward**: Moving average of rewards
- **Training/MeanLength**: Moving average of episode lengths
- **Training/MeanLoss**: Moving average of losses
- **Training/MeanQValue**: Moving average of Q-values
- **Actions/Action_X**: Count of each action taken per episode
- **Actions/Percentage_X**: Percentage of each action taken per episode

### Histograms
- **Histogram/Rewards**: Distribution of rewards
- **Histogram/EpisodeLengths**: Distribution of episode lengths
- **Histogram/QValues**: Distribution of Q-values
- **Weights/...**: Model weight distributions (if --log-model is enabled)
- **Gradients/...**: Model gradient distributions (if --log-model is enabled)

### Graphs
- Network architecture visualization (if --log-model is enabled)

### Profiler
- Detailed performance analysis (if --profile is enabled)

## Implementation Details

The TensorBoard logger replaces the previous MetricLogger with enhanced visualization capabilities. It maintains the same API for easy integration but adds additional features for deeper insights into the training process.

Key components:
- `tensorboard_logger.py`: Main implementation of the TensorBoard logger
- `supermario_train.py`: Updated to use the TensorBoard logger

## Requirements

- PyTorch
- TensorBoard
- NumPy 