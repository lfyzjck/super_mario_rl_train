# super_mario_rl_train


### Environment

- Python 3.12
- PyTorch 2.1.0
- Gymnasium 0.29.0
- Gymnasium-Super Mario Bros 1.0.0


### Features

- 支持 DDQN 和 PPO 两个训练算法

- 支持使用 MPS(MacOS) 设备进行训练
- 支持TensorBoard记录训练过程
- 支持自动保存和加载 checkpoint

##### DQN

- 使用重要性采样权重更新在线Q网络
- 优先经验回放缓冲区采样

##### PPO

TODO

### Installation

```bash
uv sync
```

that's all.

### Training

```bash
$ python supermario_train.py --help
Super Mario Bros Training

options:
  -h, --help            show this help message and exit
  --batch-size BATCH_SIZE
                        batch size for training (default: 64)
  --render              render the game while training
  --burnin BURNIN       number of steps to fill replay buffer before learning starts (default: 100000)
  --episodes EPISODES   number of episodes to train (default: 2000)
  --profile             enable PyTorch profiler for performance analysis
  --profile-episodes PROFILE_EPISODES
                        number of episodes to profile (default: 5)
  --log-model           log model weights and gradients to TensorBoard
  --log-interval LOG_INTERVAL
                        interval between logging metrics (default: 20)
```

training with default parameters:

```bash
python supermario_train.py
```

### Evaluation

#### Evaluating PPO Models with Human Rendering

To evaluate a trained PPO model with human rendering (to watch the agent play), use the evaluation script:

```bash
./eval_mario.sh
```

The script supports various options:

```
Usage: ./eval_mario.sh [options]
Options:
  --model PATH      Path to the model file (default: latest model)
  --vecnorm PATH    Path to the VecNormalize statistics file (default: latest stats)
  --episodes N      Number of episodes to evaluate (default: 3)
  --video           Record videos of the gameplay (default: false)
  --no-render       Don't render in human mode, use for headless evaluation
  --device DEVICE   Device to use for inference (cpu, cuda, auto) (default: auto)
  --help            Show this help message
```

Examples:

1. Watch the agent play with default settings:
   ```bash
   ./eval_mario.sh
   ```

2. Record videos of gameplay:
   ```bash
   ./eval_mario.sh --video
   ```

3. Evaluate a specific model for 5 episodes:
   ```bash
   ./eval_mario.sh --model path/to/model.zip --episodes 5
   ```

4. Run headless evaluation (without rendering):
   ```bash
   ./eval_mario.sh --no-render
   ```

#### Legacy Evaluation

For older models:

```bash
python replay.py
```




