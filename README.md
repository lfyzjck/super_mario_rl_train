# super_mario_rl_train


### Environment

- Python 3.12
- PyTorch 2.1.0
- Gymnasium 0.29.0
- Gymnasium-Super Mario Bros 1.0.0


### Features

- 使用重要性采样权重更新在线Q网络
- 支持使用 MPS(MacOS) 设备进行训练
- 支持PyTorch Profiler进行性能分析
- 支持TensorBoard记录训练过程
- 支持自动保存和加载 checkpoint

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

```bash
python replay.py
```




