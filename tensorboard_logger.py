import time
import numpy as np
from pathlib import Path
from torch.utils.tensorboard.writer import SummaryWriter


class TensorBoardLogger:
    """
    TensorBoard-based logger for tracking metrics during training.
    Replaces the MetricLogger with TensorBoard visualization.
    """
    def __init__(self, save_dir):
        """
        Initialize the TensorBoard logger.
        
        Args:
            save_dir (Path): Directory to save TensorBoard logs
        """
        # Create TensorBoard writer
        self.log_dir = save_dir / "tensorboard"
        self.log_dir.mkdir(exist_ok=True)
        self.writer = SummaryWriter(log_dir=str(self.log_dir))
        
        # History metrics
        self.ep_rewards = []
        self.ep_lengths = []
        self.ep_avg_losses = []
        self.ep_avg_qs = []
        
        # Action distribution tracking
        self.action_counts = {}
        
        # Current episode metrics
        self.init_episode()
        
        # Timing
        self.record_time = time.time()
        
        print(f"TensorBoard logs will be saved to {self.log_dir}")
        print(f"To view logs, run: tensorboard --logdir={self.log_dir}")
    
    def log_step(self, reward, loss, q, action=None):
        """
        Log metrics for a single step.
        
        Args:
            reward (float): Reward received at this step
            loss (float): Loss value (if learning occurred, else None)
            q (float): Q-value (if learning occurred, else None)
            action (int, optional): Action taken at this step
        """
        self.curr_ep_reward += reward
        self.curr_ep_length += 1
        
        # Track action distribution if provided
        if action is not None:
            if action not in self.action_counts:
                self.action_counts[action] = 0
            self.action_counts[action] += 1
        
        if loss:
            self.curr_ep_loss += loss
            self.curr_ep_q += q
            self.curr_ep_loss_length += 1
    
    def log_episode(self):
        """Mark end of episode and log episode-level metrics to TensorBoard"""
        self.ep_rewards.append(self.curr_ep_reward)
        self.ep_lengths.append(self.curr_ep_length)
        
        if self.curr_ep_loss_length == 0:
            ep_avg_loss = 0
            ep_avg_q = 0
        else:
            ep_avg_loss = np.round(self.curr_ep_loss / self.curr_ep_loss_length, 5)
            ep_avg_q = np.round(self.curr_ep_q / self.curr_ep_loss_length, 5)
        
        self.ep_avg_losses.append(ep_avg_loss)
        self.ep_avg_qs.append(ep_avg_q)
        
        # Log episode metrics to TensorBoard
        episode_idx = len(self.ep_rewards) - 1
        self.writer.add_scalar('Episode/Reward', self.curr_ep_reward, episode_idx)
        self.writer.add_scalar('Episode/Length', self.curr_ep_length, episode_idx)
        self.writer.add_scalar('Episode/AvgLoss', ep_avg_loss, episode_idx)
        self.writer.add_scalar('Episode/AvgQValue', ep_avg_q, episode_idx)
        
        # Log action distribution as individual scalars
        if self.action_counts:
            # Log as a bar chart using scalars
            for action, count in self.action_counts.items():
                self.writer.add_scalar(f'Actions/Action_{action}', count, episode_idx)
            
            # Also log the total actions taken
            total_actions = sum(self.action_counts.values())
            if total_actions > 0:
                # Log action distribution as percentages
                for action, count in self.action_counts.items():
                    percentage = (count / total_actions) * 100
                    self.writer.add_scalar(f'Actions/Percentage_{action}', percentage, episode_idx)
            
            # Reset action counts for next episode
            self.action_counts = {}
        
        # Reset for next episode
        self.init_episode()
    
    def init_episode(self):
        """Initialize metrics for a new episode"""
        self.curr_ep_reward = 0.0
        self.curr_ep_length = 0
        self.curr_ep_loss = 0.0
        self.curr_ep_q = 0.0
        self.curr_ep_loss_length = 0
    
    def record(self, episode, epsilon, step):
        """
        Record metrics periodically during training.
        
        Args:
            episode (int): Current episode number
            epsilon (float): Current exploration rate
            step (int): Current global step count
        """
        # Calculate moving averages (last 100 episodes)
        mean_ep_reward = np.round(np.mean(self.ep_rewards[-100:]), 3)
        mean_ep_length = np.round(np.mean(self.ep_lengths[-100:]), 3)
        mean_ep_loss = np.round(np.mean(self.ep_avg_losses[-100:]), 3)
        mean_ep_q = np.round(np.mean(self.ep_avg_qs[-100:]), 3)
        
        # Calculate time since last record
        last_record_time = self.record_time
        self.record_time = time.time()
        time_since_last_record = np.round(self.record_time - last_record_time, 3)
        
        # Log to TensorBoard
        self.writer.add_scalar('Training/Epsilon', epsilon, step)
        self.writer.add_scalar('Training/MeanReward', mean_ep_reward, step)
        self.writer.add_scalar('Training/MeanLength', mean_ep_length, step)
        self.writer.add_scalar('Training/MeanLoss', mean_ep_loss, step)
        self.writer.add_scalar('Training/MeanQValue', mean_ep_q, step)
        self.writer.add_scalar('Training/TimePerRecord', time_since_last_record, step)
        
        # Add histograms of rewards and episode lengths
        if len(self.ep_rewards) > 0:
            self.writer.add_histogram('Histogram/Rewards', np.array(self.ep_rewards[-100:]), step)
            self.writer.add_histogram('Histogram/EpisodeLengths', np.array(self.ep_lengths[-100:]), step)
            self.writer.add_histogram('Histogram/QValues', np.array(self.ep_avg_qs[-100:]), step)
        
        # Print to console
        print(
            f"Episode {episode} - "
            f"Step {step} - "
            f"Epsilon {epsilon:.3f} - "
            f"Mean Reward {mean_ep_reward:.3f} - "
            f"Mean Length {mean_ep_length:.3f} - "
            f"Mean Loss {mean_ep_loss:.3f} - "
            f"Mean Q Value {mean_ep_q:.3f} - "
            f"Time Delta {time_since_last_record:.3f}s"
        )
    
    def log_model_gradients(self, model, step):
        """
        Log model gradients to TensorBoard.
        
        Args:
            model: PyTorch model with parameters
            step (int): Current global step count
        """
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                self.writer.add_histogram(f'Gradients/{name}', param.grad.data, step)
    
    def log_model_weights(self, model, step):
        """
        Log model weights to TensorBoard.
        
        Args:
            model: PyTorch model with parameters
            step (int): Current global step count
        """
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.writer.add_histogram(f'Weights/{name}', param.data, step)
    
    def log_q_values(self, q_values, step):
        """
        Log Q-values distribution to TensorBoard.
        
        Args:
            q_values: Tensor of Q-values
            step (int): Current global step count
        """
        if q_values is not None:
            self.writer.add_histogram('QValues/Distribution', q_values, step)
    
    def add_graph(self, model, input_tensor):
        """
        Add model graph to TensorBoard.
        
        Args:
            model: PyTorch model
            input_tensor: Example input tensor for the model
        """
        try:
            self.writer.add_graph(model, input_tensor)
        except Exception as e:
            print(f"Failed to add model graph to TensorBoard: {e}")
    
    def close(self):
        """Close the TensorBoard writer"""
        self.writer.close() 