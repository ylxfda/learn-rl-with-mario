"""
Training logger module.
Responsible for recording training metrics, including:
- Reward curves
- Loss values
- Training statistics
- TensorBoard visualization
"""

import os
import json
import time
import numpy as np
from collections import deque, defaultdict
from datetime import datetime

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    print("Warning: TensorBoard not available. Install with 'pip install tensorboard'")
    TENSORBOARD_AVAILABLE = False

from config import Config


class TrainingLogger:
    """
    Training logger.
    
    Features:
    1) Record metrics during training
    2) Compute moving averages
    3) Save logs to file
    4) TensorBoard support
    5) System performance stats
    """
    
    def __init__(self, log_dir=None, experiment_name=None):
        """
        Initialize logger.
        
        Args:
            log_dir (str): directory to save logs
            experiment_name (str): experiment name
        """
        # Choose base log directory
        if log_dir is None:
            log_dir = Config.LOG_DIR
        
        # Create experiment-specific log directory
        if experiment_name is None:
            experiment_name = f"mario_ppo_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        self.log_dir = os.path.join(log_dir, experiment_name)
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Initialize TensorBoard writer
        self.tensorboard_writer = None
        if Config.TENSORBOARD_LOG and TENSORBOARD_AVAILABLE:
            self.tensorboard_writer = SummaryWriter(self.log_dir)
        
        # Metric storage
        self.metrics = defaultdict(list)
        self.episode_metrics = defaultdict(list)
        
        # Moving averages (for smoothing plots)
        self.running_averages = defaultdict(lambda: deque(maxlen=100))
        
        # Training counters
        self.start_time = time.time()
        self.episode_count = 0
        self.step_count = 0
        self.update_count = 0
        
        # Best performance tracking
        self.best_reward = float('-inf')
        self.best_episode = 0
        
        print(f"Logs will be saved to: {self.log_dir}")
        
    def log_episode(self, episode_reward, episode_length, info=None):
        """
        Log episode-end information.
        
        Args:
            episode_reward (float): total return
            episode_length (int): episode length (steps)
            info (dict): additional game info
        """
        self.episode_count += 1
        
        # Basic metrics
        self.episode_metrics['reward'].append(episode_reward)
        self.episode_metrics['length'].append(episode_length)
        
        # Update moving averages
        self.running_averages['reward'].append(episode_reward)
        self.running_averages['length'].append(episode_length)
        
        # Log game-specific scalars
        if info:
            for key, value in info.items():
                if isinstance(value, (int, float)):
                    self.episode_metrics[key].append(value)
                    self.running_averages[key].append(value)
        
        # Update best-so-far
        if episode_reward > self.best_reward:
            self.best_reward = episode_reward
            self.best_episode = self.episode_count
        
        # TensorBoard
        if self.tensorboard_writer:
            self.tensorboard_writer.add_scalar('Episode/Reward', episode_reward, self.episode_count)
            self.tensorboard_writer.add_scalar('Episode/Length', episode_length, self.episode_count)
            self.tensorboard_writer.add_scalar('Episode/Reward_MA', np.mean(self.running_averages['reward']), self.episode_count)
            
            if info:
                for key, value in info.items():
                    if isinstance(value, (int, float)):
                        self.tensorboard_writer.add_scalar(f'Episode/{key}', value, self.episode_count)
    
    def log_training_step(self, **metrics):
        """
        Record metrics during a training step.
        
        Args:
            **metrics: training metrics (loss, learning_rate, etc.)
        """
        self.step_count += 1
        
        for key, value in metrics.items():
            if isinstance(value, (int, float, np.floating, np.integer)):
                self.metrics[key].append(float(value))
                
                # TensorBoard
                if self.tensorboard_writer:
                    self.tensorboard_writer.add_scalar(f'Training/{key}', value, self.step_count)
    
    def log_update(self, **metrics):
        """
        Record metrics at PPO update time.
        
        Args:
            **metrics: PPO metrics (policy_loss, value_loss, entropy, ...)
        """
        self.update_count += 1
        
        for key, value in metrics.items():
            if isinstance(value, (int, float, np.floating, np.integer)):
                self.metrics[f'update_{key}'].append(float(value))
                
                # TensorBoard
                if self.tensorboard_writer:
                    self.tensorboard_writer.add_scalar(f'Update/{key}', value, self.update_count)
    
    def log_system_info(self, **info):
        """
        Record system info (memory, GPU usage, etc.).
        
        Args:
            **info: system info dict
        """
        if self.tensorboard_writer:
            for key, value in info.items():
                if isinstance(value, (int, float, np.floating, np.integer)):
                    self.tensorboard_writer.add_scalar(f'System/{key}', value, self.step_count)
    
    def get_recent_average(self, metric_name, window=100):
        """
        Get moving average of a metric.
        
        Args:
            metric_name (str): metric name
            window (int): averaging window
            
        Returns:
            float: average or None
        """
        if metric_name in self.running_averages and len(self.running_averages[metric_name]) > 0:
            return np.mean(list(self.running_averages[metric_name])[-window:])
        return None
    
    def print_training_stats(self):
        """
        Print training statistics.
        """
        if self.episode_count == 0:
            return
        
        current_time = time.time()
        elapsed_time = current_time - self.start_time
        
        # Compute averages
        avg_reward = self.get_recent_average('reward', 100)
        avg_length = self.get_recent_average('length', 100)
        
        print(f"\n{'='*60}")
        print(f"Training Stats (episode {self.episode_count})")
        print(f"{'='*60}")
        print(f"Elapsed: {elapsed_time/3600:.2f} hours")
        print(f"Total steps: {self.step_count:,}")
        print(f"Total updates: {self.update_count}")
        print(f"Avg reward (last 100): {avg_reward:.2f}" if avg_reward else "Avg reward: N/A")
        print(f"Avg length (last 100): {avg_length:.1f}" if avg_length else "Avg length: N/A")
        print(f"Best reward: {self.best_reward:.2f} (episode {self.best_episode})")
        
        # Show recent losses
        recent_losses = ['update_policy_loss', 'update_value_loss', 'update_total_loss']
        for loss_name in recent_losses:
            if loss_name in self.metrics and self.metrics[loss_name]:
                recent_loss = self.metrics[loss_name][-1]
                display_name = loss_name.replace('update_', '').replace('_', ' ').title()
                print(f"{display_name}: {recent_loss:.4f}")
        
        print(f"{'='*60}\n")
    
    def save_training_log(self):
        """
        Save training log to JSON file.
        """
        log_data = {
            'experiment_info': {
                'start_time': datetime.fromtimestamp(self.start_time).isoformat(),
                'episode_count': self.episode_count,
                'step_count': self.step_count,
                'update_count': self.update_count,
                'best_reward': self.best_reward,
                'best_episode': self.best_episode,
            },
            'episode_metrics': dict(self.episode_metrics),
            'training_metrics': dict(self.metrics),
            'config': {
                'num_envs': Config.NUM_ENVS,
                'learning_rate': Config.LEARNING_RATE,
                'ppo_epochs': Config.PPO_EPOCHS,
                'clip_epsilon': Config.CLIP_EPSILON,
                'frame_stack': Config.FRAME_STACK,
            }
        }
        
        # Save to JSON file
        log_file = os.path.join(self.log_dir, 'training_log.json')
        with open(log_file, 'w') as f:
            json.dump(log_data, f, indent=2)
        
        print(f"Training log saved to: {log_file}")
    
    def close(self):
        """
        Close the logger and cleanup resources.
        """
        if self.tensorboard_writer:
            self.tensorboard_writer.close()
        
        # Save final log
        self.save_training_log()
        
    def __del__(self):
        """
        Ensure resources are cleaned up.
        """
        self.close()


class PerformanceMonitor:
    """
    Performance monitor for system resource usage during training.
    """
    
    def __init__(self):
        """
        Initialize performance monitor.
        """
        self.gpu_available = False
        
        # Try enabling GPU monitoring
        try:
            import torch
            if torch.cuda.is_available():
                self.gpu_available = True
                self.device_count = torch.cuda.device_count()
        except ImportError:
            pass
    
    def get_gpu_memory_usage(self):
        """
        Get GPU memory usage.
        
        Returns:
            dict: GPU memory info
        """
        if not self.gpu_available:
            return {}
        
        import torch
        gpu_info = {}
        
        for i in range(self.device_count):
            memory_allocated = torch.cuda.memory_allocated(i) / 1024**3  # GB
            memory_cached = torch.cuda.memory_reserved(i) / 1024**3     # GB
            
            gpu_info[f'gpu_{i}_memory_allocated'] = memory_allocated
            gpu_info[f'gpu_{i}_memory_cached'] = memory_cached
        
        return gpu_info
    
    def get_system_info(self):
        """
        Get system info.
        
        Returns:
            dict: resource usage info
        """
        import psutil
        
        # CPU usage
        cpu_percent = psutil.cpu_percent(interval=1)
        
        # Memory usage
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        memory_used_gb = memory.used / 1024**3
        
        system_info = {
            'cpu_percent': cpu_percent,
            'memory_percent': memory_percent,
            'memory_used_gb': memory_used_gb,
        }
        
        # Add GPU info
        system_info.update(self.get_gpu_memory_usage())
        
        return system_info


class ProgressTracker:
    """
    Training progress tracker.
    """
    
    def __init__(self, target_reward=3000, patience=100):
        """
        Initialize.
        
        Args:
            target_reward (float): target average reward
            patience (int): early-stop patience in episodes
        """
        self.target_reward = target_reward
        self.patience = patience
        
        self.best_avg_reward = float('-inf')
        self.episodes_without_improvement = 0
        self.target_achieved = False
        
        self.reward_history = deque(maxlen=100)  # last 100 episode rewards
    
    def update(self, episode_reward):
        """
        Update tracker with a new episode.
        
        Args:
            episode_reward (float): reward for the episode
            
        Returns:
            dict: tracking info
        """
        self.reward_history.append(episode_reward)
        
        # Compute moving average once enough data collected
        if len(self.reward_history) >= 10:
            avg_reward = np.mean(self.reward_history)
            
            # Check improvement
            if avg_reward > self.best_avg_reward:
                self.best_avg_reward = avg_reward
                self.episodes_without_improvement = 0
            else:
                self.episodes_without_improvement += 1
            
            # Check if target achieved
            if avg_reward >= self.target_reward and not self.target_achieved:
                self.target_achieved = True
                print(f"\n🎉 Target achieved! Avg reward {avg_reward:.2f} >= {self.target_reward}")
        
        return {
            'avg_reward': np.mean(self.reward_history) if self.reward_history else 0,
            'best_avg_reward': self.best_avg_reward,
            'episodes_without_improvement': self.episodes_without_improvement,
            'target_achieved': self.target_achieved,
            'should_stop': self.episodes_without_improvement >= self.patience
        }
