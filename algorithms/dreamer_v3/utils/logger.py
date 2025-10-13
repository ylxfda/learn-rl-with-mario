"""
Logging utilities for DreamerV3 training.

Supports:
- TensorBoard logging
- Weights & Biases (optional)
- Console output
- Metric aggregation
"""

import torch
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from typing import Dict, Optional, Union
import numpy as np


class Logger:
    """
    Logger for training metrics and visualizations.
    
    Aggregates metrics over logging intervals and writes to:
    - TensorBoard
    - Weights & Biases (if enabled)
    - Console
    """
    
    def __init__(self, config: dict):
        """
        Args:
            config: Configuration dictionary
        """
        self.config = config
        
        # Create log directory
        self.log_dir = Path(config['logging']['log_dir'])
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # TensorBoard
        self.use_tensorboard = config['logging']['use_tensorboard']
        if self.use_tensorboard:
            self.tb_writer = SummaryWriter(str(self.log_dir / 'tensorboard'))
            print(f"TensorBoard logging to: {self.log_dir / 'tensorboard'}")
        
        # Weights & Biases
        self.use_wandb = config['logging'].get('use_wandb', False)
        if self.use_wandb:
            import wandb
            wandb.init(
                project=config['logging']['wandb_project'],
                entity=config['logging'].get('wandb_entity'),
                config=config,
                name=f"dreamerv3_mario_{config['training']['seed']}"
            )
            print("Weights & Biases logging enabled")
        
        # Metric buffers (for aggregation)
        self.metrics_buffer = {
            'world_model': {},
            'actor_critic': {},
            'episode': {},
            'evaluation': {}
        }
        
        # Best metrics tracking
        self.best_eval_reward = -float('inf')
        self.best_success_rate = 0.0
    
    def log_world_model(self, losses: Dict[str, torch.Tensor], step: int):
        """
        Log world model losses.
        
        Args:
            losses: Dictionary of loss tensors
            step: Global training step
        """
        # Convert to scalars and aggregate
        for key, value in losses.items():
            if isinstance(value, torch.Tensor):
                value = value.item()
            
            if key not in self.metrics_buffer['world_model']:
                self.metrics_buffer['world_model'][key] = []
            self.metrics_buffer['world_model'][key].append(value)
    
    def log_actor_critic(self, metrics: Dict[str, float], step: int):
        """
        Log actor-critic metrics.
        
        Args:
            metrics: Dictionary of metrics
            step: Global training step
        """
        for key, value in metrics.items():
            if key not in self.metrics_buffer['actor_critic']:
                self.metrics_buffer['actor_critic'][key] = []
            self.metrics_buffer['actor_critic'][key].append(value)
    
    def log_episode(self, episode_info: Dict[str, float], step: int, episode_num: Optional[int] = None):
        """
        Log episode statistics.

        Args:
            episode_info: Episode information dictionary
            step: Global training step
            episode_num: Episode number (optional)
        """
        for key, value in episode_info.items():
            if key not in self.metrics_buffer['episode']:
                self.metrics_buffer['episode'][key] = []
            self.metrics_buffer['episode'][key].append(value)

        # Print episode summary
        episode_prefix = f"Training Episode #{episode_num}" if episode_num is not None else "Training Episode"
        print(f"\n[{episode_prefix}]")
        print(f"  Global Step: {step}")
        print(f"  Reward: {episode_info['reward']:.2f}")
        print(f"  Length: {episode_info['length']}")
        print(f"  Max X: {episode_info['max_x_pos']}")
        print(f"  Success: {episode_info['flag_get']}")
    
    def log_evaluation(self, eval_metrics: Dict[str, float], step: int):
        """
        Log evaluation metrics.
        
        Args:
            eval_metrics: Evaluation metrics dictionary
            step: Global training step
        """
        self.metrics_buffer['evaluation'] = eval_metrics
        
        # Check if best model
        if eval_metrics['mean_reward'] > self.best_eval_reward:
            self.best_eval_reward = eval_metrics['mean_reward']
            print("  *** New best reward! ***")
        
        if eval_metrics['success_rate'] > self.best_success_rate:
            self.best_success_rate = eval_metrics['success_rate']
            print("  *** New best success rate! ***")
    
    def flush(self, step: int):
        """
        Write aggregated metrics to loggers and clear buffers.
        
        Args:
            step: Global training step
        """
        # Aggregate and log world model metrics
        if self.metrics_buffer['world_model']:
            aggregated = {}
            for key, values in self.metrics_buffer['world_model'].items():
                aggregated[f'world_model/{key}'] = np.mean(values)
            
            self._write_metrics(aggregated, step)
            self.metrics_buffer['world_model'] = {}
        
        # Aggregate and log actor-critic metrics
        if self.metrics_buffer['actor_critic']:
            aggregated = {}
            for key, values in self.metrics_buffer['actor_critic'].items():
                aggregated[f'actor_critic/{key}'] = np.mean(values)
            
            self._write_metrics(aggregated, step)
            self.metrics_buffer['actor_critic'] = {}
        
        # Aggregate and log episode metrics
        if self.metrics_buffer['episode']:
            aggregated = {}
            for key, values in self.metrics_buffer['episode'].items():
                aggregated[f'episode/{key}'] = np.mean(values)
                aggregated[f'episode/{key}_max'] = np.max(values)
            
            self._write_metrics(aggregated, step)
            self.metrics_buffer['episode'] = {}
        
        # Log evaluation metrics (no aggregation needed)
        if self.metrics_buffer['evaluation']:
            metrics = {f'eval/{k}': v for k, v in self.metrics_buffer['evaluation'].items()}
            self._write_metrics(metrics, step)
            self.metrics_buffer['evaluation'] = {}
    
    def _write_metrics(self, metrics: Dict[str, float], step: int):
        """
        Write metrics to all enabled loggers.
        
        Args:
            metrics: Dictionary of metric names and values
            step: Global training step
        """
        # TensorBoard
        if self.use_tensorboard:
            for key, value in metrics.items():
                self.tb_writer.add_scalar(key, value, step)
        
        # Weights & Biases
        if self.use_wandb:
            import wandb
            wandb.log(metrics, step=step)

    def log_video(
        self,
        tag: str,
        video: Union[torch.Tensor, np.ndarray],
        step: int,
        fps: int = 12
    ):
        """
        Log a video directly to TensorBoard.

        Args:
            tag: TensorBoard tag (e.g., "eval/reconstruction_0")
            video: Video tensor with shape (T, C, H, W) or (1, T, C, H, W)
                   Values can be uint8 [0, 255] or float [0, 1].
            step: Global training step
            fps: Playback FPS for TensorBoard
        """
        if not self.use_tensorboard:
            return

        if isinstance(video, np.ndarray):
            tensor = torch.from_numpy(video)
        else:
            tensor = video.detach()

        tensor = tensor.cpu()

        if tensor.ndim == 4:
            # Expand missing batch dimension expected by add_video
            tensor = tensor.unsqueeze(0)
        elif tensor.ndim != 5:
            raise ValueError(f"Video tensor must have 4 or 5 dims, got shape {tuple(tensor.shape)}")

        if tensor.dtype != torch.uint8:
            # Convert floating videos in [0,1] to uint8 per TensorBoard requirements
            tensor = torch.clamp(tensor, 0.0, 1.0)
            tensor = (tensor * 255.0).to(torch.uint8)

        self.tb_writer.add_video(tag, tensor, step, fps=fps)
    
    def close(self):
        """Close all loggers."""
        if self.use_tensorboard:
            self.tb_writer.close()
        
        if self.use_wandb:
            import wandb
            wandb.finish()


# ============================================================================
# Visualization Utilities
# ============================================================================

def visualize_reconstruction(
    original: torch.Tensor,
    reconstructed: torch.Tensor,
    save_path: Optional[str] = None
) -> np.ndarray:
    """
    Create side-by-side visualization of original and reconstructed images.
    
    Args:
        original: Original images, shape: (B, C, H, W)
        reconstructed: Reconstructed images, shape: (B, C, H, W)
        save_path: Optional path to save visualization
        
    Returns:
        Visualization as numpy array
    """
    import matplotlib.pyplot as plt
    
    # Take first 4 samples
    n_samples = min(4, original.shape[0])
    
    fig, axes = plt.subplots(2, n_samples, figsize=(n_samples * 3, 6))
    
    for i in range(n_samples):
        # Original
        img_orig = original[i].cpu().numpy()
        if img_orig.shape[0] == 1:  # Grayscale
            img_orig = img_orig[0]
            axes[0, i].imshow(img_orig, cmap='gray')
        else:  # RGB
            img_orig = np.transpose(img_orig, (1, 2, 0))
            axes[0, i].imshow(img_orig)
        axes[0, i].axis('off')
        if i == 0:
            axes[0, i].set_title('Original', fontsize=12)
        
        # Reconstructed
        img_recon = reconstructed[i].cpu().numpy()
        if img_recon.shape[0] == 1:  # Grayscale
            img_recon = img_recon[0]
            axes[1, i].imshow(img_recon, cmap='gray')
        else:  # RGB
            img_recon = np.transpose(img_recon, (1, 2, 0))
            axes[1, i].imshow(img_recon)
        axes[1, i].axis('off')
        if i == 0:
            axes[1, i].set_title('Reconstructed', fontsize=12)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    # Convert to numpy for logging
    fig.canvas.draw()
    img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    
    plt.close(fig)
    
    return img


def plot_training_curves(log_dir: str, save_path: Optional[str] = None):
    """
    Plot training curves from tensorboard logs.
    
    Args:
        log_dir: Directory containing tensorboard logs
        save_path: Optional path to save plot
    """
    import matplotlib.pyplot as plt
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    
    # Load tensorboard data
    event_acc = EventAccumulator(log_dir)
    event_acc.Reload()
    
    # Get available tags
    tags = event_acc.Tags()['scalars']
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot episode reward
    if 'episode/reward' in tags:
        data = event_acc.Scalars('episode/reward')
        steps = [x.step for x in data]
        values = [x.value for x in data]
        axes[0, 0].plot(steps, values, alpha=0.3)
        # Smooth with moving average
        window = min(50, len(values) // 10)
        if window > 1:
            smoothed = np.convolve(values, np.ones(window)/window, mode='valid')
            axes[0, 0].plot(steps[window-1:], smoothed, linewidth=2)
        axes[0, 0].set_title('Episode Reward')
        axes[0, 0].set_xlabel('Steps')
        axes[0, 0].set_ylabel('Reward')
        axes[0, 0].grid(True, alpha=0.3)
    
    # Plot world model loss
    if 'world_model/total_loss' in tags:
        data = event_acc.Scalars('world_model/total_loss')
        steps = [x.step for x in data]
        values = [x.value for x in data]
        axes[0, 1].plot(steps, values)
        axes[0, 1].set_title('World Model Loss')
        axes[0, 1].set_xlabel('Steps')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].grid(True, alpha=0.3)
    
    # Plot actor loss
    if 'actor_critic/actor_loss' in tags:
        data = event_acc.Scalars('actor_critic/actor_loss')
        steps = [x.step for x in data]
        values = [x.value for x in data]
        axes[1, 0].plot(steps, values)
        axes[1, 0].set_title('Actor Loss')
        axes[1, 0].set_xlabel('Steps')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].grid(True, alpha=0.3)
    
    # Plot evaluation metrics
    if 'eval/success_rate' in tags:
        data = event_acc.Scalars('eval/success_rate')
        steps = [x.step for x in data]
        values = [x.value for x in data]
        axes[1, 1].plot(steps, values, marker='o')
        axes[1, 1].set_title('Success Rate (Eval)')
        axes[1, 1].set_xlabel('Steps')
        axes[1, 1].set_ylabel('Success Rate')
        axes[1, 1].set_ylim([0, 1.1])
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()


# ============================================================================
# Progress Tracking
# ============================================================================

class ProgressTracker:
    """
    Track training progress and estimate time remaining.
    """
    
    def __init__(self, total_steps: int):
        """
        Args:
            total_steps: Total number of training steps
        """
        self.total_steps = total_steps
        self.start_time = None
        self.steps_completed = 0
    
    def start(self):
        """Start tracking."""
        import time
        self.start_time = time.time()
    
    def update(self, current_step: int):
        """Update progress."""
        self.steps_completed = current_step
    
    def get_eta(self) -> str:
        """
        Get estimated time remaining.
        
        Returns:
            Formatted ETA string
        """
        if self.start_time is None or self.steps_completed == 0:
            return "Unknown"
        
        import time
        elapsed = time.time() - self.start_time
        steps_per_sec = self.steps_completed / elapsed
        remaining_steps = self.total_steps - self.steps_completed
        eta_seconds = remaining_steps / steps_per_sec if steps_per_sec > 0 else 0
        
        # Format as HH:MM:SS
        hours = int(eta_seconds // 3600)
        minutes = int((eta_seconds % 3600) // 60)
        seconds = int(eta_seconds % 60)
        
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    
    def get_progress_str(self) -> str:
        """
        Get progress string.
        
        Returns:
            Formatted progress string
        """
        percent = 100 * self.steps_completed / self.total_steps
        return f"{self.steps_completed}/{self.total_steps} ({percent:.1f}%) - ETA: {self.get_eta()}"
