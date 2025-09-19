"""
Base classes for Reinforcement Learning algorithms.
Defines common interfaces and utilities shared by RL algorithms
to provide a unified framework for implementing variants (e.g., DQN, A2C, PPO).
"""

from abc import ABC, abstractmethod
import torch
import torch.nn as nn
import os
from typing import Dict, Any, Tuple, List

from config import Config
from utils.logger import TrainingLogger


class BaseRLAlgorithm(ABC):
    """
    Abstract base class for RL algorithms.
    
    Required interfaces:
    1) update step
    2) action selection
    3) save/load model
    4) performance reporting
    """
    
    def __init__(self, 
                 observation_space, 
                 action_space,
                 device=None,
                 logger=None):
        """
        Initialize base class state.
        
        Args:
            observation_space: Observation space
            action_space: Action space
            device (torch.device): Compute device
            logger (TrainingLogger): Logger instance
        """
        self.observation_space = observation_space
        self.action_space = action_space
        self.device = device if device else Config.DEVICE
        self.logger = logger
        
        # Training state
        self.training = True
        self.total_steps = 0
        self.total_episodes = 0
        self.total_updates = 0
        
        # Performance metrics
        self.best_reward = float('-inf')
        self.recent_rewards = []
        
        # Networks and optimizers (set by subclasses)
        self.networks = {}
        self.optimizers = {}
        
        print(f"Initialized {self.__class__.__name__}")
        print(f"Device: {self.device}")
        print(f"Observation space: {observation_space}")
        print(f"Action space: {action_space}")
    
    @abstractmethod
    def update(self, batch_data: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Run one optimization step.
        
        Args:
            batch_data (dict): Mini-batch of training data (states/actions/rewards/etc.)
            
        Returns:
            dict: Update statistics (e.g., losses)
        """
        pass
    
    @abstractmethod
    def act(self, 
            observations: torch.Tensor, 
            deterministic: bool = False) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Select actions given observations.
        
        Args:
            observations (torch.Tensor): Observations
            deterministic (bool): Whether to act deterministically
            
        Returns:
            tuple: (actions, extra_info)
        """
        pass
    
    @abstractmethod
    def save_model(self, filepath: str) -> None:
        """
        Save model parameters to file.
        
        Args:
            filepath (str): Destination path
        """
        pass
    
    @abstractmethod
    def load_model(self, filepath: str) -> None:
        """
        Load model parameters from file.
        
        Args:
            filepath (str): Model file path
        """
        pass
    
    def train(self):
        """
        Set training mode.
        """
        self.training = True
        for network in self.networks.values():
            if hasattr(network, 'train'):
                network.train()
    
    def eval(self):
        """
        Set evaluation mode.
        """
        self.training = False
        for network in self.networks.values():
            if hasattr(network, 'eval'):
                network.eval()
    
    def get_network_info(self) -> Dict[str, Any]:
        """
        Get network parameter statistics.
        
        Returns:
            dict: Network info
        """
        info = {}
        for name, network in self.networks.items():
            if hasattr(network, 'parameters'):
                total_params = sum(p.numel() for p in network.parameters())
                trainable_params = sum(p.numel() for p in network.parameters() if p.requires_grad)
                info[name] = {
                    'total_parameters': total_params,
                    'trainable_parameters': trainable_params,
                    'device': next(network.parameters()).device.type if total_params > 0 else 'N/A'
                }
        return info
    
    def update_training_stats(self, episode_reward: float):
        """
        Update training statistics.
        
        Args:
            episode_reward (float): Episode return
        """
        self.total_episodes += 1
        self.recent_rewards.append(episode_reward)
        
        # Keep last 100 episode rewards
        if len(self.recent_rewards) > 100:
            self.recent_rewards.pop(0)
        
        # Update best reward
        if episode_reward > self.best_reward:
            self.best_reward = episode_reward
        
        # Log to external logger
        if self.logger:
            avg_reward = sum(self.recent_rewards) / len(self.recent_rewards)
            self.logger.log_training_step(
                episode_reward=episode_reward,
                avg_reward_100=avg_reward,
                best_reward=self.best_reward,
                total_episodes=self.total_episodes
            )
    
    def get_training_stats(self) -> Dict[str, Any]:
        """
        Get aggregated training statistics.
        
        Returns:
            dict: Training statistics
        """
        stats = {
            'total_steps': self.total_steps,
            'total_episodes': self.total_episodes,
            'total_updates': self.total_updates,
            'best_reward': self.best_reward,
        }
        
        if self.recent_rewards:
            stats.update({
                'avg_reward_10': sum(self.recent_rewards[-10:]) / min(10, len(self.recent_rewards)),
                'avg_reward_100': sum(self.recent_rewards) / len(self.recent_rewards),
                'recent_reward': self.recent_rewards[-1]
            })
        
        return stats
    
    def should_save_model(self, current_reward: float) -> bool:
        """
        Decide whether current model should be saved.
        
        Args:
            current_reward (float): Current average reward
            
        Returns:
            bool: True if should save
        """
        return current_reward > self.best_reward
    
    def create_checkpoint(self) -> Dict[str, Any]:
        """
        Create a checkpoint dict containing state.
        
        Returns:
            dict: Checkpoint
        """
        checkpoint = {
            'algorithm_name': self.__class__.__name__,
            'total_steps': self.total_steps,
            'total_episodes': self.total_episodes,
            'total_updates': self.total_updates,
            'best_reward': self.best_reward,
            'recent_rewards': self.recent_rewards[-10:],  # keep last 10 rewards
            'config': {
                'learning_rate': Config.LEARNING_RATE,
                'device': str(self.device),
            }
        }
        
        # Add network state
        for name, network in self.networks.items():
            if hasattr(network, 'state_dict'):
                checkpoint[f'{name}_state_dict'] = network.state_dict()
        
        # Add optimizer state
        for name, optimizer in self.optimizers.items():
            if hasattr(optimizer, 'state_dict'):
                checkpoint[f'{name}_optimizer_state_dict'] = optimizer.state_dict()
        
        return checkpoint
    
    def load_checkpoint(self, checkpoint: Dict[str, Any]):
        """
        Restore state from checkpoint.
        
        Args:
            checkpoint (dict): Checkpoint data
        """
        # Restore training stats
        self.total_steps = checkpoint.get('total_steps', 0)
        self.total_episodes = checkpoint.get('total_episodes', 0)
        self.total_updates = checkpoint.get('total_updates', 0)
        self.best_reward = checkpoint.get('best_reward', float('-inf'))
        self.recent_rewards = checkpoint.get('recent_rewards', [])
        
        # Load network parameters
        for name, network in self.networks.items():
            state_key = f'{name}_state_dict'
            if state_key in checkpoint and hasattr(network, 'load_state_dict'):
                network.load_state_dict(checkpoint[state_key])
                print(f"Loaded {name} network state")
        
        # Load optimizer state
        for name, optimizer in self.optimizers.items():
            state_key = f'{name}_optimizer_state_dict'
            if state_key in checkpoint and hasattr(optimizer, 'load_state_dict'):
                optimizer.load_state_dict(checkpoint[state_key])
                print(f"Loaded {name} optimizer state")
    
    def print_algorithm_info(self):
        """
        Print detailed algorithm information.
        """
        print(f"\n{'='*60}")
        print(f"{self.__class__.__name__} Algorithm Info")
        print(f"{'='*60}")
        
        # Basic info
        print(f"Device: {self.device}")
        print(f"Observation space: {self.observation_space}")
        print(f"Action space: {self.action_space}")
        
        # Training stats
        stats = self.get_training_stats()
        print(f"\nTraining Stats:")
        for key, value in stats.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")
        
        # Network info
        network_info = self.get_network_info()
        if network_info:
            print(f"\nNetwork Info:")
            for name, info in network_info.items():
                print(f"  {name}:")
                print(f"    total_parameters: {info['total_parameters']:,}")
                print(f"    trainable_parameters: {info['trainable_parameters']:,}")
                print(f"    device: {info['device']}")
        
        print(f"{'='*60}\n")


class ModelManager:
    """
    Model manager for saving/loading and inspecting checkpoints.
    """
    
    def __init__(self, model_dir: str = None):
        """
        Initialize the model manager.
        
        Args:
            model_dir (str): Directory to store model files
        """
        self.model_dir = model_dir if model_dir else Config.MODEL_DIR
        os.makedirs(self.model_dir, exist_ok=True)
    
    def save_model(self, 
                   algorithm: BaseRLAlgorithm, 
                   filename: str = None, 
                   is_best: bool = False):
        """
        Save a model checkpoint.
        
        Args:
            algorithm (BaseRLAlgorithm): Algorithm instance to save
            filename (str): File name
            is_best (bool): Whether this is the best model so far
        """
        if filename is None:
            filename = f"{algorithm.__class__.__name__.lower()}_model.pth"
        
        filepath = os.path.join(self.model_dir, filename)
        
        # Create checkpoint
        checkpoint = algorithm.create_checkpoint()
        
        # Persist to disk
        torch.save(checkpoint, filepath)
        
        # If best, also write a best_ copy
        if is_best:
            best_filepath = os.path.join(self.model_dir, f"best_{filename}")
            torch.save(checkpoint, best_filepath)
            print(f"Best model saved to {best_filepath}")
        
        print(f"Model saved to {filepath}")
    
    def load_model(self, 
                   algorithm: BaseRLAlgorithm, 
                   filename: str = None, 
                   load_best: bool = False):
        """
        Load a model checkpoint.
        
        Args:
            algorithm (BaseRLAlgorithm): Algorithm to restore into
            filename (str): File name
            load_best (bool): Whether to load the best_ file
            
        Returns:
            dict: Loaded checkpoint
        """
        if filename is None:
            filename = f"{algorithm.__class__.__name__.lower()}_model.pth"
        
        if load_best:
            filename = f"best_{filename}"
        
        filepath = os.path.join(self.model_dir, filename)
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        # Load checkpoint
        checkpoint = torch.load(filepath, map_location=algorithm.device)
        
        # Restore algorithm state
        algorithm.load_checkpoint(checkpoint)
        
        print(f"Model loaded from {filepath}")
        
        return checkpoint
    
    def list_saved_models(self) -> List[str]:
        """
        List saved model files.
        
        Returns:
            list: Model files
        """
        model_files = [f for f in os.listdir(self.model_dir) if f.endswith('.pth')]
        return sorted(model_files)
    
    def get_model_info(self, filename: str) -> Dict[str, Any]:
        """
        Get information about a stored checkpoint.
        
        Args:
            filename (str): Model file name
            
        Returns:
            dict: Model info
        """
        filepath = os.path.join(self.model_dir, filename)
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        checkpoint = torch.load(filepath, map_location='cpu')
        
        # Extract key info
        info = {
            'filename': filename,
            'filepath': filepath,
            'file_size_mb': os.path.getsize(filepath) / (1024 * 1024),
            'algorithm_name': checkpoint.get('algorithm_name', 'Unknown'),
            'total_steps': checkpoint.get('total_steps', 0),
            'total_episodes': checkpoint.get('total_episodes', 0),
            'best_reward': checkpoint.get('best_reward', 0),
            'config': checkpoint.get('config', {}),
        }
        
        return info
