"""
强化学习算法基类
定义了所有RL算法的通用接口和基础功能
为后续实现DQN等其他算法提供统一的框架
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
    强化学习算法抽象基类
    
    定义了所有RL算法必须实现的接口：
    1. 训练步骤
    2. 动作选择
    3. 模型保存和加载
    4. 性能评估
    """
    
    def __init__(self, 
                 observation_space, 
                 action_space,
                 device=None,
                 logger=None):
        """
        初始化基类
        
        Args:
            observation_space: 观察空间
            action_space: 动作空间
            device (torch.device): 计算设备
            logger (TrainingLogger): 日志记录器
        """
        self.observation_space = observation_space
        self.action_space = action_space
        self.device = device if device else Config.DEVICE
        self.logger = logger
        
        # 训练状态
        self.training = True
        self.total_steps = 0
        self.total_episodes = 0
        self.total_updates = 0
        
        # 性能指标
        self.best_reward = float('-inf')
        self.recent_rewards = []
        
        # 网络和优化器（由子类实现）
        self.networks = {}
        self.optimizers = {}
        
        print(f"Initialized {self.__class__.__name__}")
        print(f"Device: {self.device}")
        print(f"Observation space: {observation_space}")
        print(f"Action space: {action_space}")
    
    @abstractmethod
    def update(self, batch_data: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        执行一次网络参数更新
        
        Args:
            batch_data (dict): 批次数据，包含状态、动作、奖励等
            
        Returns:
            dict: 更新统计信息（损失值等）
        """
        pass
    
    @abstractmethod
    def act(self, 
            observations: torch.Tensor, 
            deterministic: bool = False) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        根据观察选择动作
        
        Args:
            observations (torch.Tensor): 观察
            deterministic (bool): 是否选择确定性动作
            
        Returns:
            tuple: (动作, 额外信息字典)
        """
        pass
    
    @abstractmethod
    def save_model(self, filepath: str) -> None:
        """
        保存模型参数
        
        Args:
            filepath (str): 保存路径
        """
        pass
    
    @abstractmethod
    def load_model(self, filepath: str) -> None:
        """
        加载模型参数
        
        Args:
            filepath (str): 模型文件路径
        """
        pass
    
    def train(self):
        """
        设置为训练模式
        """
        self.training = True
        for network in self.networks.values():
            if hasattr(network, 'train'):
                network.train()
    
    def eval(self):
        """
        设置为评估模式
        """
        self.training = False
        for network in self.networks.values():
            if hasattr(network, 'eval'):
                network.eval()
    
    def get_network_info(self) -> Dict[str, Any]:
        """
        获取网络信息
        
        Returns:
            dict: 网络信息字典
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
        更新训练统计信息
        
        Args:
            episode_reward (float): 回合奖励
        """
        self.total_episodes += 1
        self.recent_rewards.append(episode_reward)
        
        # 保持最近100回合的奖励记录
        if len(self.recent_rewards) > 100:
            self.recent_rewards.pop(0)
        
        # 更新最佳奖励
        if episode_reward > self.best_reward:
            self.best_reward = episode_reward
        
        # 记录到日志
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
        获取训练统计信息
        
        Returns:
            dict: 训练统计信息
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
        判断是否应该保存当前模型
        
        Args:
            current_reward (float): 当前平均奖励
            
        Returns:
            bool: 是否保存
        """
        return current_reward > self.best_reward
    
    def create_checkpoint(self) -> Dict[str, Any]:
        """
        创建检查点数据
        
        Returns:
            dict: 检查点数据
        """
        checkpoint = {
            'algorithm_name': self.__class__.__name__,
            'total_steps': self.total_steps,
            'total_episodes': self.total_episodes,
            'total_updates': self.total_updates,
            'best_reward': self.best_reward,
            'recent_rewards': self.recent_rewards[-10:],  # 保存最近10个奖励
            'config': {
                'learning_rate': Config.LEARNING_RATE,
                'device': str(self.device),
            }
        }
        
        # 添加网络状态
        for name, network in self.networks.items():
            if hasattr(network, 'state_dict'):
                checkpoint[f'{name}_state_dict'] = network.state_dict()
        
        # 添加优化器状态
        for name, optimizer in self.optimizers.items():
            if hasattr(optimizer, 'state_dict'):
                checkpoint[f'{name}_optimizer_state_dict'] = optimizer.state_dict()
        
        return checkpoint
    
    def load_checkpoint(self, checkpoint: Dict[str, Any]):
        """
        从检查点恢复状态
        
        Args:
            checkpoint (dict): 检查点数据
        """
        # 恢复训练统计
        self.total_steps = checkpoint.get('total_steps', 0)
        self.total_episodes = checkpoint.get('total_episodes', 0)
        self.total_updates = checkpoint.get('total_updates', 0)
        self.best_reward = checkpoint.get('best_reward', float('-inf'))
        self.recent_rewards = checkpoint.get('recent_rewards', [])
        
        # 加载网络状态
        for name, network in self.networks.items():
            state_key = f'{name}_state_dict'
            if state_key in checkpoint and hasattr(network, 'load_state_dict'):
                network.load_state_dict(checkpoint[state_key])
                print(f"Loaded {name} network state")
        
        # 加载优化器状态
        for name, optimizer in self.optimizers.items():
            state_key = f'{name}_optimizer_state_dict'
            if state_key in checkpoint and hasattr(optimizer, 'load_state_dict'):
                optimizer.load_state_dict(checkpoint[state_key])
                print(f"Loaded {name} optimizer state")
    
    def print_algorithm_info(self):
        """
        打印算法详细信息
        """
        print(f"\n{'='*60}")
        print(f"{self.__class__.__name__} 算法信息")
        print(f"{'='*60}")
        
        # 基本信息
        print(f"设备: {self.device}")
        print(f"观察空间: {self.observation_space}")
        print(f"动作空间: {self.action_space}")
        
        # 训练统计
        stats = self.get_training_stats()
        print(f"\n训练统计:")
        for key, value in stats.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")
        
        # 网络信息
        network_info = self.get_network_info()
        if network_info:
            print(f"\n网络信息:")
            for name, info in network_info.items():
                print(f"  {name}:")
                print(f"    总参数: {info['total_parameters']:,}")
                print(f"    可训练参数: {info['trainable_parameters']:,}")
                print(f"    设备: {info['device']}")
        
        print(f"{'='*60}\n")


class ModelManager:
    """
    模型管理器
    
    负责模型的保存、加载、版本管理等功能
    """
    
    def __init__(self, model_dir: str = None):
        """
        初始化模型管理器
        
        Args:
            model_dir (str): 模型保存目录
        """
        self.model_dir = model_dir if model_dir else Config.MODEL_DIR
        os.makedirs(self.model_dir, exist_ok=True)
    
    def save_model(self, 
                   algorithm: BaseRLAlgorithm, 
                   filename: str = None, 
                   is_best: bool = False):
        """
        保存模型
        
        Args:
            algorithm (BaseRLAlgorithm): 要保存的算法
            filename (str): 文件名
            is_best (bool): 是否是最佳模型
        """
        if filename is None:
            filename = f"{algorithm.__class__.__name__.lower()}_model.pth"
        
        filepath = os.path.join(self.model_dir, filename)
        
        # 创建检查点
        checkpoint = algorithm.create_checkpoint()
        
        # 保存
        torch.save(checkpoint, filepath)
        
        # 如果是最佳模型，也保存一份best版本
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
        加载模型
        
        Args:
            algorithm (BaseRLAlgorithm): 目标算法
            filename (str): 文件名
            load_best (bool): 是否加载最佳模型
            
        Returns:
            dict: 加载的检查点信息
        """
        if filename is None:
            filename = f"{algorithm.__class__.__name__.lower()}_model.pth"
        
        if load_best:
            filename = f"best_{filename}"
        
        filepath = os.path.join(self.model_dir, filename)
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        # 加载检查点
        checkpoint = torch.load(filepath, map_location=algorithm.device)
        
        # 恢复算法状态
        algorithm.load_checkpoint(checkpoint)
        
        print(f"Model loaded from {filepath}")
        
        return checkpoint
    
    def list_saved_models(self) -> List[str]:
        """
        列出所有保存的模型
        
        Returns:
            list: 模型文件列表
        """
        model_files = [f for f in os.listdir(self.model_dir) if f.endswith('.pth')]
        return sorted(model_files)
    
    def get_model_info(self, filename: str) -> Dict[str, Any]:
        """
        获取模型文件信息
        
        Args:
            filename (str): 模型文件名
            
        Returns:
            dict: 模型信息
        """
        filepath = os.path.join(self.model_dir, filename)
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        checkpoint = torch.load(filepath, map_location='cpu')
        
        # 提取关键信息
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