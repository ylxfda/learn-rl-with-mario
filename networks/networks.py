"""
神经网络模块
定义PPO算法使用的Actor-Critic网络结构
包括CNN特征提取器、策略网络和价值网络
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Dict

from config import Config


class CNNFeatureExtractor(nn.Module):
    """
    CNN特征提取器
    
    用于从游戏图像中提取有用的特征
    设计参考了DQN和A3C中常用的CNN架构
    """
    
    def __init__(self, input_channels=4, output_dim=512):
        """
        初始化CNN特征提取器
        
        Args:
            input_channels (int): 输入通道数（帧堆叠数）
            output_dim (int): 输出特征维度
        """
        super(CNNFeatureExtractor, self).__init__()
        
        # 第一层卷积：处理大的空间模式
        # 输入: (batch, 4, 84, 84)
        # 输出: (batch, 32, 20, 20)
        self.conv1 = nn.Conv2d(
            in_channels=input_channels,
            out_channels=Config.CNN_CHANNELS[0],  # 32
            kernel_size=Config.CNN_KERNELS[0],    # 8
            stride=Config.CNN_STRIDES[0],         # 4
            padding=0
        )
        
        # 第二层卷积：提取中等尺度特征
        # 输入: (batch, 32, 20, 20)  
        # 输出: (batch, 64, 9, 9)
        self.conv2 = nn.Conv2d(
            in_channels=Config.CNN_CHANNELS[0],   # 32
            out_channels=Config.CNN_CHANNELS[1],  # 64
            kernel_size=Config.CNN_KERNELS[1],    # 4
            stride=Config.CNN_STRIDES[1],         # 2
            padding=0
        )
        
        # 第三层卷积：提取细节特征
        # 输入: (batch, 64, 9, 9)
        # 输出: (batch, 64, 7, 7)
        self.conv3 = nn.Conv2d(
            in_channels=Config.CNN_CHANNELS[1],   # 64
            out_channels=Config.CNN_CHANNELS[2],  # 64
            kernel_size=Config.CNN_KERNELS[2],    # 3
            stride=Config.CNN_STRIDES[2],         # 1
            padding=0
        )
        
        # 计算展平后的特征维度
        # 对于84x84输入，经过上述卷积后得到64*7*7=3136维特征
        self.feature_dim = self._calculate_conv_output_size(input_channels)
        
        # 全连接层：将卷积特征映射到固定维度
        self.fc = nn.Linear(self.feature_dim, output_dim)
        
        # 权重初始化
        self._initialize_weights()
    
    def _calculate_conv_output_size(self, input_channels):
        """
        计算卷积层输出的特征维度
        
        Args:
            input_channels (int): 输入通道数
            
        Returns:
            int: 展平后的特征维度
        """
        # 创建一个测试输入来计算输出尺寸
        test_input = torch.zeros(1, input_channels, Config.FRAME_SIZE, Config.FRAME_SIZE)
        
        with torch.no_grad():
            x = F.relu(self.conv1(test_input))
            x = F.relu(self.conv2(x))
            x = F.relu(self.conv3(x))
            feature_dim = x.numel()  # 总元素数量
        
        return feature_dim
    
    def _initialize_weights(self):
        """
        初始化网络权重
        使用正交初始化，这在强化学习中通常效果较好
        """
        for module in self.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                # 正交初始化
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                # 偏置初始化为0
                nn.init.constant_(module.bias, 0)
    
    def forward(self, x):
        """
        前向传播
        
        Args:
            x (torch.Tensor): 输入图像，形状 (batch, channels, height, width)
            
        Returns:
            torch.Tensor: 特征向量，形状 (batch, output_dim)
        """
        # 确保输入在[0, 1]范围内
        x = x.float() / 255.0 if x.max() > 1.0 else x.float()
        
        # 卷积层 + ReLU激活
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        
        # 展平特征
        x = x.view(x.size(0), -1)
        
        # 全连接层 + ReLU激活
        x = F.relu(self.fc(x))
        
        return x


class PolicyNetwork(nn.Module):
    """
    策略网络（Actor）
    
    根据当前状态输出动作概率分布
    在PPO中，我们需要能够：
    1. 采样动作
    2. 计算动作的概率
    3. 计算策略熵（用于鼓励探索）
    """
    
    def __init__(self, feature_dim, action_dim, hidden_dim=None):
        """
        初始化策略网络
        
        Args:
            feature_dim (int): 输入特征维度
            action_dim (int): 动作空间大小
            hidden_dim (int): 隐藏层维度
        """
        super(PolicyNetwork, self).__init__()
        
        if hidden_dim is None:
            hidden_dim = Config.HIDDEN_SIZE
        
        self.action_dim = action_dim
        
        # 策略网络：特征 -> 隐藏层 -> 动作logits
        self.policy_head = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        
        # 权重初始化
        self._initialize_weights()
    
    def _initialize_weights(self):
        """
        初始化网络权重
        """
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # 对于策略网络，使用较小的权重初始化
                # 这有助于初期的探索
                nn.init.orthogonal_(module.weight, gain=0.01)
                nn.init.constant_(module.bias, 0)
    
    def forward(self, features):
        """
        前向传播，输出动作logits
        
        Args:
            features (torch.Tensor): 输入特征，形状 (batch, feature_dim)
            
        Returns:
            torch.Tensor: 动作logits，形状 (batch, action_dim)
        """
        return self.policy_head(features)
    
    def get_action_distribution(self, features):
        """
        获取动作概率分布
        
        Args:
            features (torch.Tensor): 输入特征
            
        Returns:
            torch.distributions.Categorical: 动作概率分布
        """
        logits = self.forward(features)
        return torch.distributions.Categorical(logits=logits)
    
    def act(self, features, deterministic=False):
        """
        根据当前状态选择动作
        
        Args:
            features (torch.Tensor): 输入特征
            deterministic (bool): 是否选择确定性动作（用于测试）
            
        Returns:
            tuple: (动作, 动作对数概率)
        """
        dist = self.get_action_distribution(features)
        
        if deterministic:
            # 选择概率最大的动作
            action = dist.probs.argmax(dim=-1)
        else:
            # 从分布中采样
            action = dist.sample()
        
        log_prob = dist.log_prob(action)
        
        return action, log_prob
    
    def evaluate_actions(self, features, actions):
        """
        评估给定动作的概率和熵
        
        Args:
            features (torch.Tensor): 输入特征
            actions (torch.Tensor): 动作
            
        Returns:
            tuple: (动作对数概率, 熵)
        """
        dist = self.get_action_distribution(features)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        
        return log_probs, entropy


class ValueNetwork(nn.Module):
    """
    价值网络（Critic）
    
    估计给定状态的价值函数V(s)
    用于计算优势函数和更新策略
    """
    
    def __init__(self, feature_dim, hidden_dim=None):
        """
        初始化价值网络
        
        Args:
            feature_dim (int): 输入特征维度
            hidden_dim (int): 隐藏层维度
        """
        super(ValueNetwork, self).__init__()
        
        if hidden_dim is None:
            hidden_dim = Config.HIDDEN_SIZE
        
        # 价值网络：特征 -> 隐藏层 -> 标量价值
        self.value_head = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)  # 输出标量价值
        )
        
        # 权重初始化
        self._initialize_weights()
    
    def _initialize_weights(self):
        """
        初始化网络权重
        """
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=1.0)
                nn.init.constant_(module.bias, 0)
    
    def forward(self, features):
        """
        前向传播，输出状态价值
        
        Args:
            features (torch.Tensor): 输入特征，形状 (batch, feature_dim)
            
        Returns:
            torch.Tensor: 状态价值，形状 (batch, 1)
        """
        return self.value_head(features)


class ActorCriticNetwork(nn.Module):
    """
    Actor-Critic网络
    
    整合特征提取器、策略网络和价值网络
    这是PPO算法的核心网络结构
    """
    
    def __init__(self, observation_shape, action_dim):
        """
        初始化Actor-Critic网络
        
        Args:
            observation_shape (tuple): 观察空间形状 (channels, height, width)
            action_dim (int): 动作空间大小
        """
        super(ActorCriticNetwork, self).__init__()
        
        self.observation_shape = observation_shape
        self.action_dim = action_dim
        
        # 特征提取器
        input_channels = observation_shape[0]  # 通道数（帧堆叠数）
        feature_dim = Config.HIDDEN_SIZE
        
        self.feature_extractor = CNNFeatureExtractor(
            input_channels=input_channels,
            output_dim=feature_dim
        )
        
        # 策略网络
        self.policy_network = PolicyNetwork(
            feature_dim=feature_dim,
            action_dim=action_dim
        )
        
        # 价值网络
        self.value_network = ValueNetwork(
            feature_dim=feature_dim
        )
        
        print(f"Created ActorCritic network:")
        print(f"  Observation shape: {observation_shape}")
        print(f"  Action dim: {action_dim}")
        print(f"  Feature dim: {feature_dim}")
        print(f"  Total parameters: {self.count_parameters():,}")
    
    def count_parameters(self):
        """
        计算网络总参数数量
        
        Returns:
            int: 参数总数
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def forward(self, observations):
        """
        前向传播，同时输出动作分布和状态价值
        
        Args:
            observations (torch.Tensor): 观察，形状 (batch, *observation_shape)
            
        Returns:
            tuple: (动作分布, 状态价值)
        """
        # 提取特征
        features = self.feature_extractor(observations)
        
        # 获取动作分布
        action_dist = self.policy_network.get_action_distribution(features)
        
        # 获取状态价值
        values = self.value_network(features)
        
        return action_dist, values.squeeze(-1)  # 移除最后一个维度
    
    def act(self, observations, deterministic=False):
        """
        根据观察选择动作
        
        Args:
            observations (torch.Tensor): 观察
            deterministic (bool): 是否选择确定性动作
            
        Returns:
            tuple: (动作, 动作对数概率, 状态价值)
        """
        # 提取特征
        features = self.feature_extractor(observations)
        
        # 选择动作
        actions, log_probs = self.policy_network.act(features, deterministic)
        
        # 获取状态价值
        values = self.value_network(features).squeeze(-1)
        
        return actions, log_probs, values
    
    def evaluate(self, observations, actions):
        """
        评估给定观察和动作
        
        Args:
            observations (torch.Tensor): 观察
            actions (torch.Tensor): 动作
            
        Returns:
            tuple: (动作对数概率, 状态价值, 熵)
        """
        # 提取特征
        features = self.feature_extractor(observations)
        
        # 评估动作
        log_probs, entropy = self.policy_network.evaluate_actions(features, actions)
        
        # 获取状态价值
        values = self.value_network(features).squeeze(-1)
        
        return log_probs, values, entropy
    
    def get_value(self, observations):
        """
        仅获取状态价值（用于优势计算）
        
        Args:
            observations (torch.Tensor): 观察
            
        Returns:
            torch.Tensor: 状态价值
        """
        features = self.feature_extractor(observations)
        values = self.value_network(features).squeeze(-1)
        return values
    
    def save(self, filepath):
        """
        保存模型参数
        
        Args:
            filepath (str): 保存路径
        """
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'observation_shape': self.observation_shape,
            'action_dim': self.action_dim,
            'config': {
                'cnn_channels': Config.CNN_CHANNELS,
                'cnn_kernels': Config.CNN_KERNELS,
                'cnn_strides': Config.CNN_STRIDES,
                'hidden_size': Config.HIDDEN_SIZE,
                'frame_size': Config.FRAME_SIZE,
                'frame_stack': Config.FRAME_STACK,
            }
        }
        torch.save(checkpoint, filepath)
        print(f"Model saved to {filepath}")
    
    def load(self, filepath, device=None):
        """
        加载模型参数
        
        Args:
            filepath (str): 模型文件路径
            device (torch.device): 目标设备
        """
        if device is None:
            device = next(self.parameters()).device
        
        checkpoint = torch.load(filepath, map_location=device)
        
        # 检查模型配置是否匹配
        if 'observation_shape' in checkpoint:
            if checkpoint['observation_shape'] != self.observation_shape:
                print(f"Warning: observation shape mismatch. "
                      f"Expected {self.observation_shape}, got {checkpoint['observation_shape']}")
        
        if 'action_dim' in checkpoint:
            if checkpoint['action_dim'] != self.action_dim:
                print(f"Warning: action dim mismatch. "
                      f"Expected {self.action_dim}, got {checkpoint['action_dim']}")
        
        # 加载模型参数
        self.load_state_dict(checkpoint['model_state_dict'])
        print(f"Model loaded from {filepath}")
        
        return checkpoint.get('config', {})


def create_actor_critic_network(observation_shape, action_dim, device=None):
    """
    创建Actor-Critic网络的工厂函数
    
    Args:
        observation_shape (tuple): 观察空间形状
        action_dim (int): 动作空间大小
        device (torch.device): 目标设备
        
    Returns:
        ActorCriticNetwork: 创建的网络
    """
    if device is None:
        device = Config.DEVICE
    
    network = ActorCriticNetwork(observation_shape, action_dim)
    network = network.to(device)
    
    return network


def test_networks():
    """
    测试网络结构是否正确
    """
    print("测试网络结构...")
    
    # 模拟马里奥游戏的观察空间
    observation_shape = (Config.FRAME_STACK, Config.FRAME_SIZE, Config.FRAME_SIZE)  # (4, 84, 84)
    action_dim = 7  # 马里奥游戏的动作数量
    batch_size = 2
    
    # 创建网络
    network = create_actor_critic_network(observation_shape, action_dim)
    
    # 创建测试数据
    test_obs = torch.randn(batch_size, *observation_shape)
    test_actions = torch.randint(0, action_dim, (batch_size,))
    
    print(f"测试数据形状:")
    print(f"  观察: {test_obs.shape}")
    print(f"  动作: {test_actions.shape}")
    
    # 测试前向传播
    with torch.no_grad():
        # 测试act方法
        actions, log_probs, values = network.act(test_obs)
        print(f"\nact方法输出:")
        print(f"  动作: {actions.shape} = {actions}")
        print(f"  对数概率: {log_probs.shape} = {log_probs}")
        print(f"  价值: {values.shape} = {values}")
        
        # 测试evaluate方法
        eval_log_probs, eval_values, entropy = network.evaluate(test_obs, test_actions)
        print(f"\nevaluate方法输出:")
        print(f"  对数概率: {eval_log_probs.shape} = {eval_log_probs}")
        print(f"  价值: {eval_values.shape} = {eval_values}")
        print(f"  熵: {entropy.shape} = {entropy}")
        
        # 测试get_value方法
        values_only = network.get_value(test_obs)
        print(f"\nget_value方法输出:")
        print(f"  价值: {values_only.shape} = {values_only}")
    
    print("\n网络结构测试完成！")


if __name__ == "__main__":
    test_networks()