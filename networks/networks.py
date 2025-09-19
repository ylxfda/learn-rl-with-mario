"""
Neural network module.
Defines the Actor-Critic network used by PPO, including
CNN feature extractor, policy (actor) and value (critic) heads.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Dict

from config import Config


class CNNFeatureExtractor(nn.Module):
    """
    CNN feature extractor.
    
    Extracts features from raw game frames.
    The design follows common CNN backbones used by DQN/A3C.
    """
    
    def __init__(self, input_channels=4, output_dim=512):
        """
        Initialize the CNN feature extractor.
        
        Args:
            input_channels (int): number of channels (frame stack)
            output_dim (int): feature dimension
        """
        super(CNNFeatureExtractor, self).__init__()
        
        # Conv1: capture large spatial patterns
        # Input: (batch, 4, 84, 84)
        # Output: (batch, 32, 20, 20)
        self.conv1 = nn.Conv2d(
            in_channels=input_channels,
            out_channels=Config.CNN_CHANNELS[0],  # 32
            kernel_size=Config.CNN_KERNELS[0],    # 8
            stride=Config.CNN_STRIDES[0],         # 4
            padding=0
        )
        
        # Conv2: mid-scale features
        # Input: (batch, 32, 20, 20)
        # Output: (batch, 64, 9, 9)
        self.conv2 = nn.Conv2d(
            in_channels=Config.CNN_CHANNELS[0],   # 32
            out_channels=Config.CNN_CHANNELS[1],  # 64
            kernel_size=Config.CNN_KERNELS[1],    # 4
            stride=Config.CNN_STRIDES[1],         # 2
            padding=0
        )
        
        # Conv3: finer details
        # Input: (batch, 64, 9, 9)
        # Output: (batch, 64, 7, 7)
        self.conv3 = nn.Conv2d(
            in_channels=Config.CNN_CHANNELS[1],   # 64
            out_channels=Config.CNN_CHANNELS[2],  # 64
            kernel_size=Config.CNN_KERNELS[2],    # 3
            stride=Config.CNN_STRIDES[2],         # 1
            padding=0
        )
        
        # Compute flattened feature size
        # For 84x84 input, this yields 64*7*7=3136 dims
        self.feature_dim = self._calculate_conv_output_size(input_channels)
        
        # FC layer: map conv features to fixed dim
        self.fc = nn.Linear(self.feature_dim, output_dim)
        
        # Init weights
        self._initialize_weights()
    
    def _calculate_conv_output_size(self, input_channels):
        """
        Compute flattened feature dimension after convs.
        
        Args:
            input_channels (int): number of channels
            
        Returns:
            int: flattened feature dimension
        """
        # Use a dummy input to infer shape
        test_input = torch.zeros(1, input_channels, Config.FRAME_SIZE, Config.FRAME_SIZE)
        
        with torch.no_grad():
            x = F.relu(self.conv1(test_input))
            x = F.relu(self.conv2(x))
            x = F.relu(self.conv3(x))
            feature_dim = x.numel()  # total number of elements
        
        return feature_dim
    
    def _initialize_weights(self):
        """
        Initialize network weights using orthogonal init,
        which is commonly effective in RL.
        """
        for module in self.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                # orthogonal init
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                # bias = 0
                nn.init.constant_(module.bias, 0)
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x (torch.Tensor): images (batch, channels, height, width)
            
        Returns:
            torch.Tensor: features (batch, output_dim)
        """
        # Ensure input in [0, 1]
        x = x.float() / 255.0 if x.max() > 1.0 else x.float()
        
        # Convs + ReLU
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # FC + ReLU
        x = F.relu(self.fc(x))
        
        return x


class PolicyNetwork(nn.Module):
    """
    Policy (Actor) network.
    
    Outputs action probabilities and supports:
    1) sampling actions
    2) computing action log-probs
    3) computing policy entropy (for exploration)
    """
    
    def __init__(self, feature_dim, action_dim, hidden_dim=None):
        """
        Initialize the policy network.
        
        Args:
            feature_dim (int): input feature dim
            action_dim (int): action space size
            hidden_dim (int): hidden size
        """
        super(PolicyNetwork, self).__init__()
        
        if hidden_dim is None:
            hidden_dim = Config.HIDDEN_SIZE
        
        self.action_dim = action_dim
        
        # MLP head: features -> hidden -> logits
        self.policy_head = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        
        # Init
        self._initialize_weights()
    
    def _initialize_weights(self):
        """
        Initialize weights
        """
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Smaller init gain helps early exploration
                nn.init.orthogonal_(module.weight, gain=0.01)
                nn.init.constant_(module.bias, 0)
    
    def forward(self, features):
        """
        Forward pass, returning action logits.
        
        Args:
            features (torch.Tensor): features (batch, feature_dim)
            
        Returns:
            torch.Tensor: logits (batch, action_dim)
        """
        return self.policy_head(features)
    
    def get_action_distribution(self, features):
        """
        Get action distribution.
        
        Args:
            features (torch.Tensor): features
            
        Returns:
            torch.distributions.Categorical: categorical distribution
        """
        logits = self.forward(features)
        return torch.distributions.Categorical(logits=logits)
    
    def act(self, features, deterministic=False):
        """
        Select action given features.
        
        Args:
            features (torch.Tensor): features
            deterministic (bool): use argmax instead of sampling
            
        Returns:
            tuple: (action, log_prob)
        """
        dist = self.get_action_distribution(features)
        
        if deterministic:
            # choose argmax
            action = dist.probs.argmax(dim=-1)
        else:
            # sample from distribution
            action = dist.sample()
        
        log_prob = dist.log_prob(action)
        
        return action, log_prob
    
    def evaluate_actions(self, features, actions):
        """
        Evaluate given actions: log-prob and entropy.
        
        Args:
            features (torch.Tensor): features
            actions (torch.Tensor): actions
            
        Returns:
            tuple: (log_probs, entropy)
        """
        dist = self.get_action_distribution(features)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        
        return log_probs, entropy


class ValueNetwork(nn.Module):
    """
    Value (Critic) network.
    
    Estimates V(s) for computing advantages and losses.
    """
    
    def __init__(self, feature_dim, hidden_dim=None):
        """
        Initialize the value network.
        
        Args:
            feature_dim (int): input feature dim
            hidden_dim (int): hidden size
        """
        super(ValueNetwork, self).__init__()
        
        if hidden_dim is None:
            hidden_dim = Config.HIDDEN_SIZE
        
        # MLP head: features -> hidden -> scalar value
        self.value_head = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)  # scalar value
        )
        
        # Init
        self._initialize_weights()
    
    def _initialize_weights(self):
        """
        Initialize network weights.
        """
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=1.0)
                nn.init.constant_(module.bias, 0)
    
    def forward(self, features):
        """
        Forward pass for state values.
        
        Args:
            features (torch.Tensor): (batch, feature_dim)
            
        Returns:
            torch.Tensor: (batch, 1)
        """
        return self.value_head(features)


class ActorCriticNetwork(nn.Module):
    """
    Actor-Critic network composed of feature extractor,
    policy head, and value head (PPO core).
    """
    
    def __init__(self, observation_shape, action_dim):
        """
        Initialize Actor-Critic network.
        
        Args:
            observation_shape (tuple): (channels, height, width)
            action_dim (int): action space size
        """
        super(ActorCriticNetwork, self).__init__()
        
        self.observation_shape = observation_shape
        self.action_dim = action_dim
        
        # Feature extractor
        input_channels = observation_shape[0]  # frame stack
        feature_dim = Config.HIDDEN_SIZE
        
        self.feature_extractor = CNNFeatureExtractor(
            input_channels=input_channels,
            output_dim=feature_dim
        )
        
        # Policy head
        self.policy_network = PolicyNetwork(
            feature_dim=feature_dim,
            action_dim=action_dim
        )
        
        # Value head
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
        Count trainable parameters.
        
        Returns:
            int: number of parameters
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def forward(self, observations):
        """
        Forward pass returning action distribution and values.
        
        Args:
            observations (torch.Tensor): (batch, *observation_shape)
            
        Returns:
            tuple: (action_dist, values)
        """
        # Extract features
        features = self.feature_extractor(observations)
        
        # Action distribution
        action_dist = self.policy_network.get_action_distribution(features)
        
        # State values
        values = self.value_network(features)
        
        return action_dist, values.squeeze(-1)  # drop last dim
    
    def act(self, observations, deterministic=False):
        """
        Select action given observations.
        
        Args:
            observations (torch.Tensor): observations
            deterministic (bool): deterministic or stochastic
            
        Returns:
            tuple: (actions, log_probs, values)
        """
        # Extract features
        features = self.feature_extractor(observations)
        
        # Select action
        actions, log_probs = self.policy_network.act(features, deterministic)
        
        # Get value
        values = self.value_network(features).squeeze(-1)
        
        return actions, log_probs, values
    
    def evaluate(self, observations, actions):
        """
        Evaluate given observations and actions.
        
        Args:
            observations (torch.Tensor): observations
            actions (torch.Tensor): actions
            
        Returns:
            tuple: (log_probs, values, entropy)
        """
        # Extract features
        features = self.feature_extractor(observations)
        
        # Evaluate actions
        log_probs, entropy = self.policy_network.evaluate_actions(features, actions)
        
        # Get values
        values = self.value_network(features).squeeze(-1)
        
        return log_probs, values, entropy
    
    def get_value(self, observations):
        """
        Get values only (for advantage computation).
        
        Args:
            observations (torch.Tensor): observations
            
        Returns:
            torch.Tensor: values
        """
        features = self.feature_extractor(observations)
        values = self.value_network(features).squeeze(-1)
        return values
    
    def save(self, filepath):
        """
        Save parameters.
        
        Args:
            filepath (str): destination path
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
        Load parameters.
        
        Args:
            filepath (str): model path
            device (torch.device): target device
        """
        if device is None:
            device = next(self.parameters()).device
        
        checkpoint = torch.load(filepath, map_location=device)
        
        # Check compatibility
        if 'observation_shape' in checkpoint:
            if checkpoint['observation_shape'] != self.observation_shape:
                print(f"Warning: observation shape mismatch. "
                      f"Expected {self.observation_shape}, got {checkpoint['observation_shape']}")
        
        if 'action_dim' in checkpoint:
            if checkpoint['action_dim'] != self.action_dim:
                print(f"Warning: action dim mismatch. "
                      f"Expected {self.action_dim}, got {checkpoint['action_dim']}")
        
        # Load
        self.load_state_dict(checkpoint['model_state_dict'])
        print(f"Model loaded from {filepath}")
        
        return checkpoint.get('config', {})


def create_actor_critic_network(observation_shape, action_dim, device=None):
    """
    Factory function to build an Actor-Critic network.
    
    Args:
        observation_shape (tuple): observation shape
        action_dim (int): number of actions
        device (torch.device): target device
        
    Returns:
        ActorCriticNetwork: network instance
    """
    if device is None:
        device = Config.DEVICE
    
    network = ActorCriticNetwork(observation_shape, action_dim)
    network = network.to(device)
    
    return network


def test_networks():
    """
    Quick sanity check for shapes.
    """
    print("Testing network shapes...")
    
    # Mock Mario observation space
    observation_shape = (Config.FRAME_STACK, Config.FRAME_SIZE, Config.FRAME_SIZE)  # (4, 84, 84)
    action_dim = 7  # number of actions
    batch_size = 2
    
    # Create network
    network = create_actor_critic_network(observation_shape, action_dim)
    
    # Create test data
    test_obs = torch.randn(batch_size, *observation_shape)
    test_actions = torch.randint(0, action_dim, (batch_size,))
    
    print(f"Test data shapes:")
    print(f"  observations: {test_obs.shape}")
    print(f"  actions: {test_actions.shape}")
    
    # Test forward path
    with torch.no_grad():
        # Test act()
        actions, log_probs, values = network.act(test_obs)
        print(f"\nact() output:")
        print(f"  actions: {actions.shape} = {actions}")
        print(f"  log_probs: {log_probs.shape} = {log_probs}")
        print(f"  values: {values.shape} = {values}")
        
        # Test evaluate()
        eval_log_probs, eval_values, entropy = network.evaluate(test_obs, test_actions)
        print(f"\nevaluate() output:")
        print(f"  log_probs: {eval_log_probs.shape} = {eval_log_probs}")
        print(f"  values: {eval_values.shape} = {eval_values}")
        print(f"  entropy: {entropy.shape} = {entropy}")
        
        # Test get_value()
        values_only = network.get_value(test_obs)
        print(f"\nget_value() output:")
        print(f"  values: {values_only.shape} = {values_only}")
    
    print("\nNetwork shape test completed!")


if __name__ == "__main__":
    test_networks()
