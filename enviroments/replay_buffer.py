"""
Rollout buffer for on-policy algorithms (e.g., PPO).
Stores trajectories collected from environments and provides
utilities to compute advantages/returns and iterate mini-batches.
"""

import torch
import numpy as np
from typing import List, Tuple, Dict
from configs.ppo_config import Config


class RolloutBuffer:
    """
    Trajectory buffer for PPO.
    
    Unlike replay buffers in off-policy methods (e.g., DQN), PPO is on-policy
    and only uses recent data. The buffer holds:
    - states
    - actions
    - rewards
    - value estimates
    - action log-probs
    - done flags
    - advantages (after computation)
    - returns (after computation)
    """
    
    def __init__(self, buffer_size, num_envs, obs_shape, action_dim, device):
        """
        Initialize rollout buffer.
        
        Args:
            buffer_size (int): steps per env to store
            num_envs (int): number of parallel envs
            obs_shape (tuple): observation shape
            action_dim (int): action dimension
            device (torch.device): device for storage
        """
        self.buffer_size = buffer_size
        self.num_envs = num_envs
        self.device = device
        
        # Total items = buffer_size × num_envs
        self.total_size = buffer_size * num_envs
        
        # Storage tensors
        # States: (buffer_size, num_envs, *obs_shape)
        self.states = torch.zeros(
            (buffer_size, num_envs, *obs_shape), 
            dtype=torch.float32, 
            device=device
        )
        
        # Actions (discrete)
        self.actions = torch.zeros(
            (buffer_size, num_envs), 
            dtype=torch.long, 
            device=device
        )
        
        # Rewards
        self.rewards = torch.zeros(
            (buffer_size, num_envs), 
            dtype=torch.float32, 
            device=device
        )
        
        # Value estimates
        self.values = torch.zeros(
            (buffer_size, num_envs), 
            dtype=torch.float32, 
            device=device
        )
        
        # Action log-probs
        self.log_probs = torch.zeros(
            (buffer_size, num_envs), 
            dtype=torch.float32, 
            device=device
        )
        
        # Done flags
        self.dones = torch.zeros(
            (buffer_size, num_envs), 
            dtype=torch.bool, 
            device=device
        )
        
        # Advantages and returns (computed after collection)
        self.advantages = torch.zeros(
            (buffer_size, num_envs), 
            dtype=torch.float32, 
            device=device
        )
        
        self.returns = torch.zeros(
            (buffer_size, num_envs), 
            dtype=torch.float32, 
            device=device
        )
        
        # Current write position
        self.pos = 0
        self.full = False  # whether buffer is full
    
    def add(self, states, actions, rewards, values, log_probs, dones):
        """
        Add one step of experience to the buffer.
        
        Args:
            states (torch.Tensor): (num_envs, *obs_shape)
            actions (torch.Tensor): (num_envs,)
            rewards (torch.Tensor): (num_envs,)
            values (torch.Tensor): (num_envs,)
            log_probs (torch.Tensor): (num_envs,)
            dones (torch.Tensor): (num_envs,)
        """
        # Ensure tensors are on the correct device
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        values = values.to(self.device)
        log_probs = log_probs.to(self.device)
        dones = dones.to(self.device)
        
        # Store data
        self.states[self.pos] = states
        self.actions[self.pos] = actions
        self.rewards[self.pos] = rewards
        self.values[self.pos] = values
        self.log_probs[self.pos] = log_probs
        self.dones[self.pos] = dones
        
        # Advance cursor
        self.pos += 1
        
        # Mark full when filled
        if self.pos >= self.buffer_size:
            self.full = True
    
    def compute_advantages_and_returns(self, next_values, gamma=0.99, gae_lambda=0.95):
        """
        Compute GAE advantages and returns.
        
        GAE balances bias and variance by exponentially weighting TD errors.
        
        Args:
            next_values (torch.Tensor): (num_envs,)
            gamma (float): discount factor
            gae_lambda (float): GAE lambda
        """
        next_values = next_values.to(self.device)
        
        # Init
        advantages = torch.zeros_like(self.rewards)
        gae = 0
        
        # Backward pass for GAE
        for t in reversed(range(self.buffer_size)):
            if t == self.buffer_size - 1:
                # Next value at last step
                next_non_terminal = 1.0 - self.dones[t].float()
                next_value = next_values
            else:
                next_non_terminal = 1.0 - self.dones[t].float()
                next_value = self.values[t + 1]
            
            # TD error
            delta = (self.rewards[t] + 
                    gamma * next_value * next_non_terminal - 
                    self.values[t])
            
            # Accumulate GAE
            gae = (delta + 
                  gamma * gae_lambda * next_non_terminal * gae)
            
            advantages[t] = gae
        
        # Returns = advantages + values
        returns = advantages + self.values
        
        # Store
        self.advantages = advantages
        self.returns = returns
        
        # Normalize advantages (stability)
        self.advantages = (self.advantages - self.advantages.mean()) / (
            self.advantages.std() + 1e-8)
    
    def get_batch_iterator(self, minibatch_size):
        """
        Yield mini-batches for PPO updates.
        
        Args:
            minibatch_size (int): minibatch size
            
        Yields:
            dict: mini-batch dict
        """
        if not self.full:
            print("Warning: Buffer is not full, using partial data")
        
        # Flatten to (total_size, ...)
        indices = torch.randperm(self.total_size, device=self.device)
        
        states = self.states.flatten(0, 1)  # (total_size, *obs_shape)
        actions = self.actions.flatten(0, 1)  # (total_size,)
        values = self.values.flatten(0, 1)
        log_probs = self.log_probs.flatten(0, 1)
        advantages = self.advantages.flatten(0, 1)
        returns = self.returns.flatten(0, 1)
        
        # Yield randomized mini-batches
        for start_idx in range(0, self.total_size, minibatch_size):
            end_idx = start_idx + minibatch_size
            batch_indices = indices[start_idx:end_idx]
            
            yield {
                'states': states[batch_indices],
                'actions': actions[batch_indices],
                'old_values': values[batch_indices],
                'old_log_probs': log_probs[batch_indices],
                'advantages': advantages[batch_indices],
                'returns': returns[batch_indices],
            }
    
    def reset(self):
        """
        Reset buffer for a new round of rollouts.
        """
        self.pos = 0
        self.full = False
        
        # Zero tensors (optional; useful for debugging)
        self.states.zero_()
        self.actions.zero_()
        self.rewards.zero_()
        self.values.zero_()
        self.log_probs.zero_()
        self.dones.zero_()
        self.advantages.zero_()
        self.returns.zero_()
    
    def size(self):
        """
        Return number of valid items currently stored.
        
        Returns:
            int: number of items
        """
        if self.full:
            return self.total_size
        else:
            return self.pos * self.num_envs
    
    def get_statistics(self):
        """
        Get basic statistics of stored data for debugging/monitoring.
        
        Returns:
            dict: stats
        """
        if self.size() == 0:
            return {}
        
        stats = {}
        
        # Reward stats
        valid_rewards = self.rewards[:self.pos] if not self.full else self.rewards
        stats['reward_mean'] = float(valid_rewards.mean())
        stats['reward_std'] = float(valid_rewards.std())
        stats['reward_min'] = float(valid_rewards.min())
        stats['reward_max'] = float(valid_rewards.max())
        
        # Value stats
        valid_values = self.values[:self.pos] if not self.full else self.values
        stats['value_mean'] = float(valid_values.mean())
        stats['value_std'] = float(valid_values.std())
        
        # Advantage stats
        if hasattr(self, 'advantages') and self.advantages.numel() > 0:
            valid_advantages = self.advantages[:self.pos] if not self.full else self.advantages
            stats['advantage_mean'] = float(valid_advantages.mean())
            stats['advantage_std'] = float(valid_advantages.std())
        
        # Action distribution
        valid_actions = self.actions[:self.pos] if not self.full else self.actions
        action_counts = torch.bincount(valid_actions.flatten())
        total_actions = valid_actions.numel()
        
        for i, count in enumerate(action_counts):
            stats[f'action_{i}_ratio'] = float(count) / total_actions
        
        return stats
    
    def __len__(self):
        """
        Buffer capacity (total potential items).
        
        Returns:
            int: capacity
        """
        return self.total_size
    
    def __repr__(self):
        """
        String representation with key info.
        
        Returns:
            str: buffer info
        """
        return (f"RolloutBuffer(size={self.buffer_size}, "
                f"num_envs={self.num_envs}, "
                f"current_size={self.size()}, "
                f"full={self.full})")
