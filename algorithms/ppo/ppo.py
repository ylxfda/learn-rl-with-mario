"""
PPO (Proximal Policy Optimization) implementation.

PPO is an on-policy policy gradient algorithm that uses a clipped objective
to limit the magnitude of policy updates, improving training stability.

Key traits:
1) Clipped surrogate objective to prevent large updates
2) Actor-Critic training with value function
3) Advantages computed via GAE
4) Multiple epochs per batch to improve sample efficiency
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from typing import Dict, Any, Tuple, List

from .base import BaseRLAlgorithm, ModelManager
from networks.networks import create_actor_critic_network
from enviroments.replay_buffer import RolloutBuffer
from configs.ppo_config import Config


class PPOAlgorithm(BaseRLAlgorithm):
    """
    PPO algorithm implementation.
    
    Core idea:
    - Importance ratio r(θ) = π_θ(a|s) / π_θ_old(a|s)
    - Clip ratio within [1-ε, 1+ε]
    - Optimize clipped objective + value loss + entropy bonus
    """
    
    def __init__(self, 
                 observation_space, 
                 action_space,
                 device=None,
                 logger=None):
        """
        Initialize PPO.
        
        Args:
            observation_space: observation space
            action_space: action space
            device: compute device
            logger: logger
        """
        super().__init__(observation_space, action_space, device, logger)
        
        # Extract space info
        self.obs_shape = observation_space.shape
        self.action_dim = action_space.n
        
        # Create Actor-Critic network
        self.actor_critic = create_actor_critic_network(
            observation_shape=self.obs_shape,
            action_dim=self.action_dim,
            device=self.device
        )
        
        # Optimizer
        self.optimizer = optim.Adam(
            self.actor_critic.parameters(),
            lr=Config.LEARNING_RATE,
            eps=1e-5  # numerical stability
        )
        
        # LR scheduler (optional)
        self.lr_scheduler = optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=1000,  # decay every 1000 updates
            gamma=0.9
        )
        
        # PPO hyperparameters
        self.clip_epsilon = Config.CLIP_EPSILON
        self.ppo_epochs = Config.PPO_EPOCHS
        self.value_loss_coeff = Config.VALUE_LOSS_COEFF
        self.entropy_coeff = Config.ENTROPY_COEFF
        self.max_grad_norm = Config.MAX_GRAD_NORM
        
        # GAE params
        self.gamma = Config.GAMMA
        self.gae_lambda = Config.GAE_LAMBDA
        
        # References for base class helpers
        self.networks = {'actor_critic': self.actor_critic}
        self.optimizers = {'main': self.optimizer}
        
        # Training stats
        self.policy_losses = []
        self.value_losses = []
        self.entropies = []
        self.clip_fractions = []
        
        print("PPO algorithm initialized")
        print(f"Total network parameters: {self.actor_critic.count_parameters():,}")
        
    def act(self, observations, deterministic=False):
        """
        Select actions according to current policy.
        
        Args:
            observations (torch.Tensor): (batch_size, *obs_shape)
            deterministic (bool): choose argmax instead of sampling
            
        Returns:
            tuple: (actions, extra_info)
        """
        with torch.no_grad():
            # Ensure correct device
            observations = observations.to(self.device)
            
            # Actions, log-probs and values
            actions, log_probs, values = self.actor_critic.act(
                observations, 
                deterministic=deterministic
            )
            
            # Extras
            extra_info = {
                'log_probs': log_probs,
                'values': values
            }
            
            return actions, extra_info
    
    def compute_gae(self, rewards, values, dones, next_values):
        """
        Compute GAE advantages.
        
        Args:
            rewards (torch.Tensor): rewards (T, N)
            values (torch.Tensor): values (T, N)
            dones (torch.Tensor): done flags (T, N)
            next_values (torch.Tensor): last values (N,)
            
        Returns:
            tuple: (advantages, returns)
        """
        T, N = rewards.shape
        advantages = torch.zeros_like(rewards)
        
        # Backward recursion
        gae = 0
        for t in reversed(range(T)):
            if t == T - 1:
                # Next value at last step
                next_non_terminal = 1.0 - dones[t].float()
                next_value = next_values
            else:
                next_non_terminal = 1.0 - dones[t].float()  
                next_value = values[t + 1]
            
            # TD error δ = r + γV(s') - V(s)
            delta = rewards[t] + self.gamma * next_value * next_non_terminal - values[t]
            
            # GAE: A = δ + γλ * next_non_terminal * gae
            gae = delta + self.gamma * self.gae_lambda * next_non_terminal * gae
            advantages[t] = gae
        
        # Returns R = A + V
        returns = advantages + values
        
        return advantages, returns
    
    def update(self, rollout_buffer):
        """
        Update policy and value networks using rollout data.
        
        Args:
            rollout_buffer (RolloutBuffer): buffer with trajectories
            
        Returns:
            dict: update statistics
        """
        # Aggregated stats
        update_stats = {
            'policy_loss': 0.0,
            'value_loss': 0.0, 
            'entropy': 0.0,
            'total_loss': 0.0,
            'clip_fraction': 0.0,
            'kl_divergence': 0.0,
            'explained_variance': 0.0
        }
        
        # Multi-epoch PPO update
        for epoch in range(self.ppo_epochs):
            epoch_stats = {key: 0.0 for key in update_stats.keys()}
            batch_count = 0
            
            # Iterate mini-batches
            for batch in rollout_buffer.get_batch_iterator(Config.MINIBATCH_SIZE):
                batch_count += 1
                
                # Unpack batch
                states = batch['states']
                actions = batch['actions'] 
                old_log_probs = batch['old_log_probs']
                old_values = batch['old_values']
                advantages = batch['advantages']
                returns = batch['returns']
                
                # Re-evaluate action log-probabilities and values
                new_log_probs, new_values, entropy = self.actor_critic.evaluate(
                    states, actions
                )
                
                # Importance sampling ratio
                ratio = torch.exp(new_log_probs - old_log_probs)
                
                # PPO clipped objective
                surr1 = ratio * advantages
                surr2 = torch.clamp(
                    ratio, 
                    1.0 - self.clip_epsilon, 
                    1.0 + self.clip_epsilon
                ) * advantages
                
                # Policy loss = -min(surr1, surr2)
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss (optionally clipped)
                if Config.CLIP_EPSILON > 0:
                    # Clipped value loss
                    value_pred_clipped = old_values + torch.clamp(
                        new_values - old_values,
                        -self.clip_epsilon,
                        self.clip_epsilon
                    )
                    value_loss_1 = (new_values - returns).pow(2)
                    value_loss_2 = (value_pred_clipped - returns).pow(2)
                    value_loss = 0.5 * torch.max(value_loss_1, value_loss_2).mean()
                else:
                    # Standard MSE
                    value_loss = 0.5 * (new_values - returns).pow(2).mean()
                
                # Entropy bonus (exploration)
                entropy_loss = entropy.mean()
                
                # Total loss
                total_loss = (policy_loss + 
                             self.value_loss_coeff * value_loss - 
                             self.entropy_coeff * entropy_loss)
                
                # Backprop
                self.optimizer.zero_grad()
                total_loss.backward()
                # print(f"debug: policy_loss={policy_loss.item()}, value_loss={value_loss.item()}, entropy_loss={entropy_loss.item()}, total_loss={total_loss.item()}")
                
                # Gradient clipping
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.actor_critic.parameters(), 
                    self.max_grad_norm
                )
                
                # Optimizer step
                self.optimizer.step()
                
                # Collect stats
                with torch.no_grad():
                    # Clip fraction
                    clipped = torch.abs(ratio - 1.0) > self.clip_epsilon
                    clip_fraction = clipped.float().mean()
                    
                    # Approximate KL divergence
                    kl_div = ((ratio - 1.0) - (new_log_probs - old_log_probs)).mean()
                    
                    # Explained variance
                    y_true = returns
                    y_pred = new_values
                    var_y = y_true.var()
                    explained_var = 1 - (y_true - y_pred).var() / (var_y + 1e-8)
                
                # Accumulate
                epoch_stats['policy_loss'] += policy_loss.item()
                epoch_stats['value_loss'] += value_loss.item()
                epoch_stats['entropy'] += entropy_loss.item()
                epoch_stats['total_loss'] += total_loss.item()
                epoch_stats['clip_fraction'] += clip_fraction.item()
                epoch_stats['kl_divergence'] += kl_div.item()
                epoch_stats['explained_variance'] += explained_var.item()
            
            # Average per epoch
            if batch_count > 0:
                for key in epoch_stats:
                    epoch_stats[key] /= batch_count
                    update_stats[key] += epoch_stats[key]
        
        # Final averages
        for key in update_stats:
            update_stats[key] /= self.ppo_epochs
        
        # LR step
        self.lr_scheduler.step()
        update_stats['learning_rate'] = self.optimizer.param_groups[0]['lr']
        
        # Update counters
        self.total_updates += 1
        
        # Log
        if self.logger:
            self.logger.log_update(**update_stats)
        
        return update_stats
    
    def save_model(self, filepath):
        """
        Save PPO model.
        
        Args:
            filepath (str): Destination path
        """
        checkpoint = self.create_checkpoint()
        
        # Add PPO-specific metadata
        checkpoint.update({
            'ppo_config': {
                'clip_epsilon': self.clip_epsilon,
                'ppo_epochs': self.ppo_epochs,
                'value_loss_coeff': self.value_loss_coeff,
                'entropy_coeff': self.entropy_coeff,
                'gamma': self.gamma,
                'gae_lambda': self.gae_lambda,
            },
            'obs_shape': self.obs_shape,
            'action_dim': self.action_dim,
        })
        
        torch.save(checkpoint, filepath)
        print(f"PPO model saved to: {filepath}")
    
    def load_model(self, filepath):
        """
        Load PPO model.
        
        Args:
            filepath (str): Model file path
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        
        # Validate model compatibility
        if 'obs_shape' in checkpoint:
            if checkpoint['obs_shape'] != self.obs_shape:
                print(f"Warning: observation shape mismatch. Expected {self.obs_shape}, got {checkpoint['obs_shape']}")
        
        if 'action_dim' in checkpoint:
            if checkpoint['action_dim'] != self.action_dim:
                print(f"Warning: action space mismatch. Expected {self.action_dim}, got {checkpoint['action_dim']}")
        
        # Load checkpoint into base
        self.load_checkpoint(checkpoint)
        
        # Restore PPO-specific config
        if 'ppo_config' in checkpoint:
            ppo_config = checkpoint['ppo_config']
            self.clip_epsilon = ppo_config.get('clip_epsilon', self.clip_epsilon)
            self.ppo_epochs = ppo_config.get('ppo_epochs', self.ppo_epochs)
            self.value_loss_coeff = ppo_config.get('value_loss_coeff', self.value_loss_coeff)
            self.entropy_coeff = ppo_config.get('entropy_coeff', self.entropy_coeff)
            self.gamma = ppo_config.get('gamma', self.gamma)
            self.gae_lambda = ppo_config.get('gae_lambda', self.gae_lambda)
        
        print(f"PPO model loaded from {filepath}")
    
    def get_action_probabilities(self, observations):
        """
        Get action probabilities (for analysis).
        
        Args:
            observations (torch.Tensor): Observations
            
        Returns:
            torch.Tensor: Probabilities (batch_size, action_dim)
        """
        with torch.no_grad():
            observations = observations.to(self.device)
            action_dist, _ = self.actor_critic(observations)
            return action_dist.probs
    
    def compute_value(self, observations):
        """
        Compute state values (for analysis).
        
        Args:
            observations (torch.Tensor): Observations
            
        Returns:
            torch.Tensor: Values (batch_size,)
        """
        with torch.no_grad():
            observations = observations.to(self.device)
            values = self.actor_critic.get_value(observations)
            return values


def create_ppo_algorithm(observation_space, 
                        action_space, 
                        device=None, 
                        logger=None):
    """
    Factory function to create a PPO algorithm instance.
    
    Args:
        observation_space: Observation space
        action_space: Action space
        device: Compute device
        logger: Logger
        
    Returns:
        PPOAlgorithm: PPO instance
    """
    return PPOAlgorithm(
        observation_space=observation_space,
        action_space=action_space,
        device=device,
        logger=logger
    )


def test_ppo_algorithm():
    """
    Quick self-test for PPO wiring.
    """
    print("Testing PPO...")
    
    # Mock spaces
    import gym
    
    class MockObsSpace:
        def __init__(self):
            self.shape = (Config.FRAME_STACK, Config.FRAME_SIZE, Config.FRAME_SIZE)
    
    class MockActionSpace:
        def __init__(self):
            self.n = 7  # number of Mario actions
    
    obs_space = MockObsSpace()
    action_space = MockActionSpace()
    
    # Create PPO
    ppo = create_ppo_algorithm(obs_space, action_space)
    
    # Test data
    batch_size = 4
    test_obs = torch.randn(batch_size, *obs_space.shape)
    
    print(f"Test observation shape: {test_obs.shape}")
    
    # Test action selection
    actions, extra_info = ppo.act(test_obs)
    print(f"Chosen actions: {actions}")
    print(f"Extra info: {list(extra_info.keys())}")
    
    # Test value computation
    values = ppo.compute_value(test_obs)
    print(f"State values: {values}")
    
    # Test probability distribution
    probs = ppo.get_action_probabilities(test_obs)
    print(f"Action probs shape: {probs.shape}")
    
    print("PPO test completed!")


if __name__ == "__main__":
    test_ppo_algorithm()
