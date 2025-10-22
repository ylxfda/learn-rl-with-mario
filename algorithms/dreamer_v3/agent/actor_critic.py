"""
Actor and Critic networks for DreamerV3.

Actor: π_θ(a_t | s_t) - Policy network
Critic: v_ψ(R | s_t) - Distributional value network with two-hot encoding

Both operate on state s_t = {h_t, z_t} from the world model.

Follows Section 3.3 and Appendix B.3 of DreamerV3 paper.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple

from algorithms.dreamer_v3.models.networks import MLP, init_weights
from algorithms.dreamer_v3.models.distributions import (
    DiscreteActionDist,
    TwoHotEncoding,
    symlog,
    symexp,
    sg,
)


class Actor(nn.Module):
    """
    Actor network: π_θ(a_t | s_t)
    
    Takes state s_t = {h_t, z_t} and outputs action distribution.
    For Mario: discrete action space with 7 actions.
    
    Uses Unimix for exploration: mix policy with uniform distribution.
    """
    
    def __init__(self, config: dict):
        super().__init__()
        
        # State dimensions
        self.hidden_size = config['model']['hidden_size']
        self.stoch_size = config['model']['stoch_size']
        self.discrete_size = config['model']['discrete_size']
        self.stoch_dim = self.stoch_size * self.discrete_size
        
        # Action space
        self.action_size = 7  # Mario discrete actions
        
        # MLP: state -> action logits
        self.mlp = MLP(
            input_dim=self.hidden_size + self.stoch_dim,
            hidden_dim=config['model']['mlp_hidden'],
            output_dim=self.action_size,
            num_layers=config['model']['mlp_layers'],
            activation=config['model']['activation'],
            layer_norm=config['model']['layer_norm']
        )
        
        # Exploration parameters
        self.unimix_ratio = config['training']['unimix_ratio']
        self.entropy_scale = config['training']['entropy_scale']
        
        # Store config
        self.config = config
        
        # Initialize weights
        self.apply(lambda m: init_weights(m, gain=1.0))
    
    def forward(
        self,
        h: torch.Tensor,
        z: torch.Tensor,
        deterministic: bool = False
    ) -> DiscreteActionDist:
        """
        Compute action distribution π_θ(a | s_t).
        
        Args:
            h: Deterministic state, shape: (B, hidden_size) or (B, T, hidden_size)
            z: Stochastic state (one-hot), shape: (B, stoch_size, discrete_size) or (B, T, ...)
            deterministic: If True, return mode instead of sampling
            
        Returns:
            DiscreteActionDist for sampling actions
        """
        # Flatten z
        z_flat = z.reshape(*z.shape[:-2], -1)  # (B, stoch_dim) or (B, T, stoch_dim)
        
        # Concatenate state
        state = torch.cat([h, z_flat], dim=-1)  # (B, hidden+stoch) or (B, T, hidden+stoch)
        
        # Get action logits
        logits = self.mlp(state)  # (B, action_size) or (B, T, action_size)
        
        # Create distribution (with Unimix if not deterministic)
        unimix = 0.0 if deterministic else self.unimix_ratio
        action_dist = DiscreteActionDist(logits, unimix_ratio=unimix)
        
        return action_dist
    
    def get_action(
        self,
        h: torch.Tensor,
        z: torch.Tensor,
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample action and return log probability.
        
        Args:
            h: Deterministic state, shape: (B, hidden_size)
            z: Stochastic state, shape: (B, stoch_size, discrete_size)
            deterministic: If True, return argmax action
            
        Returns:
            action: Action index, shape: (B,)
            log_prob: Log probability of action, shape: (B,)
        """
        dist = self.forward(h, z, deterministic)
        
        if deterministic:
            action = dist.mode()
        else:
            action = dist.sample()
        
        log_prob = dist.log_prob(action)
        
        return action, log_prob


class Critic(nn.Module):
    """
    Critic network: v_ψ(R | s_t)
    
    Distributional value learning using two-hot encoding (DreamerV3 Section B.3).
    
    Instead of predicting scalar value E[R], predicts distribution over R.
    Benefits:
        - More expressive (can represent multi-modal returns)
        - Better gradients (smoother than MSE on scalar)
        - Handles stochasticity naturally
    
    Process:
        1. MLP: state -> logits for value bins
        2. Softmax -> probability distribution over bins
        3. Two-hot decoding -> expected value
    """
    
    def __init__(self, config: dict):
        super().__init__()
        
        # State dimensions
        self.hidden_size = config['model']['hidden_size']
        self.stoch_size = config['model']['stoch_size']
        self.discrete_size = config['model']['discrete_size']
        self.stoch_dim = self.stoch_size * self.discrete_size
        
        # Two-hot encoding parameters
        self.num_bins = config['training']['num_bins']
        self.value_min = config['training']['value_min']
        self.value_max = config['training']['value_max']
        
        # Two-hot encoder/decoder
        self.twohot = TwoHotEncoding(
            num_bins=self.num_bins,
            value_min=self.value_min,
            value_max=self.value_max
        )
        
        # MLP: state -> logits for value distribution
        self.mlp = MLP(
            input_dim=self.hidden_size + self.stoch_dim,
            hidden_dim=config['model']['mlp_hidden'],
            output_dim=self.num_bins,  # Logits for bins
            num_layers=config['model']['mlp_layers'],
            activation=config['model']['activation'],
            layer_norm=config['model']['layer_norm']
        )
        
        # Store config
        self.config = config
        
        # Initialize weights
        self.apply(lambda m: init_weights(m, gain=1.0))
    
    def forward(self, h: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """
        Predict value distribution v_ψ(R | s_t).
        
        Args:
            h: Deterministic state, shape: (B, hidden_size) or (B, T, hidden_size)
            z: Stochastic state, shape: (B, stoch_size, discrete_size) or (B, T, ...)
            
        Returns:
            Value distribution (probs), shape: (B, num_bins) or (B, T, num_bins)
        """
        # Flatten z
        z_flat = z.reshape(*z.shape[:-2], -1)
        
        # Concatenate state
        state = torch.cat([h, z_flat], dim=-1)
        
        # Get logits
        logits = self.mlp(state)  # (B, num_bins) or (B, T, num_bins)
        
        # Convert to probabilities
        probs = F.softmax(logits, dim=-1)
        
        return probs
    
    def get_value(self, h: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """
        Get expected value E[v_ψ(R | s_t)].
        
        Args:
            h: Deterministic state, shape: (B, hidden_size) or (B, T, hidden_size)
            z: Stochastic state, shape: (B, stoch_size, discrete_size)
            
        Returns:
            Expected value, shape: (B,) or (B, T)
        """
        # Get distribution
        value_dist = self.forward(h, z)  # (B, num_bins) or (B, T, num_bins)
        
        # Decode to scalar value
        value = self.twohot.decode(value_dist)
        
        return value
    
    def compute_loss(
        self,
        h: torch.Tensor,
        z: torch.Tensor,
        target_values: torch.Tensor,
        slow_target_dist: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Compute distributional value loss using two-hot targets with slow target regularization.

        Loss has two components:
            1. Main loss: fit λ-returns
            2. Regularization loss: stay close to slow target critic's predictions

        The regularization stabilizes learning by preventing the critic from changing too quickly.

        Args:
            h: Deterministic state, shape: (B, hidden_size) or (B, T, hidden_size)
            z: Stochastic state, shape: (B, stoch_size, discrete_size)
            target_values: Target values (λ-returns), shape: (B,) or (B, T)
            slow_target_dist: Optional slow target distribution, shape: (B, num_bins) or (B, T, num_bins)
                             If provided, adds regularization term to keep predictions close to slow target

        Returns:
            Scalar loss
        """
        # Get predicted distribution
        pred_dist = self.forward(h, z)  # (B, num_bins) or (B, T, num_bins)

        # ====================================================================
        # Main Loss: Fit λ-returns
        # ====================================================================
        # Encode target values to two-hot
        target_dist = self.twohot.encode(target_values)  # (B, num_bins) or (B, T, num_bins)

        # Cross-entropy loss between prediction and two-hot target (treat as classification)
        # L_main = -∑ target_dist * log(pred_dist)
        loss_main = -torch.sum(target_dist * torch.log(pred_dist + 1e-8), dim=-1)

        # ====================================================================
        # Regularization Loss: Stay close to slow target (if provided)
        # ====================================================================
        # This implements the key idea from DreamerV3 paper:
        # "we stabilize learning by regularizing the critic towards predicting
        #  the outputs of an exponentially moving average of its own parameters"
        if slow_target_dist is not None:
            # L_reg = -∑ slow_target_dist * log(pred_dist)
            # This encourages pred_dist to be close to slow_target_dist
            loss_reg = -torch.sum(slow_target_dist * torch.log(pred_dist + 1e-8), dim=-1)

            # Total loss: main loss + regularization loss
            # Note: In the reference implementation, this is done by subtracting log_prob,
            # which is equivalent to adding cross-entropy
            total_loss = loss_main + loss_reg
        else:
            total_loss = loss_main

        return total_loss.mean()


class EMATargetCritic:
    """
    Exponential Moving Average (EMA) target critic.
    
    Maintains a slowly-updated copy of the critic for stable bootstrapping.
    Used in computing λ-returns to prevent moving target problem.
    
    Update: θ_target = (1 - τ) * θ_target + τ * θ
    """
    
    def __init__(self, critic: Critic, tau: float = 0.02):
        """
        Args:
            critic: Main critic network to track
            tau: EMA coefficient (0.02 means 2% update per step)
        """
        self.tau = tau
        
        # Create a copy of critic (no gradients)
        self.target_critic = Critic(critic.config).to(next(critic.parameters()).device)
        self.target_critic.load_state_dict(critic.state_dict())
        
        # Freeze target critic
        for param in self.target_critic.parameters():
            param.requires_grad = False
    
    def update(self, critic: Critic):
        """
        Update target critic using EMA.
        
        Args:
            critic: Current critic network
        """
        with torch.no_grad():
            for target_param, param in zip(
                self.target_critic.parameters(),
                critic.parameters()
            ):
                target_param.data.copy_(
                    (1 - self.tau) * target_param.data + self.tau * param.data
                )
    
    def get_value(self, h: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """
        Get value from target critic.
        
        Args:
            h: Deterministic state
            z: Stochastic state
            
        Returns:
            Target value (no gradients)
        """
        with torch.no_grad():
            return self.target_critic.get_value(h, z)


# ============================================================================
# Lambda Returns Computation
# ============================================================================

def compute_lambda_returns(
    rewards: torch.Tensor,
    continues: torch.Tensor,
    values: torch.Tensor,
    bootstrap: torch.Tensor,
    gamma: float = 0.997,
    lambda_: float = 0.95
) -> torch.Tensor:
    """
    Compute λ-returns for actor-critic training (Section 3.3).
    
    λ-return is a mixture of n-step returns:
        G_t^λ = (1-λ) * Σ_{n=1}^∞ λ^{n-1} * G_t^{(n)}
    
    Recursive formulation:
        G_t^λ = r_t + γ * c_t * ((1-λ) * V(s_{t+1}) + λ * G_{t+1}^λ)
    
    Where:
        - r_t: reward at time t
        - c_t: continuation flag (0 if terminal, 1 otherwise)
        - γ: discount factor
        - λ: trace decay parameter
        - V(s_{t+1}): value estimate (for TD backup)
    
    Args:
        rewards: Predicted rewards (symlog space), shape: (B, T)
        continues: Predicted continuation probs, shape: (B, T)
        values: Value estimates V(s_t), shape: (B, T)
        bootstrap: Bootstrap value V(s_T), shape: (B,)
        gamma: Discount factor
        lambda_: λ parameter for TD(λ)
        
    Returns:
        λ-returns, shape: (B, T)
    """
    # Convert rewards from symlog to normal space
    rewards = symexp(rewards)
    
    B, T = rewards.shape
    device = rewards.device
    
    # Initialize returns with bootstrap value
    # We'll compute returns backward from T to 0
    returns = []
    G = bootstrap  # G_T = V(s_T) (bootstrap from beyond horizon)
    
    # Work backwards through time
    for t in reversed(range(T)):
        # Value at time t (for TD backup)
        V_t = values[:, t]
        
        # λ-return at time t:
        # G_t = r_t + γ * c_t * ((1-λ) * V_t + λ * G_{t+1})
        G = rewards[:, t] + gamma * continues[:, t] * (
            (1 - lambda_) * V_t + lambda_ * G
        )
        
        returns.insert(0, G)
    
    # Stack returns
    returns = torch.stack(returns, dim=1)  # (B, T)
    
    # Convert back to symlog space for stable learning
    returns = symlog(returns)
    
    return returns


def compute_advantages(
    returns: torch.Tensor,
    values: torch.Tensor
) -> torch.Tensor:
    """
    Compute advantages for actor training.
    
    A_t = G_t^λ - V(s_t)
    
    Advantages are normalized for stability.
    
    Args:
        returns: λ-returns (symlog), shape: (B, T)
        values: Value estimates (symlog), shape: (B, T)
        
    Returns:
        Normalized advantages, shape: (B, T)
    """
    # Compute raw advantages
    advantages = returns - values
    
    # Normalize (mean=0, std=1) for stability
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    
    return advantages


# ============================================================================
# Actor Loss Computation
# ============================================================================

def compute_actor_loss(
    log_probs: torch.Tensor,
    advantages: torch.Tensor,
    entropy: torch.Tensor,
    entropy_scale: float = 1e-3
) -> torch.Tensor:
    """
    Compute actor loss (policy gradient with entropy regularization).
    
    L_actor = -E[log π_θ(a|s) * A(s,a)] - β_entropy * H[π_θ]
    
    Where:
        - log π_θ(a|s): log probability of taken action
        - A(s,a): advantage (how much better than average)
        - H[π_θ]: entropy of policy (encourages exploration)
    
    Args:
        log_probs: Log probabilities of actions, shape: (B, T)
        advantages: Advantages (stop-gradient), shape: (B, T)
        entropy: Policy entropy, shape: (B, T)
        entropy_scale: Weight for entropy bonus
        
    Returns:
        Scalar loss
    """
    # Policy gradient loss
    pg_loss = -(log_probs * sg(advantages)).mean()
    
    # Entropy regularization (encourage exploration)
    entropy_loss = -entropy_scale * entropy.mean()
    
    # Total actor loss
    actor_loss = pg_loss + entropy_loss
    
    return actor_loss


# ============================================================================
# Helper Functions
# ============================================================================

def init_actor_critic(config: dict, device: torch.device):
    """
    Initialize actor, critic, and target critic.
    
    Args:
        config: Configuration dictionary
        device: Device to place models on
        
    Returns:
        actor, critic, target_critic
    """
    # Create actor
    actor = Actor(config).to(device)
    
    # Create critic
    critic = Critic(config).to(device)
    
    # Create target critic (EMA copy)
    target_critic = EMATargetCritic(
        critic,
        tau=config['training']['tau']
    )
    
    return actor, critic, target_critic
