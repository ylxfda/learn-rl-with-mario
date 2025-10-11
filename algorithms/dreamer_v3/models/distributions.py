"""
Probability distributions for DreamerV3.

This module implements the key distributions used in DreamerV3:
1. SymLog/SymExp for stable reward/value encoding
2. TwoHot encoding for distributional value learning
3. Categorical distribution for stochastic state z_t

Mathematical notation follows the DreamerV3 paper.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional


# ============================================================================
# SymLog Transform (Section B.4)
# ============================================================================

def symlog(x: torch.Tensor) -> torch.Tensor:
    """
    Symmetric logarithm transform for stable value/reward encoding.
    
    SymLog(x) = sign(x) * log(|x| + 1)
    
    This transform:
    - Preserves sign of input
    - Reduces large values' scale
    - Is invertible via symexp
    
    Args:
        x: Input tensor (any shape)
        
    Returns:
        Transformed tensor (same shape as x)
        
    Example:
        >>> symlog(torch.tensor([0.0, 1.0, -10.0, 100.0]))
        tensor([0.0, 0.693, -2.398, 4.615])
    """
    return torch.sign(x) * torch.log(torch.abs(x) + 1.0)


def symexp(x: torch.Tensor) -> torch.Tensor:
    """
    Inverse of symlog: SymExp(x) = sign(x) * (exp(|x|) - 1)
    
    Args:
        x: Symlog-encoded tensor
        
    Returns:
        Original scale tensor
    """
    return torch.sign(x) * (torch.exp(torch.abs(x)) - 1.0)


# ============================================================================
# Two-Hot Encoding (Section B.3)
# ============================================================================

class TwoHotEncoding:
    """
    Two-hot encoding for distributional value learning (DreamerV3 Section B.3).
    
    Instead of single bin (one-hot), spread probability over two adjacent bins.
    This provides smoother gradients and better generalization.
    
    Process:
        1. Create uniform bins in symlog space: [value_min, value_max]
        2. For value v, find bins k and k+1 where v falls between
        3. Split probability between bins proportional to distance
    
    Example:
        If bins are [0, 1, 2, 3] and value is 1.3:
        - Falls between bin 1 (value=1) and bin 2 (value=2)
        - Weight on bin 1: 0.7, weight on bin 2: 0.3
    """
    
    def __init__(self, num_bins: int = 255, value_min: float = -20, value_max: float = 20):
        """
        Args:
            num_bins: Number of discretization bins (DreamerV3 uses 255)
            value_min: Minimum value in symlog space
            value_max: Maximum value in symlog space
        """
        self.num_bins = num_bins
        self.value_min = value_min
        self.value_max = value_max
        
        # Create uniform bins in symlog space
        # bins shape: (num_bins,)
        self.bins = torch.linspace(value_min, value_max, num_bins)
        
    def encode(self, value: torch.Tensor) -> torch.Tensor:
        """
        Encode scalar values into two-hot distribution.
        
        Args:
            value: Scalar values, shape (..., )
            
        Returns:
            Two-hot encoded distribution, shape (..., num_bins)
        """
        # Move bins to same device as value
        bins = self.bins.to(value.device)
        
        # Transform value to symlog space
        # value_symlog shape: (..., )
        value_symlog = symlog(value)
        
        # Clip to bin range
        value_symlog = torch.clamp(value_symlog, self.value_min, self.value_max)
        
        # Find which bin each value falls into
        # We want to find bin k such that bins[k] <= value < bins[k+1]
        # Shape: (..., num_bins) - boolean mask where True means bin value < input value
        above = value_symlog[..., None] > bins[None, :]

        # Count how many bins have values strictly less than our value
        # This gives us the index k where bins[k] <= value < bins[k+1]
        # index shape: (..., )
        index = torch.sum(above, dim=-1) - 1
        index = torch.clamp(index, 0, self.num_bins - 2)  # Ensure we have k and k+1
        
        # Get values at bin k and k+1
        # bin_low, bin_high shape: (..., )
        bin_low = bins[index]
        bin_high = bins[index + 1]
        
        # Compute interpolation weight
        # If value is at bin_low, weight_high = 0
        # If value is at bin_high, weight_high = 1
        # weight_high shape: (..., )
        weight_high = (value_symlog - bin_low) / (bin_high - bin_low + 1e-8)
        weight_high = torch.clamp(weight_high, 0.0, 1.0)
        weight_low = 1.0 - weight_high
        
        # Create two-hot distribution
        # target shape: (..., num_bins)
        target = torch.zeros(*value.shape, self.num_bins, device=value.device)
        
        # Scatter weights to appropriate bins
        # We need to handle batched indexing carefully
        batch_indices = torch.arange(value.numel(), device=value.device)
        flat_index = index.flatten()
        flat_weight_low = weight_low.flatten()
        flat_weight_high = weight_high.flatten()
        
        target_flat = target.reshape(-1, self.num_bins)
        target_flat[batch_indices, flat_index] = flat_weight_low
        target_flat[batch_indices, flat_index + 1] = flat_weight_high
        
        return target.reshape(*value.shape, self.num_bins)
    
    def decode(self, distribution: torch.Tensor) -> torch.Tensor:
        """
        Decode two-hot distribution back to scalar values.
        
        Args:
            distribution: Probability distribution, shape (..., num_bins)
            
        Returns:
            Expected values in original space, shape (..., )
        """
        # Move bins to same device
        bins = self.bins.to(distribution.device)
        
        # Compute expected value in symlog space
        # value_symlog = Σ p_i * bin_i
        # Shape: (..., )
        value_symlog = torch.sum(distribution * bins, dim=-1)
        
        # Transform back to original space
        return symexp(value_symlog)


# ============================================================================
# Categorical Distribution for Stochastic State z_t
# ============================================================================

class CategoricalDist:
    """
    Categorical distribution for stochastic state z_t in RSSM.
    
    In DreamerV3, z_t is a vector of 'stoch_size' independent categorical variables,
    each with 'discrete_size' classes. This gives a factorized representation.
    
    Example:
        stoch_size=32, discrete_size=32
        => z_t has 32 categorical variables, each with 32 classes
        => Total representation: 32 * 32 = 1024 dimensions (one-hot)
        => But only 32 * log2(32) = 160 bits of information
    """
    
    def __init__(self, logits: torch.Tensor, unimix_ratio: float = 0.01):
        """
        Args:
            logits: Unnormalized log probabilities
                    Shape: (..., stoch_size, discrete_size)
            unimix_ratio: Mix with uniform for exploration (Unimix trick)
        """
        # Apply Unimix: mix with uniform distribution for exploration
        # p_unimix = (1 - ε) * p_model + ε * uniform
        # In logit space: logit_unimix = log((1-ε)*exp(logit) + ε/K)
        if unimix_ratio > 0:
            uniform_logits = torch.zeros_like(logits)
            probs = torch.softmax(logits, dim=-1)
            self.probs = (1 - unimix_ratio) * probs + unimix_ratio / logits.shape[-1]
            self.logits = torch.log(self.probs + 1e-8)
        else:
            self.logits = logits
            self.probs = torch.softmax(logits, dim=-1)
        
    def sample(self, sample_shape: torch.Size = torch.Size()) -> torch.Tensor:
        """
        Sample from categorical distribution using straight-through gradients.
        
        Forward pass: one-hot sample (discrete)
        Backward pass: use soft probabilities (continuous)
        
        Args:
            sample_shape: Additional dimensions for sampling
            
        Returns:
            One-hot samples, shape: sample_shape + (..., stoch_size, discrete_size)
        """
        # Sample indices from categorical
        # Shape: sample_shape + (..., stoch_size)
        indices = torch.multinomial(
            self.probs.reshape(-1, self.probs.shape[-1]),
            num_samples=1
        ).reshape(self.probs.shape[:-1])
        
        # Convert to one-hot
        # Shape: sample_shape + (..., stoch_size, discrete_size)
        one_hot = F.one_hot(indices, num_classes=self.probs.shape[-1]).float()
        
        # Straight-through estimator: forward=one_hot, backward=probs
        one_hot_st = one_hot + self.probs - self.probs.detach()
        
        return one_hot_st
    
    def mode(self) -> torch.Tensor:
        """
        Return most likely category (for deterministic evaluation).
        
        Returns:
            One-hot encoding of argmax, shape: (..., stoch_size, discrete_size)
        """
        indices = torch.argmax(self.probs, dim=-1)
        one_hot = F.one_hot(indices, num_classes=self.probs.shape[-1]).float()
        return one_hot
    
    def log_prob(self, sample: torch.Tensor) -> torch.Tensor:
        """
        Compute log probability of sample.
        
        Args:
            sample: One-hot encoded sample, shape: (..., stoch_size, discrete_size)
            
        Returns:
            Log probabilities, shape: (..., stoch_size)
        """
        # sample · log(probs) for each categorical variable
        log_probs_per_category = torch.sum(
            sample * torch.log(self.probs + 1e-8), 
            dim=-1
        )
        return log_probs_per_category
    
    def entropy(self) -> torch.Tensor:
        """
        Compute entropy H[p] = -Σ p log p for each categorical.
        
        Returns:
            Entropy per categorical variable, shape: (..., stoch_size)
        """
        return -torch.sum(self.probs * torch.log(self.probs + 1e-8), dim=-1)
    
    def kl_divergence(self, other: 'CategoricalDist') -> torch.Tensor:
        """
        Compute KL[self || other] for each categorical variable.
        
        KL[p || q] = Σ p(x) log(p(x) / q(x))
        
        Args:
            other: Another CategoricalDist with same shape
            
        Returns:
            KL divergence per categorical, shape: (..., stoch_size)
        """
        # KL[p || q] = Σ p * (log p - log q)
        kl = torch.sum(
            self.probs * (torch.log(self.probs + 1e-8) - torch.log(other.probs + 1e-8)),
            dim=-1
        )
        return kl


# ============================================================================
# Discrete Action Distribution
# ============================================================================

class DiscreteActionDist:
    """
    Discrete action distribution for Mario (7 or 12 possible actions).
    
    Uses categorical distribution with optional entropy regularization.
    """
    
    def __init__(self, logits: torch.Tensor, unimix_ratio: float = 0.01):
        """
        Args:
            logits: Action logits, shape: (..., num_actions)
            unimix_ratio: Mix with uniform for exploration
        """
        # Apply Unimix
        if unimix_ratio > 0:
            probs = torch.softmax(logits, dim=-1)
            self.probs = (1 - unimix_ratio) * probs + unimix_ratio / logits.shape[-1]
            self.logits = torch.log(self.probs + 1e-8)
        else:
            self.logits = logits
            self.probs = torch.softmax(logits, dim=-1)
    
    def sample(self) -> torch.Tensor:
        """Sample action index."""
        # Shape: (..., )
        return torch.multinomial(self.probs, num_samples=1).squeeze(-1)
    
    def mode(self) -> torch.Tensor:
        """Return most likely action (for evaluation)."""
        return torch.argmax(self.probs, dim=-1)
    
    def log_prob(self, action: torch.Tensor) -> torch.Tensor:
        """
        Compute log probability of action.
        
        Args:
            action: Action indices, shape: (..., )
            
        Returns:
            Log probabilities, shape: (..., )
        """
        # Gather log prob for selected action
        log_probs = torch.log(self.probs + 1e-8)
        return log_probs.gather(-1, action.unsqueeze(-1).long()).squeeze(-1)
    
    def entropy(self) -> torch.Tensor:
        """
        Compute entropy H[π] = -Σ π(a) log π(a).
        
        Returns:
            Entropy, shape: (..., )
        """
        return -torch.sum(self.probs * torch.log(self.probs + 1e-8), dim=-1)


# ============================================================================
# Helper Functions
# ============================================================================

def free_bits_kl(kl: torch.Tensor, free_nats: float = 1.0) -> torch.Tensor:
    """
    Apply free bits constraint to KL divergence (Section B.2).
    
    Free bits allows KL to be small (< free_nats) without penalty.
    This prevents posterior collapse to prior.
    
    KL_free = max(KL, free_nats)
    
    Args:
        kl: KL divergence per dimension, shape: (..., stoch_size)
        free_nats: Minimum KL per dimension (nats, not bits)
        
    Returns:
        KL with free bits applied, shape: (..., stoch_size)
    """
    # Only penalize KL if it exceeds free_nats
    return torch.maximum(kl, torch.ones_like(kl) * free_nats)


def sg(tensor: torch.Tensor) -> torch.Tensor:
    """
    Stop gradient (detach). Used extensively in DreamerV3.
    
    Short notation: sg(x) = x.detach()
    """
    return tensor.detach()