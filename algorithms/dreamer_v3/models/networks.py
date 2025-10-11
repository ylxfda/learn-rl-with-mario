"""
Neural network building blocks for DreamerV3.

Implements the architectural components used throughout the model:
- MLP with LayerNorm and SiLU activation
- CNN Encoder/Decoder for images
- GRU for sequential modeling

All components follow DreamerV3 specifications (Section B.1).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List


# ============================================================================
# Multi-Layer Perceptron (MLP)
# ============================================================================

class MLP(nn.Module):
    """
    Multi-layer perceptron with LayerNorm and SiLU activation.
    
    Architecture per layer:
        x -> Linear -> LayerNorm -> SiLU -> x_next
    
    DreamerV3 uses:
        - SiLU (Swish) activation: x * sigmoid(x)
        - LayerNorm for training stability
        - Typically 4 layers of 512 units
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int = 4,
        activation: str = "SiLU",
        layer_norm: bool = True
    ):
        """
        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden layer dimension
            output_dim: Output dimension
            num_layers: Number of hidden layers (default: 4)
            activation: Activation function (default: "SiLU")
            layer_norm: Whether to use LayerNorm (default: True)
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        
        # Select activation function
        if activation == "SiLU":
            self.activation = nn.SiLU()
        elif activation == "ReLU":
            self.activation = nn.ReLU()
        elif activation == "ELU":
            self.activation = nn.ELU()
        else:
            raise ValueError(f"Unknown activation: {activation}")
        
        # Build layers
        layers = []
        
        # Input layer
        layers.append(nn.Linear(input_dim, hidden_dim))
        if layer_norm:
            layers.append(nn.LayerNorm(hidden_dim))
        layers.append(self.activation)
        
        # Hidden layers
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            if layer_norm:
                layers.append(nn.LayerNorm(hidden_dim))
            layers.append(self.activation)
        
        # Output layer (no activation, will be added by specific heads)
        layers.append(nn.Linear(hidden_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor, shape: (..., input_dim)
            
        Returns:
            Output tensor, shape: (..., output_dim)
        """
        return self.network(x)


# ============================================================================
# Convolutional Neural Network (CNN) Encoder
# ============================================================================

class CNNEncoder(nn.Module):
    """
    CNN encoder for processing images into feature vectors.
    
    Architecture:
        Image (C, H, W) -> Conv blocks -> Flatten -> Feature vector
        
    Each conv block:
        Conv2d(kernel=4, stride=2) -> SiLU
        
    With 3 blocks and input 64x64:
        64x64 -> 32x32 -> 16x16 -> 8x8
        
    Final: 8x8 * channels -> flatten to vector
    """
    
    def __init__(
        self,
        input_channels: int = 1,  # 1 for grayscale, 3 for RGB
        cnn_depth: int = 32,      # Base number of channels
        num_blocks: int = 3,       # Number of conv blocks
        activation: str = "SiLU"
    ):
        """
        Args:
            input_channels: Number of input channels (1=grayscale, 3=RGB)
            cnn_depth: Base depth (channels double each block)
            num_blocks: Number of convolutional blocks
            activation: Activation function
        """
        super().__init__()
        
        self.input_channels = input_channels
        self.cnn_depth = cnn_depth
        self.num_blocks = num_blocks
        
        # Select activation
        if activation == "SiLU":
            self.activation = nn.SiLU()
        elif activation == "ReLU":
            self.activation = nn.ReLU()
        else:
            raise ValueError(f"Unknown activation: {activation}")
        
        # Build conv blocks
        layers = []
        in_channels = input_channels
        
        for i in range(num_blocks):
            out_channels = cnn_depth * (2 ** i)  # 32, 64, 128, ...
            layers.append(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=4,
                    stride=2,
                    padding=1  # To maintain size // 2
                )
            )
            layers.append(self.activation)
            in_channels = out_channels
        
        self.conv_net = nn.Sequential(*layers)
        
        # Calculate output dimension
        # After num_blocks with stride 2: H -> H // (2^num_blocks)
        # For 64x64 input with 3 blocks: 64 -> 32 -> 16 -> 8
        # Output: (cnn_depth * 2^(num_blocks-1)) * 8 * 8
        self.output_spatial = 8  # Assuming 64x64 input
        self.output_channels = cnn_depth * (2 ** (num_blocks - 1))
        self.output_dim = self.output_channels * self.output_spatial * self.output_spatial
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Image tensor, shape: (B, C, H, W) or (B, T, C, H, W)
            
        Returns:
            Feature vector, shape: (B, output_dim) or (B, T, output_dim)
        """
        # Handle both (B, C, H, W) and (B, T, C, H, W)
        has_time_dim = x.ndim == 5
        if has_time_dim:
            B, T, C, H, W = x.shape
            x = x.reshape(B * T, C, H, W)
        
        # Apply conv net
        x = self.conv_net(x)  # Shape: (B*T, channels, h, w)
        
        # Flatten spatial dimensions
        x = x.reshape(x.shape[0], -1)  # Shape: (B*T, output_dim)
        
        # Restore time dimension if needed
        if has_time_dim:
            x = x.reshape(B, T, -1)
        
        return x


# ============================================================================
# Convolutional Neural Network (CNN) Decoder
# ============================================================================

class CNNDecoder(nn.Module):
    """
    CNN decoder for reconstructing images from feature vectors.
    
    Architecture:
        Feature vector -> Linear -> Reshape -> Deconv blocks -> Image
        
    Each deconv block:
        ConvTranspose2d(kernel=4, stride=2) -> SiLU
        
    Final layer outputs image with sigmoid (for [0,1] pixel values).
    """
    
    def __init__(
        self,
        input_dim: int,           # Feature dimension
        output_channels: int = 1, # 1 for grayscale, 3 for RGB
        cnn_depth: int = 32,
        num_blocks: int = 3,
        activation: str = "SiLU",
        initial_spatial: int = 8  # Initial spatial size after reshape
    ):
        """
        Args:
            input_dim: Input feature dimension
            output_channels: Number of output channels (1=grayscale, 3=RGB)
            cnn_depth: Base depth
            num_blocks: Number of deconv blocks
            activation: Activation function
            initial_spatial: Initial spatial dimension (8 for 64x64 final)
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.output_channels = output_channels
        self.cnn_depth = cnn_depth
        self.num_blocks = num_blocks
        self.initial_spatial = initial_spatial
        
        # Select activation
        if activation == "SiLU":
            self.activation = nn.SiLU()
        elif activation == "ReLU":
            self.activation = nn.ReLU()
        else:
            raise ValueError(f"Unknown activation: {activation}")
        
        # Calculate initial channels (reverse of encoder)
        initial_channels = cnn_depth * (2 ** (num_blocks - 1))
        initial_size = initial_channels * initial_spatial * initial_spatial
        
        # Linear layer to reshape
        self.fc = nn.Linear(input_dim, initial_size)
        self.initial_channels = initial_channels
        
        # Build deconv blocks (reverse order of encoder)
        layers = []
        in_channels = initial_channels
        
        for i in range(num_blocks - 1):
            out_channels = in_channels // 2  # Halve channels each block
            layers.append(
                nn.ConvTranspose2d(
                    in_channels,
                    out_channels,
                    kernel_size=4,
                    stride=2,
                    padding=1
                )
            )
            layers.append(self.activation)
            in_channels = out_channels
        
        # Final layer: output image
        layers.append(
            nn.ConvTranspose2d(
                in_channels,
                output_channels,
                kernel_size=4,
                stride=2,
                padding=1
            )
        )
        # No activation here - will use appropriate dist (e.g., Bernoulli/MSE)
        
        self.deconv_net = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Feature tensor, shape: (B, input_dim) or (B, T, input_dim)
            
        Returns:
            Reconstructed image, shape: (B, C, H, W) or (B, T, C, H, W)
        """
        # Handle time dimension
        has_time_dim = x.ndim == 3
        if has_time_dim:
            B, T, D = x.shape
            x = x.reshape(B * T, D)
        
        # Linear projection
        x = self.fc(x)  # Shape: (B*T, initial_size)
        
        # Reshape to spatial
        x = x.reshape(
            x.shape[0],
            self.initial_channels,
            self.initial_spatial,
            self.initial_spatial
        )  # Shape: (B*T, channels, h, w)
        
        # Apply deconv net
        x = self.deconv_net(x)  # Shape: (B*T, output_channels, H, W)
        
        # Restore time dimension if needed
        if has_time_dim:
            x = x.reshape(B, T, *x.shape[1:])
        
        return x


# ============================================================================
# Gated Recurrent Unit (GRU) for Sequence Modeling
# ============================================================================

class GRUCell(nn.Module):
    """
    GRU cell for deterministic state h_t in RSSM.
    
    GRU update equations:
        r_t = σ(W_r [h_{t-1}, x_t])           # Reset gate
        u_t = σ(W_u [h_{t-1}, x_t])           # Update gate
        h_tilde = tanh(W_h [r_t * h_{t-1}, x_t])  # Candidate
        h_t = (1 - u_t) * h_{t-1} + u_t * h_tilde  # New state
    
    In RSSM: x_t = [z_{t-1}, a_{t-1}]
    """
    
    def __init__(self, input_dim: int, hidden_dim: int, layer_norm: bool = True):
        """
        Args:
            input_dim: Input dimension (e.g., stoch_size + action_size)
            hidden_dim: Hidden state dimension
            layer_norm: Whether to use LayerNorm
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # GRU parameters
        self.gru = nn.GRUCell(input_dim, hidden_dim)
        
        # Optional layer norm on output
        self.layer_norm = nn.LayerNorm(hidden_dim) if layer_norm else None
    
    def forward(
        self,
        x: torch.Tensor,
        h_prev: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            x: Input tensor, shape: (B, input_dim)
            h_prev: Previous hidden state, shape: (B, hidden_dim)
            
        Returns:
            New hidden state, shape: (B, hidden_dim)
        """
        h_next = self.gru(x, h_prev)
        
        if self.layer_norm is not None:
            h_next = self.layer_norm(h_next)
        
        return h_next


# ============================================================================
# Helper Functions
# ============================================================================

def init_weights(module: nn.Module, gain: float = 1.0):
    """
    Initialize network weights using Xavier uniform.
    
    Args:
        module: PyTorch module to initialize
        gain: Scaling factor for initialization
    """
    if isinstance(module, (nn.Linear, nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.xavier_uniform_(module.weight, gain=gain)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.GRUCell):
        for name, param in module.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param, gain=gain)
            elif 'bias' in name:
                nn.init.zeros_(param)