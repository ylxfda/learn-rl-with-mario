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

    With 4 blocks and input 64x64:
        64x64 -> 32x32 -> 16x16 -> 8x8 -> 4x4

    Final: 4x4 * channels -> flatten to vector
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
        # For 64x64 input with 4 blocks: 64 -> 32 -> 16 -> 8 -> 4
        # Output: (cnn_depth * 2^(num_blocks-1)) * spatial * spatial
        self.output_spatial = 64 // (2 ** num_blocks)  # Dynamically calculate spatial size
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
            initial_spatial: Initial spatial dimension (4 for 64x64 final with 4 blocks)
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

    Standard GRU update equations:
        r_t = σ(W_r [h_{t-1}, x_t] + b_r)           # Reset gate
        u_t = σ(W_u [h_{t-1}, x_t] + b_u)           # Update gate
        h̃_t = tanh(W_h [r_t ⊙ h_{t-1}, x_t] + b_h) # Candidate
        h_t = (1 - u_t) ⊙ h_{t-1} + u_t ⊙ h̃_t      # New state

    With update_bias trick (DreamerV3):
        u_t = σ(W_u [h_{t-1}, x_t] + b_u + update_bias)

    The update_bias (typically -1) makes the update gate more conservative,
    encouraging the GRU to preserve long-term memory in h_t. This is crucial
    for RSSM's deterministic state, which should maintain stable context across
    many time steps during imagination rollouts.

    Why update_bias = -1?
    - sigmoid(x - 1) shifts the activation curve to the right
    - Lower update gate values → more retention of old state
    - Example: sigmoid(0) = 0.5 → sigmoid(-1) = 0.27
    - This means: instead of updating 50%, only update 27%
    - Result: ~2.8x longer memory retention over multiple steps

    In RSSM context:
        x_t = [z_{t-1}, a_{t-1}]  (preprocessed through Linear+LayerNorm+SiLU)
        h_t = deterministic recurrent state (should be stable)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        layer_norm: bool = True,
        update_bias: float = -1.0
    ):
        """
        Args:
            input_dim: Input dimension (e.g., stoch_size + action_size)
            hidden_dim: Hidden state dimension
            layer_norm: Whether to use LayerNorm on gate logits (before activation)
            update_bias: Bias added to update gate logits (default: -1.0)
                - 0.0: Standard GRU (50% update at neutral input)
                - -1.0: Conservative GRU (27% update at neutral input)
                - -2.0: Very conservative (12% update at neutral input)
                DreamerV3 uses -1.0 for better long-term memory
        """
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.update_bias = update_bias

        # Use PyTorch's built-in GRUCell
        self.gru = nn.GRUCell(input_dim, hidden_dim)

        # Optional layer norm on gate logits (before activation)
        # Applied to all 3 gates (reset, update, new) at once
        self.layer_norm = nn.LayerNorm(3 * hidden_dim) if layer_norm else None

    def forward(
        self,
        x: torch.Tensor,
        h_prev: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass with optional update gate bias and LayerNorm on logits.

        Args:
            x: Input tensor, shape: (B, input_dim)
            h_prev: Previous hidden state, shape: (B, hidden_dim)

        Returns:
            New hidden state, shape: (B, hidden_dim)
        """
        # Always use manual forward pass when layer_norm or update_bias is active
        if self.layer_norm is not None or self.update_bias != 0.0:
            h_next = self._forward_with_bias(x, h_prev)
        else:
            # Only use standard GRU when both are disabled
            h_next = self.gru(x, h_prev)

        return h_next

    def _forward_with_bias(self, x: torch.Tensor, h_prev: torch.Tensor) -> torch.Tensor:
        """
        GRU forward with LayerNorm on logits and update_bias applied to update gate.

        This manually computes GRU equations to:
        1. Apply LayerNorm to gate logits BEFORE activation (Project A's approach)
        2. Add update_bias to the update gate

        Standard GRU computation (from PyTorch source):
            gi = W_ir @ x + b_ir  (input contributions for all gates)
            gh = W_hr @ h + b_hr  (hidden contributions for all gates)

            For reset & update gates:
                r = σ(gi[0] + gh[0])
                u = σ(gi[1] + gh[1])

            For new gate (with reset modulation):
                n = tanh(gi[2] + r * gh[2])

            Final update:
                h' = (1 - u) * h + u * n

        Our modifications:
            1. gates = LayerNorm(gi + gh) for reset & update  ← Apply LayerNorm to logits
            2. u = σ(gates[1] + update_bias)  ← Add bias to update gate

        Args:
            x: Input, shape: (B, input_dim)
            h_prev: Previous state, shape: (B, hidden_dim)

        Returns:
            New state, shape: (B, hidden_dim)
        """
        # Get GRU parameters
        # PyTorch GRUCell stores weights as:
        # weight_ih: (3 * hidden_dim, input_dim) for input transformations
        # weight_hh: (3 * hidden_dim, hidden_dim) for hidden transformations
        # bias_ih, bias_hh: (3 * hidden_dim,) for input and hidden biases

        w_ih = self.gru.weight_ih  # (3*H, I)
        w_hh = self.gru.weight_hh  # (3*H, H)
        b_ih = self.gru.bias_ih    # (3*H,)
        b_hh = self.gru.bias_hh    # (3*H,)

        # Input transformation: W_i @ x + b_i
        gi = F.linear(x, w_ih, b_ih)  # (B, 3*H)
        # Hidden transformation: W_h @ h + b_h
        gh = F.linear(h_prev, w_hh, b_hh)  # (B, 3*H)

        # Split into three gates: [reset, update, new]
        i_r, i_u, i_n = gi.chunk(3, dim=1)
        h_r, h_u, h_n = gh.chunk(3, dim=1)

        # Combine logits for reset and update gates
        logits_r = i_r + h_r  # (B, H)
        logits_u = i_u + h_u  # (B, H)

        # Apply LayerNorm to gate logits BEFORE activation
        # This normalizes the pre-activation values, stabilizing training
        if self.layer_norm is not None:
            # Stack the logits to apply LayerNorm jointly
            # This ensures the normalization statistics are computed across all gates
            logits_combined = torch.cat([logits_r, logits_u, i_n], dim=1)  # (B, 3*H)
            logits_combined = self.layer_norm(logits_combined)  # (B, 3*H)
            logits_r, logits_u, i_n = logits_combined.chunk(3, dim=1)

        # Reset gate: r_t = σ(logits_r)
        reset_gate = torch.sigmoid(logits_r)

        # Update gate with bias: u_t = σ(logits_u + update_bias)
        # This is the update_bias trick!
        update_gate = torch.sigmoid(logits_u + self.update_bias)

        # New gate: n_t = tanh(i_n + reset_gate * h_n)
        # Note: Only the input part (i_n) goes through LayerNorm
        # The hidden part (h_n) is modulated by reset_gate as per standard GRU
        new_gate = torch.tanh(i_n + reset_gate * h_n)

        # Final state: h_t = (1 - u_t) * h_{t-1} + u_t * n_t
        #               └───────┬────────┘         └────┬────┘
        #                 keep old state         use new info
        h_next = (1 - update_gate) * h_prev + update_gate * new_gate

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
        # Initialize GRU weights
        for name, param in module.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param, gain=gain)
            elif 'bias' in name:
                # Initialize biases to zero
                # Note: If using update_bias in our custom GRUCell wrapper,
                # the update gate bias will be effectively modified during forward pass
                nn.init.zeros_(param)