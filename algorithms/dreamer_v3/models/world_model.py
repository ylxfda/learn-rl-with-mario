"""
World Model (RSSM) for DreamerV3.

Recurrent State Space Model with:
- Encoder: q_φ(z_t | h_t, x_t) - posterior over stochastic state
- Sequence: h_t = f_φ(h_{t-1}, z_{t-1}, a_{t-1}) - deterministic recurrence
- Prior: p_φ(z_t | h_t) - stochastic transition model
- Decoder: p_φ(x_t | h_t, z_t) - observation reconstruction
- Reward: p_φ(r_t | h_t, z_t) - reward prediction
- Continue: p_φ(c_t | h_t, z_t) - episode continuation

Follows Section 3.2 and Equation 2 from DreamerV3 paper.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional

from algorithms.dreamer_v3.models.networks import (
    CNNEncoder,
    CNNDecoder,
    MLP,
    GRUCell,
    init_weights,
)
from algorithms.dreamer_v3.models.distributions import (
    CategoricalDist,
    symlog,
    symexp,
    free_bits_kl,
    sg,
)


class RSSM(nn.Module):
    """
    Recurrent State Space Model - the core world model in DreamerV3.
    
    State space:
        s_t = {h_t, z_t}  where:
        - h_t: deterministic recurrent state (GRU hidden), shape: (hidden_size,)
        - z_t: stochastic state (categorical), shape: (stoch_size, discrete_size)
    
    The world model learns to:
        1. Encode observations into stochastic states (closed-loop with images)
        2. Predict future states without observations (open-loop, for imagination)
        3. Decode states back to observations
        4. Predict rewards and episode termination
    """
    
    def __init__(self, config: dict):
        super().__init__()
        
        # Extract config
        self.hidden_size = config['model']['hidden_size']  # h_t dimension
        self.stoch_size = config['model']['stoch_size']    # Number of categoricals
        self.discrete_size = config['model']['discrete_size']  # Classes per categorical
        self.action_size = 7  # Mario has 7 discrete actions (or 12 for extended)
        
                # For flattened z_t representation
        self.stoch_dim = self.stoch_size * self.discrete_size
        
        # ====================================================================
        # Encoder: q_φ(z_t | h_t, x_t) - Posterior
        # ====================================================================
        # Takes observation x_t and deterministic state h_t, outputs z_t
        
        # CNN to extract image features
        self.image_encoder = CNNEncoder(
            input_channels=1 if config['env']['grayscale'] else 3,
            cnn_depth=config['model']['cnn_depth'],
            num_blocks=config['model']['cnn_blocks'],
            activation=config['model']['activation']
        )
        
        # MLP: [image_features, h_t] -> logits for z_t
        # Input: image_features (from CNN) + h_t
        # Output: logits for (stoch_size, discrete_size) categoricals
        self.encoder_mlp = MLP(
            input_dim=self.image_encoder.output_dim + self.hidden_size,
            hidden_dim=config['model']['mlp_hidden'],
            output_dim=self.stoch_size * self.discrete_size,  # Logits for all categories
            num_layers=config['model']['mlp_layers'],
            activation=config['model']['activation'],
            layer_norm=config['model']['layer_norm']
        )
        
        # ====================================================================
        # Sequence Model: h_t = f_φ(h_{t-1}, z_{t-1}, a_{t-1})
        # ====================================================================
        # GRU that updates deterministic state h_t given previous z and action

        # Input preprocessing layer: [z_{t-1}, a_{t-1}] -> hidden representation
        inp_layers = []
        inp_dim = self.stoch_dim + self.action_size
        inp_layers.append(nn.Linear(inp_dim, self.hidden_size, bias=False))
        if config['model']['layer_norm']:
            inp_layers.append(nn.LayerNorm(self.hidden_size, eps=1e-03))

        # Activation function
        if config['model']['activation'] == "SiLU":
            inp_layers.append(nn.SiLU())
        elif config['model']['activation'] == "ReLU":
            inp_layers.append(nn.ReLU())
        elif config['model']['activation'] == "ELU":
            inp_layers.append(nn.ELU())
        else:
            raise ValueError(f"Unknown activation: {config['model']['activation']}")

        self.img_in_layers = nn.Sequential(*inp_layers)

        # GRU cell processes the preprocessed input
        self.sequence_model = GRUCell(
            input_dim=self.hidden_size,  # Now takes preprocessed hidden representation
            hidden_dim=self.hidden_size,
            layer_norm=config['model']['layer_norm'],
            update_bias=config['model']['gru_update_bias']
        )
        
        # ====================================================================
        # Prior: p_φ(z_t | h_t) - Transition Model
        # ====================================================================
        # Predicts z_t from h_t alone (no observation) - used for imagination
        
        self.prior_mlp = MLP(
            input_dim=self.hidden_size,
            hidden_dim=config['model']['mlp_hidden'],
            output_dim=self.stoch_size * self.discrete_size,
            num_layers=config['model']['mlp_layers'],
            activation=config['model']['activation'],
            layer_norm=config['model']['layer_norm']
        )
        
        # ====================================================================
        # Decoder: p_φ(x_t | h_t, z_t) - Observation Reconstruction
        # ====================================================================
        # Reconstructs image from state s_t = {h_t, z_t}
        
        self.image_decoder = CNNDecoder(
            input_dim=self.hidden_size + self.stoch_dim,
            output_channels=1 if config['env']['grayscale'] else 3,
            cnn_depth=config['model']['decoder_cnn_depth'],
            num_blocks=config['model']['cnn_blocks'],
            activation=config['model']['activation'],
            initial_spatial=self.image_encoder.output_spatial  # Match encoder's output spatial size
        )
        
        # ====================================================================
        # Reward Head: p_φ(r_t | h_t, z_t)
        # ====================================================================
        # Predicts reward in symlog space (for stability with large rewards)
        
        self.reward_head = MLP(
            input_dim=self.hidden_size + self.stoch_dim,
            hidden_dim=config['model']['mlp_hidden'],
            output_dim=1,  # Scalar reward prediction
            num_layers=config['model']['decoder_mlp_layers'],
            activation=config['model']['activation'],
            layer_norm=config['model']['layer_norm']
        )
        
        # ====================================================================
        # Continue Head: p_φ(c_t | h_t, z_t)
        # ====================================================================
        # Predicts probability of episode continuation (1 - terminal)
        
        self.continue_head = MLP(
            input_dim=self.hidden_size + self.stoch_dim,
            hidden_dim=config['model']['mlp_hidden'],
            output_dim=1,  # Scalar logit for continuation probability
            num_layers=config['model']['decoder_mlp_layers'],
            activation=config['model']['activation'],
            layer_norm=config['model']['layer_norm']
        )
        
        # Initialize weights
        self.apply(lambda m: init_weights(m, gain=1.0))
        
        # Store config for loss computation
        self.config = config
    
    # ========================================================================
    # Core RSSM Operations
    # ========================================================================
    
    def encode(self, h: torch.Tensor, x: torch.Tensor) -> CategoricalDist:
        """
        Encoder: q_φ(z_t | h_t, x_t)
        
        Computes posterior distribution over z_t given observation and h_t.
        Used during training on real data (closed-loop).
        
        Args:
            h: Deterministic state, shape: (B, hidden_size) or (B, T, hidden_size)
            x: Observation image, shape: (B, C, H, W) or (B, T, C, H, W)
            
        Returns:
            CategoricalDist over z_t, shape: (B, stoch_size, discrete_size)
        """
        # Extract image features
        img_features = self.image_encoder(x)  # (B, img_feat_dim) or (B, T, img_feat_dim)
        
        # Concatenate with current deterministic state h_t
        encoder_input = torch.cat([img_features, h], dim=-1)

        # Get logits for categorical distribution
        logits = self.encoder_mlp(encoder_input)  # (B, stoch_size * discrete_size)
        
        # Reshape to (B, stoch_size, discrete_size) for categorical
        logits = logits.reshape(*logits.shape[:-1], self.stoch_size, self.discrete_size)
        
        return CategoricalDist(logits, unimix_ratio=0.01)
    
    def dynamics(
        self,
        h_prev: torch.Tensor,
        z_prev: torch.Tensor,
        action: torch.Tensor
    ) -> torch.Tensor:
        """
        Sequence model: h_t = f_φ(h_{t-1}, z_{t-1}, a_{t-1})

        Updates deterministic state h using GRU with preprocessing layer.

        Args:
            h_prev: Previous deterministic state, shape: (B, hidden_size)
            z_prev: Previous stochastic state (one-hot), shape: (B, stoch_size, discrete_size)
            action: Action (one-hot), shape: (B, action_size)

        Returns:
            New deterministic state h_t, shape: (B, hidden_size)
        """
        # Flatten z_prev to (B, stoch_dim)
        z_flat = z_prev.reshape(z_prev.shape[0], -1)

        # Concatenate [z_{t-1}, a_{t-1}]
        input_features = torch.cat([z_flat, action], dim=-1)

        # Preprocess input through linear layer 
        # This transforms [z, a] into a richer representation before GRU
        preprocessed = self.img_in_layers(input_features)

        # Update h_t using GRU with preprocessed input
        h_next = self.sequence_model(preprocessed, h_prev)

        return h_next
    
    def prior(self, h: torch.Tensor) -> CategoricalDist:
        """
        Prior: p_φ(z_t | h_t)
        
        Predicts z_t from h_t without observation.
        Used for imagination (open-loop) and to compute KL divergence.
        
        Args:
            h: Deterministic state, shape: (B, hidden_size) or (B, T, hidden_size)
            
        Returns:
            CategoricalDist over z_t, shape: (B, stoch_size, discrete_size)
        """
        # Get logits
        logits = self.prior_mlp(h)  # (B, stoch_size * discrete_size)
        
        # Reshape
        logits = logits.reshape(*logits.shape[:-1], self.stoch_size, self.discrete_size)
        
        return CategoricalDist(logits, unimix_ratio=0.01)
    
    def decode(self, h: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """
        Decoder: p_φ(x_t | h_t, z_t)
        
        Reconstructs observation from state.
        
        Args:
            h: Deterministic state, shape: (B, hidden_size) or (B, T, hidden_size)
            z: Stochastic state (one-hot), shape: (B, stoch_size, discrete_size)
            
        Returns:
            Reconstructed image, shape: (B, C, H, W) or (B, T, C, H, W)
        """
        # Flatten z
        z_flat = z.reshape(*z.shape[:-2], -1)  # (B, stoch_dim) or (B, T, stoch_dim)
        
        # Concatenate deterministic and stochastic parts
        state = torch.cat([h, z_flat], dim=-1)

        # Decode to image
        x_recon = self.image_decoder(state)
        
        return x_recon
    
    def predict_reward(self, h: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """
        Reward head: p_φ(r_t | h_t, z_t)
        
        Predicts reward in symlog space.
        
        Args:
            h: Deterministic state, shape: (B, hidden_size) or (B, T, hidden_size)
            z: Stochastic state, shape: (B, stoch_size, discrete_size)
            
        Returns:
            Predicted reward (symlog space), shape: (B,) or (B, T)
        """
        # Flatten z
        z_flat = z.reshape(*z.shape[:-2], -1)
        
        # Concatenate state
        state = torch.cat([h, z_flat], dim=-1)
        
        # Predict reward in symlog space
        reward_symlog = self.reward_head(state).squeeze(-1)
        
        return reward_symlog
    
    def predict_continue(self, h: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """
        Continue head: p_φ(c_t | h_t, z_t)

        Predicts logit for episode continuation.

        Args:
            h: Deterministic state, shape: (B, hidden_size) or (B, T, hidden_size)
            z: Stochastic state, shape: (B, stoch_size, discrete_size)

        Returns:
            Continue logit (NOT probability), shape: (B,) or (B, T)
            Use torch.sigmoid() to convert to probability if needed.
        """
        # Flatten z
        z_flat = z.reshape(*z.shape[:-2], -1)

        # Concatenate state
        state = torch.cat([h, z_flat], dim=-1)

        # Predict continue logit (return logit, not probability)
        continue_logit = self.continue_head(state).squeeze(-1)

        return continue_logit
    
    # ========================================================================
    # Rollout Functions
    # ========================================================================
    
    def observe(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        h_0: Optional[torch.Tensor] = None,
        is_first: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Closed-loop rollout: use encoder to get posterior z_t from observations.
        
        This is used during world model training on replay buffer data.
        Computes posterior q_φ(z_t | h_t, x_t) at each timestep.
        
        Args:
            observations: Image sequence, shape: (B, T, C, H, W)
            actions: Action sequence, shape: (B, T, action_size) - one-hot
            h_0: Initial hidden state, shape: (B, hidden_size) - if None, use zeros
            is_first: Optional bool mask, shape: (B, T). True marks start of new episode.
            
        Returns:
            Dictionary containing:
                - h: Deterministic states, shape: (B, T, hidden_size)
                - z_post: Posterior stochastic states, shape: (B, T, stoch_size, discrete_size)
                - z_prior: Prior stochastic states (for KL), shape: (B, T, stoch_size, discrete_size)
                - x_recon: Reconstructed observations, shape: (B, T, C, H, W)
                - reward: Predicted rewards (symlog), shape: (B, T)
                - continue: Predicted continuation logits (NOT probs), shape: (B, T)
        """
        B, T = observations.shape[:2]
        
        # Initialize hidden state
        if h_0 is None:
            h = torch.zeros(B, self.hidden_size, device=observations.device)
        else:
            h = h_0

        if is_first is not None:
            is_first = is_first.to(observations.device)
        
        # Storage for trajectory
        h_seq = []
        z_post_seq = []
        z_prior_seq = []
        
        for t in range(T):
            # Get observation and action at time t
            x_t = observations[:, t]  # (B, C, H, W)
            a_t = actions[:, t]       # (B, action_size)

            if is_first is not None:
                reset_mask = is_first[:, t].unsqueeze(-1)
                if reset_mask.any():
                    # whenever a rollout chunk introduces a new episode,
                    # reset the deterministic state for exactly those batch elements.
                    h = torch.where(reset_mask, torch.zeros_like(h), h)
            
            # Snapshot current deterministic state so recon uses matching h_t
            h_current = h
            
            # Prior: p_φ(z_t | h_t) - before seeing x_t
            z_prior_dist = self.prior(h_current)
            
            # Posterior: q_φ(z_t | h_t, x_t) - after seeing x_t
            z_post_dist = self.encode(h_current, x_t)
            
            # Sample from posterior (with straight-through gradients)
            z_post = z_post_dist.sample()  # (B, stoch_size, discrete_size)
            z_prior = z_prior_dist.sample()  # For KL computation
            
            # Update deterministic state for current timestep so reconstruction uses freshest features
            # h_{t+1} = f_φ(h_t, z_t, a_t)
            h = self.dynamics(h_current, z_post, a_t)
            
            # Store updated states
            h_seq.append(h_current)
            z_post_seq.append(z_post)
            z_prior_seq.append(z_prior)
        
        # Stack sequences
        h_seq = torch.stack(h_seq, dim=1)  # (B, T, hidden_size)
        z_post_seq = torch.stack(z_post_seq, dim=1)  # (B, T, stoch_size, discrete_size)
        z_prior_seq = torch.stack(z_prior_seq, dim=1)
        
        # Decode observations
        x_recon = self.decode(h_seq, z_post_seq)  # (B, T, C, H, W)
        
        # Predict rewards and continues
        reward_pred = self.predict_reward(h_seq, z_post_seq)  # (B, T)
        continue_pred = self.predict_continue(h_seq, z_post_seq)  # (B, T)
        
        return {
            'h': h_seq,
            'z_post': z_post_seq,
            'z_prior': z_prior_seq,
            'x_recon': x_recon,
            'reward': reward_pred,
            'continue': continue_pred,
            'h_last': h  # deterministic state after processing entire sequence
        }
    
    def imagine(
        self,
        h_0: torch.Tensor,
        z_0: torch.Tensor,
        actor: nn.Module,
        horizon: int
    ) -> Dict[str, torch.Tensor]:
        """
        Open-loop rollout: imagine future using prior (no observations).
        
        This is used for actor-critic training.
        At each step:
            1. Sample action from actor: a_t ~ π_θ(· | h_t, z_t)
            2. Update deterministic state: h_{t+1} = f_φ(h_t, z_t, a_t)
            3. Sample stochastic state: z_{t+1} ~ p_φ(· | h_{t+1})
            4. Predict reward and continue: r_t, c_t
        
        Args:
            h_0: Initial deterministic state, shape: (B, hidden_size)
            z_0: Initial stochastic state, shape: (B, stoch_size, discrete_size)
            actor: Actor network (returns action distribution)
            horizon: Number of steps to imagine
            
        Returns:
            Dictionary containing imagined trajectory:
                - h: States, shape: (B, horizon, hidden_size)
                - z: States, shape: (B, horizon, stoch_size, discrete_size)
                - actions: Sampled actions, shape: (B, horizon, action_size)
                - reward: Predicted rewards (symlog), shape: (B, horizon)
                - continue: Predicted continues, shape: (B, horizon)
                - log_probs: Action log probs, shape: (B, horizon)
        """
        B = h_0.shape[0]
        device = h_0.device
        
        h = h_0
        z = z_0
        
        # Storage
        h_seq = []
        z_seq = []
        action_seq = []
        reward_seq = []
        continue_seq = []
        log_prob_seq = []
        
        for _ in range(horizon):
            # Store current state
            h_seq.append(h)
            z_seq.append(z)
            
            # Sample action from actor
            action_dist = actor(h, z)
            action_idx = action_dist.sample()  # (B,)
            log_prob = action_dist.log_prob(action_idx)
            
            # Convert to one-hot
            action_onehot = F.one_hot(action_idx, num_classes=self.action_size).float()
            
            action_seq.append(action_onehot)
            log_prob_seq.append(log_prob)
            
            # Predict reward and continue from current state
            reward = self.predict_reward(h, z)
            cont_logit = self.predict_continue(h, z)
            cont = torch.sigmoid(cont_logit)  # Convert logit to probability for λ-returns

            reward_seq.append(reward)
            continue_seq.append(cont)
            
            # Transition to next state using prior (open-loop)
            # h_{t+1} = f_φ(h_t, z_t, a_t)
            h = self.dynamics(h, z, action_onehot)
            
            # z_{t+1} ~ p_φ(· | h_{t+1})
            z_dist = self.prior(h)
            z = z_dist.sample()
        
        return {
            'h': torch.stack(h_seq, dim=1),
            'z': torch.stack(z_seq, dim=1),
            'actions': torch.stack(action_seq, dim=1),
            'reward': torch.stack(reward_seq, dim=1),
            'continue': torch.stack(continue_seq, dim=1),
            'log_probs': torch.stack(log_prob_seq, dim=1)
        }
    
    # ========================================================================
    # Loss Computation (Equation 2)
    # ========================================================================
    
    def compute_loss(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        continues: torch.Tensor,
        *,
        is_first: Optional[torch.Tensor] = None,
        h_0: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute world model loss (Equation 2 in DreamerV3 paper).
        
        L_world = β_pred * L_pred + β_dyn * L_dyn + β_rep * L_rep
        
        Where:
            L_pred: Prediction loss (reconstruction + reward + continue)
            L_dyn: Dynamics loss (KL regularization on posterior vs prior)
            L_rep: Representation loss (bidirectional KL with stop-gradient)
        
        Args:
            observations: (B, T, C, H, W)
            actions: (B, T, action_size) - one-hot
            rewards: (B, T) - actual rewards
            continues: (B, T) - actual continuation flags (1=continue, 0=done)
            
        Returns:
            Dictionary with loss components
        """
        # Get posteriors and predictions via closed-loop rollout
        outputs = self.observe(observations, actions, h_0=h_0, is_first=is_first)
        
        h = outputs['h']  # (B, T, hidden_size)
        z_post = outputs['z_post']  # (B, T, stoch_size, discrete_size)
        z_prior = outputs['z_prior']
        x_recon = outputs['x_recon']  # (B, T, C, H, W)
        reward_pred = outputs['reward']  # (B, T) symlog space
        continue_pred = outputs['continue']  # (B, T)
        h_last = outputs['h_last']  # (B, hidden_size)
        
        # ====================================================================
        # L_pred: Prediction Loss
        # ====================================================================
        
        # Image reconstruction loss (MSE in pixel space)
        recon_loss = F.mse_loss(x_recon, observations, reduction='none')
        recon_loss = recon_loss.mean(dim=[2, 3, 4])  # Mean over C, H, W -> (B, T)
        
        # Reward prediction loss (MSE in symlog space)
        reward_symlog = symlog(rewards)
        reward_loss = F.mse_loss(reward_pred, reward_symlog, reduction='none')  # (B, T)

        # Continue prediction loss (binary cross-entropy with logits - safe for AMP)
        # continue_pred is now logits, not probabilities
        continue_loss = F.binary_cross_entropy_with_logits(
            continue_pred,
            continues,
            reduction='none'
        )  # (B, T)
        
        # Total prediction loss
        L_pred = (recon_loss + reward_loss + continue_loss).mean()
        
        # ====================================================================
        # L_dyn: Dynamics Loss (KL regularization)
        # ====================================================================
        # KL[ q_φ(z_t | h_t, x_t) || p_φ(z_t | h_t) ]
        # Regularizes posterior to stay close to prior
        
        # Compute KL for each categorical variable
        # We need to recreate distributions for KL computation
        # Flatten to (B*T, stoch_size, discrete_size)
        B, T = z_post.shape[:2]
        
        # Get logits for posterior and prior
        h_flat = h.reshape(B * T, -1)
        x_flat = observations.reshape(B * T, *observations.shape[2:])
        
        post_dist_flat = self.encode(h_flat, x_flat)
        prior_dist_flat = self.prior(h_flat)
        
        post_probs = post_dist_flat.probs
        prior_probs = prior_dist_flat.probs
        
        # KL divergence per categorical for logging (no free bits)
        kl_per_cat_raw = torch.sum(
            post_probs * (torch.log(post_probs + 1e-8) - torch.log(prior_probs + 1e-8)),
            dim=-1
        )
        
        # Apply free bits for reporting
        kl_per_cat = free_bits_kl(kl_per_cat_raw, free_nats=self.config['training']['free_nats'])
        
        # ====================================================================
        # Dynamics loss: stop gradient on posterior so only prior is updated
        # L_dyn = KL[q || sg(p)]
        # ====================================================================
        kl_dyn = torch.sum(
            sg(post_probs) * (torch.log(sg(post_probs) + 1e-8) - torch.log(prior_probs + 1e-8)),
            dim=-1
        )
        kl_dyn = free_bits_kl(kl_dyn, free_nats=self.config['training']['free_nats'])
        L_dyn = kl_dyn.mean()
        
        # ====================================================================
        # L_rep: Representation Loss (stop gradient on prior so only posterior is updated)
        # L_rep = KL[sg(q) || p]
        # ====================================================================
        kl_rep = torch.sum(
            post_probs * (torch.log(post_probs + 1e-8) - torch.log(sg(prior_probs) + 1e-8)),
            dim=-1
        )
        kl_rep = free_bits_kl(kl_rep, free_nats=self.config['training']['free_nats'])
        L_rep = kl_rep.mean()
        
        # ====================================================================
        # Total Loss
        # ====================================================================
        
        beta_pred = self.config['training']['beta_pred']
        beta_dyn = self.config['training']['beta_dyn']
        beta_rep = self.config['training']['beta_rep']
        
        total_loss = beta_pred * L_pred + beta_dyn * L_dyn + beta_rep * L_rep
        
        return {
            'total_loss': total_loss,
            'pred_loss': L_pred,
            'dyn_loss': L_dyn,
            'rep_loss': L_rep,
            # Expose final deterministic state so the trainer can stitch chunks together.
            'h_last': h_last.detach(),
            # Return full sequences for coupled training with actor-critic
            'h_seq': h.detach(),  # (B, T, hidden_size)
            'z_seq': z_post.detach(),  # (B, T, stoch_size, discrete_size)
            'recon_loss': recon_loss.mean(),
            'reward_loss': reward_loss.mean(),
            'continue_loss': continue_loss.mean(),
            'kl_divergence': kl_per_cat.mean(),
            'kl_raw': kl_per_cat_raw.mean()
        }
