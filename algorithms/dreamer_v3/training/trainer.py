"""
Main training loop for DreamerV3 on Super Mario Bros.

Implements the three-phase training process:
1. Environment Collection: Gather experience using current policy
2. World Model Learning: Train RSSM on replay sequences
3. Behavior Learning: Train actor-critic on imagined rollouts

Follows Algorithm 1 from DreamerV3 paper.
"""

import torch
import torch.nn.functional as F
import numpy as np
import yaml
from pathlib import Path
from typing import Dict, Optional
from tqdm import tqdm

from algorithms.dreamer_v3.models.world_model import RSSM
from algorithms.dreamer_v3.agent.actor_critic import (
    Actor,
    Critic,
    EMATargetCritic,
    compute_lambda_returns,
    compute_actor_loss,
)
from algorithms.dreamer_v3.training.replay_buffer import ReplayBuffer
from algorithms.dreamer_v3.envs.mario_env import make_mario_env
from algorithms.dreamer_v3.utils.logger import Logger


class DreamerV3Trainer:
    """
    Main trainer for DreamerV3 agent on Super Mario Bros.
    
    Training loop (Algorithm 1):
    ┌─────────────────────────────────────────────────────┐
    │ while not converged:                                 │
    │   1. Collect H_collect steps in environment          │
    │   2. Train world model on replay buffer              │
    │   3. Imagine trajectories and train actor-critic     │
    │   4. Update target critic via EMA                    │
    └─────────────────────────────────────────────────────┘
    """
    
    def __init__(self, config_path: str):
        """
        Args:
            config_path: Path to YAML configuration file
        """
        # Load config
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Set device
        self.device = torch.device(
            self.config['training']['device'] 
            if torch.cuda.is_available() 
            else 'cpu'
        )
        print(f"Using device: {self.device}")
        
        # Set random seeds
        self._set_seeds(self.config['training']['seed'])
        
        # Create environment
        self.env = make_mario_env(self.config)
        print(f"Environment: {self.config['env']['name']}")
        print(f"  Action space: {self.env.action_size}")
        print(f"  Observation shape: {self.env.observation_shape}")
        
        # Create replay buffer
        self.replay_buffer = ReplayBuffer(
            capacity=self.config['training']['replay_capacity'],
            observation_shape=self.env.observation_shape,
            action_size=self.env.action_size,
            device=self.device
        )
        print(f"Replay buffer capacity: {self.config['training']['replay_capacity']}")
        
        # Create models
        self.world_model = RSSM(self.config).to(self.device)
        self.actor = Actor(self.config).to(self.device)
        self.critic = Critic(self.config).to(self.device)
        self.target_critic = EMATargetCritic(
            self.critic,
            tau=self.config['training']['tau']
        )
        
        # Count parameters
        world_params = sum(p.numel() for p in self.world_model.parameters())
        actor_params = sum(p.numel() for p in self.actor.parameters())
        critic_params = sum(p.numel() for p in self.critic.parameters())
        print(f"\nModel parameters:")
        print(f"  World model: {world_params:,}")
        print(f"  Actor: {actor_params:,}")
        print(f"  Critic: {critic_params:,}")
        print(f"  Total: {world_params + actor_params + critic_params:,}")
        
        # Create optimizers
        self.optimizer_model = torch.optim.Adam(
            self.world_model.parameters(),
            lr=self.config['training']['lr_model'],
            eps=self.config['optimization']['eps']
        )
        self.optimizer_actor = torch.optim.Adam(
            self.actor.parameters(),
            lr=self.config['training']['lr_actor'],
            eps=self.config['optimization']['eps']
        )
        self.optimizer_critic = torch.optim.Adam(
            self.critic.parameters(),
            lr=self.config['training']['lr_critic'],
            eps=self.config['optimization']['eps']
        )
        
        # Mixed precision training
        self.use_amp = self.config['optimization']['mixed_precision']
        self.scaler = torch.amp.GradScaler('cuda') if self.use_amp else None
        
        # Logger
        self.logger = Logger(self.config)
        
        # Training state
        self.global_step = 0
        self.episode_count = 0
        
        # For online collection
        self.current_obs = None
        self.current_h = None
        self.current_z = None
        
    def _set_seeds(self, seed: int):
        """Set random seeds for reproducibility."""
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
    
    # ========================================================================
    # Phase 1: Environment Collection
    # ========================================================================
    
    def collect_experience(self, num_steps: int):
        """
        Collect experience from environment using current policy.
        
        FROZEN[φ, θ, ψ]: Models are in eval mode, no gradient updates.
        
        Args:
            num_steps: Number of environment steps to collect (H_collect)
        """
        self.world_model.eval()
        self.actor.eval()
        
        # Initialize episode if needed
        if self.current_obs is None:
            self.current_obs = self.env.reset()
            # Initialize hidden state
            self.current_h = torch.zeros(
                1, self.world_model.hidden_size, device=self.device
            )
            # Get initial stochastic state from observation
            with torch.no_grad():
                obs_tensor = torch.from_numpy(self.current_obs).float().unsqueeze(0).to(self.device) / 255.0
                z_dist = self.world_model.encode(self.current_h, obs_tensor)
                self.current_z = z_dist.sample()
        
        for _ in range(num_steps):
            # Select action using current policy
            with torch.no_grad():
                action_idx, log_prob = self.actor.get_action(
                    self.current_h,
                    self.current_z,
                    deterministic=False  # Use stochastic policy for exploration
                )
                action_idx = action_idx.item()
            
            # Step environment
            next_obs, reward, done, info = self.env.step(action_idx)
            
            # Add to replay buffer
            self.replay_buffer.add(
                observation=self.current_obs,
                action=action_idx,
                reward=reward,
                done=done
            )
            
            # Update state for next step (using world model)
            if not done:
                with torch.no_grad():
                    # Encode next observation
                    next_obs_tensor = torch.from_numpy(next_obs).float().unsqueeze(0).to(self.device) / 255.0
                    z_post_dist = self.world_model.encode(self.current_h, next_obs_tensor)
                    next_z = z_post_dist.sample()
                    
                    # Update deterministic state
                    action_onehot = F.one_hot(
                        torch.tensor([action_idx], device=self.device),
                        num_classes=self.env.action_size
                    ).float()
                    next_h = self.world_model.dynamics(
                        self.current_h,
                        self.current_z,
                        action_onehot
                    )
                    
                    self.current_h = next_h
                    self.current_z = next_z
                    self.current_obs = next_obs
            else:
                # Episode ended, reset
                self.current_obs = self.env.reset()
                self.current_h = torch.zeros(
                    1, self.world_model.hidden_size, device=self.device
                )
                with torch.no_grad():
                    obs_tensor = torch.from_numpy(self.current_obs).float().unsqueeze(0).to(self.device) / 255.0
                    z_dist = self.world_model.encode(self.current_h, obs_tensor)
                    self.current_z = z_dist.sample()
                
                # Log episode info
                if 'episode' in info:
                    self.episode_count += 1
                    self.logger.log_episode(info['episode'], self.global_step, self.episode_count)
            
            self.global_step += 1
    
    # ========================================================================
    # Phase 2: World Model Learning
    # ========================================================================
    
    def train_world_model(self, num_updates: int = 1):
        """
        Train world model on replay buffer sequences.
        
        UPDATE[φ]: Update world model parameters.
        FROZEN[θ, ψ]: Actor-critic parameters frozen.
        
        Args:
            num_updates: Number of gradient updates to perform
        """
        if not self.replay_buffer.is_ready(self.config['training']['replay_min_size']):
            return  # Not enough data yet
        
        self.world_model.train()
        
        for _ in range(num_updates):
            # Sample batch of sequences
            batch = self.replay_buffer.sample_sequences(
                batch_size=self.config['training']['batch_size_model'],
                seq_length=self.config['training']['batch_length']
            )
            
            observations = batch['observations']  # (B, T, C, H, W)
            actions = batch['actions']  # (B, T, action_size)
            rewards = batch['rewards']  # (B, T)
            continues = batch['continues']  # (B, T)
            
            # Compute world model loss
            with torch.autocast(device_type='cuda', enabled=self.use_amp):
                losses = self.world_model.compute_loss(
                    observations, actions, rewards, continues
                )
                loss = losses['total_loss']
            
            # Backward pass
            self.optimizer_model.zero_grad()
            if self.use_amp:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer_model)
                torch.nn.utils.clip_grad_norm_(
                    self.world_model.parameters(),
                    self.config['optimization']['grad_clip']
                )
                self.scaler.step(self.optimizer_model)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.world_model.parameters(),
                    self.config['optimization']['grad_clip']
                )
                self.optimizer_model.step()
            
            # Log losses
            self.logger.log_world_model(losses, self.global_step)
    
    # ========================================================================
    # Phase 3: Behavior Learning (Actor-Critic)
    # ========================================================================
    
    def train_actor_critic(self, num_updates: int = 1):
        """
        Train actor and critic on imagined trajectories.
        
        UPDATE[θ, ψ]: Update actor and critic.
        FROZEN[φ]: World model frozen (used for imagination).
        
        Args:
            num_updates: Number of gradient updates
        """
        if not self.replay_buffer.is_ready(self.config['training']['replay_min_size']):
            return
        
        self.world_model.eval()  # Freeze world model
        self.actor.train()
        self.critic.train()
        
        for _ in range(num_updates):
            # Sample starting states for imagination
            starts = self.replay_buffer.sample_starts(
                batch_size=self.config['training']['batch_size_actor']
            )
            
            # Initialize states from starts (using world model encoder)
            with torch.no_grad():
                obs_start = starts['observations']  # (B, C, H, W)
                B = obs_start.shape[0]
                
                # Initialize h_0
                h_0 = torch.zeros(B, self.world_model.hidden_size, device=self.device)
                
                # Get z_0 from encoder
                z_dist_0 = self.world_model.encode(h_0, obs_start)
                z_0 = z_dist_0.sample()
            
            # Imagine trajectories (open-loop with prior)
            with torch.no_grad():
                imagined = self.world_model.imagine(
                    h_0=h_0,
                    z_0=z_0,
                    actor=self.actor,
                    horizon=self.config['training']['h_imagine']
                )
            
            h_seq = imagined['h']  # (B, H, hidden_size)
            z_seq = imagined['z']  # (B, H, stoch_size, discrete_size)
            rewards = imagined['reward']  # (B, H) in symlog space
            continues = imagined['continue']  # (B, H)
            log_probs = imagined['log_probs']  # (B, H)
            
            # ================================================================
            # Train Critic
            # ================================================================
            
            # Compute value estimates
            with torch.autocast(device_type='cuda', enabled=self.use_amp):
                values = self.critic.get_value(h_seq, z_seq)  # (B, H)
            
            # Get bootstrap value from target critic
            with torch.no_grad():
                # Use last state for bootstrap
                h_last = h_seq[:, -1]  # (B, hidden_size)
                z_last = z_seq[:, -1]  # (B, stoch_size, discrete_size)
                bootstrap = self.target_critic.get_value(h_last, z_last)  # (B,)
            
            # Compute λ-returns
            with torch.no_grad():
                lambda_returns = compute_lambda_returns(
                    rewards=rewards,
                    continues=continues,
                    values=values.detach(),
                    bootstrap=bootstrap,
                    gamma=self.config['training']['gamma'],
                    lambda_=self.config['training']['lambda_']
                )  # (B, H)
            
            # Critic loss
            with torch.autocast(device_type='cuda', enabled=self.use_amp):
                critic_loss = self.critic.compute_loss(
                    h_seq.detach(),  # Stop gradient to world model
                    z_seq.detach(),
                    lambda_returns
                )
            
            # Update critic
            self.optimizer_critic.zero_grad()
            if self.use_amp:
                self.scaler.scale(critic_loss).backward()
                self.scaler.unscale_(self.optimizer_critic)
                torch.nn.utils.clip_grad_norm_(
                    self.critic.parameters(),
                    self.config['optimization']['grad_clip']
                )
                self.scaler.step(self.optimizer_critic)
                self.scaler.update()
            else:
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.critic.parameters(),
                    self.config['optimization']['grad_clip']
                )
                self.optimizer_critic.step()
            
            # ================================================================
            # Train Actor
            # ================================================================
            
            # Re-compute actions and log probs (with gradients)
            with torch.autocast(device_type='cuda', enabled=self.use_amp):
                # Flatten batch and time for actor forward
                h_flat = h_seq.reshape(-1, h_seq.shape[-1])  # (B*H, hidden_size)
                z_flat = z_seq.reshape(-1, *z_seq.shape[2:])  # (B*H, stoch_size, discrete_size)
                
                action_dist = self.actor(h_flat, z_flat)
                
                # Get actions from imagined trajectory
                actions_flat = imagined['actions'].reshape(-1, self.env.action_size)  # (B*H, action_size)
                action_indices = torch.argmax(actions_flat, dim=-1)  # (B*H,)
                
                new_log_probs = action_dist.log_prob(action_indices)  # (B*H,)
                entropy = action_dist.entropy()  # (B*H,)
                
                # Reshape back
                new_log_probs = new_log_probs.reshape(B, -1)  # (B, H)
                entropy = entropy.reshape(B, -1)  # (B, H)
            
            # Compute advantages
            with torch.no_grad():
                advantages = lambda_returns - values.detach()
                # Normalize advantages
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            
            # Actor loss
            with torch.autocast(device_type='cuda', enabled=self.use_amp):
                actor_loss = compute_actor_loss(
                    log_probs=new_log_probs,
                    advantages=advantages,
                    entropy=entropy,
                    entropy_scale=self.config['training']['entropy_scale']
                )
            
            # Update actor
            self.optimizer_actor.zero_grad()
            if self.use_amp:
                self.scaler.scale(actor_loss).backward()
                self.scaler.unscale_(self.optimizer_actor)
                torch.nn.utils.clip_grad_norm_(
                    self.actor.parameters(),
                    self.config['optimization']['grad_clip']
                )
                self.scaler.step(self.optimizer_actor)
                self.scaler.update()
            else:
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.actor.parameters(),
                    self.config['optimization']['grad_clip']
                )
                self.optimizer_actor.step()
            
            # Log actor-critic losses
            self.logger.log_actor_critic({
                'actor_loss': actor_loss.item(),
                'critic_loss': critic_loss.item(),
                'mean_value': values.mean().item(),
                'mean_return': lambda_returns.mean().item(),
                'mean_advantage': advantages.mean().item(),
                'mean_entropy': entropy.mean().item()
            }, self.global_step)
            
            # Update target critic (EMA)
            self.target_critic.update(self.critic)
    
    # ========================================================================
    # Main Training Loop
    # ========================================================================
    
    def train(self):
        """
        Main training loop (Algorithm 1 from paper).
        """
        print("\n" + "="*60)
        print("Starting DreamerV3 Training")
        print("="*60)
        
        total_steps = self.config['training']['total_steps']
        h_collect = self.config['training']['h_collect']
        train_ratio = self.config['training']['train_ratio']
        
        # Progress bar
        pbar = tqdm(total=total_steps, desc="Training")
        
        while self.global_step < total_steps:
            # Phase 1: Collect experience
            self.collect_experience(num_steps=h_collect)
            pbar.update(h_collect)
            
            # Phase 2: Train world model
            self.train_world_model(num_updates=train_ratio)
            
            # Phase 3: Train actor-critic
            self.train_actor_critic(num_updates=train_ratio)
            
            # Logging
            if self.global_step % self.config['logging']['log_every'] == 0:
                self.logger.flush(self.global_step)
            
            # Evaluation
            if self.global_step % self.config['logging']['eval_every'] == 0:
                self.evaluate()
            
            # Save checkpoint
            if self.global_step % self.config['logging']['save_every'] == 0:
                self.save_checkpoint()
        
        pbar.close()
        print("\nTraining completed!")
        self.env.close()
    
    # ========================================================================
    # Evaluation
    # ========================================================================
    
    def evaluate(self, num_episodes: int = None):
        """
        Evaluate current policy.
        
        Args:
            num_episodes: Number of episodes to evaluate (default: from config)
        """
        if num_episodes is None:
            num_episodes = self.config['logging']['num_eval_episodes']
        
        self.world_model.eval()
        self.actor.eval()
        
        eval_rewards = []
        eval_lengths = []
        eval_max_x = []
        eval_success = []
        
        for _ in range(num_episodes):
            obs = self.env.reset()
            h = torch.zeros(1, self.world_model.hidden_size, device=self.device)
            
            with torch.no_grad():
                obs_tensor = torch.from_numpy(obs).float().unsqueeze(0).to(self.device) / 255.0
                z_dist = self.world_model.encode(h, obs_tensor)
                z = z_dist.sample()
            
            episode_reward = 0
            episode_length = 0
            done = False
            
            while not done:
                # Select action (deterministic for evaluation)
                with torch.no_grad():
                    action_idx, _ = self.actor.get_action(h, z, deterministic=True)
                    action_idx = action_idx.item()
                
                # Step environment
                obs, reward, done, info = self.env.step(action_idx)
                episode_reward += reward
                episode_length += 1
                
                if not done:
                    with torch.no_grad():
                        obs_tensor = torch.from_numpy(obs).float().unsqueeze(0).to(self.device) / 255.0
                        z_dist = self.world_model.encode(h, obs_tensor)
                        z = z_dist.sample()
                        
                        action_onehot = F.one_hot(
                            torch.tensor([action_idx], device=self.device),
                            num_classes=self.env.action_size
                        ).float()
                        h = self.world_model.dynamics(h, z, action_onehot)
            
            eval_rewards.append(episode_reward)
            eval_lengths.append(episode_length)
            if 'episode' in info:
                eval_max_x.append(info['episode']['max_x_pos'])
                eval_success.append(info['episode']['flag_get'])
        
        # Log evaluation results
        self.logger.log_evaluation({
            'mean_reward': np.mean(eval_rewards),
            'mean_length': np.mean(eval_lengths),
            'mean_max_x': np.mean(eval_max_x),
            'success_rate': np.mean(eval_success)
        }, self.global_step)
        
        print(f"\n[Eval @ step {self.global_step}]")
        print(f"  Mean reward: {np.mean(eval_rewards):.2f}")
        print(f"  Mean length: {np.mean(eval_lengths):.1f}")
        print(f"  Mean max X: {np.mean(eval_max_x):.1f}")
        print(f"  Success rate: {np.mean(eval_success)*100:.1f}%")

        # Reset collection state after evaluation
        # The environment is left in a 'done' state after eval, which breaks collection
        self.current_obs = None
        self.current_h = None
        self.current_z = None
    
    # ========================================================================
    # Checkpointing
    # ========================================================================
    
    def save_checkpoint(self):
        """Save model checkpoint."""
        checkpoint_dir = Path(self.config['logging']['log_dir']) / 'checkpoints'
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint_path = checkpoint_dir / f'checkpoint_{self.global_step}.pt'
        
        torch.save({
            'global_step': self.global_step,
            'episode_count': self.episode_count,
            'world_model': self.world_model.state_dict(),
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'target_critic': self.target_critic.target_critic.state_dict(),
            'optimizer_model': self.optimizer_model.state_dict(),
            'optimizer_actor': self.optimizer_actor.state_dict(),
            'optimizer_critic': self.optimizer_critic.state_dict(),
        }, checkpoint_path)
        
        print(f"\nCheckpoint saved: {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.global_step = checkpoint['global_step']
        self.episode_count = checkpoint['episode_count']
        self.world_model.load_state_dict(checkpoint['world_model'])
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.target_critic.target_critic.load_state_dict(checkpoint['target_critic'])
        self.optimizer_model.load_state_dict(checkpoint['optimizer_model'])
        self.optimizer_actor.load_state_dict(checkpoint['optimizer_actor'])
        self.optimizer_critic.load_state_dict(checkpoint['optimizer_critic'])
        
        print(f"Checkpoint loaded: {checkpoint_path}")
        print(f"  Resuming from step {self.global_step}")


# ============================================================================
# Entry Point
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train DreamerV3 on Super Mario Bros")
    parser.add_argument(
        '--config',
        type=str,
        default='configs/mario_config.yaml',
        help='Path to config file'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        default=None,
        help='Path to checkpoint to resume from'
    )
    
    args = parser.parse_args()
    
    # Create trainer
    trainer = DreamerV3Trainer(args.config)
    
    # Load checkpoint if provided
    if args.checkpoint:
        trainer.load_checkpoint(args.checkpoint)
    
    # Start training
    trainer.train()
