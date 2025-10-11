"""
Replay Buffer for DreamerV3.

Stores sequences of (observation, action, reward, continue) tuples.
Supports efficient sampling of random sequences for world model training.

Key features:
- Circular buffer for memory efficiency
- Sequential sampling (returns chunks of consecutive timesteps)
- Handles episode boundaries correctly
"""

import numpy as np
import torch
from typing import Dict, Tuple, Optional
from collections import deque
import json
import os
from PIL import Image
import cv2


class ReplayBuffer:
    """
    Replay buffer for storing and sampling experience sequences.
    
    Unlike standard RL replay buffers that sample individual transitions,
    this buffer samples sequences (chunks) of consecutive transitions.
    This is necessary for training the recurrent world model (RSSM).
    
    Storage format:
        - observations: (capacity, C, H, W)
        - actions: (capacity, action_size) - one-hot
        - rewards: (capacity,)
        - continues: (capacity,) - 1.0 for non-terminal, 0.0 for terminal
        - episode_ids: (capacity,) - which episode each transition belongs to
    """
    
    def __init__(
        self,
        capacity: int,
        observation_shape: Tuple[int, ...],
        action_size: int,
        device: torch.device = torch.device('cpu')
    ):
        """
        Args:
            capacity: Maximum number of timesteps to store
            observation_shape: Shape of observations (C, H, W)
            action_size: Number of discrete actions
            device: Device for tensors
        """
        self.capacity = capacity
        self.observation_shape = observation_shape
        self.action_size = action_size
        self.device = device
        
        # Storage buffers (numpy for efficiency, convert to torch when sampling)
        self.observations = np.zeros((capacity, *observation_shape), dtype=np.uint8)
        self.actions = np.zeros((capacity, action_size), dtype=np.float32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.continues = np.zeros(capacity, dtype=np.float32)
        self.episode_ids = np.full(capacity, -1, dtype=np.int32)  # -1 indicates empty slot
        
        # Pointers
        self.idx = 0  # Current write position
        self.size = 0  # Current number of stored timesteps
        self.current_episode_id = 0
        
        # Episode tracking (for sequential sampling)
        self.episode_lengths = []  # Length of each completed episode
        self.episode_start_indices = []  # Start index of each episode in buffer
    
    def add(
        self,
        observation: np.ndarray,
        action: int,
        reward: float,
        done: bool
    ):
        """
        Add a single timestep to the buffer.
        
        Args:
            observation: Observation array, shape: (C, H, W), dtype: uint8
            action: Action index
            reward: Scalar reward
            done: Whether episode terminated
        """
        # Convert action to one-hot
        action_onehot = np.zeros(self.action_size, dtype=np.float32)
        action_onehot[action] = 1.0
        
        # Store transition
        self.observations[self.idx] = observation
        self.actions[self.idx] = action_onehot
        self.rewards[self.idx] = reward
        self.continues[self.idx] = 0.0 if done else 1.0
        self.episode_ids[self.idx] = self.current_episode_id
        
        # Update pointers
        self.idx = (self.idx + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        # print(self.size)
        
        # If episode done, start new episode
        if done:
            self.current_episode_id += 1
    
    def add_batch(
        self,
        observations: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        dones: np.ndarray
    ):
        """
        Add a batch of timesteps (for parallel environments).
        
        Args:
            observations: (B, C, H, W)
            actions: (B,) - action indices
            rewards: (B,)
            dones: (B,) - done flags
        """
        for obs, act, rew, done in zip(observations, actions, rewards, dones):
            self.add(obs, act, rew, done)
    
    def sample_sequences(
        self,
        batch_size: int,
        seq_length: int
    ) -> Dict[str, torch.Tensor]:
        """
        Sample random sequences from the buffer.
        
        Each sequence is a chunk of consecutive timesteps from a single episode.
        
        Args:
            batch_size: Number of sequences to sample
            seq_length: Length of each sequence (T)
            
        Returns:
            Dictionary containing:
                - observations: (B, T, C, H, W)
                - actions: (B, T, action_size)
                - rewards: (B, T)
                - continues: (B, T)
        """
        # Find valid start indices
        # A valid start index allows sampling seq_length consecutive steps
        # from the same episode
        valid_indices = []
        
        for i in range(self.size):
            # Check if we can sample seq_length steps starting from i
            end_idx = i + seq_length
            
            # Make sure we don't wrap around buffer
            if end_idx > self.size:
                continue
            
            # Make sure all steps are from same episode
            episode_id = self.episode_ids[i]
            same_episode = np.all(
                self.episode_ids[i:end_idx] == episode_id
            )
            
            if same_episode:
                valid_indices.append(i)
        
        if len(valid_indices) < batch_size:
            raise ValueError(
                f"Not enough valid sequences. "
                f"Need {batch_size}, but only {len(valid_indices)} available. "
                f"Buffer size: {self.size}, Sequence length: {seq_length}"
            )
        
        # Sample random start indices
        start_indices = np.random.choice(valid_indices, size=batch_size, replace=True)
        
        # Collect sequences
        obs_batch = []
        act_batch = []
        rew_batch = []
        cont_batch = []
        
        for start_idx in start_indices:
            end_idx = start_idx + seq_length
            
            obs_batch.append(self.observations[start_idx:end_idx])
            act_batch.append(self.actions[start_idx:end_idx])
            rew_batch.append(self.rewards[start_idx:end_idx])
            cont_batch.append(self.continues[start_idx:end_idx])
        
        # Stack and convert to torch tensors
        observations = torch.from_numpy(np.stack(obs_batch)).float() / 255.0  # Normalize to [0,1]
        actions = torch.from_numpy(np.stack(act_batch))
        rewards = torch.from_numpy(np.stack(rew_batch))
        continues = torch.from_numpy(np.stack(cont_batch))
        
        # Move to device
        observations = observations.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        continues = continues.to(self.device)
        
        return {
            'observations': observations,
            'actions': actions,
            'rewards': rewards,
            'continues': continues
        }
    
    def sample_starts(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """
        Sample random starting states for imagination.
        
        Returns single timesteps that can be used as initial states
        for imagination rollouts.
        
        Args:
            batch_size: Number of start states to sample
            
        Returns:
            Dictionary with single-timestep tensors:
                - observations: (B, C, H, W)
                - actions: (B, action_size)  # Action that led to this state
                - rewards: (B,)
                - continues: (B,)
        """
        # Sample random indices
        indices = np.random.randint(0, self.size, size=batch_size)
        
        # Collect data
        observations = torch.from_numpy(
            self.observations[indices]
        ).float() / 255.0
        actions = torch.from_numpy(self.actions[indices])
        rewards = torch.from_numpy(self.rewards[indices])
        continues = torch.from_numpy(self.continues[indices])
        
        # Move to device
        observations = observations.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        continues = continues.to(self.device)
        
        return {
            'observations': observations,
            'actions': actions,
            'rewards': rewards,
            'continues': continues
        }
    
    def __len__(self) -> int:
        """Return current size of buffer."""
        return self.size
    
    def is_ready(self, min_size: int) -> bool:
        """Check if buffer has enough samples for training."""
        return self.size >= min_size

    def save_episode(self, episode_id: int, save_dir: str = "logs/train_eps", fps: int = 30):
        """
        Debug function to save all transitions from a specific episode.

        Saves episode data to: {save_dir}/episode_{episode_id}/
        - observations as JPEG images: frame_000000.jpg, frame_000001.jpg, ...
        - observations as MP4 video: episode.mp4
        - observations as GIF animation: episode.gif
        - actions as JSON: actions.json (list of action indices)
        - rewards as JSON: rewards.json (list of floats)
        - continues as JSON: continues.json (list of floats)

        Args:
            episode_id: The episode ID to save
            save_dir: Base directory for saving episodes (default: logs/train_eps)
            fps: Frames per second for the video and GIF (default: 30)
        """
        # Find all indices belonging to this episode
        episode_mask = self.episode_ids[:self.size] == episode_id
        episode_indices = np.where(episode_mask)[0]

        if len(episode_indices) == 0:
            raise ValueError(
                f"Episode ID {episode_id} not found in buffer. "
                f"Available episode IDs: {np.unique(self.episode_ids[:self.size])}"
            )

        # Create output directory
        episode_dir = os.path.join(save_dir, f"episode_{episode_id}")
        os.makedirs(episode_dir, exist_ok=True)

        # Extract episode data
        episode_observations = self.observations[episode_indices]  # (T, C, H, W)
        episode_actions = self.actions[episode_indices]  # (T, action_size) one-hot
        episode_rewards = self.rewards[episode_indices]  # (T,)
        episode_continues = self.continues[episode_indices]  # (T,)

        # Convert one-hot actions to action indices
        action_indices = np.argmax(episode_actions, axis=1).tolist()

        # Prepare video writer
        video_path = os.path.join(episode_dir, "episode.mp4")
        first_obs = episode_observations[0]

        if first_obs.shape[0] == 1:
            # Grayscale
            height, width = first_obs.shape[1], first_obs.shape[2]
            is_color = False
        else:
            # RGB
            height, width = first_obs.shape[1], first_obs.shape[2]
            is_color = True

        # Initialize video writer (use mp4v codec)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(video_path, fourcc, fps, (width, height), is_color)

        # Save observations as JPEG images and add to video
        for t, obs in enumerate(episode_observations):
            # obs shape: (C, H, W), dtype: uint8
            # Convert to (H, W, C) for PIL and OpenCV
            if obs.shape[0] == 1:
                # Grayscale: (1, H, W) -> (H, W)
                img_array = obs[0]
                img = Image.fromarray(img_array, mode='L')
                # Convert to BGR for OpenCV (grayscale needs to be converted to BGR for color video)
                frame_bgr = cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR) if is_color else img_array
            else:
                # RGB: (C, H, W) -> (H, W, C)
                img_array = np.transpose(obs, (1, 2, 0))
                img = Image.fromarray(img_array, mode='RGB')
                # Convert RGB to BGR for OpenCV
                frame_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

            # Save JPEG
            img_path = os.path.join(episode_dir, f"frame_{t:06d}.jpg")
            img.save(img_path, quality=95)

            # Add frame to video
            video_writer.write(frame_bgr)

        # Release video writer
        video_writer.release()

        # Create GIF animation
        gif_path = os.path.join(episode_dir, "episode.gif")
        gif_frames = []

        for obs in episode_observations:
            if obs.shape[0] == 1:
                # Grayscale: (1, H, W) -> (H, W)
                img_array = obs[0]
                img = Image.fromarray(img_array, mode='L')
            else:
                # RGB: (C, H, W) -> (H, W, C)
                img_array = np.transpose(obs, (1, 2, 0))
                img = Image.fromarray(img_array, mode='RGB')

            gif_frames.append(img)

        # Save as GIF with duration in milliseconds (1000ms / fps)
        duration_ms = int(1000 / fps)
        gif_frames[0].save(
            gif_path,
            save_all=True,
            append_images=gif_frames[1:],
            duration=duration_ms,
            loop=0  # 0 means loop forever
        )

        # Save actions as JSON
        actions_path = os.path.join(episode_dir, "actions.json")
        with open(actions_path, 'w') as f:
            json.dump(action_indices, f, indent=2)

        # Save rewards as JSON
        rewards_path = os.path.join(episode_dir, "rewards.json")
        with open(rewards_path, 'w') as f:
            json.dump(episode_rewards.tolist(), f, indent=2)

        # Save continues as JSON
        continues_path = os.path.join(episode_dir, "continues.json")
        with open(continues_path, 'w') as f:
            json.dump(episode_continues.tolist(), f, indent=2)

        print(f"Saved episode {episode_id} ({len(episode_indices)} frames) to {episode_dir}")
        print(f"  - {len(episode_indices)} observation images")
        print(f"  - Video: {video_path}")
        print(f"  - GIF: {gif_path}")
        print(f"  - Actions: {actions_path}")
        print(f"  - Rewards: {rewards_path}")
        print(f"  - Continues: {continues_path}")

        return episode_dir


# ============================================================================
# Episode Buffer (for online collection)
# ============================================================================

class EpisodeBuffer:
    """
    Temporary buffer for collecting a single episode before adding to replay.
    
    Used during environment interaction to collect full episodes,
    then batch-add them to the main replay buffer.
    """
    
    def __init__(self):
        self.observations = []
        self.actions = []
        self.rewards = []
        self.dones = []
    
    def add(
        self,
        observation: np.ndarray,
        action: int,
        reward: float,
        done: bool
    ):
        """Add a single timestep."""
        self.observations.append(observation)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
    
    def get_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Get all collected data as numpy arrays.
        
        Returns:
            observations, actions, rewards, dones (all np.ndarray)
        """
        observations = np.array(self.observations)
        actions = np.array(self.actions)
        rewards = np.array(self.rewards)
        dones = np.array(self.dones)
        
        return observations, actions, rewards, dones
    
    def clear(self):
        """Clear the buffer."""
        self.observations = []
        self.actions = []
        self.rewards = []
        self.dones = []
    
    def __len__(self) -> int:
        """Return number of timesteps collected."""
        return len(self.observations)