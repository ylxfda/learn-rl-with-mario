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
from typing import Dict, Tuple, Optional, List
from collections import deque
from dataclasses import dataclass, field
import json
import os
from PIL import Image
import cv2


@dataclass
class EpisodeInfo:
    """Metadata for a completed episode stored in the buffer."""
    episode_id: int
    start_idx: int
    length: int


@dataclass
class Segment:
    """Contiguous chunk of experience belonging to a single episode."""
    observations: np.ndarray  # (L, C, H, W)
    actions: np.ndarray       # (L, action_size)
    rewards: np.ndarray       # (L,)
    continues: np.ndarray     # (L,)
    is_first: np.ndarray      # (L,) -> bool (marks episode boundary at t==0)

    @property
    def length(self) -> int:
        return self.observations.shape[0]


@dataclass
class RollState:
    """Rolling buffer that keeps enough steps to serve training chunks."""
    segments: deque = field(default_factory=deque)
    offset: int = 0  # Number of consumed steps within the first segment
    total_steps: int = 0  # Available steps after accounting for offset

    def available_steps(self) -> int:
        return self.total_steps

    def append_segment(self, segment: Segment) -> None:
        if segment.length == 0:
            return
        # Keep segments in temporal order; we only append full episodes.
        self.segments.append(segment)
        self.total_steps += segment.length

    def extract_chunk(self, length: int) -> Dict[str, np.ndarray]:
        if length > self.total_steps:
            raise ValueError(
                f"Requested {length} steps, but only {self.total_steps} available."
            )

        obs_parts: List[np.ndarray] = []
        act_parts: List[np.ndarray] = []
        rew_parts: List[np.ndarray] = []
        cont_parts: List[np.ndarray] = []
        first_parts: List[np.ndarray] = []

        remaining = length

        while remaining > 0:
            if not self.segments:
                raise RuntimeError("RollState is out of segments unexpectedly.")

            segment = self.segments[0]
            start = self.offset
            take = min(remaining, segment.length - start)

            obs_parts.append(segment.observations[start:start + take])
            act_parts.append(segment.actions[start:start + take])
            rew_parts.append(segment.rewards[start:start + take])
            cont_parts.append(segment.continues[start:start + take])
            first_parts.append(segment.is_first[start:start + take])

            remaining -= take

            if start + take == segment.length:
                # Current segment fully consumed; drop it and move to the next
                self.segments.popleft()
                self.offset = 0
            else:
                # Leave partially consumed segment at front with updated offset
                self.offset = start + take

        self.total_steps -= length

        def _concat(parts: List[np.ndarray]) -> np.ndarray:
            return np.concatenate(parts, axis=0) if len(parts) > 1 else parts[0]

        return {
            'observations': _concat(obs_parts),
            'actions': _concat(act_parts),
            'rewards': _concat(rew_parts),
            'continues': _concat(cont_parts),
            'is_first': _concat(first_parts)
        }


class ReplayBuffer:
    """
    Replay buffer for storing and sampling experience sequences.
    
    Unlike standard RL replay buffers that sample individual transitions,
    this buffer samples sequences (chunks) of consecutive transitions.
    This is necessary for training the recurrent world model (RSSM).
    
    Major design ideas:
      1. Environment interaction remains unchanged; we simply accumulate full episodes with distinct IDs and lengths.
      2. When training needs B rollouts, we sample episodes with probability proportional to their length and concatenate them
         (e.g., r_0 = ep_4 + ep_1 + ep_0). While building each rollout we mark the timestep where an episode starts via `is_first`.
      3. Once rollouts exist, we slice them along time into aligned chunks of shape (B, T, C, H, W) plus rewards/continues/is_first
         and feed those chunks to the GPU.
      4. Rollouts only keep enough material to serve the next chunk; if a rollout still lacks T steps after slicing, we append
         additional episodes on demand so that preparation can run alongside training.
      5. During training the `is_first` mask tells the world model whenever h must reset, while other timesteps carry the state forward.
      6. Because different batch elements hit `is_first` independently, downstream code maintains vectorized updates instead of loops.
    
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
        self.current_episode_length = 0
        self.current_episode_start_idx = 0

        # Completed episode metadata keyed by episode_id
        self.episodes: Dict[int, EpisodeInfo] = {}

        # Rolling batches cache for sequence sampling
        self.roll_states: Optional[List[RollState]] = None
        self.roll_batch_size: Optional[int] = None
    
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
        if self.current_episode_length == 0:
            self.current_episode_start_idx = self.idx

        if self.size == self.capacity:
            overwritten_episode_id = self.episode_ids[self.idx]
            if overwritten_episode_id in self.episodes:
                self.episodes.pop(overwritten_episode_id, None)

        # Convert action to one-hot
        action_onehot = np.zeros(self.action_size, dtype=np.float32)
        action_onehot[action] = 1.0
        
        # Store transition
        self.observations[self.idx] = observation
        self.actions[self.idx] = action_onehot
        self.rewards[self.idx] = reward
        self.continues[self.idx] = 0.0 if done else 1.0
        self.episode_ids[self.idx] = self.current_episode_id
        self.current_episode_length += 1
        
        # Update pointers
        self.idx = (self.idx + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        # print(self.size)
        
        # If episode done, start new episode
        if done:
            episode_info = EpisodeInfo(
                episode_id=self.current_episode_id,
                start_idx=self.current_episode_start_idx,
                length=self.current_episode_length
            )
            self.episodes[self.current_episode_id] = episode_info
            self.current_episode_id += 1
            self.current_episode_length = 0
    
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
        Sample sequences by concatenating full episodes into rolling buffers.

        Design ideas:
            1. Completed episodes are sampled with probability ∝ length, so longer episodes appear more often in the
               concatenated rollouts.
            2. Each rollout tracks `is_first` at the boundary where a new episode is appended.
            3. Aligned slices of length T form the actual training chunk handed to the GPU.
            4. After a chunk is emitted, rollouts top themselves up with additional episodes only if they fall short
               of the next T steps.

        Args:
            batch_size: Number of rollouts (B)
            seq_length: Number of timesteps per chunk (T)

        Returns:
            Dictionary containing tensors:
                - observations: (B, T, C, H, W)
                - actions: (B, T, action_size)
                - rewards: (B, T)
                - continues: (B, T)
                - dones: (B, T) -> bool tensor
                - is_first: (B, T) -> bool tensor marking episode starts                
        """
        if seq_length <= 0:
            raise ValueError("seq_length must be positive.")

        self._ensure_roll_states(batch_size)

        # Grow each roll until it has enough steps to emit a chunk
        for roll_state in self.roll_states:
            while roll_state.available_steps() < seq_length:
                if not self.episodes:
                    raise ValueError("No completed episodes available to extend rollouts.")
                # Length-weighted sampling so longer episodes contribute proportionally.
                episode_info = self._sample_episode_info()
                segment = self._episode_to_segment(episode_info)
                roll_state.append_segment(segment)

        obs_batch = []
        act_batch = []
        rew_batch = []
        cont_batch = []
        done_batch = []
        first_batch = []

        # Extract aligned chunks (size seq_length) from each roll
        for roll_state in self.roll_states:
            chunk = roll_state.extract_chunk(seq_length)
            obs_batch.append(chunk['observations'])
            act_batch.append(chunk['actions'])
            rew_batch.append(chunk['rewards'])
            cont_batch.append(chunk['continues'])
            # `continues` is 0.0 at terminal steps, so <= 0 flags done transitions.
            done_batch.append(chunk['continues'] <= 0.0)
            first_batch.append(chunk['is_first'])

        # Stack and convert to tensors (normalize observations to [0,1])
        observations = torch.from_numpy(np.stack(obs_batch)).float() / 255.0
        actions = torch.from_numpy(np.stack(act_batch)).float()
        rewards = torch.from_numpy(np.stack(rew_batch)).float()
        continues = torch.from_numpy(np.stack(cont_batch)).float()
        dones = torch.from_numpy(np.stack(done_batch)).bool()
        is_first = torch.from_numpy(np.stack(first_batch)).bool()

        # Move to target device
        observations = observations.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        continues = continues.to(self.device)
        dones = dones.to(self.device)
        is_first = is_first.to(self.device)

        return {
            'observations': observations,
            'actions': actions,
            'rewards': rewards,
            'continues': continues,
            'dones': dones,
            'is_first': is_first
        }

    def _ensure_roll_states(self, batch_size: int) -> None:
        """
        Ensure we have cached rollout state for the requested batch size.

        Reinitializes the per-roll buffers if the batch size changes so that
        each rollout can maintain its own queue of episode segments.

        Args:
            batch_size: Desired number of concurrent rollouts.
        """
        if self.roll_states is None or self.roll_batch_size != batch_size:
            # Reset cached state whenever batch size changes (e.g., warmup vs. training).
            self.roll_states = [RollState() for _ in range(batch_size)]
            self.roll_batch_size = batch_size

    def _sample_episode_info(self) -> EpisodeInfo:
        """
        Draw a completed episode according to length-proportional sampling.

        Longer episodes receive higher probability so that each stored timestep
        has equal chance of being used when we extend a rollout.

        Returns:
            EpisodeInfo describing the chosen episode.
        """
        if not self.episodes:
            raise ValueError("No completed episodes to sample.")

        episode_infos = list(self.episodes.values())
        lengths = np.array([info.length for info in episode_infos], dtype=np.float64)

        if lengths.sum() <= 0:
            raise ValueError("Episode lengths must be positive for sampling.")

        probabilities = lengths / lengths.sum()
        index = np.random.choice(len(episode_infos), p=probabilities)
        return episode_infos[index]

    def _episode_to_segment(self, episode_info: EpisodeInfo) -> Segment:
        """
        Materialize a contiguous segment for the given episode.

        Handles circular-buffer wraparound, converts all arrays into chronological
        order, and marks the first timestep with `is_first=True`.

        Args:
            episode_info: Metadata describing the episode to extract.

        Returns:
            Segment object containing observations/actions/rewards/etc.
        """
        start = episode_info.start_idx
        length = episode_info.length

        observations = self._slice_circular(self.observations, start, length)
        actions = self._slice_circular(self.actions, start, length)
        rewards = self._slice_circular(self.rewards, start, length)
        continues = self._slice_circular(self.continues, start, length)

        is_first = np.zeros(length, dtype=bool)
        if length > 0:
            is_first[0] = True

        return Segment(
            observations=observations,
            actions=actions,
            rewards=rewards,
            continues=continues,
            is_first=is_first
        )

    def _slice_circular(self, array: np.ndarray, start: int, length: int) -> np.ndarray:
        """
        Slice `length` items from a circular buffer beginning at `start`.

        If the range does not wrap, this is a straightforward slice; otherwise
        the tail and head of the buffer are concatenated to restore temporal order.

        Args:
            array: Underlying circular storage array.
            start: Starting index in the circular buffer.
            length: Number of items to extract.

        Returns:
            Numpy array containing the extracted slice in chronological order.
        """
        if length <= 0:
            return array[0:0].copy()

        end = start + length

        if end <= self.capacity:
            return array[start:end].copy()

        first_len = self.capacity - start
        second_len = length - first_len

        first_part = array[start:].copy()
        second_part = array[:second_len].copy()

        # Concatenate wrapped segments so callers always see chronological order.
        return np.concatenate((first_part, second_part), axis=0)
    
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
        episode_info = self.episodes.get(episode_id)

        if episode_info is None:
            raise ValueError(
                f"Episode ID {episode_id} not found in buffer metadata. "
                f"Available episode IDs: {list(self.episodes.keys())}"
            )

        # Create output directory
        episode_dir = os.path.join(save_dir, f"episode_{episode_id}")
        os.makedirs(episode_dir, exist_ok=True)

        # Extract episode data (contiguous ordering, handles circular buffer wrap)
        segment = self._episode_to_segment(episode_info)
        episode_observations = segment.observations  # (T, C, H, W)
        episode_actions = segment.actions  # (T, action_size) one-hot
        episode_rewards = segment.rewards  # (T,)
        episode_continues = segment.continues  # (T,)
        episode_length = episode_observations.shape[0]

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

        print(f"Saved episode {episode_id} ({episode_length} frames) to {episode_dir}")
        print(f"  - {episode_length} observation images")
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
