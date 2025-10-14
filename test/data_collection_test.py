#!/usr/bin/env python3
"""
Data Collection and Sampling Test for DreamerV3 Replay Buffer

Purpose:
========
This test verifies that the new length-weighted episode sampling strategy with
circular buffer produces temporally continuous sequences. Specifically, it tests
that sample_sequences() returns valid data chunks that can be concatenated across
batches to reconstruct complete episodes.

Test Strategy:
==============
1. Replicate the exact data collection pipeline from trainer.py:
   - Create environment with same preprocessing
   - Collect experience using random actions
   - Store in ReplayBuffer with same configuration

2. Sample sequences using sample_sequences(B, T):
   - Returns batch shape: (B, T, C, H, W)
   - Each of B sequences is tracked independently

3. For each sequence in the batch (b=0, 1, ..., B-1):
   - Collect frames across multiple sample_sequences() calls
   - Detect episode boundaries using is_first and dones flags
   - When a complete episode is found (is_first=True to done=True):
     * Save it as GIF animation: test_output/b{b}_ep{i}_len{len}.gif
   - Repeat until we collect 3 complete episodes per sequence

Expected Output:
================
For batch_size=4, we should get 12 GIF animations total:
- test_output/b0_ep0_len{N}.gif  # First sequence, first episode
- test_output/b0_ep1_len{N}.gif  # First sequence, second episode
- test_output/b0_ep2_len{N}.gif  # First sequence, third episode
- test_output/b1_ep0_len{N}.gif  # Second sequence, first episode
- ... (and so on for b2, b3)

Each GIF should show a complete, temporally continuous Mario episode from
start to finish (playing at 15 fps, half the original speed), confirming that:
- Episodes are stored correctly in the circular buffer
- Length-weighted sampling works properly
- is_first and done flags correctly mark episode boundaries
- Frames are concatenated correctly across chunks

If GIFs show discontinuities or weird transitions, it indicates bugs in:
- Circular buffer wrapping (_slice_circular)
- Episode boundary tracking (is_first flag)
- Rollout state management (RollState.extract_chunk)
"""

import sys
import os
import numpy as np
import torch
import yaml
import cv2
from pathlib import Path
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from algorithms.dreamer_v3.training.replay_buffer import ReplayBuffer
from algorithms.dreamer_v3.envs.mario_env import make_mario_env


class EpisodeCollector:
    """
    Collects frames from sampled sequences and assembles complete episodes.

    Tracks state for one sequence in the batch (one value of b in the B dimension).
    """

    def __init__(self, sequence_id: int, target_episodes: int = 3):
        """
        Args:
            sequence_id: Which sequence in batch this collector tracks (0 to B-1)
            target_episodes: How many complete episodes to collect
        """
        self.sequence_id = sequence_id
        self.target_episodes = target_episodes

        # State tracking
        self.completed_episodes = []  # List of (frames, length) tuples
        self.current_episode_frames = []  # Frames for episode being collected
        self.is_collecting = False  # Whether we're inside an episode
        self.num_completed = 0  # Number of completed episodes

    def process_chunk(self, observations: torch.Tensor, is_first: torch.Tensor,
                      dones: torch.Tensor) -> bool:
        """
        Process a chunk of timesteps for this sequence.

        Args:
            observations: (T, C, H, W) tensor of observations for this sequence
            is_first: (T,) bool tensor marking episode starts
            dones: (T,) bool tensor marking episode ends

        Returns:
            True if we've collected enough episodes, False otherwise
        """
        T = observations.shape[0]

        for t in range(T):
            frame = observations[t]  # (C, H, W)
            is_first_flag = is_first[t].item()
            done_flag = dones[t].item()

            # Check for episode start
            if is_first_flag:
                # If we were collecting, that episode is incomplete (boundary cut)
                # Discard it and start fresh
                if self.is_collecting and len(self.current_episode_frames) > 0:
                    print(f"  [Seq {self.sequence_id}] Discarding incomplete episode "
                          f"({len(self.current_episode_frames)} frames)")

                # Start new episode
                self.current_episode_frames = [frame]
                self.is_collecting = True
            elif self.is_collecting:
                # Continue collecting current episode
                self.current_episode_frames.append(frame)

            # Check for episode end
            if done_flag and self.is_collecting:
                # Complete episode found!
                episode_length = len(self.current_episode_frames)
                self.completed_episodes.append(
                    (self.current_episode_frames, episode_length)
                )
                self.num_completed += 1

                print(f"  [Seq {self.sequence_id}] Completed episode {self.num_completed}: "
                      f"{episode_length} frames")

                # Reset for next episode
                self.current_episode_frames = []
                self.is_collecting = False

                # Check if we've collected enough
                if self.num_completed >= self.target_episodes:
                    return True

        return False

    def save_episodes(self, output_dir: Path):
        """
        Save all completed episodes as GIF animations.

        Args:
            output_dir: Directory to save GIFs
        """
        from PIL import Image

        for ep_idx, (frames, length) in enumerate(self.completed_episodes):
            # Stack frames: list of (C, H, W) -> (T, C, H, W)
            video_tensor = torch.stack(frames, dim=0)  # (T, C, H, W)

            # Convert to numpy: (T, C, H, W) -> (T, H, W, C)
            video_np = video_tensor.permute(0, 2, 3, 1).cpu().numpy()  # (T, H, W, C)

            # Convert float [0,1] to uint8 [0,255]
            video_np = (video_np * 255.0).astype(np.uint8)

            # Determine if grayscale or RGB
            T, H, W, C = video_np.shape

            # Create GIF filename
            filename = f"b{self.sequence_id}_ep{ep_idx}_len{length}.gif"
            filepath = output_dir / filename

            # Convert frames to PIL Images
            gif_frames = []
            for t in range(T):
                frame = video_np[t]  # (H, W, C)

                if C == 1:
                    # Grayscale: (H, W, 1) -> (H, W)
                    frame = frame[:, :, 0]
                    img = Image.fromarray(frame, mode='L')
                elif C == 3:
                    # RGB
                    img = Image.fromarray(frame, mode='RGB')
                else:
                    raise ValueError(f"Unexpected number of channels: {C}")

                gif_frames.append(img)

            # Save as GIF
            # FPS = 15 (half of original 30), so duration = 1000ms / 15 ≈ 67ms per frame
            fps = 15  # Playback speed (half of original 30 fps)
            duration_ms = int(1000 / fps)

            gif_frames[0].save(
                str(filepath),
                save_all=True,
                append_images=gif_frames[1:],
                duration=duration_ms,
                loop=0,  # 0 means loop forever
                optimize=False  # Set to True to reduce file size (but slower)
            )

            print(f"    Saved: {filepath}")


def collect_random_experience(env, replay_buffer, num_steps: int):
    """
    Collect experience using random actions (mimics trainer.py's collect_experience).

    This replicates the exact data collection logic from trainer.py, but uses
    random actions instead of policy actions since we don't need a trained model.

    Args:
        env: Mario environment
        replay_buffer: ReplayBuffer instance
        num_steps: Number of timesteps to collect
    """
    obs = env.reset()

    for step in tqdm(range(num_steps), desc="Collecting experience"):
        # Random action (trainer uses actor.get_action, we use random)
        action = np.random.randint(0, env.action_size)

        # Step environment (identical to trainer.py line 185)
        next_obs, reward, done, info = env.step(action)

        # Add to replay buffer (identical to trainer.py line 188-193)
        replay_buffer.add(
            observation=obs,
            action=action,
            reward=reward,
            done=done
        )

        if done:
            # Reset environment (identical to trainer.py line 219)
            obs = env.reset()

            # Log episode info if available
            if 'episode' in info:
                ep_info = info['episode']
                print(f"\n  Episode complete: reward={ep_info['reward']:.1f}, "
                      f"length={ep_info['length']}, max_x={ep_info['max_x_pos']:.0f}")
        else:
            obs = next_obs


def test_sampling_pipeline(config_path: str = 'configs/dreamerv3_config.yaml'):
    """
    Main test function that replicates the training pipeline.

    Pipeline (from trainer.py):
    1. Create environment (trainer.__init__, line 69)
    2. Create replay buffer (trainer.__init__, line 75)
    3. Collect experience (trainer.collect_experience, line 149)
    4. Sample sequences (trainer.train_world_model, line 256)

    Args:
        config_path: Path to configuration file
    """
    print("="*80)
    print("Data Collection and Sampling Test")
    print("="*80)
    print()

    # ============================================================================
    # Step 1: Load config and create environment (mimics trainer.__init__)
    # ============================================================================

    print("[1/5] Loading configuration...")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Device: {device}")
    print(f"  Config: {config_path}")

    # Create environment (identical to trainer.py line 69)
    print("\n[2/5] Creating environment...")
    env = make_mario_env(config)
    print(f"  Environment: {config['env']['name']}")
    print(f"  Action space: {env.action_size}")
    print(f"  Observation shape: {env.observation_shape}")

    # ============================================================================
    # Step 2: Create replay buffer (mimics trainer.__init__ line 75)
    # ============================================================================

    print("\n[3/5] Creating replay buffer...")
    replay_buffer = ReplayBuffer(
        capacity=config['training']['replay_capacity'],
        observation_shape=env.observation_shape,
        action_size=env.action_size,
        device=device
    )
    print(f"  Capacity: {config['training']['replay_capacity']} timesteps")
    print(f"  Minimum size: {config['training']['replay_min_size']} timesteps")

    # ============================================================================
    # Step 3: Collect experience (mimics trainer.collect_experience)
    # ============================================================================

    print("\n[4/5] Collecting experience...")
    # Collect enough data to ensure we have several complete episodes
    num_collect_steps = config['training']['replay_min_size'] * 2
    print(f"  Collecting {num_collect_steps} timesteps...")

    collect_random_experience(env, replay_buffer, num_collect_steps)

    print(f"\n  Replay buffer stats:")
    print(f"    Total timesteps: {len(replay_buffer)}")
    print(f"    Complete episodes: {len(replay_buffer.episodes)}")
    print(f"    Episode IDs: {list(replay_buffer.episodes.keys())[:10]}...")

    # ============================================================================
    # Step 4: Sample sequences and reconstruct episodes (mimics trainer.train_world_model)
    # ============================================================================

    print("\n[5/5] Sampling sequences and reconstructing episodes...")

    # Sampling parameters (from config and trainer.py line 256-258)
    batch_size = config['training']['batch_size_model']  # B
    seq_length = config['training']['batch_length']  # T

    print(f"  Batch size (B): {batch_size}")
    print(f"  Sequence length (T): {seq_length}")
    print(f"  Batch shape: ({batch_size}, {seq_length}, {env.observation_shape})")
    print(f"  Target: 3 complete episodes per sequence")
    print()

    # Create collectors for each sequence in the batch
    collectors = [
        EpisodeCollector(sequence_id=b, target_episodes=3)
        for b in range(batch_size)
    ]

    # Sample batches until all collectors have enough episodes
    max_batches = 1000  # Safety limit
    batch_count = 0

    pbar = tqdm(total=batch_size * 3, desc="Collecting episodes")

    while batch_count < max_batches:
        # Sample batch (identical to trainer.py line 256)
        batch = replay_buffer.sample_sequences(
            batch_size=batch_size,
            seq_length=seq_length
        )

        observations = batch['observations']  # (B, T, C, H, W)
        is_first = batch['is_first']  # (B, T)
        dones = batch['dones']  # (B, T)

        batch_count += 1

        # Process each sequence in the batch
        all_done = True
        for b in range(batch_size):
            collector = collectors[b]

            if collector.num_completed < collector.target_episodes:
                # Extract this sequence's data
                obs_seq = observations[b]  # (T, C, H, W)
                is_first_seq = is_first[b]  # (T,)
                dones_seq = dones[b]  # (T,)

                # Process chunk
                is_done = collector.process_chunk(obs_seq, is_first_seq, dones_seq)

                if not is_done:
                    all_done = False
                else:
                    pbar.update(1)

        # Update progress for completed collectors
        pbar.n = sum(c.num_completed for c in collectors)
        pbar.refresh()

        if all_done:
            break

    pbar.close()

    print(f"\n  Processed {batch_count} batches")
    print(f"  Total sampled timesteps: {batch_count * batch_size * seq_length}")

    # ============================================================================
    # Step 5: Save GIF animations
    # ============================================================================

    print("\n[6/6] Saving GIF animations...")
    output_dir = Path("test_output")
    output_dir.mkdir(exist_ok=True)

    for collector in collectors:
        if collector.num_completed > 0:
            collector.save_episodes(output_dir)

    # ============================================================================
    # Summary
    # ============================================================================

    print("\n" + "="*80)
    print("Test Complete!")
    print("="*80)
    print(f"\nResults saved to: {output_dir.absolute()}")
    print(f"Total GIFs: {sum(c.num_completed for c in collectors)}")
    print(f"Playback speed: 15 fps (half of original 30 fps)")
    print()
    print("Expected naming format:")
    print("  b{sequence_id}_ep{episode_num}_len{length}.gif")
    print()
    print("Verification checklist:")
    print("  [ ] GIFs play correctly in browser/viewer")
    print("  [ ] Mario's movement is smooth and continuous")
    print("  [ ] No sudden jumps or discontinuities")
    print("  [ ] Episodes start from beginning (Mario at spawn point)")
    print("  [ ] Episodes end naturally (death, timeout, or flag)")
    print("  [ ] Playback speed is appropriate (not too fast)")
    print()
    print("If GIFs show discontinuities, check:")
    print("  - _slice_circular() in replay_buffer.py (circular buffer wrapping)")
    print("  - _episode_to_segment() (episode extraction)")
    print("  - RollState.extract_chunk() (chunk assembly)")
    print("  - is_first flag logic (episode boundary marking)")
    print()

    # Cleanup
    env.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Test data collection and sampling pipeline"
    )
    parser.add_argument(
        '--config',
        type=str,
        default='configs/dreamerv3_config.yaml',
        help='Path to config file'
    )

    args = parser.parse_args()

    try:
        test_sampling_pipeline(args.config)
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user.")
    except Exception as e:
        print(f"\n\nError during test: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
