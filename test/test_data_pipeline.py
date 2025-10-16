"""
Test data pipeline by saving sampled episodes as GIF files.

This module extracts complete episodes (from is_first to done) from batched
sequences and saves them as GIFs for visual inspection.
"""

import torch
import numpy as np
from pathlib import Path
from typing import Dict
from PIL import Image


class EpisodeCollector:
    """
    Collects complete episodes from batched sequences and saves them as GIFs.

    Tracks episodes across batch dimension and saves 3 complete episodes per batch index.
    """

    def __init__(self, output_dir: str = "./test_output", episodes_per_batch: int = 3):
        """
        Args:
            output_dir: Directory to save GIF files
            episodes_per_batch: Number of complete episodes to collect per batch index
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.episodes_per_batch = episodes_per_batch

        # Track collection state
        # episode_buffers[b] = list of frames for current episode at batch index b
        self.episode_buffers = {}
        # episode_counts[b] = number of complete episodes collected at batch index b
        self.episode_counts = {}
        # in_episode[b] = whether we're currently in an episode at batch index b
        self.in_episode = {}

    def process_batch(self, batch: Dict[str, torch.Tensor]) -> bool:
        """
        Process a batch of sequences and extract complete episodes.

        Args:
            batch: Dictionary containing:
                - observations: (B, T, C, H, W) tensor
                - is_first: (B, T) tensor indicating episode starts
                - continues: (B, T) tensor (0.0 = done, 1.0 = continue)

        Returns:
            True if collection is complete (3 episodes per batch index), False otherwise
        """
        observations = batch['observations']  # (B, T, C, H, W)
        is_first = batch['is_first']  # (B, T)
        continues = batch['continues']  # (B, T)

        B, T = observations.shape[0], observations.shape[1]

        # Process each batch index
        for b in range(B):
            # Initialize tracking for this batch index if needed
            if b not in self.episode_counts:
                self.episode_counts[b] = 0
                self.episode_buffers[b] = []
                self.in_episode[b] = False

            # Skip if we've already collected enough episodes for this batch index
            if self.episode_counts[b] >= self.episodes_per_batch:
                continue

            # Process each timestep
            for t in range(T):
                is_first_step = is_first[b, t].item() > 0.5
                is_done = continues[b, t].item() < 0.5

                # Start new episode if this is a first step
                if is_first_step:
                    # Save previous episode if we were in one
                    if self.in_episode[b] and len(self.episode_buffers[b]) > 0:
                        self._save_episode(b, self.episode_buffers[b])
                        self.episode_counts[b] += 1
                        self.episode_buffers[b] = []

                    # Start collecting new episode
                    self.in_episode[b] = True
                    self.episode_buffers[b] = []

                # Add frame to current episode buffer if we're in an episode
                if self.in_episode[b]:
                    # Convert observation to numpy (C, H, W) -> (H, W, C)
                    frame = observations[b, t].cpu().numpy()
                    frame = np.transpose(frame, (1, 2, 0))  # (H, W, C)

                    # Convert from [0, 1] float to [0, 255] uint8
                    frame = np.clip(frame * 255.0, 0, 255).astype(np.uint8)

                    # Handle grayscale images (C=1)
                    if frame.shape[-1] == 1:
                        frame = np.repeat(frame, 3, axis=-1)

                    self.episode_buffers[b].append(frame)

                # End episode if done
                if is_done and self.in_episode[b]:
                    if len(self.episode_buffers[b]) > 0:
                        self._save_episode(b, self.episode_buffers[b])
                        self.episode_counts[b] += 1
                        self.episode_buffers[b] = []
                    self.in_episode[b] = False

                    # Stop processing this batch index if we have enough episodes
                    if self.episode_counts[b] >= self.episodes_per_batch:
                        break

        # Check if collection is complete for all batch indices
        return self._is_collection_complete(B)

    def _save_episode(self, batch_idx: int, frames: list):
        """
        Save episode as GIF file.

        Args:
            batch_idx: Batch index (b)
            frames: List of numpy arrays (H, W, C) in uint8 format
        """
        if len(frames) == 0:
            return

        episode_idx = self.episode_counts[batch_idx]
        num_steps = len(frames)
        output_path = self.output_dir / f"b_{batch_idx}_episode_{episode_idx}_steps_{num_steps}.gif"

        # Convert frames to PIL Images
        pil_frames = [Image.fromarray(frame) for frame in frames]

        # Save as GIF (duration in milliseconds per frame, ~15 FPS)
        pil_frames[0].save(
            output_path,
            save_all=True,
            append_images=pil_frames[1:],
            duration=67,  # ~15 FPS
            loop=0
        )

        print(f"[DataPipelineTest] Saved episode: {output_path} ({num_steps} frames)")

    def _is_collection_complete(self, batch_size: int) -> bool:
        """
        Check if we've collected enough episodes for all batch indices.

        Args:
            batch_size: Current batch size

        Returns:
            True if all batch indices have collected the target number of episodes
        """
        for b in range(batch_size):
            if b not in self.episode_counts or self.episode_counts[b] < self.episodes_per_batch:
                return False
        return True


# Global collector instance
_collector = None


def test_data_pipeline(batch: Dict[str, torch.Tensor]) -> bool:
    """
    Test function to be called from trainer.

    Args:
        batch: Batch dictionary from sample_sequences

    Returns:
        True if collection is complete and training should stop, False otherwise
    """
    global _collector

    if _collector is None:
        _collector = EpisodeCollector()
        print("[DataPipelineTest] Started collecting episodes...")

    is_complete = _collector.process_batch(batch)

    if is_complete:
        print("[DataPipelineTest] Collection complete! Check ./test_output/ for GIF files.")
        print("[DataPipelineTest] Stopping training for test...")

    return is_complete
