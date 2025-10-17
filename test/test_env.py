"""
Test environment interactions by saving frames as JPG files.

This module saves each environment frame with reward and action information.
"""

import numpy as np
from pathlib import Path
from PIL import Image


class EnvFrameSaver:
    """
    Saves environment frames as JPG files with reward and action info.
    """

    def __init__(self, output_dir: str = "./test_output/env"):
        """
        Args:
            output_dir: Directory to save JPG files
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def save_frame(self, obs: np.ndarray, step: int, reward: float, action: int):
        """
        Save a single frame as JPG.

        Args:
            obs: Observation array (H, W, C) or (C, H, W) in uint8 or float format
            step: Step number for filename
            reward: Reward value
            action: Action index
        """
        # Convert observation to (H, W, C) format if needed
        if obs.ndim == 3:
            if obs.shape[0] in [1, 3, 4]:  # (C, H, W) format
                frame = np.transpose(obs, (1, 2, 0))
            else:  # (H, W, C) format
                frame = obs
        else:
            raise ValueError(f"Unexpected observation shape: {obs.shape}")

        # Convert from [0, 1] float to [0, 255] uint8 if needed
        if frame.dtype == np.float32 or frame.dtype == np.float64:
            frame = np.clip(frame * 255.0, 0, 255).astype(np.uint8)

        # Handle grayscale images (C=1)
        if frame.shape[-1] == 1:
            frame = np.repeat(frame, 3, axis=-1)

        # Create filename with step, reward, and action
        filename = f"obs_{step:04d}_rewd_{reward:.2f}_act_{action:02d}.jpg"
        filepath = self.output_dir / filename

        # Save as JPG
        Image.fromarray(frame).save(filepath, 'JPEG', quality=95)


# Global saver instance
_saver = None


def test_env_step(obs: np.ndarray, step: int, reward: float, action: int):
    """
    Test function to be called from trainer after env.step().

    Args:
        obs: Observation before the action was taken
        step: Global step number
        reward: Reward received from the action
        action: Action index that was taken
    """
    global _saver

    if _saver is None:
        _saver = EnvFrameSaver()
        print("[EnvTest] Started saving environment frames to ./test_output/env/")

    _saver.save_frame(obs, step, reward, action)
