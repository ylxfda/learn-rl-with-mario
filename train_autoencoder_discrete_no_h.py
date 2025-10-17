#!/usr/bin/env python3
"""
Legacy discrete autoencoder test with h fixed to zero.

This script reproduces the earlier experiment that confirmed the Dreamer
encoder/decoder can reconstruct observations without relying on the
deterministic state h. It is useful for comparing against the new
fully recurrent variant.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import imageio.v2 as imageio
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision.utils import save_image
import yaml

from algorithms.dreamer_v3.envs.mario_env import make_mario_env
from algorithms.dreamer_v3.models.world_model import RSSM


class FrameDataset(Dataset):
    """Simple dataset that returns individual frames."""

    def __init__(self, frames: torch.Tensor):
        self.frames = frames

    def __len__(self) -> int:
        return self.frames.shape[0]

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.frames[idx]


def collect_frames(env, num_frames: int, warmup_right: int) -> torch.Tensor:
    """Collect frames with a scripted policy that moves right initially."""
    frames: List[torch.Tensor] = []
    obs = env.reset()
    step = 0

    while len(frames) < num_frames:
        frames.append(torch.from_numpy(obs).float() / 255.0)
        if step < warmup_right:
            action_idx = 1  # move right
        else:
            action_idx = np.random.randint(0, env.action_size)

        obs, _, done, _ = env.step(action_idx)
        step += 1

        if done:
            obs = env.reset()
            step = 0

    return torch.stack(frames)


def save_preview(frames: torch.Tensor, recon: torch.Tensor, output_image: Path, output_gif: Path) -> None:
    """Save reconstruction grid and GIF for quick inspection."""
    output_image.parent.mkdir(parents=True, exist_ok=True)

    top = frames[:16]
    bottom = recon[:16]
    grid = torch.cat([top, bottom], dim=0)
    save_image(grid, output_image, nrow=16, value_range=(0.0, 1.0))

    gif_frames = []
    for truth, pred in zip(frames, recon):
        diff = torch.clamp(torch.abs(pred - truth) * 2.0, 0.0, 1.0)
        if truth.shape[0] == 1:
            truth = truth.repeat(3, 1, 1)
            pred = pred.repeat(3, 1, 1)
            diff = diff.repeat(3, 1, 1)
        stacked = torch.cat([truth, pred, diff], dim=1)
        gif_frames.append((stacked.permute(1, 2, 0).numpy() * 255).astype(np.uint8))

    imageio.mimsave(output_gif, gif_frames, fps=12)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Discrete Dreamer autoencoder without h.")
    parser.add_argument("--config", type=str, default="configs/dreamerv3_config.yaml", help="Config file path.")
    parser.add_argument("--frames", type=int, default=10000, help="Number of frames to collect.")
    parser.add_argument("--epochs", type=int, default=20, help="Training epochs.")
    parser.add_argument("--batch-size", type=int, default=128, help="Mini-batch size.")
    parser.add_argument("--warmup-right", type=int, default=400, help="Steps forcing Mario to move right.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--output", type=str, default="logs/discrete_autoencoder_no_h.png", help="Preview image path.")
    parser.add_argument("--gif-output", type=str, default="logs/discrete_autoencoder_no_h.gif", help="GIF output path.")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    env = make_mario_env(config)
    print(f"Collecting {args.frames} frames...")
    frames = collect_frames(env, args.frames, args.warmup_right)
    env.close()

    dataset = FrameDataset(frames)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

    model = RSSM(config).to(device)
    params = list(model.image_encoder.parameters()) + \
             list(model.encoder_mlp.parameters()) + \
             list(model.image_decoder.parameters())
    optimizer = torch.optim.Adam(params, lr=args.lr)
    criterion = nn.MSELoss()

    hidden_size = model.hidden_size

    print("Starting training...")
    for epoch in range(1, args.epochs + 1):
        epoch_loss = 0.0
        for batch in loader:
            batch = batch.to(device)
            B = batch.size(0)
            h = torch.zeros(B, hidden_size, device=device)

            z_dist = model.encode(h, batch)
            z = z_dist.mode()
            recon = model.decode(h, z)

            loss = criterion(recon, batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        epoch_loss /= len(loader)
        print(f"Epoch {epoch:02d}: loss={epoch_loss:.6f}")

    model.eval()
    with torch.no_grad():
        sample = frames.to(device)
        h = torch.zeros(sample.size(0), hidden_size, device=device)
        z = model.encode(h, sample).mode()
        recon = model.decode(h, z).cpu()

    mse = torch.mean((recon - frames) ** 2).item()
    save_preview(frames, recon, Path(args.output), Path(args.gif_output))
    print(f"Saved reconstruction preview to {args.output}")
    print(f"Saved reconstruction GIF to {args.gif_output}")
    print(f"Mean pixel MSE: {mse:.5f}")


if __name__ == "__main__":
    main()
