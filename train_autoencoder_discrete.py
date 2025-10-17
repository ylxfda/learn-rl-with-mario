#!/usr/bin/env python3
"""
Simplified DreamerV3 reconstruction training.

This script collects a scripted dataset of Mario frames/actions,
trains the full RSSM on a pure pixel-reconstruction objective,
and then evaluates reconstructions along a fixed action rollout.

It is meant to isolate encoder/decoder + recurrent state behaviour
without the extra losses used in the full Dreamer training loop.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import imageio.v2 as imageio
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision.utils import save_image
import yaml

from algorithms.dreamer_v3.envs.mario_env import make_mario_env
from algorithms.dreamer_v3.models.world_model import RSSM


class SequenceDataset(Dataset):
    """Sliding-window dataset of observation/action sequences."""

    def __init__(
        self,
        observations: torch.Tensor,  # (N, C, H, W)
        actions: torch.Tensor,       # (N, action_size)
        is_first: torch.Tensor,      # (N,)
        seq_len: int
    ):
        self.observations = observations
        self.actions = actions
        self.is_first = is_first
        self.seq_len = seq_len

    def __len__(self) -> int:
        return self.observations.shape[0] - self.seq_len + 1

    def __getitem__(self, idx: int):
        obs = self.observations[idx:idx + self.seq_len]
        act = self.actions[idx:idx + self.seq_len]
        first = self.is_first[idx:idx + self.seq_len]
        return obs, act, first


def collect_sequences(env, num_frames: int, warmup_right: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Roll out scripted policy to gather observation/action sequences.
    """
    observations = []
    actions = []
    first_flags = []

    obs = env.reset()
    first = True
    step = 0

    while len(observations) < num_frames:
        observations.append(torch.from_numpy(obs).float() / 255.0)
        first_flags.append(first)

        if step < warmup_right:
            action_idx = 1  # move right
        else:
            action_idx = np.random.randint(0, env.action_size)

        onehot = torch.zeros(env.action_size, dtype=torch.float32)
        onehot[action_idx] = 1.0
        actions.append(onehot)

        next_obs, _, done, _ = env.step(action_idx)
        obs = next_obs
        first = done
        step += 1

        if done:
            obs = env.reset()
            first = True
            step = 0

    observations = torch.stack(observations)  # (N, C, H, W)
    actions = torch.stack(actions)            # (N, action_size)
    first_flags = torch.tensor(first_flags, dtype=torch.bool)  # (N,)
    return observations, actions, first_flags


def evaluate_model(
    model: RSSM,
    config: dict,
    device: torch.device,
    output_image: Path,
    output_gif: Path,
    frames: int,
    warmup_right: int
) -> None:
    """Generate reconstruction grid and GIF for qualitative inspection."""
    env = make_mario_env(config)
    obs = env.reset()
    h = torch.zeros(1, model.hidden_size, device=device)

    truth_frames = []
    recon_frames = []
    kl_values = []
    entropy_values = []

    for step in range(frames):
        obs_tensor = torch.from_numpy(obs).float().unsqueeze(0).to(device) / 255.0
        with torch.no_grad():
            z_dist = model.encode(h, obs_tensor)
            z = z_dist.mode()
            recon = model.decode(h, z)
            prior_dist = model.prior(h)

            post_probs = z_dist.probs
            prior_probs = prior_dist.probs
            kl = torch.sum(
                post_probs * (torch.log(post_probs + 1e-8) - torch.log(prior_probs + 1e-8)),
                dim=[-1, -2]
            ).mean().item()
            entropy = z_dist.entropy().mean().item()

        truth_frames.append(obs_tensor.squeeze(0).cpu())
        recon_frames.append(recon.squeeze(0).cpu())
        kl_values.append(kl)
        entropy_values.append(entropy)

        if step < warmup_right:
            action_idx = 1
        else:
            action_idx = 1  # keep holding right to avoid oscillations

        next_obs, _, done, _ = env.step(action_idx)

        if not done:
            with torch.no_grad():
                action_onehot = F.one_hot(
                    torch.tensor([action_idx], device=device),
                    num_classes=env.action_size
                ).float()
                h = model.dynamics(h, z, action_onehot)
            obs = next_obs
        else:
            obs = env.reset()
            h = torch.zeros(1, model.hidden_size, device=device)

    env.close()

    # Save grid (first 16 frames)
    model.eval()
    top = torch.stack(truth_frames[:16])
    bottom = torch.stack(recon_frames[:16])
    grid = torch.cat([top, bottom], dim=0)
    output_image.parent.mkdir(parents=True, exist_ok=True)
    save_image(grid, output_image, nrow=16, value_range=(0.0, 1.0))

    # Save GIF
    gif_frames = []
    for truth, recon in zip(truth_frames, recon_frames):
        truth = torch.clamp(truth, 0.0, 1.0)
        recon = torch.clamp(recon, 0.0, 1.0)
        diff = torch.clamp(torch.abs(recon - truth) * 2.0, 0.0, 1.0)

        if truth.shape[0] == 1:
            truth = truth.repeat(3, 1, 1)
            recon = recon.repeat(3, 1, 1)
            diff = diff.repeat(3, 1, 1)

        stacked = torch.cat([truth, recon, diff], dim=1)
        gif_frames.append((stacked.permute(1, 2, 0).numpy() * 255).astype(np.uint8))

    imageio.mimsave(output_gif, gif_frames, fps=12)

    mse = np.mean([(t.numpy() - r.numpy()) ** 2 for t, r in zip(truth_frames, recon_frames)])
    print(f"Saved reconstruction preview to {output_image}")
    print(f"Saved reconstruction GIF to {output_gif}")
    print(f"Mean pixel MSE: {mse:.5f}")
    print(f"Mean KL: {np.mean(kl_values):.4f}, Std KL: {np.std(kl_values):.4f}")
    print(f"Mean posterior entropy: {np.mean(entropy_values):.4f}")


def evaluate_training_recon(
    model: RSSM,
    observations: torch.Tensor,
    actions: torch.Tensor,
    is_first: torch.Tensor,
    device: torch.device,
    output_image: Path,
    output_gif: Path,
    max_frames: int
) -> None:
    """Evaluate reconstructions on training sequences using observe()."""
    model.eval()
    length = min(max_frames, observations.shape[0])
    obs = observations[:length].unsqueeze(0).to(device)
    act = actions[:length].unsqueeze(0).to(device)
    first = is_first[:length].unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model.observe(
            observations=obs,
            actions=act,
            h_0=None,
            is_first=first
        )
    recon = outputs["x_recon"].squeeze(0).cpu()
    truth = obs.squeeze(0).cpu()

    mse = torch.mean((recon - truth) ** 2).item()

    # Save grid (first 16 frames)
    grid_truth = truth[:16]
    grid_recon = recon[:16]
    grid = torch.cat([grid_truth, grid_recon], dim=0)
    output_image.parent.mkdir(parents=True, exist_ok=True)
    save_image(grid, output_image, nrow=16, value_range=(0.0, 1.0))

    # Save GIF
    frames = []
    for t, r in zip(truth, recon):
        diff = torch.clamp(torch.abs(r - t) * 2.0, 0.0, 1.0)
        if t.shape[0] == 1:
            t = t.repeat(3, 1, 1)
            r = r.repeat(3, 1, 1)
            diff = diff.repeat(3, 1, 1)
        stacked = torch.cat([t, r, diff], dim=1)
        frames.append((stacked.permute(1, 2, 0).numpy() * 255).astype(np.uint8))
    imageio.mimsave(output_gif, frames, fps=12)

    print(f"[TrainRecon] Saved grid to {output_image}")
    print(f"[TrainRecon] Saved GIF to {output_gif}")
    print(f"[TrainRecon] Mean pixel MSE: {mse:.5f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simplified Dreamer reconstruction training.")
    parser.add_argument("--config", type=str, default="configs/dreamerv3_config.yaml", help="Config file.")
    parser.add_argument("--frames", type=int, default=20000, help="Number of frames to collect for training.")
    parser.add_argument("--epochs", type=int, default=20, help="Training epochs.")
    parser.add_argument("--batch-size", type=int, default=32, help="Mini-batch size.")
    parser.add_argument("--seq-len", type=int, default=64, help="Sequence length for BPTT.")
    parser.add_argument("--warmup-right", type=int, default=400, help="Steps forcing Mario to move right.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--output", type=str, default="logs/discrete_autoencoder_recon.png", help="Preview image path.")
    parser.add_argument("--train-output", type=str, default="logs/discrete_autoencoder_train_recon.png",
                        help="Preview image (training data) path.")
    parser.add_argument("--gif-output", type=str, default="logs/discrete_autoencoder_recon.gif", help="GIF output path.")
    parser.add_argument("--train-gif-output", type=str, default="logs/discrete_autoencoder_train_recon.gif",
                        help="GIF for training data reconstruction.")
    parser.add_argument("--gif-frames", type=int, default=400, help="Frames for evaluation GIF.")
    parser.add_argument("--train-eval-frames", type=int, default=400,
                        help="Number of frames from training data for evaluation.")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    env = make_mario_env(config)
    print(f"Collecting {args.frames} frames...")
    observations, actions, is_first = collect_sequences(env, args.frames, args.warmup_right)
    env.close()

    dataset = SequenceDataset(
        observations=observations,
        actions=actions,
        is_first=is_first,
        seq_len=args.seq_len
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

    model = RSSM(config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    print("Starting training...")
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        for obs_seq, act_seq, first_seq in loader:
            obs_seq = obs_seq.to(device)  # (B, T, C, H, W)
            act_seq = act_seq.to(device)  # (B, T, action_size)
            first_seq = first_seq.to(device)  # (B, T)

            outputs = model.observe(
                observations=obs_seq,
                actions=act_seq,
                h_0=None,
                is_first=first_seq
            )

            recon = outputs["x_recon"]
            loss = F.mse_loss(recon, obs_seq)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            optimizer.step()

            epoch_loss += loss.item()

        epoch_loss /= len(loader)
        print(f"Epoch {epoch:02d}: loss={epoch_loss:.6f}")

    model.eval()
    evaluate_training_recon(
        model=model,
        observations=observations,
        actions=actions,
        is_first=is_first,
        device=device,
        output_image=Path(args.train_output),
        output_gif=Path(args.train_gif_output),
        max_frames=args.train_eval_frames
    )
    evaluate_model(
        model=model,
        config=config,
        device=device,
        output_image=Path(args.output),
        output_gif=Path(args.gif_output),
        frames=args.gif_frames,
        warmup_right=args.warmup_right
    )


if __name__ == "__main__":
    main()
