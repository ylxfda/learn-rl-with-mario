#!/usr/bin/env python3
"""
Simplified DreamerV3 reconstruction training.

Goal:
    Validate whether the RSSM encoder/decoder and deterministic state h
    can accurately reconstruct observations without involving reward,
    KL, or other auxiliary losses.

Workflow:
    1. Collect sequences with a scripted policy (move right first).
    2. Train the full RSSM (h, z, dynamics) using only pixel MSE loss.
    3. After training, generate reconstruction GIFs in both teacher-forced
       (training data) and open-loop settings to diagnose issues.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple, Deque

from collections import deque

import imageio.v2 as imageio
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision.utils import save_image
import yaml

from algorithms.dreamer_v3.envs.mario_env import make_mario_env
from algorithms.dreamer_v3.models.world_model import RSSM
from algorithms.dreamer_v3.models.distributions import symexp
from algorithms.dreamer_v3.agent.actor_critic import (
    Actor,
    Critic,
    EMATargetCritic,
    compute_lambda_returns,
    compute_actor_loss,
)


class SequenceDataset(Dataset):
    """
    Sliding-window dataset over stored trajectories.
    - Input is the entire trajectory (observations/actions/rewards/continues/is_first).
    - Each __getitem__ returns a chunk of length seq_len for truncated
      BPTT, matching the tensor shapes used in the main Dreamer trainer.
    """

    def __init__(
        self,
        observations: torch.Tensor,  # (N, C, H, W)
        actions: torch.Tensor,       # (N, action_size)
        rewards: torch.Tensor,       # (N,)
        continues: torch.Tensor,     # (N,)
        is_first: torch.Tensor,      # (N,)
        seq_len: int
    ):
        self.observations = observations
        self.actions = actions
        self.rewards = rewards
        self.continues = continues
        self.is_first = is_first
        self.seq_len = seq_len

    def __len__(self) -> int:
        return self.observations.shape[0] - self.seq_len + 1

    def __getitem__(self, idx: int):
        obs = self.observations[idx:idx + self.seq_len]
        act = self.actions[idx:idx + self.seq_len]
        rew = self.rewards[idx:idx + self.seq_len]
        cont = self.continues[idx:idx + self.seq_len]
        first = self.is_first[idx:idx + self.seq_len]
        return obs, act, rew, cont, first


def collect_sequences(
    env,
    num_frames: int,
    warmup_right: int,
    stall_window: int = 12,
    stall_threshold: float = 1.0
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Collect (obs, action, is_first) sequences using a scripted policy.

    Args:
        env: Mario environment wrapper.
        num_frames: Total number of frames to gather.
        warmup_right: Steps to force the "move right" action so Mario reaches key areas.

    Returns:
        observations: (N, C, H, W) tensor normalized to [0, 1].
        actions: (N, action_size) one-hot actions.
        rewards: (N,) float tensor of environment rewards.
        continues: (N,) float tensor where 1.0 means the episode continues.
        first_flags: (N,) bool tensor marking the start of each episode.
    """
    observations = []
    actions = []
    rewards = []
    continues = []
    first_flags = []
    delta_buffer: Deque[float] = deque(maxlen=stall_window)

    obs = env.reset()
    first = True
    step = 0
    prev_x = 0.0
    delta_buffer.clear()

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

        next_obs, reward, done, info = env.step(action_idx)
        obs = next_obs
        rewards.append(torch.tensor(reward, dtype=torch.float32))
        continues.append(torch.tensor(0.0 if done else 1.0, dtype=torch.float32))
        first = done
        step += 1
        current_x = float(info.get("x_pos", 0))
        delta_x = current_x - prev_x
        prev_x = current_x
        delta_buffer.append(delta_x)

        if done:
            obs = env.reset()
            first = True
            step = 0
            prev_x = 0.0
            delta_buffer.clear()
            continue

        if len(delta_buffer) == stall_window and all(
            abs(dx) < stall_threshold for dx in delta_buffer
        ):
            obs = env.reset()
            first = True
            step = 0
            prev_x = 0.0
            delta_buffer.clear()

    observations = torch.stack(observations)  # (N, C, H, W)
    actions = torch.stack(actions)            # (N, action_size)
    rewards = torch.stack(rewards)            # (N,)
    continues = torch.stack(continues)        # (N,)
    first_flags = torch.tensor(first_flags, dtype=torch.bool)  # (N,)
    return observations, actions, rewards, continues, first_flags


def evaluate_model(
    model: RSSM,
    actor: Actor,
    config: dict,
    device: torch.device,
    output_image: Path,
    output_gif: Path,
    frames: int,
    warmup_right: int
) -> None:
    """
    Perform an open-loop rollout in the environment to log reconstructions.

    - Each step uses encoder(h_t, obs_t) to obtain the posterior, then decode(h_t, z_t).
    - h_{t+1} is updated via dynamics(h_t, z_t, action) to mimic online operation.
    - Additionally track posterior vs prior KL and posterior entropy to monitor collapse.
    """
    env = make_mario_env(config)
    obs = env.reset()
    h = torch.zeros(1, model.hidden_size, device=device)
    actor.eval()

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
            with torch.no_grad():
                action_dist = actor(h, z, deterministic=True)
                action_idx = action_dist.mode().item()

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

    imageio.mimsave(
        output_gif,
        gif_frames,
        format="GIF",
        loop=0,
        duration=1 / 12
    )

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
    """
    Evaluate reconstructions on the training data (teacher-forced).

    - Directly call observe(), identical to training, so the encoder sees real observations.
    - Verifies that the model behaves correctly under teacher forcing.
    """
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
    imageio.mimsave(
        output_gif,
        frames,
        format="GIF",
        loop=0,
        duration=1 / 12
    )

    print(f"[TrainRecon] Saved grid to {output_image}")
    print(f"[TrainRecon] Saved GIF to {output_gif}")
    print(f"[TrainRecon] Mean pixel MSE: {mse:.5f}")


def parse_args() -> argparse.Namespace:
    """
    Command-line arguments:
        --frames            Number of frames to collect for the dataset.
        --epochs            Training epochs (pure MSE; increase if needed).
        --seq-len           BPTT sequence length; longer sequences stress h more.
        --warmup-right      Steps with forced "move right" action during collection.
        --output / --gif-output
                            Outputs for open-loop evaluation.
        --train-output / --train-gif-output
                            Outputs for closed-loop (training data) evaluation.
    """
    parser = argparse.ArgumentParser(description="Simplified Dreamer reconstruction training.")
    parser.add_argument("--config", type=str, default="configs/dreamerv3_config.yaml", help="Config file.")
    parser.add_argument("--frames", type=int, default=20000, help="Total frames to collect for training.")
    parser.add_argument("--epochs", type=int, default=20, help="Training epochs.")
    parser.add_argument("--batch-size", type=int, default=32, help="Mini-batch size.")
    parser.add_argument("--seq-len", type=int, default=64, help="Sequence length for BPTT.")
    parser.add_argument("--warmup-right", type=int, default=400, help="Steps forcing Mario to move right.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--output", type=str, default="logs/discrete_autoencoder_recon.png", help="Open-loop preview image.")
    parser.add_argument("--train-output", type=str, default="logs/discrete_autoencoder_train_recon.png",
                        help="Closed-loop (training data) preview image.")
    parser.add_argument("--gif-output", type=str, default="logs/discrete_autoencoder_recon.gif", help="Open-loop GIF path.")
    parser.add_argument("--train-gif-output", type=str, default="logs/discrete_autoencoder_train_recon.gif",
                        help="Closed-loop GIF path.")
    parser.add_argument("--gif-frames", type=int, default=400, help="Frames for open-loop GIF.")
    parser.add_argument("--train-eval-frames", type=int, default=400,
                        help="Frames used in closed-loop evaluation.")
    parser.add_argument("--checkpoint-dir", type=str, default="logs/autoencoder_ckpts",
                        help="Directory to save checkpoints.")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    env = make_mario_env(config)  # reuse Dreamer main program Mario warpper
    print(f"Collecting {args.frames} frames...")
    observations, actions, rewards, continues, is_first = collect_sequences(
        env, args.frames, args.warmup_right
    )
    env.close()

    dataset = SequenceDataset(
        observations=observations,
        actions=actions,
        rewards=rewards,
        continues=continues,
        is_first=is_first,
        seq_len=args.seq_len
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    model = RSSM(config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)  # only world model being updated
    actor = Actor(config).to(device)
    critic = Critic(config).to(device)
    target_critic = EMATargetCritic(critic, tau=config['training']['tau'])
    optimizer_actor = torch.optim.Adam(
        actor.parameters(),
        lr=config['training']['lr_actor'],
        eps=config['optimization']['eps']
    )
    optimizer_critic = torch.optim.Adam(
        critic.parameters(),
        lr=config['training']['lr_critic'],
        eps=config['optimization']['eps']
    )

    print("Starting training...")
    for epoch in range(1, args.epochs + 1):
        model.train()
        actor.train()
        critic.train()
        epoch_loss = 0.0
        epoch_pred = 0.0
        epoch_dyn = 0.0
        epoch_rep = 0.0
        epoch_recon = 0.0
        epoch_reward = 0.0
        epoch_continue = 0.0
        epoch_actor_loss = 0.0
        epoch_critic_loss = 0.0
        for obs_seq, act_seq, rew_seq, cont_seq, first_seq in loader:
            obs_seq = obs_seq.to(device)  # (B, T, C, H, W)
            act_seq = act_seq.to(device)  # (B, T, action_size)
            rew_seq = rew_seq.to(device)  # (B, T)
            cont_seq = cont_seq.to(device)  # (B, T)
            first_seq = first_seq.to(device)  # (B, T)

            losses = model.compute_loss(
                observations=obs_seq,
                actions=act_seq,
                rewards=rew_seq,
                continues=cont_seq,
                is_first=first_seq,
                h_0=None
            )
            loss = losses["total_loss"]

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            optimizer.step()

            epoch_loss += loss.item()
            epoch_pred += losses["pred_loss"].detach().item()
            epoch_dyn += losses["dyn_loss"].detach().item()
            epoch_rep += losses["rep_loss"].detach().item()
            epoch_recon += losses["recon_loss"].detach().item()
            epoch_reward += losses["reward_loss"].detach().item()
            epoch_continue += losses["continue_loss"].detach().item()

            # Actor-Critic update (single step per batch, mirroring main flow)
            with torch.no_grad():
                h_seq = losses["h_seq"]
                z_seq = losses["z_seq"]
                h_start = h_seq[:, -1]
                z_start = z_seq[:, -1]
                imagined = model.imagine(
                    h_0=h_start,
                    z_0=z_start,
                    actor=actor,
                    horizon=config['training']['h_imagine']
                )
                h_imag = imagined['h']
                z_imag = imagined['z']
                rewards_imag = imagined['reward']  # symlog space
                continues_imag = imagined['continue']

            values = critic.get_value(h_imag, z_imag)
            with torch.no_grad():
                bootstrap = target_critic.get_value(h_imag[:, -1], z_imag[:, -1])
                lambda_returns = compute_lambda_returns(
                    rewards=rewards_imag,
                    continues=continues_imag,
                    values=values.detach(),
                    bootstrap=bootstrap,
                    gamma=config['training']['gamma'],
                    lambda_=config['training']['lambda_']
                )

            critic_loss = critic.compute_loss(
                h_imag.detach(),
                z_imag.detach(),
                lambda_returns
            )
            optimizer_critic.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(critic.parameters(), config['optimization']['grad_clip'])
            optimizer_critic.step()

            h_flat = h_imag.reshape(-1, h_imag.shape[-1])
            z_flat = z_imag.reshape(-1, *z_imag.shape[2:])
            action_dist = actor(h_flat, z_flat)
            actions_flat = imagined['actions'].reshape(-1, model.action_size)
            action_indices = torch.argmax(actions_flat, dim=-1)
            log_probs = action_dist.log_prob(action_indices).reshape(h_imag.shape[0], -1)
            entropy = action_dist.entropy().reshape(h_imag.shape[0], -1)

            advantages = lambda_returns - values.detach()
            flat_returns = lambda_returns.reshape(-1)
            p95 = torch.quantile(flat_returns, 0.95)
            p5 = torch.quantile(flat_returns, 0.05)
            denom = (p95 - p5).clamp_min(1e-6)
            advantages = (advantages - advantages.mean()) / denom

            actor_loss = compute_actor_loss(
                log_probs=log_probs,
                advantages=advantages,
                entropy=entropy,
                entropy_scale=config['training']['entropy_scale']
            )
            optimizer_actor.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(actor.parameters(), config['optimization']['grad_clip'])
            optimizer_actor.step()
            target_critic.update(critic)

            epoch_actor_loss += actor_loss.detach().item()
            epoch_critic_loss += critic_loss.detach().item()

        epoch_loss /= len(loader)
        epoch_pred /= len(loader)
        epoch_dyn /= len(loader)
        epoch_rep /= len(loader)
        epoch_recon /= len(loader)
        epoch_reward /= len(loader)
        epoch_continue /= len(loader)
        epoch_actor_loss /= len(loader)
        epoch_critic_loss /= len(loader)
        print(
            f"Epoch {epoch:02d}: total={epoch_loss:.6f}, "
            f"pred={epoch_pred:.6f}, dyn={epoch_dyn:.6f}, rep={epoch_rep:.6f}, "
            f"recon={epoch_recon:.6f}, reward={epoch_reward:.6f}, cont={epoch_continue:.6f}, "
            f"actor={epoch_actor_loss:.6f}, critic={epoch_critic_loss:.6f}"
        )
        if epoch % 2 == 0:
            ckpt_suffix = f"epoch_{epoch:02d}"
            evaluate_training_recon(
                model=model,
                observations=observations,
                actions=actions,
                is_first=is_first,
                device=device,
                output_image=Path(args.train_output).with_name(f"{Path(args.train_output).stem}_{ckpt_suffix}.png"),
                output_gif=Path(args.train_gif_output).with_name(f"{Path(args.train_gif_output).stem}_{ckpt_suffix}.gif"),
                max_frames=args.train_eval_frames
            )
            evaluate_model(
                model=model,
                actor=actor,
                config=config,
                device=device,
                output_image=Path(args.output).with_name(f"{Path(args.output).stem}_{ckpt_suffix}.png"),
                output_gif=Path(args.gif_output).with_name(f"{Path(args.gif_output).stem}_{ckpt_suffix}.gif"),
                frames=args.gif_frames,
                warmup_right=args.warmup_right
            )
        torch.save(
            {
                "epoch": epoch,
                "world_model": model.state_dict(),
                "actor": actor.state_dict(),
                "critic": critic.state_dict(),
                "optimizer_model": optimizer.state_dict(),
                "optimizer_actor": optimizer_actor.state_dict(),
                "optimizer_critic": optimizer_critic.state_dict(),
            },
            checkpoint_dir / f"checkpoint_epoch_{epoch:02d}.pt"
        )

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
        actor=actor,
        config=config,
        device=device,
        output_image=Path(args.output),
        output_gif=Path(args.gif_output),
        frames=args.gif_frames,
        warmup_right=args.warmup_right
    )


if __name__ == "__main__":
    main()
