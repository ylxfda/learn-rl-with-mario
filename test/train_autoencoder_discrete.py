#!/usr/bin/env python3
"""
Iterative DreamerV3 world-model/actor training in a controlled experiment.

This script lives under `test/` so it can evolve independently from the
main training loop while still reusing the same components.  The workflow:

1. Collect an initial dataset with a scripted exploration policy.
2. Train the world model + actor/critic for a fixed number of epochs.
3. For subsequent iterations:
      a. Remove a fraction of the oldest episodes.
      b. Roll out the current actor (with exploration) to refresh the buffer.
      c. Retrain for another block of epochs.
4. After every training block, log reconstructions/GIFs/checkpoints that
   are tagged with both the iteration and epoch index.

The goal is to mirror the data freshness dynamics of the main Dreamer
training loop in a smaller, easier-to-instrument environment.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Deque, List, Optional, Tuple
from collections import deque
from pathlib import Path
import sys
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision.utils import save_image
from PIL import Image
import yaml

# Make sure project root is on PYTHONPATH so `algorithms.*` imports work when executed from `test/`.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from algorithms.dreamer_v3.envs.mario_env import make_mario_env
from algorithms.dreamer_v3.models.world_model import RSSM
from algorithms.dreamer_v3.agent.actor_critic import (
    Actor,
    Critic,
    EMATargetCritic,
    compute_lambda_returns,
    compute_actor_loss,
)


# --------------------------------------------------------------------------- #
# Dataset helpers
# --------------------------------------------------------------------------- #


class SequenceDataset(Dataset):
    """
    Sliding-window dataset built from flattened episode tensors.

    The world model expects batches of shape (B, T, ...).  We keep a
    single concatenated tensor for each modality and slice sequential
    windows of length `seq_len`.  Shuffling is handled by the DataLoader.
    """

    def __init__(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        continues: torch.Tensor,
        is_first: torch.Tensor,
        seq_len: int,
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
        end = idx + self.seq_len
        obs = self.observations[idx:end]
        act = self.actions[idx:end]
        rew = self.rewards[idx:end]
        cont = self.continues[idx:end]
        first = self.is_first[idx:end]
        return obs, act, rew, cont, first


@dataclass
class Episode:
    """Simple container holding one complete environment episode."""

    observations: torch.Tensor  # (T, C, H, W)
    actions: torch.Tensor       # (T, action_size) one-hot
    rewards: torch.Tensor       # (T,)
    continues: torch.Tensor     # (T,)
    is_first: torch.Tensor      # (T,)

    @property
    def length(self) -> int:
        return int(self.rewards.shape[0])


def episodes_to_tensors(episodes: List[Episode]) -> Tuple[torch.Tensor, ...]:
    """Concatenate a list of `Episode` objects into flat tensors."""

    observations = torch.cat([ep.observations for ep in episodes], dim=0)
    actions = torch.cat([ep.actions for ep in episodes], dim=0)
    rewards = torch.cat([ep.rewards for ep in episodes], dim=0)
    continues = torch.cat([ep.continues for ep in episodes], dim=0)
    is_first = torch.cat([ep.is_first for ep in episodes], dim=0)
    return observations, actions, rewards, continues, is_first


# --------------------------------------------------------------------------- #
# Data collection
# --------------------------------------------------------------------------- #


def collect_episodes(
    *,
    config: dict,
    num_frames: int,
    warmup_right: int,
    stall_window: int,
    stall_threshold: float,
    actor: Optional[Actor],
    world_model: Optional[RSSM],
    device: torch.device,
    explore_prob: float,
) -> List[Episode]:
    """
    Collect episodes until the accumulated timesteps reach `num_frames`.

    When `actor/world_model` are provided, actions are sampled from the
    current policy with an ε-greedy exploration probability.  Otherwise
    actions follow the scripted policy (move-right warmup + random).  To
    avoid logging hundreds of identical frames when Mario is stuck, we
    monitor x-position deltas and force a reset if progress stalls.
    """

    env = make_mario_env(config)
    episodes: List[Episode] = []
    total_steps = 0
    use_actor = actor is not None and world_model is not None

    try:
        while total_steps < num_frames:
            obs = env.reset()
            prev_x = 0.0
            delta_buffer: Deque[float] = deque(maxlen=stall_window)
            step = 0

            obs_buf: List[torch.Tensor] = []
            act_idx_buf: List[int] = []
            rew_buf: List[torch.Tensor] = []
            cont_buf: List[torch.Tensor] = []
            first_buf: List[bool] = []

            if use_actor:
                world_model.eval()
                actor.eval()
                with torch.no_grad():
                    h = torch.zeros(1, world_model.hidden_size, device=device)

            while True:
                obs_norm = torch.from_numpy(obs).float() / 255.0
                obs_buf.append(obs_norm)
                first_buf.append(len(first_buf) == 0)

                if use_actor:
                    with torch.no_grad():
                        obs_tensor = obs_norm.unsqueeze(0).to(device)
                        z = world_model.encode(h, obs_tensor).sample()

                    if np.random.rand() < explore_prob:
                        action_idx = int(np.random.randint(env.action_size))
                    else:
                        with torch.no_grad():
                            action_dist = actor(h, z)
                            action_idx = int(action_dist.sample().item())

                    action_onehot_model = F.one_hot(
                        torch.tensor([action_idx], device=device),
                        num_classes=env.action_size,
                    ).float()
                else:
                    if step < warmup_right:
                        action_idx = 1
                    else:
                        action_idx = int(np.random.randint(env.action_size))

                act_idx_buf.append(action_idx)

                next_obs, reward, done, info = env.step(action_idx)
                current_x = float(info.get("x_pos", 0))
                delta_x = current_x - prev_x
                prev_x = current_x
                delta_buffer.append(delta_x)

                # Treat lack of progress as terminal to refresh exploration.
                stalled = (
                    not done
                    and len(delta_buffer) == stall_window
                    and all(abs(dx) < stall_threshold for dx in delta_buffer)
                )
                done_flag = done or stalled

                rew_buf.append(torch.tensor(reward, dtype=torch.float32))
                cont_buf.append(torch.tensor(0.0 if done_flag else 1.0, dtype=torch.float32))

                if use_actor:
                    with torch.no_grad():
                        h = world_model.dynamics(h, z, action_onehot_model)

                obs = next_obs
                step += 1

                if done_flag:
                    break

            actions_tensor = F.one_hot(
                torch.tensor(act_idx_buf, dtype=torch.long),
                num_classes=env.action_size,
            ).float()

            episode = Episode(
                observations=torch.stack(obs_buf),
                actions=actions_tensor,
                rewards=torch.stack(rew_buf),
                continues=torch.stack(cont_buf),
                is_first=torch.tensor(first_buf, dtype=torch.bool),
            )
            episodes.append(episode)
            total_steps += episode.length

    finally:
        env.close()

    return episodes


# --------------------------------------------------------------------------- #
# Evaluation utilities
# --------------------------------------------------------------------------- #


def _convert_frames_to_shared_palette(frames: List[np.ndarray]) -> List[Image.Image]:
    """Convert RGB frames to paletted images sharing the first frame's palette."""
    if not frames:
        return []

    base_img = Image.fromarray(frames[0])
    base_paletted = base_img.convert("P", palette=Image.ADAPTIVE, colors=256, dither=Image.NONE)
    paletted_frames = [base_paletted]

    for arr in frames[1:]:
        img = Image.fromarray(arr)
        pal = img.quantize(palette=base_paletted, dither=Image.NONE)
        paletted_frames.append(pal)

    return paletted_frames


def _save_gif(frames: List[Image.Image], path: Path, fps: int = 12) -> None:
    """Persist paletted frames as a looping GIF without triggering palette flicker."""
    if not frames:
        return
    duration_ms = int(1000 / max(1, fps))
    frames[0].save(
        path,
        save_all=True,
        append_images=frames[1:],
        loop=0,
        duration=duration_ms,
        optimize=False,
    )


def evaluate_model(
    model: RSSM,
    actor: Actor,
    config: dict,
    device: torch.device,
    output_image: Path,
    output_gif: Path,
    frames: int,
    warmup_right: int,
) -> None:
    """
    Log open-loop reconstructions for diagnostic purposes.

    The actor drives the environment (with deterministic actions) so we
    can visualise how well the model reconstructs online observations.
    """

    env = make_mario_env(config)
    obs = env.reset()
    h = torch.zeros(1, model.hidden_size, device=device)
    actor.eval()

    truth_frames: List[torch.Tensor] = []
    recon_frames: List[torch.Tensor] = []
    kl_values: List[float] = []
    entropy_values: List[float] = []

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
            dim=[-1, -2],
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
                    num_classes=env.action_size,
                ).float()
                h = model.dynamics(h, z, action_onehot)
            obs = next_obs
        else:
            obs = env.reset()
            h = torch.zeros(1, model.hidden_size, device=device)

    env.close()

    model.eval()
    top = torch.stack(truth_frames[:16])
    bottom = torch.stack(recon_frames[:16])
    grid = torch.cat([top, bottom], dim=0)
    output_image.parent.mkdir(parents=True, exist_ok=True)
    save_image(grid, output_image, nrow=16, value_range=(0.0, 1.0))

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

    paletted_frames = _convert_frames_to_shared_palette(gif_frames)
    _save_gif(paletted_frames, output_gif)

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
    max_frames: int,
) -> None:
    """
    Closed-loop reconstruction on training data (teacher forcing).
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
            is_first=first,
        )
    recon = outputs["x_recon"].squeeze(0).cpu()
    truth = obs.squeeze(0).cpu()
    mse = torch.mean((recon - truth) ** 2).item()

    grid_truth = truth[:16]
    grid_recon = recon[:16]
    grid = torch.cat([grid_truth, grid_recon], dim=0)
    output_image.parent.mkdir(parents=True, exist_ok=True)
    save_image(grid, output_image, nrow=16, value_range=(0.0, 1.0))

    frames = []
    for t, r in zip(truth, recon):
        diff = torch.clamp(torch.abs(r - t) * 2.0, 0.0, 1.0)
        if t.shape[0] == 1:
            t = t.repeat(3, 1, 1)
            r = r.repeat(3, 1, 1)
            diff = diff.repeat(3, 1, 1)
        stacked = torch.cat([t, r, diff], dim=1)
        frames.append((stacked.permute(1, 2, 0).numpy() * 255).astype(np.uint8))

    paletted_frames = _convert_frames_to_shared_palette(frames)
    _save_gif(paletted_frames, output_gif)

    print(f"[TrainRecon] Saved grid to {output_image}")
    print(f"[TrainRecon] Saved GIF to {output_gif}")
    print(f"[TrainRecon] Mean pixel MSE: {mse:.5f}")


# --------------------------------------------------------------------------- #
# Argument parsing
# --------------------------------------------------------------------------- #


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Iterative Dreamer autoencoder experiment.")
    parser.add_argument("--config", type=str, default="configs/dreamerv3_config.yaml", help="Config file.")
    parser.add_argument("--frames", type=int, default=20000, help="Target frames to collect per iteration.")
    parser.add_argument("--epochs", type=int, default=10, help="Training epochs per iteration.")
    parser.add_argument("--iterations", type=int, default=5, help="Number of collection/training iterations.")
    parser.add_argument("--batch-size", type=int, default=32, help="Mini-batch size.")
    parser.add_argument("--seq-len", type=int, default=64, help="Sequence length for truncated BPTT.")
    parser.add_argument("--warmup-right", type=int, default=0, help="Forced move-right steps for scripted rollout.")
    parser.add_argument("--lr", type=float, default=1e-3, help="World model learning rate.")
    parser.add_argument("--output", type=str, default="logs/discrete_autoencoder_recon.png", help="Open-loop preview image.")
    parser.add_argument("--train-output", type=str, default="logs/discrete_autoencoder_train_recon.png",
                        help="Closed-loop (training data) preview image.")
    parser.add_argument("--gif-output", type=str, default="logs/discrete_autoencoder_recon.gif", help="Open-loop GIF path.")
    parser.add_argument("--train-gif-output", type=str, default="logs/discrete_autoencoder_train_recon.gif",
                        help="Closed-loop GIF path.")
    parser.add_argument("--gif-frames", type=int, default=400, help="Frames for open-loop GIF.")
    parser.add_argument("--train-eval-frames", type=int, default=400, help="Frames for closed-loop GIF.")
    parser.add_argument("--checkpoint-dir", type=str, default="logs/autoencoder_ckpts", help="Checkpoint directory.")
    parser.add_argument("--explore-prob", type=float, default=0.1,
                        help="Policy exploration probability during data collection.")
    parser.add_argument("--stall-window", type=int, default=15,
                        help="Steps sampled for stall detection based on x-position deltas.")
    parser.add_argument("--stall-threshold", type=float, default=1.0,
                        help="Minimum |Δx| considered as progress (in pixels).")
    parser.add_argument("--data-update-ratio", type=float, default=0.1,
                        help="Fraction of timesteps to replace each iteration.")
    return parser.parse_args()


# --------------------------------------------------------------------------- #
# Training loop
# --------------------------------------------------------------------------- #


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    model = RSSM(config).to(device)
    optimizer_model = torch.optim.Adam(model.parameters(), lr=args.lr)
    actor = Actor(config).to(device)
    critic = Critic(config).to(device)
    target_critic = EMATargetCritic(critic, tau=config['training']['tau'])
    optimizer_actor = torch.optim.Adam(actor.parameters(), lr=config['training']['lr_actor'],
                                       eps=config['optimization']['eps'])
    optimizer_critic = torch.optim.Adam(critic.parameters(), lr=config['training']['lr_critic'],
                                        eps=config['optimization']['eps'])

    episodes: List[Episode] = []

    for collect_iter in range(args.iterations):
        if collect_iter == 0:
            print(f"[Collect] Iteration {collect_iter + 1}: scripted exploration rollouts")
            episodes = collect_episodes(
                config=config,
                num_frames=args.frames,
                warmup_right=args.warmup_right,
                stall_window=args.stall_window,
                stall_threshold=args.stall_threshold,
                actor=None,
                world_model=None,
                device=device,
                explore_prob=args.explore_prob,
            )
        else:
            total_steps_before = sum(ep.length for ep in episodes)
            target_remove = max(1, int(total_steps_before * args.data_update_ratio))
            removed = 0
            while episodes and removed < target_remove:
                removed += episodes.pop(0).length

            target_add = max(1, int(total_steps_before * args.data_update_ratio))
            print(f"[Collect] Iteration {collect_iter + 1}: policy-guided rollouts with explore_prob={args.explore_prob}")
            new_eps = collect_episodes(
                config=config,
                num_frames=target_add,
                warmup_right=0,
                stall_window=args.stall_window,
                stall_threshold=args.stall_threshold,
                actor=actor,
                world_model=model,
                device=device,
                explore_prob=args.explore_prob,
            )
            episodes.extend(new_eps)

        observations, actions, rewards, continues, is_first = episodes_to_tensors(episodes)
        dataset = SequenceDataset(
            observations=observations,
            actions=actions,
            rewards=rewards,
            continues=continues,
            is_first=is_first,
            seq_len=args.seq_len,
        )
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

        print(f"[Train] Iteration {collect_iter + 1}/{args.iterations}")
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
                obs_seq = obs_seq.to(device)
                act_seq = act_seq.to(device)
                rew_seq = rew_seq.to(device)
                cont_seq = cont_seq.to(device)
                first_seq = first_seq.to(device)

                losses = model.compute_loss(
                    observations=obs_seq,
                    actions=act_seq,
                    rewards=rew_seq,
                    continues=cont_seq,
                    is_first=first_seq,
                    h_0=None,
                )
                loss = losses["total_loss"]

                optimizer_model.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
                optimizer_model.step()

                epoch_loss += loss.item()
                epoch_pred += losses["pred_loss"].detach().item()
                epoch_dyn += losses["dyn_loss"].detach().item()
                epoch_rep += losses["rep_loss"].detach().item()
                epoch_recon += losses["recon_loss"].detach().item()
                epoch_reward += losses["reward_loss"].detach().item()
                epoch_continue += losses["continue_loss"].detach().item()

                with torch.no_grad():
                    h_seq = losses["h_seq"]
                    z_seq = losses["z_seq"]
                    h_start = h_seq[:, -1]
                    z_start = z_seq[:, -1]
                    imagined = model.imagine(
                        h_0=h_start,
                        z_0=z_start,
                        actor=actor,
                        horizon=config['training']['h_imagine'],
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
                        lambda_=config['training']['lambda_'],
                    )

                critic_loss = critic.compute_loss(
                    h_imag.detach(),
                    z_imag.detach(),
                    lambda_returns,
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
                    entropy_scale=config['training']['entropy_scale'],
                )
                optimizer_actor.zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(actor.parameters(), config['optimization']['grad_clip'])
                optimizer_actor.step()
                target_critic.update(critic)

                epoch_actor_loss += actor_loss.detach().item()
                epoch_critic_loss += critic_loss.detach().item()

            num_batches = len(loader)
            epoch_loss /= num_batches
            epoch_pred /= num_batches
            epoch_dyn /= num_batches
            epoch_rep /= num_batches
            epoch_recon /= num_batches
            epoch_reward /= num_batches
            epoch_continue /= num_batches
            epoch_actor_loss /= num_batches
            epoch_critic_loss /= num_batches

            print(
                f"Epoch {epoch:02d}: total={epoch_loss:.6f}, pred={epoch_pred:.6f}, "
                f"dyn={epoch_dyn:.6f}, rep={epoch_rep:.6f}, recon={epoch_recon:.6f}, "
                f"reward={epoch_reward:.6f}, cont={epoch_continue:.6f}, "
                f"actor={epoch_actor_loss:.6f}, critic={epoch_critic_loss:.6f}"
            )

            if epoch % 2 == 0:
                suffix = f"iter_{collect_iter + 1}_epoch_{epoch:02d}"
                evaluate_training_recon(
                    model=model,
                    observations=observations,
                    actions=actions,
                    is_first=is_first,
                    device=device,
                    output_image=Path(args.train_output).with_name(f"{Path(args.train_output).stem}_{suffix}.png"),
                    output_gif=Path(args.train_gif_output).with_name(f"{Path(args.train_gif_output).stem}_{suffix}.gif"),
                    max_frames=args.train_eval_frames,
                )
                evaluate_model(
                    model=model,
                    actor=actor,
                    config=config,
                    device=device,
                    output_image=Path(args.output).with_name(f"{Path(args.output).stem}_{suffix}.png"),
                    output_gif=Path(args.gif_output).with_name(f"{Path(args.gif_output).stem}_{suffix}.gif"),
                    frames=args.gif_frames,
                    warmup_right=args.warmup_right,
                )

            torch.save(
                {
                    "iteration": collect_iter,
                    "epoch": epoch,
                    "world_model": model.state_dict(),
                    "actor": actor.state_dict(),
                    "critic": critic.state_dict(),
                    "optimizer_model": optimizer_model.state_dict(),
                    "optimizer_actor": optimizer_actor.state_dict(),
                    "optimizer_critic": optimizer_critic.state_dict(),
                },
                checkpoint_dir / f"checkpoint_iter_{collect_iter + 1}_epoch_{epoch:02d}.pt",
            )

    # Final evaluation without suffix (latest snapshot).
    observations, actions, rewards, continues, is_first = episodes_to_tensors(episodes)
    evaluate_training_recon(
        model=model,
        observations=observations,
        actions=actions,
        is_first=is_first,
        device=device,
        output_image=Path(args.train_output),
        output_gif=Path(args.train_gif_output),
        max_frames=args.train_eval_frames,
    )
    evaluate_model(
        model=model,
        actor=actor,
        config=config,
        device=device,
        output_image=Path(args.output),
        output_gif=Path(args.gif_output),
        frames=args.gif_frames,
        warmup_right=args.warmup_right,
    )


if __name__ == "__main__":
    main()
