"""
Play Super Mario Bros using a trained DreamerV3 agent.

This script loads a trained checkpoint and plays Mario, showing the game screen
in real-time. Useful for visualizing agent performance.

Usage:
    python play_dreamerv3.py --checkpoint logs/checkpoints/checkpoint_50000.pt
    python play_dreamerv3.py --checkpoint logs/checkpoints/checkpoint_50000.pt --episodes 5
    python play_dreamerv3.py --checkpoint logs/checkpoints/checkpoint_50000.pt --deterministic
"""

import torch
import torch.nn.functional as F
import numpy as np
import yaml
import argparse
import time
from pathlib import Path

from algorithms.dreamer_v3.models.world_model import RSSM
from algorithms.dreamer_v3.agent.actor_critic import Actor
from algorithms.dreamer_v3.envs.mario_env import make_mario_env


def play_mario(checkpoint_path: str, config_path: str, num_episodes: int = 1,
               deterministic: bool = True, render_delay: float = 0.0):
    """
    Play Mario using trained agent.

    Args:
        checkpoint_path: Path to trained checkpoint
        config_path: Path to config file
        num_episodes: Number of episodes to play
        deterministic: Use deterministic policy (default: True for best performance)
        render_delay: Delay in seconds between frames (0 = no delay, 0.01 = ~100 FPS)
    """
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create environment
    env = make_mario_env(config)
    print(f"\nEnvironment: {config['env']['name']}")
    print(f"  Action space: {env.action_size}")
    print(f"  Observation shape: {env.observation_shape}")

    # Create models
    world_model = RSSM(config).to(device)
    actor = Actor(config).to(device)

    # Load checkpoint
    print(f"\nLoading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    world_model.load_state_dict(checkpoint['world_model'])
    actor.load_state_dict(checkpoint['actor'])

    print(f"  Checkpoint from step: {checkpoint['global_step']}")
    print(f"  Episodes completed: {checkpoint['episode_count']}")

    # Set to eval mode
    world_model.eval()
    actor.eval()

    # Print controls
    print("\n" + "="*60)
    print("Playing Mario with trained agent")
    print("="*60)
    print(f"Policy: {'Deterministic (argmax)' if deterministic else 'Stochastic (sampling)'}")
    print(f"Episodes to play: {num_episodes}")
    print("\nPress Ctrl+C to stop early")
    print("="*60 + "\n")

    # Statistics
    all_rewards = []
    all_lengths = []
    all_max_x = []
    all_successes = []

    for episode in range(num_episodes):
        print(f"\n{'='*60}")
        print(f"Episode {episode + 1}/{num_episodes}")
        print(f"{'='*60}")

        # Reset environment
        obs = env.reset()
        h = torch.zeros(1, world_model.hidden_size, device=device)

        # Get initial stochastic state
        with torch.no_grad():
            obs_tensor = torch.from_numpy(obs).float().unsqueeze(0).to(device) / 255.0
            z_dist = world_model.encode(h, obs_tensor)
            z = z_dist.sample()

        episode_reward = 0
        episode_length = 0
        done = False

        # Play episode
        while not done:
            # Render the game screen
            env.render()

            # Add delay if specified (for better visualization)
            if render_delay > 0:
                time.sleep(render_delay)

            # Select action
            with torch.no_grad():
                action_idx, log_prob = actor.get_action(h, z, deterministic=deterministic)
                action_idx = action_idx.item()

            # Step environment
            obs, reward, done, info = env.step(action_idx)
            episode_reward += reward
            episode_length += 1

            # Update state for next step (if not done)
            if not done:
                with torch.no_grad():
                    # Encode next observation
                    obs_tensor = torch.from_numpy(obs).float().unsqueeze(0).to(device) / 255.0
                    z_dist = world_model.encode(h, obs_tensor)
                    z = z_dist.sample()

                    # Update deterministic state
                    action_onehot = F.one_hot(
                        torch.tensor([action_idx], device=device),
                        num_classes=env.action_size
                    ).float()
                    h = world_model.dynamics(h, z, action_onehot)

        # Episode finished
        episode_info = info.get('episode', {})
        max_x = episode_info.get('max_x_pos', 0)
        success = episode_info.get('flag_get', False)

        all_rewards.append(episode_reward)
        all_lengths.append(episode_length)
        all_max_x.append(max_x)
        all_successes.append(success)

        print(f"\n{'='*60}")
        print(f"Episode {episode + 1} Results:")
        print(f"  Reward: {episode_reward:.2f}")
        print(f"  Length: {episode_length} steps")
        print(f"  Max X position: {max_x}")
        print(f"  Success: {'✓ FLAG REACHED!' if success else '✗ Failed'}")
        print(f"{'='*60}")

    # Print overall statistics
    print(f"\n\n{'='*60}")
    print("Overall Statistics")
    print(f"{'='*60}")
    print(f"Episodes played: {num_episodes}")
    print(f"Mean reward: {np.mean(all_rewards):.2f} ± {np.std(all_rewards):.2f}")
    print(f"Mean length: {np.mean(all_lengths):.1f} ± {np.std(all_lengths):.1f}")
    print(f"Mean max X: {np.mean(all_max_x):.1f} ± {np.std(all_max_x):.1f}")
    print(f"Success rate: {np.mean(all_successes)*100:.1f}% ({sum(all_successes)}/{num_episodes})")
    print(f"{'='*60}\n")

    # Close environment
    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Play Super Mario Bros with trained DreamerV3 agent"
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to trained checkpoint (e.g., logs/checkpoints/checkpoint_50000.pt)'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='configs/dreamerv3_config.yaml',
        help='Path to config file (default: configs/dreamerv3_config.yaml)'
    )
    parser.add_argument(
        '--episodes',
        type=int,
        default=1,
        help='Number of episodes to play (default: 1)'
    )
    parser.add_argument(
        '--stochastic',
        action='store_true',
        help='Use stochastic policy instead of deterministic (default: deterministic)'
    )
    parser.add_argument(
        '--delay',
        type=float,
        default=0.0,
        help='Delay between frames in seconds for slower playback (default: 0.0)'
    )

    args = parser.parse_args()

    # Check if checkpoint exists
    if not Path(args.checkpoint).exists():
        print(f"Error: Checkpoint not found: {args.checkpoint}")
        print("\nAvailable checkpoints:")
        checkpoint_dir = Path("logs/checkpoints")
        if checkpoint_dir.exists():
            checkpoints = sorted(checkpoint_dir.glob("checkpoint_*.pt"))
            if checkpoints:
                for ckpt in checkpoints:
                    print(f"  {ckpt}")
            else:
                print("  No checkpoints found in logs/checkpoints/")
        else:
            print("  Checkpoint directory does not exist: logs/checkpoints/")
        exit(1)

    # Play!
    try:
        play_mario(
            checkpoint_path=args.checkpoint,
            config_path=args.config,
            num_episodes=args.episodes,
            deterministic=not args.stochastic,
            render_delay=args.delay
        )
    except KeyboardInterrupt:
        print("\n\nPlayback interrupted by user.")
    except Exception as e:
        print(f"\n\nError during playback: {e}")
        import traceback
        traceback.print_exc()
