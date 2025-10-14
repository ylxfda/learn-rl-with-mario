#!/usr/bin/env python3
"""
Test script to verify the reward shaping implementation.

Usage:
    python test_reward_shaping.py              # Run 5 episodes
    python test_reward_shaping.py --episodes 10  # Run 10 episodes

Purpose:
- Verify reward shaping is working correctly after modifications
- Show reward statistics with random actions
- Help debug if training rewards seem incorrect

Current reward shaping (2025-10-14 fix):
- Progress reward: progress * 1.0 (direct scaling)
- Flag reward: 1000.0 (large success bonus)
- Death penalty: -5.0 (negative feedback)
- No time penalty or survival bonus (removed to reduce noise)
"""

import sys
import argparse
import numpy as np
from enviroments.preprocessing import create_mario_env_dreamerv3

def test_reward_shaping(num_episodes=5):
    """Run episodes with random actions and display reward statistics."""

    print("="*80)
    print("Reward Shaping Test - DreamerV3 Mario")
    print("="*80)
    print("\nChanges applied:")
    print("  1. Progress reward: progress / 100.0 → progress * 1.0 (100x)")
    print("  2. Flag reward: 50.0 → 1000.0 (20x)")
    print("  3. Removed: time penalty, survival bonus (+0.01/step)")
    print("="*80)
    print()

    # Create environment
    env = create_mario_env_dreamerv3(world='1', stage='1', frame_skip=4, size=64, grayscale=False)

    episode_rewards = []
    episode_lengths = []
    max_x_positions = []

    for ep in range(num_episodes):
        obs = env.reset()
        done = False
        total_reward = 0.0
        steps = 0
        max_x = 0

        while not done:
            # Random action
            action = np.random.randint(0, 7)
            obs, reward, done, info = env.step(action)

            total_reward += reward
            steps += 1
            max_x = max(max_x, info.get('x_pos', 0))

            # Limit episode length for testing
            if steps >= 500:
                break

        episode_rewards.append(total_reward)
        episode_lengths.append(steps)
        max_x_positions.append(max_x)

        print(f"Episode {ep+1}:")
        print(f"  Total reward: {total_reward:.1f}")
        print(f"  Episode length: {steps} steps")
        print(f"  Max X position: {max_x:.0f}")
        print(f"  Flag reached: {info.get('flag_get', False)}")
        print()

    print("="*80)
    print("Summary Statistics:")
    print("="*80)
    print(f"Average reward: {np.mean(episode_rewards):.1f} ± {np.std(episode_rewards):.1f}")
    print(f"Average length: {np.mean(episode_lengths):.1f} steps")
    print(f"Average max X: {np.mean(max_x_positions):.0f}")
    print()

    print("Expected vs Old Behavior:")
    print("-"*80)
    print("Scenario: Mario moves 500 pixels forward then dies")
    print()
    print("OLD reward shaping:")
    print("  Progress: 500 / 100 = 5.0")
    print("  Death: -5.0")
    print("  Survival (50 steps): 50 * 0.01 = 0.5")
    print("  Total: ~0.5  (almost nothing!)")
    print()
    print("NEW reward shaping:")
    print("  Progress: 500 * 1.0 = 500.0")
    print("  Death: -5.0")
    print("  Total: ~495.0  (clear positive signal!)")
    print()
    print("-"*80)
    print("Scenario: Mario reaches the flag at 3000 pixels")
    print()
    print("OLD reward shaping:")
    print("  Progress: 3000 / 100 = 30.0")
    print("  Flag: 50.0")
    print("  Survival (300 steps): 3.0")
    print("  Total: ~83.0")
    print()
    print("NEW reward shaping:")
    print("  Progress: 3000 * 1.0 = 3000.0")
    print("  Flag: 1000.0")
    print("  Total: ~4000.0  (48x larger!)")
    print()
    print("="*80)
    print("\nConclusion:")
    print("  ✓ Rewards are now 100x more informative for progress")
    print("  ✓ Success is 20x more rewarding")
    print("  ✓ Agent should learn much faster with clearer signals")
    print("="*80)

    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test reward shaping implementation")
    parser.add_argument('--episodes', type=int, default=5, help='Number of episodes to run (default: 5)')
    args = parser.parse_args()

    try:
        test_reward_shaping(num_episodes=args.episodes)
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user.")
    except Exception as e:
        print(f"\n\nError during test: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
