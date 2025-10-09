"""
Generic training script

This script implements a generic training pipeline:
1) Parses command-line arguments to select an algorithm and config file.
2) Dynamically imports the appropriate trainer class.
3) Initializes and runs the trainer.

Usage:
python train.py --algo ppo --config config.py
"""

import os
import time
import argparse
import importlib


def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description='Generic RL Training')
    
    # Algorithm and config
    parser.add_argument('--algo', type=str, required=True, help='Algorithm to use (e.g., ppo, dreamer-v3)')
    parser.add_argument('--config', type=str, default='configs/ppo_config.py', help='Path to configuration file')

    # Environment params
    parser.add_argument('--num_envs', type=int, help='number of parallel environments')
    parser.add_argument('--render_env', type=int, help='ID of the env to render (for live preview)')
    
    # Training params
    parser.add_argument('--max_episodes', type=int, help='maximum number of training episodes')
    parser.add_argument('--max_steps', type=int, help='maximum number of training steps')
    parser.add_argument('--save_freq', type=int, help='save model every N updates')
    parser.add_argument('--log_freq', type=int, help='log stats every N updates')
    
    # Algorithm-specific params (will be passed to the trainer)
    # PPO params
    parser.add_argument('--learning_rate', type=float, help='learning rate')
    parser.add_argument('--ppo_epochs', type=int, help='PPO epochs per update')
    parser.add_argument('--clip_epsilon', type=float, help='PPO clip epsilon')
    parser.add_argument('--steps_per_update', type=int, help='steps to collect per update')
    
    # System params
    parser.add_argument('--device', type=str, help='compute device (cuda/cpu)')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--resume', type=str, help='path to resume training from a saved model')
    parser.add_argument('--experiment_name', type=str, help='experiment name (used for logs)')
    
    return parser.parse_args()

def main():
    """Program entry point"""
    args = parse_args()

    # Dynamically import the trainer
    try:
        trainer_module = importlib.import_module(f"algorithms.{args.algo}.trainer")
        trainer_class = getattr(trainer_module, f"{args.algo.upper()}Trainer")
    except (ImportError, AttributeError) as e:
        print(f"Error: Could not find trainer for algorithm '{args.algo}'."
              f"Make sure you have a 'trainer.py' file with a '{args.algo.upper()}Trainer' class in the 'algorithms/{args.algo}' directory.")
        import traceback
        traceback.print_exc()
        return

    # Load config
    # Note: This is a simple way to load config. A more robust solution might be needed for complex projects.
    # For now, we assume the config file is a python file that modifies the `Config` class.
    if os.path.exists(args.config):
        import sys
        sys.path.append(os.path.dirname(os.path.abspath(args.config)))
        importlib.import_module(os.path.basename(args.config).replace('.py', ''))

    print(f"Starting {args.algo.upper()} training...")
    
    # Create and run the trainer
    trainer = trainer_class(args)
    trainer.train()


if __name__ == "__main__":
    main()