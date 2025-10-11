#!/usr/bin/env python3
"""
Main training script for DreamerV3 on Super Mario Bros.

Usage:
    python train_dreamerv3.py --config configs/dreamerv3_config.yaml
    python train_dreamerv3.py --config configs/dreamerv3_config.yaml --checkpoint logs/checkpoints/checkpoint_100000.pt

This script initializes the trainer and starts the training loop.
"""

import argparse
import sys
from pathlib import Path

from algorithms.dreamer_v3.training.trainer import DreamerV3Trainer


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Train DreamerV3 on Super Mario Bros World 1-1",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '--config',
        type=str,
        default='configs/dreamerv3_config.yaml',
        help='Path to configuration YAML file'
    )

    parser.add_argument(
        '--checkpoint',
        type=str,
        default=None,
        help='Path to checkpoint file to resume training from'
    )

    parser.add_argument(
        '--eval-only',
        action='store_true',
        help='Only run evaluation (requires --checkpoint)'
    )

    parser.add_argument(
        '--num-eval-episodes',
        type=int,
        default=None,
        help='Number of evaluation episodes (overrides config)'
    )

    args = parser.parse_args()

    # Validate arguments
    if args.eval_only and not args.checkpoint:
        parser.error("--eval-only requires --checkpoint")

    print("="*70)
    print("DreamerV3 for Super Mario Bros - World 1-1")
    print("="*70)
    print(f"\nConfiguration: {args.config}")
    if args.checkpoint:
        print(f"Checkpoint: {args.checkpoint}")
    print()

    # Create trainer
    try:
        trainer = DreamerV3Trainer(args.config)
    except Exception as e:
        print(f"Error creating trainer: {e}")
        sys.exit(1)

    # Load checkpoint if provided
    if args.checkpoint:
        try:
            trainer.load_checkpoint(args.checkpoint)
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            sys.exit(1)

    # Run evaluation only or full training
    if args.eval_only:
        print("\n" + "="*70)
        print("Running Evaluation Only")
        print("="*70 + "\n")

        num_episodes = args.num_eval_episodes
        trainer.evaluate(num_episodes=num_episodes)
    else:
        # Start training
        try:
            trainer.train()
        except KeyboardInterrupt:
            print("\n\nTraining interrupted by user.")
            print("Saving checkpoint before exit...")
            trainer.save_checkpoint()
            print("Checkpoint saved. Exiting.")
        except Exception as e:
            print(f"\nError during training: {e}")
            import traceback
            traceback.print_exc()
            print("\nSaving emergency checkpoint...")
            trainer.save_checkpoint()
            sys.exit(1)
        finally:
            trainer.env.close()

    print("\n" + "="*70)
    print("Done!")
    print("="*70)


if __name__ == "__main__":
    main()
