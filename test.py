"""
Generic testing script

This script implements a generic testing pipeline:
1) Parses command-line arguments to select an algorithm and config file.
2) Dynamically imports the appropriate tester class.
3) Initializes and runs the tester.

Usage:
python test.py --algo ppo --config config.py --model_path models/best_ppo_mario_model.pth --episodes 5
"""

import os
import argparse
import importlib

def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description='Generic RL Testing')

    # Algorithm and config
    parser.add_argument('--algo', type=str, required=True, help='Algorithm to use (e.g., ppo, dreamer-v3)')
    parser.add_argument('--config', type=str, default='configs/ppo_config.py', help='Path to configuration file')

    # Model
    parser.add_argument('--model_path', type=str, required=True,
                       help='path to trained model file')
    parser.add_argument('--load_best', action='store_true',
                       help='load the best_ model variant if available')
    
    # Evaluation
    parser.add_argument('--episodes', type=int, default=5,
                       help='number of test episodes')
    parser.add_argument('--world', type=str, default='1-1',
                       help='test world (e.g., 1-1, 1-2)')
    parser.add_argument('--worlds', nargs='+', default=None,
                       help='multi-world testing')
    parser.add_argument('--deterministic', action='store_true',
                       help='use deterministic policy')
    parser.add_argument('--render', action='store_true', default=True,
                       help='render gameplay')
    parser.add_argument('--render_mode', type=str, default='human',
                       choices=['human', 'rgb_array'],
                       help='render mode')
    parser.add_argument('--render_delay', type=float, default=0.02,
                       help='sleep between frames (seconds)')
    
    # Analysis
    parser.add_argument('--save_video', action='store_true',
                       help='save gameplay video')
    parser.add_argument('--analyze_actions', action='store_true',
                       help='analyze action distribution')
    parser.add_argument('--show_values', action='store_true',
                       help='show state values')
    
    # System
    parser.add_argument('--device', type=str, default=None,
                       help='compute device (cuda/cpu)')
    parser.add_argument('--max_steps', type=int, default=10000,
                       help='max steps per episode (failsafe)')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    
    return parser.parse_args()

def main():
    """Program entry point"""
    args = parse_args()

    # Dynamically import the tester
    try:
        tester_module = importlib.import_module(f"algorithms.{args.algo}.tester")
        tester_class = getattr(tester_module, f"{args.algo.upper()}Tester")
    except (ImportError, AttributeError):
        print(f"Error: Could not find tester for algorithm '{args.algo}'."
              f"Make sure you have a 'tester.py' file with a '{args.algo.upper()}Tester' class in the 'algorithms/{args.algo}' directory.")
        return

    # Load config
    if os.path.exists(args.config):
        import sys
        sys.path.append(os.path.dirname(os.path.abspath(args.config)))
        importlib.import_module(os.path.basename(args.config).replace('.py', ''))

    print(f"Starting {args.algo.upper()} testing...")
    
    # Create and run the tester
    tester = tester_class(args)
    tester.test()


if __name__ == "__main__":
    main()