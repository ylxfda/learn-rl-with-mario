"""
PPO Mario testing script

Loads a trained PPO model and runs gameplay tests with live rendering.
Supports single/multi-episode and multi-world evaluations.

Usage:
python test.py --model_path models/best_ppo_mario_model.pth --episodes 5
"""

import os
import time
import argparse
import numpy as np
import torch
from tqdm import tqdm

# Project modules
from config import Config
from enviroments.mario_env import create_mario_environment, MultiWorldMarioEnvironment
from algorithms.ppo import create_ppo_algorithm
from algorithms.base import ModelManager


def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description='Test trained PPO Mario model')
    
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
    
    return parser.parse_args()


class PPOTester:
    """PPO tester"""
    
    def __init__(self, args):
        """
        Initialize tester.
        
        Args:
            args: Parsed CLI args
        """
        self.args = args
        self.device = torch.device(args.device) if args.device else Config.DEVICE
        
        print("Initializing PPO Mario tester...")
        print("=" * 50)
        print(f"Model path: {args.model_path}")
        print(f"Device: {self.device}")
        print(f"Episodes: {args.episodes}")
        print(f"Deterministic: {args.deterministic}")
        print("=" * 50)
        
        # Load model
        self._load_model()
        
        # Create environment
        self._create_environment()
        
        # Test statistics container
        self.test_stats = []
        
    def _load_model(self):
        """Load trained model"""
        print("Loading PPO model...")
        
        if not os.path.exists(self.args.model_path):
            raise FileNotFoundError(f"Model file not found: {self.args.model_path}")
        
        # Peek checkpoint to get model info
        checkpoint = torch.load(self.args.model_path, map_location=self.device)
        
        # Extract observation/action info from checkpoint
        obs_shape = checkpoint.get('obs_shape', (Config.FRAME_STACK, Config.FRAME_SIZE, Config.FRAME_SIZE))
        action_dim = 12
        
        # Mock spaces
        class MockObsSpace:
            def __init__(self, shape):
                self.shape = shape
        
        class MockActionSpace:
            def __init__(self, n):
                self.n = n
        
        obs_space = MockObsSpace(obs_shape)
        action_space = MockActionSpace(action_dim)
        
        # Create PPO instance
        self.ppo = create_ppo_algorithm(
            observation_space=obs_space,
            action_space=action_space,
            device=self.device
        )
        
        # Load model weights
        model_manager = ModelManager()
        try:
            model_manager.load_model(
                self.ppo, 
                filename=os.path.basename(self.args.model_path), 
                load_best=self.args.load_best
            )
        except:
            # Fallback: load directly through algorithm
            self.ppo.load_model(self.args.model_path)
        
        # Eval mode
        self.ppo.eval()
        
        print("Model loaded!")
        
        # Print checkpoint info
        if 'total_steps' in checkpoint:
            print(f"Total steps: {checkpoint['total_steps']:,}")
        if 'total_episodes' in checkpoint:
            print(f"Total episodes: {checkpoint['total_episodes']:,}")
        if 'best_reward' in checkpoint:
            print(f"Best reward: {checkpoint['best_reward']:.2f}")
    
    def _create_environment(self):
        """Create test environment"""
        print("Creating test environment...")
        
        render_mode = self.args.render_mode if self.args.render else None
        
        if self.args.worlds:
            # Multi-world testing
            self.env = MultiWorldMarioEnvironment(
                worlds=self.args.worlds,
                render_mode=render_mode,
                random_start=True
            )
            print(f"Multi-worlds: {self.args.worlds}")
        else:
            # Single world
            self.env = create_mario_environment(
                world=self.args.world,
                render_mode=render_mode
            )
            print(f"World: {self.args.world}")
        
        print("Environment ready!")
    
    def test_episode(self, episode_num):
        """
        Run a single test episode.
        
        Args:
            episode_num (int): Episode index
            
        Returns:
            dict: Episode statistics
        """
        print(f"\nStarting episode {episode_num + 1}")
        
        # Reset env
        observation = self.env.reset()
        observation = torch.FloatTensor(observation).unsqueeze(0).to(self.device)
        
        # Episode stats
        episode_reward = 0
        episode_length = 0
        actions_taken = []
        rewards_received = []
        values_estimated = []
        action_probs_history = []
        
        done = False
        start_time = time.time()
        
        while not done and episode_length < self.args.max_steps:
            # Get action
            with torch.no_grad():
                actions, extra_info = self.ppo.act(
                    observation, 
                    deterministic=self.args.deterministic
                )
                
                # Optional analysis traces
                if self.args.analyze_actions or self.args.show_values:
                    action_probs = self.ppo.get_action_probabilities(observation)
                    values = extra_info.get('values', torch.tensor([0.0]))
                    
                    action_probs_history.append(action_probs.cpu().numpy()[0])
                    values_estimated.append(values.cpu().numpy()[0] if hasattr(values, 'cpu') else values)
            
            action = actions.item()
            actions_taken.append(action)
            
            # Step env
            next_observation, reward, done, info = self.env.step(action)
            
            # Render
            if self.args.render:
                self.env.render()
                time.sleep(self.args.render_delay)  # slow down for visibility
            
            # Update counters
            episode_reward += reward
            episode_length += 1
            rewards_received.append(reward)
            
            # Live info
            if episode_length % 100 == 0:
                print(f"  Steps: {episode_length}, Reward: {episode_reward:.2f}, Action: {action}")
            
            # Prepare next step
            observation = torch.FloatTensor(next_observation).unsqueeze(0).to(self.device)
        
        # Episode summary
        episode_time = time.time() - start_time
        
        episode_stats = {
            'episode': episode_num + 1,
            'reward': episode_reward,
            'length': episode_length,
            'time': episode_time,
            'fps': episode_length / episode_time if episode_time > 0 else 0,
            'success': info.get('flag_get', False) if 'flag_get' in info else False,
            'x_pos': info.get('x_pos', 0) if 'x_pos' in info else 0,
            'world': getattr(self.env, 'world', self.args.world),
            'actions': actions_taken,
            'rewards': rewards_received,
        }
        
        # Add analysis fields
        if self.args.analyze_actions and action_probs_history:
            episode_stats['action_distribution'] = np.mean(action_probs_history, axis=0)
            episode_stats['action_entropy'] = np.mean([
                -np.sum(probs * np.log(probs + 1e-8)) for probs in action_probs_history
            ])
        
        if self.args.show_values and values_estimated:
            episode_stats['avg_value'] = np.mean(values_estimated)
            episode_stats['max_value'] = np.max(values_estimated)
            episode_stats['min_value'] = np.min(values_estimated)
        
        # Print episode result
        print(f"Episode {episode_num + 1} done:")
        print(f"  Reward: {episode_reward:.2f}")
        print(f"  Steps: {episode_length}")
        print(f"  Time: {episode_time:.2f}s")
        print(f"  Success: {'yes' if episode_stats['success'] else 'no'}")
        print(f"  Final x_pos: {episode_stats['x_pos']:.1f}")
        
        return episode_stats
    
    def analyze_results(self):
        """Analyze test results"""
        if not self.test_stats:
            return
        
        print(f"\n{'='*60}")
        print("Test Results Summary")
        print(f"{'='*60}")
        
        # Basic stats
        rewards = [stat['reward'] for stat in self.test_stats]
        lengths = [stat['length'] for stat in self.test_stats]
        times = [stat['time'] for stat in self.test_stats]
        successes = [stat['success'] for stat in self.test_stats]
        x_positions = [stat['x_pos'] for stat in self.test_stats]
        
        print(f"Episodes: {len(self.test_stats)}")
        print(f"Avg reward: {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
        print(f"Max reward: {np.max(rewards):.2f}")
        print(f"Min reward: {np.min(rewards):.2f}")
        print(f"Avg length: {np.mean(lengths):.1f} ± {np.std(lengths):.1f}")
        print(f"Success rate: {np.mean(successes)*100:.1f}% ({sum(successes)}/{len(successes)})")
        print(f"Avg final x_pos: {np.mean(x_positions):.1f}")
        print(f"Avg FPS: {np.mean([s['fps'] for s in self.test_stats]):.1f}")
        
        # Action analysis
        if self.args.analyze_actions:
            print(f"\nAction analysis:")
            
            # Merge all actions
            all_actions = []
            for stat in self.test_stats:
                all_actions.extend(stat['actions'])
            
            # Distribution
            action_meanings = [
                'NOOP', 'right', 'right_A', 'right_B', 'right_A_B', 'A', 'left'
            ]
            
            action_counts = np.bincount(all_actions, minlength=7)
            action_ratios = action_counts / len(all_actions)
            
            for i, (meaning, ratio) in enumerate(zip(action_meanings, action_ratios)):
                print(f"  {meaning}: {ratio*100:.1f}% ({action_counts[i]} times)")
            
            # Average entropy
            avg_entropies = [stat.get('action_entropy', 0) for stat in self.test_stats]
            if any(avg_entropies):
                print(f"Avg action entropy: {np.mean(avg_entropies):.3f}")
        
        # Value function analysis
        if self.args.show_values:
            print(f"\nValue function analysis:")
            avg_values = [stat.get('avg_value', 0) for stat in self.test_stats]
            max_values = [stat.get('max_value', 0) for stat in self.test_stats]
            min_values = [stat.get('min_value', 0) for stat in self.test_stats]
            
            print(f"Avg state value: {np.mean(avg_values):.2f} ± {np.std(avg_values):.2f}")
            print(f"Max state value: {np.mean(max_values):.2f}")
            print(f"Min state value: {np.mean(min_values):.2f}")
        
        # Per-world analysis (multi-world only)
        if hasattr(self.env, 'worlds'):
            print(f"\nWorld analysis:")
            world_stats = {}
            for stat in self.test_stats:
                world = stat['world']
                if world not in world_stats:
                    world_stats[world] = {'rewards': [], 'successes': []}
                world_stats[world]['rewards'].append(stat['reward'])
                world_stats[world]['successes'].append(stat['success'])
            
            for world, data in world_stats.items():
                avg_reward = np.mean(data['rewards'])
                success_rate = np.mean(data['successes']) * 100
                print(f"  {world}: avg_reward={avg_reward:.2f}, success_rate={success_rate:.1f}%")
    
    def test(self):
        """Run the full test loop"""
        print(f"\nStarting test for {self.args.episodes} episodes...")
        
        try:
            # Run test episodes
            for episode in tqdm(range(self.args.episodes), desc="Testing"):
                episode_stats = self.test_episode(episode)
                self.test_stats.append(episode_stats)
            
            # Analyze results
            self.analyze_results()
            
            # Optional: save results
            if self.args.save_video or len(self.args.model_path.split('/')) > 1:
                results_dir = os.path.dirname(self.args.model_path)
                results_file = os.path.join(results_dir, 'test_results.txt')
                
                with open(results_file, 'w', encoding='utf-8') as f:
                    f.write("PPO Mario Test Results\n")
                    f.write("="*50 + "\n")
                    f.write(f"Model: {self.args.model_path}\n")
                    f.write(f"Episodes: {self.args.episodes}\n")
                    f.write(f"Deterministic: {self.args.deterministic}\n")
                    f.write(f"World(s): {self.args.world if not self.args.worlds else self.args.worlds}\n")
                    f.write("\nEpisode details:\n")
                    
                    for i, stat in enumerate(self.test_stats):
                        f.write(f"Episode {i+1}: reward={stat['reward']:.2f}, "
                               f"length={stat['length']}, success={stat['success']}\n")
                
                print(f"\nResults written to: {results_file}")
        
        except KeyboardInterrupt:
            print(f"\nTest interrupted, completed {len(self.test_stats)} episodes")
            if self.test_stats:
                self.analyze_results()
        
        except Exception as e:
            print(f"\nError during testing: {e}")
            import traceback
            traceback.print_exc()
        
        finally:
            # Cleanup
            if hasattr(self, 'env'):
                self.env.close()
            
            print(f"\nTest finished. Thanks for using the PPO Mario tester!")

def main():
    """Program entry point"""
    # Parse CLI args
    args = parse_args()
    
    print("PPO Mario")
    # print("=" * 60)
    # print(f"Device: {torch.device(args.device) if args.device else Config.DEVICE}")
    # # print(f"Num envs: {args.num_envs}")
    # print(f"Worlds: {args.worlds}")
    # print(f"Max episodes: {args.max_episodes:,}")
    # print(f"Max steps: {args.max_steps:,}")
    # print("=" * 60)
    
    # Create tester
    tester = PPOTester(args)
    
    # Run tests
    tester.test()


if __name__ == "__main__":
    main()
