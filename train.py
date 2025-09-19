"""
PPO Mario training script

This script implements a complete PPO training pipeline:
1) Create parallel Mario environments
2) Initialize PPO algorithm
3) Collect rollouts
4) Update networks
5) Monitor performance and save models

Usage:
python train.py
"""

import os
import time
import argparse
import numpy as np
import torch
from tqdm import tqdm

# Project modules
from config import Config
from enviroments.parallel_envs import create_parallel_mario_envs
from algorithms.ppo import create_ppo_algorithm
from utils.replay_buffer import RolloutBuffer
from utils.logger import TrainingLogger, PerformanceMonitor, ProgressTracker
from algorithms.base import ModelManager

def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description='PPO Mario Training')
    
    # Environment params
    parser.add_argument('--num_envs', type=int, default=Config.NUM_ENVS,
                       help='number of parallel environments')
    # World selection is controlled by Config.WORLD_STAGE
    parser.add_argument('--render_env', type=int, default=None,
                       help='ID of the env to render (for live preview)')
    
    # Training params
    parser.add_argument('--max_episodes', type=int, default=Config.MAX_EPISODES,
                       help='maximum number of training episodes')
    parser.add_argument('--max_steps', type=int, default=Config.MAX_STEPS,
                       help='maximum number of training steps')
    parser.add_argument('--save_freq', type=int, default=Config.SAVE_FREQ,
                       help='save model every N updates')
    parser.add_argument('--log_freq', type=int, default=Config.LOG_FREQ,
                       help='log stats every N updates')
    
    # PPO params
    parser.add_argument('--learning_rate', type=float, default=Config.LEARNING_RATE,
                       help='learning rate')
    parser.add_argument('--ppo_epochs', type=int, default=Config.PPO_EPOCHS,
                       help='PPO epochs per update')
    parser.add_argument('--clip_epsilon', type=float, default=Config.CLIP_EPSILON,
                       help='PPO clip epsilon')
    parser.add_argument('--steps_per_update', type=int, default=Config.STEPS_PER_UPDATE,
                       help='steps to collect per update')
    
    # System params
    parser.add_argument('--device', type=str, default=None,
                       help='compute device (cuda/cpu)')
    parser.add_argument('--seed', type=int, default=Config.SEED,
                       help='random seed')
    parser.add_argument('--resume', type=str, default=None,
                       help='path to resume training from a saved model')
    parser.add_argument('--experiment_name', type=str, default=None,
                       help='experiment name (used for logs)')
    
    return parser.parse_args()


def set_seed(seed):
    """Set random seeds"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    # Deterministic settings (may impact performance)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False


class PPOTrainer:
    """PPO trainer"""
    
    def __init__(self, args):
        """
        Initialize trainer.
        
        Args:
            args: Parsed CLI args
        """
        self.args = args
        self.device = torch.device(args.device) if args.device else Config.DEVICE
        print(f"Using device: {self.device}")
        
        # Set seeds
        set_seed(args.seed)
        
        # Apply CLI overrides to Config
        Config.LEARNING_RATE = args.learning_rate
        Config.PPO_EPOCHS = args.ppo_epochs
        Config.CLIP_EPSILON = args.clip_epsilon
        Config.STEPS_PER_UPDATE = args.steps_per_update
        
        # Print config
        Config.print_config()
        
        # Prepare directories
        os.makedirs(Config.MODEL_DIR, exist_ok=True)
        os.makedirs(Config.LOG_DIR, exist_ok=True)
        
        # Initialize components
        self._init_environment()
        self._init_algorithm() 
        self._init_buffer()
        self._init_logging()
        self._init_monitoring()
        
        print("PPO trainer initialized!")
    
    def _init_environment(self):
        """Initialize environments"""
        print("Creating parallel Mario environments...")
        
        self.envs = create_parallel_mario_envs(
            num_envs=self.args.num_envs,
            worlds=Config.WORLD_STAGE,
            use_subprocess=True,  # use subprocesses for better throughput
            render_env_id=self.args.render_env
        )
        
        self.observation_space = self.envs.observation_space
        self.action_space = self.envs.action_space
        
        print(f"Environments ready: {len(self.envs)} parallel envs")
    
    def _init_algorithm(self):
        """Initialize PPO"""
        print("Initializing PPO algorithm...")
        
        self.ppo = create_ppo_algorithm(
            observation_space=self.observation_space,
            action_space=self.action_space,
            device=self.device,
            logger=None  # set later
        )
        
        # Load checkpoint if provided
        if self.args.resume:
            print(f"Resuming from {self.args.resume} ...")
            model_manager = ModelManager()
            model_manager.load_model(self.ppo, self.args.resume)
    
    def _init_buffer(self):
        """Initialize rollout buffer"""
        print("Initializing rollout buffer...")
        
        self.rollout_buffer = RolloutBuffer(
            buffer_size=Config.STEPS_PER_UPDATE,
            num_envs=self.args.num_envs,
            obs_shape=self.observation_space.shape,
            action_dim=1,  # discrete action
            device=self.device
        )
        
        print(f"Buffer capacity: {len(self.rollout_buffer):,} transitions")
    
    def _init_logging(self):
        """Initialize logging"""
        print("Initializing logging...")
        
        self.logger = TrainingLogger(
            log_dir=Config.LOG_DIR,
            experiment_name=self.args.experiment_name
        )
        
        # Connect logger to algorithm
        self.ppo.logger = self.logger
        
        # Progress tracker
        self.progress_tracker = ProgressTracker(
            target_reward=Config.TARGET_REWARD,
            patience=Config.PATIENCE  # early stop after no improvement
        )
    
    def _init_monitoring(self):
        """Initialize performance monitoring"""
        self.performance_monitor = PerformanceMonitor()
        self.model_manager = ModelManager()
        
        # Training stats
        self.episode_rewards = []
        self.episode_lengths = []
        self.best_avg_reward = float('-inf')
        self.episodes_since_best = 0

    def _compute_world_sampling_weights(self, eval_stats):
        """
        Compute sampling weights per world based on eval results (harder worlds get higher weight).
        
        Strategy:
        - Read per-world avg reward from eval_stats (eval_avg_reward_X_Y)
        - Use (max_reward - reward) as difficulty score
        - Apply alpha and min-weight, then normalize
        """
        worlds = Config.WORLD_STAGE
        if isinstance(worlds, str):
            worlds = [worlds]
        if not worlds or len(worlds) == 1:
            return None

        # Collect per-world average rewards
        avg_rewards = {}
        for w in worlds:
            tag = w.replace('-', '_')
            key = f'eval_avg_reward_{tag}'
            avg_rewards[w] = float(eval_stats.get(key, eval_stats.get('eval_avg_reward', 0.0)))

        # Compute difficulty scores
        max_avg = max(avg_rewards.values()) if avg_rewards else 0.0
        eps = 1e-6
        alpha = getattr(Config, 'WORLD_SAMPLING_ALPHA', 1.0)
        base = getattr(Config, 'WORLD_SAMPLING_MIN_WEIGHT', 0.05)

        raw_weights = {}
        for w, r in avg_rewards.items():
            score = max_avg - r  # lower reward -> higher score
            weight = (score + eps) ** alpha + base
            raw_weights[w] = weight

        # Normalize
        total = sum(raw_weights.values())
        if total <= 0:
            return None
        weights = {w: (raw_weights[w] / total) for w in worlds}
        return weights
    
    def collect_rollouts(self):
        """
        Collect one batch of rollouts.
        
        Returns:
            dict: Collection statistics
        """
        self.ppo.eval()  # evaluation mode (disable dropout etc.)
        
        # Reset buffer
        self.rollout_buffer.reset()
        
        # Reset envs and get initial observations
        observations = self.envs.reset()
        
        # Collection stats
        collect_stats = {
            'episodes_completed': 0,
            'total_reward': 0.0,
            'avg_episode_length': 0.0,
        }
        
        episode_rewards = []
        episode_lengths = []
        current_episode_rewards = np.zeros(self.args.num_envs)
        current_episode_lengths = np.zeros(self.args.num_envs)
        
        # Collect fixed number of steps
        for step in range(Config.STEPS_PER_UPDATE):
            # Select actions
            with torch.no_grad():
                actions, extra_info = self.ppo.act(observations)
                values = extra_info['values']
                log_probs = extra_info['log_probs']
            
            # Step environments
            next_observations, rewards, dones, infos = self.envs.step(actions)
            
            # Store transition
            self.rollout_buffer.add(
                states=observations,
                actions=actions,
                rewards=rewards,
                values=values,
                log_probs=log_probs,
                dones=dones
            )
            
            # Update stats
            current_episode_rewards += rewards.cpu().numpy()
            current_episode_lengths += 1
            
            # Handle episode termination
            for i, done in enumerate(dones):
                if done:
                    episode_reward = current_episode_rewards[i]
                    episode_length = current_episode_lengths[i]
                    
                    episode_rewards.append(episode_reward)
                    episode_lengths.append(episode_length)
                    
                    # Log to logger
                    info = infos[i] if i < len(infos) else {}
                    self.logger.log_episode(episode_reward, episode_length, info)
                    
                    # Update progress tracker
                    progress_info = self.progress_tracker.update(episode_reward)
                    
                    # Reset counters
                    current_episode_rewards[i] = 0
                    current_episode_lengths[i] = 0
                    
                    collect_stats['episodes_completed'] += 1
            
            # Next observations
            observations = next_observations
        
        # Value of last observations (for GAE)
        with torch.no_grad():
            next_values = self.ppo.compute_value(next_observations)
        
        # Compute advantages and returns
        self.rollout_buffer.compute_advantages_and_returns(
            next_values=next_values,
            gamma=Config.GAMMA,
            gae_lambda=Config.GAE_LAMBDA
        )
        
        # Finalize collection stats
        if episode_rewards:
            collect_stats['total_reward'] = sum(episode_rewards)
            collect_stats['avg_episode_length'] = np.mean(episode_lengths)
            
            # Update global stats
            self.episode_rewards.extend(episode_rewards)
            self.episode_lengths.extend(episode_lengths)
        
        return collect_stats
    
    def train_step(self):
        """
        Run one full training step.
        
        Returns:
            dict: Training stats
        """
        # 1) Collect data
        collect_stats = self.collect_rollouts()
        
        # 2) Update policy
        self.ppo.train()  # set train mode
        update_stats = self.ppo.update(self.rollout_buffer)
        
        # 3) Merge stats
        train_stats = {**collect_stats, **update_stats}
        
        # 4) Bump total steps
        self.ppo.total_steps += Config.STEPS_PER_UPDATE * self.args.num_envs
        
        return train_stats
    
    def evaluate_model(self, num_episodes=5):
        """
        Evaluate current model.
        
        - Evaluate each configured world for num_episodes
        - Aggregate overall and per-world stats; overall guides early stop/best model
        
        Args:
            num_episodes (int): episodes per world
            
        Returns:
            dict: overall and per-world statistics
        """
        self.ppo.eval()
        from enviroments.mario_env import create_mario_environment

        # Worlds to evaluate (string or list)
        worlds = Config.WORLD_STAGE
        if isinstance(worlds, str):
            worlds = [worlds]

        print(f"Evaluating model (per-world {num_episodes} episodes): {worlds}")

        # Overall accumulators
        all_rewards = []
        all_lengths = []

        # Per-world stats
        per_world_stats = {}

        for world in worlds:
            eval_env = create_mario_environment(world=world, render_mode=None)
            world_rewards = []
            world_lengths = []

            for episode in range(num_episodes):
                obs = eval_env.reset()
                obs = torch.FloatTensor(obs).unsqueeze(0).to(self.device)

                episode_reward = 0.0
                episode_length = 0
                done = False

                while not done:
                    with torch.no_grad():
                        actions, _ = self.ppo.act(obs, deterministic=True)
                    obs, reward, done, info = eval_env.step(actions.item())
                    obs = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
                    episode_reward += reward
                    episode_length += 1
                    if episode_length > 5000:
                        break

                world_rewards.append(episode_reward)
                world_lengths.append(episode_length)
                print(f"  [{world}] Episode {episode+1}: reward={episode_reward:.2f}, length={episode_length}")

            eval_env.close()

            # Save per-world stats
            per_world_stats[world] = {
                'avg_reward': float(np.mean(world_rewards)) if world_rewards else 0.0,
                'std_reward': float(np.std(world_rewards)) if world_rewards else 0.0,
                'max_reward': float(np.max(world_rewards)) if world_rewards else 0.0,
                'min_reward': float(np.min(world_rewards)) if world_rewards else 0.0,
                'avg_length': float(np.mean(world_lengths)) if world_lengths else 0.0,
            }

            all_rewards.extend(world_rewards)
            all_lengths.extend(world_lengths)

        # Aggregate overall stats (for best/early stop)
        eval_stats = {
            'eval_avg_reward': float(np.mean(all_rewards)) if all_rewards else 0.0,
            'eval_std_reward': float(np.std(all_rewards)) if all_rewards else 0.0,
            'eval_max_reward': float(np.max(all_rewards)) if all_rewards else 0.0,
            'eval_min_reward': float(np.min(all_rewards)) if all_rewards else 0.0,
            'eval_avg_length': float(np.mean(all_lengths)) if all_lengths else 0.0,
        }

        # Log per-world metrics as well (replace '-' with '_' for TB tags)
        for world, stats in per_world_stats.items():
            tag = world.replace('-', '_')
            eval_stats[f'eval_avg_reward_{tag}'] = stats['avg_reward']
            eval_stats[f'eval_std_reward_{tag}'] = stats['std_reward']
            eval_stats[f'eval_max_reward_{tag}'] = stats['max_reward']
            eval_stats[f'eval_min_reward_{tag}'] = stats['min_reward']
            eval_stats[f'eval_avg_length_{tag}'] = stats['avg_length']

        print(f"Evaluation done: overall avg reward={eval_stats['eval_avg_reward']:.2f} ± {eval_stats['eval_std_reward']:.2f}")
        return eval_stats
    
    def should_stop_training(self):
        """
        Decide whether to stop training.
        
        Returns:
            tuple: (should_stop, reason)
        """
        # Max steps
        if self.ppo.total_steps >= self.args.max_steps:
            return True, f"Reached max steps {self.args.max_steps:,}"
        
        # Max episodes
        if self.ppo.total_episodes >= self.args.max_episodes:
            return True, f"Reached max episodes {self.args.max_episodes:,}"
        
        # Target reward
        if len(self.episode_rewards) >= 100:
            recent_avg = np.mean(self.episode_rewards[-100:])
            if recent_avg >= Config.TARGET_REWARD:
                return True, f"Reached target reward {Config.TARGET_REWARD} (current: {recent_avg:.2f})"
        
        # Early stopping
        progress_info = self.progress_tracker.update(
            self.episode_rewards[-1] if self.episode_rewards else 0
        )
        if progress_info['should_stop']:
            return True, f"Early stop: no improvement for {progress_info['episodes_without_improvement']} episodes"
        
        return False, ""
    
    def train(self):
        """Main training loop"""
        print("\nStarting PPO training...")
        print("=" * 60)
        
        start_time = time.time()
        update_count = 0
        
        try:
            while True:
                update_start_time = time.time()
                
                # Do one training step
                train_stats = self.train_step()
                update_count += 1
                
                # Log system info
                if self.performance_monitor:
                    system_info = self.performance_monitor.get_system_info()
                    self.logger.log_system_info(**system_info)
                
                # Periodic console stats
                if update_count % self.args.log_freq == 0:
                    update_time = time.time() - update_start_time
                    total_time = time.time() - start_time
                    
                    print(f"\nUpdate #{update_count}")
                    print(f"Total steps: {self.ppo.total_steps:,}")
                    print(f"Total episodes: {self.ppo.total_episodes:,}")
                    print(f"Update time: {update_time:.2f}s")
                    print(f"Elapsed: {total_time/3600:.2f}h")
                    
                    if train_stats.get('episodes_completed', 0) > 0:
                        print(f"Episodes finished: {train_stats['episodes_completed']}")
                        print(f"Avg reward: {train_stats['total_reward']/train_stats['episodes_completed']:.2f}")
                    
                    print(f"Policy loss: {train_stats.get('policy_loss', 0):.4f}")
                    print(f"Value loss: {train_stats.get('value_loss', 0):.4f}")
                    print(f"Entropy: {train_stats.get('entropy', 0):.4f}")
                    print(f"Clip fraction: {train_stats.get('clip_fraction', 0):.3f}")
                    print(f"Learning rate: {train_stats.get('learning_rate', 0):.2e}")
                    
                    # Recent performance
                    if len(self.episode_rewards) >= 10:
                        recent_10 = np.mean(self.episode_rewards[-10:])
                        recent_100 = np.mean(self.episode_rewards[-100:]) if len(self.episode_rewards) >= 100 else np.mean(self.episode_rewards)
                        print(f"Avg reward (last 10): {recent_10:.2f}")
                        print(f"Avg reward (last 100): {recent_100:.2f}")
                        print(f"Best reward so far: {max(self.episode_rewards):.2f}")
                
                # Periodically save model
                if update_count % self.args.save_freq == 0:
                    # Evaluate
                    eval_stats = self.evaluate_model(num_episodes=3)
                    
                    # Check if best so far
                    current_avg = eval_stats['eval_avg_reward']
                    is_best = current_avg > self.best_avg_reward
                    
                    if is_best:
                        self.best_avg_reward = current_avg
                        self.episodes_since_best = 0
                        print(f"🎉 New best model! Avg reward: {current_avg:.2f}")
                    else:
                        self.episodes_since_best += 1
                    
                    # Save
                    model_filename = f"ppo_mario_update_{update_count}.pth"
                    self.model_manager.save_model(
                        self.ppo, 
                        filename=model_filename,
                        is_best=is_best
                    )
                    
                    # Dynamic world sampling: 1) weighted per-episode switch; 2) per-env allocation
                    weights = self._compute_world_sampling_weights(eval_stats)
                    if weights:
                        try:
                            if getattr(Config, 'DYNAMIC_WORLD_SAMPLING', False) and not getattr(Config, 'USE_DYNAMIC_WORLD_COUNTS', False):
                                self.envs.set_world_weights(weights)
                                print(f"Updated world sampling weights: {weights}")
                            if getattr(Config, 'USE_DYNAMIC_WORLD_COUNTS', False):
                                self.envs.set_world_allocation(weights)
                        except Exception as e:
                            print(f"Failed to update world sampling config: {e}")
                    
                    # Log eval stats
                    self.logger.log_training_step(**eval_stats)
                
                # Check stopping criteria
                should_stop, stop_reason = self.should_stop_training()
                if should_stop:
                    print(f"\nTraining stopped: {stop_reason}")
                    break
                
                # Every 100 updates: detailed stats
                if update_count % 100 == 0:
                    self.logger.print_training_stats()
                    
                    # Show environment stats
                    env_stats = self.envs.get_statistics()
                    print("Environment stats:")
                    for key, value in env_stats.items():
                        if isinstance(value, float):
                            print(f"  {key}: {value:.4f}")
                        else:
                            print(f"  {key}: {value}")
        
        except KeyboardInterrupt:
            print("\nInterrupted, saving model and exiting...")
            
        except Exception as e:
            print(f"\nError during training: {e}")
            import traceback
            traceback.print_exc()
        
        finally:
            # Final save
            final_model_path = "ppo_mario_final.pth"
            self.model_manager.save_model(self.ppo, filename=final_model_path)
            
            # Final evaluation
            final_eval = self.evaluate_model(num_episodes=10)
            print(f"\nFinal evaluation:")
            for key, value in final_eval.items():
                print(f"  {key}: {value:.4f}")
            
            # Cleanup
            self.envs.close()
            self.logger.close()
            
            total_time = time.time() - start_time
            print(f"\nTraining finished! Elapsed: {total_time/3600:.2f} hours")
            print(f"Final model saved to: {final_model_path}")


def main():
    """Program entry point"""
    # Parse CLI args
    args = parse_args()
    
    print("PPO Mario Training")
    print("=" * 60)
    print(f"Device: {torch.device(args.device) if args.device else Config.DEVICE}")
    print(f"Num envs: {args.num_envs}")
    print(f"Training worlds: {Config.WORLD_STAGE}")
    print(f"Max episodes: {args.max_episodes:,}")
    print(f"Max steps: {args.max_steps:,}")
    print("=" * 60)
    
    # Create trainer
    trainer = PPOTrainer(args)
    
    # Start training
    trainer.train()


if __name__ == "__main__":
    main()
