"""
Mario environment module.
Wrapper around gym-super-mario-bros providing standardized interfaces
including environment creation, preprocessing, and reward shaping.
"""

import gym
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
import numpy as np
import torch
from typing import Tuple, Dict, Any

try:
    import gym_super_mario_bros
    from nes_py.wrappers import JoypadSpace
    from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
    MARIO_AVAILABLE = True
except ImportError:
    print("Warning: gym-super-mario-bros not installed. Install with 'pip install gym-super-mario-bros'")
    MARIO_AVAILABLE = False

from enviroments.preprocessing import MarioWrapper, create_mario_env
from configs.ppo_config import Config


class MarioEnvironment:
    """
    Mario environment wrapper.
    
    Provides a unified interface to create and manage Mario
    across different worlds/levels.
    """
    
    def __init__(self, world='1-1', render_mode=None):
        """
        Initialize Mario environment.
        
        Args:
            world (str): world-level like '1-1'
            render_mode (str): None for headless, 'human' to render
        """
        if not MARIO_AVAILABLE:
            raise ImportError("gym-super-mario-bros is required but not installed")
        
        self.world = world
        self.render_mode = render_mode
        
        # Create environment
        self.env = self._create_env()
        
        # Spaces
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        
        # Stats
        self.episode_count = 0
        self.total_steps = 0
        
    def _create_env(self):
        """
        Create a Mario environment instance.
        
        Returns:
            gym.Env: configured environment
        """
        # Build env name from world
        if '-' in self.world:
            world_num, level_num = self.world.split('-')
            env_name = f'SuperMarioBros-{world_num}-{level_num}-v0'
        else:
            env_name = f'SuperMarioBros-{self.world}-v0'
        
        # Create base env
        try:
            env = gym_super_mario_bros.make(env_name)
        except Exception as e:
            print(f"Failed to create environment {env_name}, using default SuperMarioBros-v0")
            env = gym_super_mario_bros.make('SuperMarioBros-v0')
        
        # Complex action space
        env = JoypadSpace(env, COMPLEX_MOVEMENT)
        
        # Apply preprocessing wrapper
        env = MarioWrapper(env)
        
        return env
    
    def reset(self):
        """
        Reset environment and return initial state.
        
        Returns:
            np.array: initial state
        """
        state = self.env.reset()
        self.episode_count += 1
        return state
    
    def step(self, action):
        """
        Perform one step.
        
        Args:
            action (int): action to execute
            
        Returns:
            tuple: (next_state, reward, done, info)
        """
        next_state, reward, done, info = self.env.step(action)
        self.total_steps += 1
        
        return next_state, reward, done, info
    
    def render(self, mode='human'):
        """
        Render environment.
        
        Args:
            mode (str): render mode
        """
        return self.env.render(mode)
    
    def close(self):
        """
        Close environment
        """
        self.env.close()

    def reconfigure_world(self, world: str, render_mode=None):
        """
        Reconfigure to a new world (single-world env).
        
        Args:
            world (str): new world-level like '1-2'
            render_mode (str): render mode
        """
        try:
            if hasattr(self, 'env') and self.env is not None:
                self.env.close()
        except Exception:
            pass
        self.world = world
        if render_mode is not None:
            self.render_mode = render_mode
        self.env = self._create_env()
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
    
    def get_action_meanings(self):
        """
        Get action meanings.
        
        Returns:
            list: action meaning list
        """
        return [
            'NOOP',           # 0: no-op
            'right',          # 1: move right
            'right_A',        # 2: right + A (jump)
            'right_B',        # 3: right + B (run)
            'right_A_B',      # 4: right + A + B (run jump)
            'A',              # 5: jump
            'left'            # 6: move left
        ]
    
    def get_random_action(self):
        """
        Sample a random action.
        
        Returns:
            int: random action
        """
        return self.action_space.sample()
    
    def get_info(self):
        """
        Get environment info snapshot.
        
        Returns:
            dict: info dict
        """
        return {
            'world': self.world,
            'observation_space': self.observation_space,
            'action_space': self.action_space,
            'action_meanings': self.get_action_meanings(),
            'episode_count': self.episode_count,
            'total_steps': self.total_steps,
        }


class MultiWorldMarioEnvironment(MarioEnvironment):
    """
    Multi-world Mario environment.
    
    Supports switching between worlds to increase training diversity.
    """
    
    def __init__(self, worlds=['1-1', '1-2', '1-3'], render_mode=None, random_start=True, world_weights=None):
        """
        Initialize multi-world environment.
        
        Args:
            worlds (list): available worlds
            render_mode (str): render mode
            random_start (bool): randomly select starting world
        """
        self.worlds = worlds
        self.current_world_idx = 0
        self.random_start = random_start
        # Switch probability (from config)
        self.switch_prob = getattr(Config, 'WORLD_SWITCH_PROB', 1.0)
        
        # Sampling weights
        self.world_weights = None
        self.set_world_weights(world_weights)
        
        # If random start, pick a random world index
        if random_start:
            self.current_world_idx = np.random.randint(len(worlds))
        
        # Initialize base class with current world
        super().__init__(worlds[self.current_world_idx], render_mode)
        
        # Per-world stats
        self.world_episode_counts = {world: 0 for world in worlds}
        self.world_success_counts = {world: 0 for world in worlds}

    def _normalized_weights(self):
        if self.world_weights is None:
            # Default uniform
            return np.ones(len(self.worlds)) / len(self.worlds)
        w = np.array(self.world_weights, dtype=np.float64)
        w = np.clip(w, 1e-8, None)
        w = w / w.sum()
        return w

    def set_world_weights(self, weights):
        """
        Set/update sampling weights per world.
        
        Args:
            weights (list|dict|None):
                - list/ndarray: weights in self.worlds order
                - dict: {world: weight}
                - None: uniform weights
        """
        if weights is None:
            self.world_weights = None
            return
        if isinstance(weights, dict):
            self.world_weights = [float(weights.get(w, 1.0)) for w in self.worlds]
        else:
            self.world_weights = list(map(float, weights))

    def switch_world(self, world_idx=None):
        """
        Switch to a target world.
        
        Args:
            world_idx (int): world index; None to sample by weights
        """
        if world_idx is None:
            # Sample by weights
            probs = self._normalized_weights()
            world_idx = int(np.random.choice(len(self.worlds), p=probs))
        
        if world_idx != self.current_world_idx:
            self.current_world_idx = world_idx
            self.world = self.worlds[world_idx]
            
            # Recreate env
            self.env.close()
            self.env = self._create_env()
            
            # print(f"Switched to world: {self.world}")
    
    def reset(self, switch_world=None):
        """
        Reset environment, optionally switching world.
        
        Args:
            switch_world (bool): switch world at reset
            
        Returns:
            np.array: initial state
        """
        # Decide whether to switch world
        if switch_world is None:
            switch_world = self.random_start
        
        # With probability, re-sample world by weights
        if switch_world and np.random.random() < self.switch_prob:
            self.switch_world()
        
        # Update per-world episode count
        self.world_episode_counts[self.world] += 1
        
        return super().reset()
    
    def step(self, action):
        """
        Step and record success stats.
        
        Args:
            action (int): action
            
        Returns:
            tuple: transition result
        """
        next_state, reward, done, info = super().step(action)
        
        # Record success (flag)
        if done and info.get('flag_get', False):
            self.world_success_counts[self.world] += 1
        
        return next_state, reward, done, info
    
    def get_world_statistics(self):
        """
        Get per-world statistics.
        
        Returns:
            dict: statistics
        """
        stats = {}
        for world in self.worlds:
            episodes = self.world_episode_counts[world]
            successes = self.world_success_counts[world]
            success_rate = successes / episodes if episodes > 0 else 0.0
            
            stats[world] = {
                'episodes': episodes,
                'successes': successes,
                'success_rate': success_rate
            }
        
        return stats
    
    def get_info(self):
        """
        Get multi-world env info.
        
        Returns:
            dict: info
        """
        info = super().get_info()
        info.update({
            'available_worlds': self.worlds,
            'current_world': self.world,
            'world_statistics': self.get_world_statistics(),
        })
        return info


def create_mario_environment(world='1-1', multi_world=False, worlds=None, render_mode=None, world_weights=None):
    """
    Factory function to create a Mario environment.
    
    Args:
        world (str): world id for single-world mode
        multi_world (bool): enable multi-world mode
        worlds (list): list of worlds for multi-world mode
        render_mode (str): render mode
        
    Returns:
        MarioEnvironment: environment instance
    """
    if multi_world:
        if worlds is None:
            # Default multi-world configuration
            worlds = ['1-1', '1-2', '1-3', '1-4']
        return MultiWorldMarioEnvironment(worlds, render_mode, random_start=True, world_weights=world_weights)
    else:
        return MarioEnvironment(world, render_mode)


def test_mario_environment():
    """
    Quick test to verify environment setup.
    """
    print("Testing Mario environment...")
    
    if not MARIO_AVAILABLE:
        print("Mario environment not available, skipping test")
        return
    
    try:
        # Create env
        env = create_mario_environment('1-1')
        print(f"Environment created: {env.get_info()}")
        
        # Reset
        state = env.reset()
        print(f"Reset OK, state shape: {state.shape}")
        
        # Step a few times
        for i in range(5):
            action = env.get_random_action()
            next_state, reward, done, info = env.step(action)
            print(f"Step {i+1}: action={action}, reward={reward:.2f}, done={done}")
            
            if done:
                break
        
        # Close
        env.close()
        print("Environment test completed")
        
    except Exception as e:
        print(f"Environment test failed: {e}")


if __name__ == "__main__":
    test_mario_environment()
