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

from utils.preprocessing import MarioWrapper, create_mario_env


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
        
        if not isinstance(info, dict):
            info = {'raw_info': info}
        info.setdefault('world', self.world)

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


def create_mario_environment(world='1-1', render_mode=None):
    """
    Factory function to create a Mario environment.
    
    Args:
        world (str): world id for single-world mode
        render_mode (str): render mode
        
    Returns:
        MarioEnvironment: environment instance
    """
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
