"""
Super Mario Bros environment wrapper for DreamerV3.

This module integrates with the shared preprocessing utilities in
enviroments/preprocessing.py to provide a consistent Mario environment
across different algorithms.
"""

from enviroments.preprocessing import create_mario_env_dreamerv3


class MarioEnv:
    """
    Wrapper for Super Mario Bros with DreamerV3-specific preprocessing.

    This is a compatibility layer that uses the shared preprocessing
    from enviroments/preprocessing.py while providing the interface
    expected by DreamerV3's trainer.
    """

    def __init__(self, config: dict):
        """
        Args:
            config: Configuration dictionary containing env settings
        """
        self.config = config

        # Extract environment parameters
        env_name = config['env']['name']  # e.g., "SuperMarioBros-1-1-v0"

        # Parse world from env name
        # Format: "SuperMarioBros-1-1-v0" -> world="1", stage="1"
        if '-' in env_name:
            parts = env_name.split('-')
            if len(parts) >= 3:
                world = parts[1]
                stage = parts[2]
            else:
                world = '1'
                stage = '1'
        else:
            world = '1'
            stage = '1'

        # Get preprocessing parameters
        frame_skip = config['env'].get('frame_skip', 4)
        resize = config['env'].get('resize', [64, 64])
        size = resize[0]  # Assume square images
        grayscale = config['env'].get('grayscale', True)

        # Create environment using shared preprocessing
        self.env = create_mario_env_dreamerv3(
            world=world,
            stage=stage,
            frame_skip=frame_skip,
            size=size,
            grayscale=grayscale
        )

        self.action_size = self.env.action_space.n  # 7 for SIMPLE_MOVEMENT

        # Observation shape (for replay buffer)
        self.observation_shape = self.env.observation_space.shape  # (1, H, W)

        # Episode tracking
        self.episode_reward = 0
        self.episode_length = 0
        self.max_x_pos = 0

    def reset(self):
        """
        Reset environment and return initial observation.

        Returns:
            Processed observation, shape: (C, H, W), dtype: uint8
        """
        obs = self.env.reset()

        # Reset episode tracking
        self.episode_reward = 0
        self.episode_length = 0
        self.max_x_pos = 0

        return obs

    def step(self, action: int):
        """
        Take action and return next observation, reward, done, info.

        Args:
            action: Action index (0 to action_size-1)

        Returns:
            observation: Processed obs, shape: (C, H, W)
            reward: Shaped reward
            done: Episode termination flag
            info: Dictionary with episode statistics
        """
        obs, reward, done, info = self.env.step(action)

        # Update episode tracking
        self.episode_reward += reward
        self.episode_length += 1

        # Update max x position
        if 'x_pos' in info:
            self.max_x_pos = max(self.max_x_pos, info['x_pos'])

        # Add episode statistics to info
        if done:
            info['episode'] = {
                'reward': self.episode_reward,
                'length': self.episode_length,
                'max_x_pos': self.max_x_pos,
                'flag_get': info.get('flag_get', False)
            }

        return obs, reward, done, info

    def render(self, mode: str = 'human'):
        """Render the environment."""
        return self.env.render(mode)

    def close(self):
        """Close the environment."""
        self.env.close()

    def get_observation_space(self):
        """Get observation space shape."""
        return self.observation_shape

    def get_action_space(self):
        """Get action space size."""
        return self.action_size


# ============================================================================
# Helper function to create environment
# ============================================================================

def make_mario_env(config: dict):
    """
    Create Mario environment with config.

    Args:
        config: Configuration dictionary

    Returns:
        MarioEnv instance
    """
    return MarioEnv(config)


# ============================================================================
# Example usage and testing
# ============================================================================

if __name__ == "__main__":
    # Test environment wrapper
    import yaml
    import numpy as np

    # Load config
    with open('configs/dreamerv3_config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # Create environment
    env = make_mario_env(config)

    print(f"Action space size: {env.action_size}")
    print(f"Observation shape: {env.observation_shape}")

    # Test episode
    obs = env.reset()
    print(f"Initial observation shape: {obs.shape}, dtype: {obs.dtype}")

    total_reward = 0
    steps = 0
    done = False

    while not done and steps < 100:
        # Random action
        action = np.random.randint(0, env.action_size)

        obs, reward, done, info = env.step(action)
        total_reward += reward
        steps += 1

        if done:
            print(f"\nEpisode finished!")
            print(f"  Total reward: {total_reward:.2f}")
            print(f"  Steps: {steps}")
            if 'episode' in info:
                print(f"  Max X position: {info['episode']['max_x_pos']}")
                print(f"  Flag reached: {info['episode']['flag_get']}")

    env.close()
    print("\nEnvironment test completed successfully!")
