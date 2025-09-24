"""
Parallel environment management.
Runs multiple Mario environments in parallel (subprocess or in-process)
to accelerate data collection and training.
"""

import multiprocessing as mp
import numpy as np
import torch
from typing import List, Tuple, Dict, Any
import time
import traceback

from .mario_env import create_mario_environment, MARIO_AVAILABLE
from config import Config


class SubprocVecEnv:
    """
    Subprocess vectorized environment.
    
    Each env runs in its own process to bypass the GIL and achieve
    true parallelism. Useful for CPU-heavy environments (image preprocessing).
    """
    
    def __init__(self, env_fns):
        """
        Initialize subprocess vectorized env.
        
        Args:
            env_fns (list): list of env factory functions
        """
        self.waiting = False
        self.closed = False
        self.num_envs = len(env_fns)
        
        # Create pipes for IPC
        self.remotes, self.work_remotes = zip(*[mp.Pipe() for _ in range(self.num_envs)])
        
        # Spawn workers
        self.processes = []
        for work_remote, remote, env_fn in zip(self.work_remotes, self.remotes, env_fns):
            args = (work_remote, remote, env_fn)
            process = mp.Process(target=self._worker, args=args)
            process.daemon = True  # terminate workers when parent exits
            process.start()
            self.processes.append(process)
            work_remote.close()  # close work end in parent
        
        # Probe spaces
        self.remotes[0].send(('get_spaces', None))
        observation_space, action_space = self.remotes[0].recv()
        self.observation_space = observation_space
        self.action_space = action_space
    
    @staticmethod
    def _worker(remote, parent_remote, env_fn):
        """
        Worker loop running in subprocess.
        
        Args:
            remote: child process pipe
            parent_remote: parent process pipe
            env_fn: env factory
        """
        parent_remote.close()  # close parent end in child process
        
        try:
            env = env_fn()
        except Exception as e:
            remote.send(('error', f'Environment creation failed: {e}'))
            remote.close()
            return
        
        while True:
            try:
                cmd, data = remote.recv()
                
                if cmd == 'step':
                    # Step
                    observation, reward, done, info = env.step(data)
                    if done:
                        # Auto-reset when done
                        observation = env.reset()
                    remote.send((observation, reward, done, info))
                
                elif cmd == 'reset':
                    # Reset
                    observation = env.reset()
                    remote.send(observation)
                
                elif cmd == 'close':
                    # Close
                    env.close()
                    remote.close()
                    break
                
                elif cmd == 'get_spaces':
                    # Get spaces
                    remote.send((env.observation_space, env.action_space))
                
                elif cmd == 'render':
                    # Render
                    img = env.render(mode=data)
                    remote.send(img)
                
                elif cmd == 'get_info':
                    # Get env info
                    info = env.get_info() if hasattr(env, 'get_info') else {}
                    remote.send(info)
                
                elif cmd == 'set_world':
                    # Reconfigure to specific world (single-world env)
                    new_world = data
                    if hasattr(env, 'reconfigure_world'):
                        try:
                            env.reconfigure_world(new_world)
                            remote.send('ok')
                        except Exception as e:
                            remote.send(('error', f'Failed to set world: {e}'))
                    else:
                        remote.send('ignored')
                
                else:
                    raise NotImplementedError(f"Command {cmd} not implemented")
            
            except Exception as e:
                error_msg = f"Worker error: {e}\n{traceback.format_exc()}"
                remote.send(('error', error_msg))
                break
    
    def step_async(self, actions):
        """
        Asynchronously dispatch actions (non-blocking).
        
        Args:
            actions (list): list of actions
        """
        if self.waiting:
            raise RuntimeError("Already waiting for step results")
        
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
        
        self.waiting = True

    def set_worlds(self, worlds):
        """
        Set fixed world for each sub-env.
        
        Args:
            worlds (list[str]): per-env world names (len == num_envs)
        """
        if len(worlds) != self.num_envs:
            raise ValueError("worlds length must equal num_envs")
        for remote, w in zip(self.remotes, worlds):
            remote.send(('set_world', w))
        for remote in self.remotes:
            try:
                remote.recv()
            except Exception:
                pass
    
    def step_wait(self):
        """
        Wait for async steps to complete.
        
        Returns:
            tuple: batch of (obs, reward, done, info)
        """
        if not self.waiting:
            raise RuntimeError("Not waiting for step results")
        
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        
        # Check for worker errors
        for i, result in enumerate(results):
            if isinstance(result, tuple) and len(result) == 2 and result[0] == 'error':
                raise RuntimeError(f"Environment {i} error: {result[1]}")
        
        # Unzip results
        observations, rewards, dones, infos = zip(*results)
        
        return np.array(observations), np.array(rewards), np.array(dones), list(infos)
    
    def step(self, actions):
        """
        Synchronous step helper.
        
        Args:
            actions (list): list of actions
            
        Returns:
            tuple: batch of (obs, reward, done, info)
        """
        self.step_async(actions)
        return self.step_wait()
    
    def reset(self):
        """
        Reset all environments.
        
        Returns:
            np.array: initial observation batch
        """
        for remote in self.remotes:
            remote.send(('reset', None))
        
        observations = [remote.recv() for remote in self.remotes]
        
        return np.array(observations)
    
    def close(self):
        """
        Close all envs and worker processes.
        """
        if self.closed:
            return
        
        if self.waiting:
            # Wait for current op to complete
            for remote in self.remotes:
                remote.recv()
            self.waiting = False
        
        # Send close
        for remote in self.remotes:
            remote.send(('close', None))
        
        # Join workers
        for process in self.processes:
            process.join(timeout=5)
            if process.is_alive():
                process.terminate()  # force terminate
        
        for remote in self.remotes:
            remote.close()
        
        self.closed = True
    
    def render(self, mode='human', env_id=0):
        """
        Render a specific environment.
        
        Args:
            mode (str): render mode
            env_id (int): environment id
            
        Returns:
            Any: render result
        """
        self.remotes[env_id].send(('render', mode))
        return self.remotes[env_id].recv()
    
    def get_env_info(self, env_id=0):
        """
        Get info for a specific environment.
        
        Args:
            env_id (int): environment id
            
        Returns:
            dict: info dict
        """
        self.remotes[env_id].send(('get_info', None))
        return self.remotes[env_id].recv()
    
    def __len__(self):
        return self.num_envs
    
    def __del__(self):
        if not self.closed:
            self.close()


class DummyVecEnv:
    """
    In-process vectorized environment (single process).
    
    All envs run in the same process; useful for debugging or when
    env creation is expensive.
    """
    
    def __init__(self, env_fns):
        """
        Initialize dummy vectorized env.
        
        Args:
            env_fns (list): list of env factories
        """
        self.envs = [fn() for fn in env_fns]
        self.num_envs = len(self.envs)
        
        # Probe spaces
        self.observation_space = self.envs[0].observation_space
        self.action_space = self.envs[0].action_space
        self.closed = False
    
    def step(self, actions):
        """
        Step all envs.
        
        Args:
            actions (list): list of actions
            
        Returns:
            tuple: batch of (obs, reward, done, info)
        """
        results = []
        for env, action in zip(self.envs, actions):
            observation, reward, done, info = env.step(action)
            if done:
                observation = env.reset()
            results.append((observation, reward, done, info))
        
        observations, rewards, dones, infos = zip(*results)
        return np.array(observations), np.array(rewards), np.array(dones), list(infos)
    
    def reset(self):
        """
        Reset all envs.
        
        Returns:
            np.array: initial observation batch
        """
        observations = [env.reset() for env in self.envs]
        return np.array(observations)

    def set_worlds(self, worlds):
        """
        Set a fixed world for each env.
        """
        if len(worlds) != self.num_envs:
            raise ValueError("worlds length must equal num_envs")
        for env, w in zip(self.envs, worlds):
            if hasattr(env, 'reconfigure_world'):
                env.reconfigure_world(w)
    
    def close(self):
        """
        Close all environments.
        """
        if not self.closed:
            for env in self.envs:
                env.close()
            self.closed = True
    
    def render(self, mode='human', env_id=0):
        """
        Render a specific environment.
        
        Args:
            mode (str): render mode
            env_id (int): environment id
        """
        return self.envs[env_id].render(mode)
    
    def get_env_info(self, env_id=0):
        """
        Get info for a specific env.
        
        Args:
            env_id (int): environment id
            
        Returns:
            dict: info dict
        """
        return self.envs[env_id].get_info() if hasattr(self.envs[env_id], 'get_info') else {}
    
    def __len__(self):
        return self.num_envs
    
    def __del__(self):
        self.close()


class ParallelMarioEnvironments:
    """
    High-level manager over vectorized Mario environments.
    """
    
    def __init__(self, num_envs=16, worlds=None, use_subprocess=True, render_env_id=None):
        """
        Initialize parallel environments.
        
        Args:
            num_envs (int): number of envs
            worlds (list): list of worlds (None = default)
            use_subprocess (bool): use subprocess vectorization
            render_env_id (int): env id to render (or None)
        """
        self.num_envs = num_envs
        self.use_subprocess = use_subprocess
        self.render_env_id = render_env_id
        
        if not MARIO_AVAILABLE:
            raise ImportError("Mario environment not available")
        
        # Default world list
        if worlds is None:
            worlds = ['1-1', '1-2', '1-3', '1-4']
        
        # Build env factory list
        env_fns = []
        for i in range(num_envs):
            # Only the specified env renders
            render_mode = 'human' if i == render_env_id else None
            world = worlds[i % len(worlds)]
            env_fn = lambda w=world, r=render_mode: create_mario_environment(
                world=w,
                render_mode=r
            )
            env_fns.append(env_fn)
        
        # Create vectorized env
        if use_subprocess and num_envs > 1:
            self.vec_env = SubprocVecEnv(env_fns)
        else:
            self.vec_env = DummyVecEnv(env_fns)
        
        # Spaces
        self.observation_space = self.vec_env.observation_space
        self.action_space = self.vec_env.action_space
        
        # Stats
        self.total_episodes = 0
        self.total_steps = 0
        self.episode_rewards = []
        self.episode_lengths = []
        
        print(f"Created {num_envs} parallel Mario environments")
        print(f"Observation space: {self.observation_space}")
        print(f"Action space: {self.action_space}")
        # Save current world allocation (for dynamic adjustments)
        self.available_worlds = list(worlds)
        self.current_worlds = [worlds[i % len(worlds)] for i in range(num_envs)]
    
    def reset(self):
        """
        Reset all envs and return tensor observations.
        
        Returns:
            torch.Tensor: (num_envs, *obs_shape)
        """
        observations = self.vec_env.reset()
        
        # To tensor
        return torch.FloatTensor(observations).to(Config.DEVICE)
    
    def step(self, actions):
        """
        Step all envs in parallel.
        
        Args:
            actions (torch.Tensor): (num_envs,)
            
        Returns:
            tuple: (next_obs, rewards, dones, info_list)
        """
        # Convert actions to numpy
        if isinstance(actions, torch.Tensor):
            actions = actions.cpu().numpy()
        
        # Step
        observations, rewards, dones, infos = self.vec_env.step(actions)
        
        # Update counters
        self.total_steps += self.num_envs
        
        for i, (reward, done, info) in enumerate(zip(rewards, dones, infos)):
            if done:
                self.total_episodes += 1
                # Record episode stats if present
                if 'episode_reward' in info:
                    self.episode_rewards.append(info['episode_reward'])
                if 'episode_length' in info:
                    self.episode_lengths.append(info['episode_length'])
        
        # To tensor
        observations = torch.FloatTensor(observations).to(Config.DEVICE)
        rewards = torch.FloatTensor(rewards).to(Config.DEVICE)
        dones = torch.BoolTensor(dones).to(Config.DEVICE)
        
        return observations, rewards, dones, infos
    
    def render(self, env_id=None):
        """
        Render a specific environment.
        
        Args:
            env_id (int): environment id (None uses default render env)
        """
        if env_id is None:
            env_id = self.render_env_id
        
        if env_id is not None and env_id < self.num_envs:
            return self.vec_env.render(env_id=env_id)
    
    def set_world_allocation(self, weights_or_counts):
        """
        Dynamically adjust number of sub-envs per world (each sub-env fixed to one world).
        
        Args:
            weights_or_counts (dict|list):
                - dict {world: weight or count}
                - list weights or counts matching self.available_worlds order
        """
        worlds = self.available_worlds
        n = self.num_envs
        # Parse input to weights/counts array
        if isinstance(weights_or_counts, dict):
            arr = np.array([float(weights_or_counts.get(w, 0.0)) for w in worlds], dtype=np.float64)
        else:
            arr = np.array(list(map(float, weights_or_counts)), dtype=np.float64)
            if arr.size != len(worlds):
                raise ValueError("weights_or_counts size must match number of available worlds")
        # If sum > n scale proportionally; if <=1 treat as weights * n
        min_envs = int(getattr(Config, 'WORLD_MIN_ENVS_PER_WORLD', 1))
        if arr.sum() <= 1.0 + 1e-9:
            weights = arr / (arr.sum() + 1e-9)
            counts = np.floor(weights * n).astype(int)
        else:
            counts = arr.astype(int)
        # At least min_envs per world
        counts = np.maximum(counts, min_envs)
        # Adjust total to n
        diff = counts.sum() - n
        if diff != 0:
            # Adjustment order: reduce from larger counts first
            order = np.argsort(counts)[::-1] if diff > 0 else np.argsort(counts)
            i = 0
            while diff != 0 and i < len(order):
                idx = order[i]
                if diff > 0 and counts[idx] > min_envs:
                    counts[idx] -= 1
                    diff -= 1
                    i = 0
                    continue
                if diff < 0:
                    counts[idx] += 1
                    diff += 1
                    i = 0
                    continue
                i += 1
        # Expand to per-env world list
        new_assignment = []
        for w, c in zip(worlds, counts):
            new_assignment.extend([w] * c)
        # Fix rounding mismatch if any
        if len(new_assignment) > n:
            new_assignment = new_assignment[:n]
        elif len(new_assignment) < n:
            new_assignment.extend([worlds[0]] * (n - len(new_assignment)))
        # Apply to vectorized env
        if hasattr(self.vec_env, 'set_worlds'):
            self.vec_env.set_worlds(new_assignment)
            self.current_worlds = list(new_assignment)
            print(f"Updated per-env world allocation: {dict(zip(worlds, counts))}")
    
    def close(self):
        """
        Close all environments.
        """
        self.vec_env.close()
    
    def get_statistics(self):
        """
        Get training statistics.
        
        Returns:
            dict: stats
        """
        stats = {
            'total_episodes': self.total_episodes,
            'total_steps': self.total_steps,
            'num_envs': self.num_envs,
        }
        
        if self.episode_rewards:
            stats.update({
                'avg_episode_reward': np.mean(self.episode_rewards[-100:]),  # last 100 episodes
                'max_episode_reward': np.max(self.episode_rewards),
                'min_episode_reward': np.min(self.episode_rewards),
            })
        
        if self.episode_lengths:
            stats.update({
                'avg_episode_length': np.mean(self.episode_lengths[-100:]),
                'max_episode_length': np.max(self.episode_lengths),
                'min_episode_length': np.min(self.episode_lengths),
            })
        
        return stats
    
    def __len__(self):
        return self.num_envs
    
    def __del__(self):
        self.close()


def create_parallel_mario_envs(num_envs=16, worlds=None, use_subprocess=True, render_env_id=None):
    """
    Factory to create a ParallelMarioEnvironments manager.
    
    Args:
        num_envs (int): number of envs
        worlds (list): world list
        use_subprocess (bool): use subprocess vectorization
        render_env_id (int): env id to render
        
    Returns:
        ParallelMarioEnvironments: manager instance
    """
    return ParallelMarioEnvironments(
        num_envs=num_envs,
        worlds=worlds,
        use_subprocess=use_subprocess,
        render_env_id=render_env_id
    )


def test_parallel_environments():
    """
    Quick test for parallel envs.
    """
    print("Testing parallel Mario environments...")
    
    if not MARIO_AVAILABLE:
        print("Mario environment not available, skipping test")
        return
    
    try:
        # Create a couple of envs for testing
        envs = create_parallel_mario_envs(num_envs=2, use_subprocess=False)
        print(f"Parallel envs created: {len(envs)} envs")
        
        # Reset
        states = envs.reset()
        print(f"Reset OK, states shape: {states.shape}")
        
        # Run a few steps
        for i in range(3):
            actions = torch.randint(0, envs.action_space.n, (len(envs),))
            next_states, rewards, dones, infos = envs.step(actions)
            
            print(f"Step {i+1}:")
            print(f"  actions: {actions.tolist()}")
            print(f"  rewards: {rewards.tolist()}")
            print(f"  dones: {dones.tolist()}")
        
        # Stats
        stats = envs.get_statistics()
        print(f"Stats: {stats}")
        
        # Close
        envs.close()
        print("Parallel envs test completed")
        
    except Exception as e:
        print(f"Parallel envs test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_parallel_environments()
