"""
Image preprocessing for Mario.
Transforms raw frames into a network-friendly format:
- crop HUD
- grayscale
- resize
- normalize
- frame stacking
"""

import cv2
import numpy as np
import torch
from collections import deque
import gym
from config import Config

class FrameStack:
    """
    Frame stacker that builds a (num_stack, H, W) state from sequential frames.
    Provides temporal context so the model can infer motion.
    """
    
    def __init__(self, num_stack=4):
        """
        Initialize frame stacker.
        
        Args:
            num_stack (int): number of stacked frames
        """
        self.num_stack = num_stack
        self.frames = deque(maxlen=num_stack)
        
    def reset(self, frame):
        """
        Reset stack for a new episode.
        
        Args:
            frame (np.array): preprocessed initial frame
        """
        # Fill stack with the initial frame to avoid zeros at start
        for _ in range(self.num_stack):
            self.frames.append(frame)
            
    def add_frame(self, frame):
        """
        Push a new frame into the stack.
        
        Args:
            frame (np.array): preprocessed frame
        """
        self.frames.append(frame)
        
    def get_state(self):
        """
        Get current stacked state.
        
        Returns:
            np.array: (num_stack, height, width)
        """
        return np.stack(self.frames, axis=0)


def preprocess_frame(frame):
    """
    Preprocess a single game frame.
    
    Steps:
    1) crop HUD
    2) convert to grayscale
    3) resize
    4) normalize to [0, 1]
    
    Args:
        frame (np.array): raw frame, typically (240, 256, 3)
        
    Returns:
        np.array: (84, 84) float32 in [0, 1]
    """
    
    # Crop game area to remove top HUD (typical useful area y in [30, 210])
    frame = frame[30:210, :]
    
    # Grayscale — shape is more important than color here
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    
    # Resize to 84x84 (standard in many DRL setups)
    frame = cv2.resize(frame, (Config.FRAME_SIZE, Config.FRAME_SIZE))
    
    # Normalize to [0, 1]
    frame = frame.astype(np.float32) / 255.0
    
    return frame


class MarioWrapper(gym.Wrapper):
    """
    Mario environment wrapper that adds:
    1) automatic image preprocessing
    2) frame stacking
    3) action repeat (frame skip)
    4) reward shaping
    5) episode termination tweaks
    """
    
    def __init__(self, env):
        """
        Initialize wrapper.
        
        Args:
            env: original Mario env
        """
        super(MarioWrapper, self).__init__(env)
        
        # Frame stacker
        self.frame_stack = FrameStack(Config.FRAME_STACK)
        
        # Redefine observation space from RGB to stacked grayscale frames
        self.observation_space = gym.spaces.Box(
            low=0.0, 
            high=1.0, 
            shape=(Config.FRAME_STACK, Config.FRAME_SIZE, Config.FRAME_SIZE),
            dtype=np.float32
        )
        
        # Variables used for reward shaping
        self.prev_x_pos = 0         # previous x position
        self.prev_time = 400        # previous remaining time
        self.death_penalty = -15    # death penalty
        self.time_penalty = -0.1    # time passing penalty
        
    def reset(self, **kwargs):
        """
        Reset and return initial processed, stacked state.
        
        Returns:
            np.array: stacked initial state
        """
        # Reset underlying env
        obs = self.env.reset(**kwargs)
        
        # Preprocess initial frame
        processed_frame = preprocess_frame(obs)
        
        # Reset frame stack
        self.frame_stack.reset(processed_frame)
        
        # Reset shaping state
        self.prev_x_pos = 0
        self.prev_time = 400
        
        return self.frame_stack.get_state()
    
    def step(self, action):
        """
        Step with preprocessing, stacking and reward shaping.
        
        Args:
            action: action to perform
            
        Returns:
            tuple: (state, reward, done, info)
        """
        total_reward = 0
        
        # Action repeat to reduce decision frequency
        for _ in range(Config.SKIP_FRAMES):
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            
            if done:
                break
        
        # Preprocess frame
        processed_frame = preprocess_frame(obs)
        
        # Add to frame stack
        self.frame_stack.add_frame(processed_frame)
        
        # Reward shaping to densify signal
        shaped_reward = self._shape_reward(total_reward, info, done)
        
        return self.frame_stack.get_state(), shaped_reward, done, info
    
    def _shape_reward(self, reward, info, done):
        """
        Reward shaping function.
        
        Original rewards are sparse; add intermediate signals:
        1) reward rightward progress
        2) small time penalty
        3) death penalty
        
        Args:
            reward (float): original reward
            info (dict): env info
            done (bool): episode done
            
        Returns:
            float: shaped reward
        """
        shaped_reward = reward
        
        # Current info
        current_x = info.get('x_pos', 0)
        current_time = info.get('time', 400)
        
        # Movement reward: encourage moving right
        x_reward = (current_x - self.prev_x_pos) * 0.1
        shaped_reward += x_reward
        
        # Time penalty: encourage faster completion
        time_reward = (current_time - self.prev_time) * self.time_penalty
        shaped_reward += time_reward
        
        # Death penalty
        if done and info.get('life', 2) < 2:
            shaped_reward += self.death_penalty
        
        # Update state
        self.prev_x_pos = current_x
        self.prev_time = current_time
        
        return shaped_reward


def create_mario_env(world='1-1', stage=None):
    """
    Factory for a single-world Mario env with preprocessing wrapper.
    
    Args:
        world (str): world id like '1-1'
        stage (str): optional stage
        
    Returns:
        gym.Env: wrapped env
    """
    import gym_super_mario_bros
    from nes_py.wrappers import JoypadSpace
    from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
    
    # Base env
    if stage:
        env_name = f'SuperMarioBros-{world}-{stage}-v0'
    else:
        env_name = f'SuperMarioBros-{world}-v0'
    
    env = gym_super_mario_bros.make(env_name)
    
    # Simplified action space
    # SIMPLE_MOVEMENT = [['NOOP'], ['right'], ['right', 'A'], ['right', 'B'], ['right', 'A', 'B'], ['A'], ['left']]
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    
    # Apply preprocessing wrapper
    env = MarioWrapper(env)
    
    return env


def preprocess_batch_frames(frames):
    """
    Preprocess a batch of frames (for vectorized envs).
    
    Args:
        frames (list): list of raw frames
        
    Returns:
        np.array: batch of preprocessed frames
    """
    processed_frames = []
    
    for frame in frames:
        processed_frame = preprocess_frame(frame)
        processed_frames.append(processed_frame)
    
    return np.array(processed_frames)


def frames_to_tensor(frames, device=None):
    """
    Convert frames ndarray to a PyTorch tensor.
    
    Args:
        frames (np.array): frames array
        device (torch.device): target device
        
    Returns:
        torch.Tensor: tensor on device
    """
    if device is None:
        device = Config.DEVICE
    
    # Ensure dtype
    if frames.dtype != np.float32:
        frames = frames.astype(np.float32)
    
    # Convert and move to device
    tensor = torch.from_numpy(frames).to(device)
    
    return tensor


def tensor_to_frames(tensor):
    """
    Convert PyTorch tensor back to numpy array.
    
    Args:
        tensor (torch.Tensor): input tensor
        
    Returns:
        np.array: numpy array
    """
    return tensor.cpu().numpy()
