"""
图像预处理模块
处理马里奥游戏的原始图像，转换为适合神经网络训练的格式
包括：灰度化、缩放、归一化、帧堆叠等操作
"""

import cv2
import numpy as np
import torch
from collections import deque
import gym
from config import Config

class FrameStack:
    """
    帧堆叠类 - 将连续的多帧图像堆叠起来，为模型提供时序信息
    这样模型可以"看到"马里奥的运动方向和速度变化
    """
    
    def __init__(self, num_stack=4):
        """
        初始化帧堆叠器
        
        Args:
            num_stack (int): 堆叠的帧数，默认4帧
        """
        self.num_stack = num_stack
        self.frames = deque(maxlen=num_stack)
        
    def reset(self, frame):
        """
        重置帧堆叠器，用于新回合开始
        
        Args:
            frame (np.array): 预处理后的初始帧
        """
        # 用相同的帧填满整个栈，避免初始时的零填充
        for _ in range(self.num_stack):
            self.frames.append(frame)
            
    def add_frame(self, frame):
        """
        添加新帧到堆叠中
        
        Args:
            frame (np.array): 新的预处理帧
        """
        self.frames.append(frame)
        
    def get_state(self):
        """
        获取当前堆叠的帧作为状态
        
        Returns:
            np.array: 堆叠后的状态，形状为 (num_stack, height, width)
        """
        return np.stack(self.frames, axis=0)


def preprocess_frame(frame):
    """
    预处理单个游戏帧
    
    这个函数将原始的彩色游戏画面转换为神经网络友好的格式：
    1. 裁剪掉不重要的区域（如分数显示）
    2. 转换为灰度图像（减少数据量，游戏不需要颜色信息）
    3. 缩放到固定大小
    4. 归一化像素值
    
    Args:
        frame (np.array): 原始游戏帧，形状通常为 (240, 256, 3)
        
    Returns:
        np.array: 预处理后的帧，形状为 (84, 84)，像素值在[0, 1]范围内
    """
    
    # 裁剪游戏区域，去除顶部的分数和信息显示区域
    # 马里奥游戏的有效游戏区域通常在y轴的30-210像素之间
    frame = frame[30:210, :]  # 保留高度180像素，去除顶部30像素的UI
    
    # 转换为灰度图像
    # 马里奥游戏主要依赖形状而不是颜色来识别物体
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    
    # 缩放到84x84像素
    # 这是深度强化学习中常用的标准尺寸，在保持重要信息的同时减少计算量
    frame = cv2.resize(frame, (Config.FRAME_SIZE, Config.FRAME_SIZE))
    
    # 归一化像素值到[0, 1]范围
    # 这有助于神经网络的训练稳定性
    frame = frame.astype(np.float32) / 255.0
    
    return frame


class MarioWrapper(gym.Wrapper):
    """
    马里奥环境包装器
    
    这个包装器对原始的马里奥环境进行了以下增强：
    1. 自动进行图像预处理
    2. 实现帧堆叠
    3. 动作重复（跳过帧）
    4. 奖励塑形
    5. 回合结束条件优化
    """
    
    def __init__(self, env):
        """
        初始化环境包装器
        
        Args:
            env: 原始的马里奥游戏环境
        """
        super(MarioWrapper, self).__init__(env)
        
        # 初始化帧堆叠器
        self.frame_stack = FrameStack(Config.FRAME_STACK)
        
        # 重新定义观察空间
        # 从原来的RGB图像空间变为灰度帧堆叠空间
        self.observation_space = gym.spaces.Box(
            low=0.0, 
            high=1.0, 
            shape=(Config.FRAME_STACK, Config.FRAME_SIZE, Config.FRAME_SIZE),
            dtype=np.float32
        )
        
        # 用于奖励塑形的变量
        self.prev_x_pos = 0         # 上一步的x坐标位置
        self.prev_time = 400        # 上一步的剩余时间
        self.death_penalty = -15    # 死亡惩罚
        self.time_penalty = -0.1    # 时间流逝惩罚
        
    def reset(self, **kwargs):
        """
        重置环境并返回初始状态
        
        Returns:
            np.array: 预处理并堆叠后的初始状态
        """
        # 重置原始环境
        obs = self.env.reset(**kwargs)
        
        # 预处理初始帧
        processed_frame = preprocess_frame(obs)
        
        # 重置帧堆叠器
        self.frame_stack.reset(processed_frame)
        
        # 重置奖励塑形变量
        self.prev_x_pos = 0
        self.prev_time = 400
        
        return self.frame_stack.get_state()
    
    def step(self, action):
        """
        执行动作并返回结果
        
        Args:
            action: 要执行的动作
            
        Returns:
            tuple: (状态, 奖励, 是否结束, 信息)
        """
        total_reward = 0
        
        # 动作重复 - 同一个动作执行多帧
        # 这减少了需要做决策的频率，加快训练速度
        for _ in range(Config.SKIP_FRAMES):
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            
            if done:
                break
        
        # 预处理新帧
        processed_frame = preprocess_frame(obs)
        
        # 添加到帧堆叠中
        self.frame_stack.add_frame(processed_frame)
        
        # 奖励塑形 - 改进原始奖励信号
        shaped_reward = self._shape_reward(total_reward, info, done)
        
        return self.frame_stack.get_state(), shaped_reward, done, info
    
    def _shape_reward(self, reward, info, done):
        """
        奖励塑形函数
        
        原始的马里奥游戏奖励很稀疏，只有在特定事件时才给奖励。
        我们增加一些中间奖励来指导学习：
        1. 向右移动给予奖励
        2. 时间流逝给予小惩罚
        3. 死亡给予惩罚
        
        Args:
            reward (float): 原始奖励
            info (dict): 游戏信息
            done (bool): 是否回合结束
            
        Returns:
            float: 塑形后的奖励
        """
        shaped_reward = reward
        
        # 获取当前游戏信息
        current_x = info.get('x_pos', 0)
        current_time = info.get('time', 400)
        
        # 移动奖励：向右移动给予奖励，向左移动给予惩罚
        x_reward = (current_x - self.prev_x_pos) * 0.1
        shaped_reward += x_reward
        
        # 时间惩罚：鼓励快速完成关卡
        time_reward = (current_time - self.prev_time) * self.time_penalty
        shaped_reward += time_reward
        
        # 死亡惩罚
        if done and info.get('life', 2) < 2:
            shaped_reward += self.death_penalty
        
        # 更新记录
        self.prev_x_pos = current_x
        self.prev_time = current_time
        
        return shaped_reward


def create_mario_env(world='1-1', stage=None):
    """
    创建马里奥游戏环境的工厂函数
    
    Args:
        world (str): 世界编号，如 '1-1', '1-2' 等
        stage (str): 舞台类型，None表示使用默认
        
    Returns:
        gym.Env: 包装后的马里奥环境
    """
    import gym_super_mario_bros
    from nes_py.wrappers import JoypadSpace
    from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
    
    # 创建基础环境
    if stage:
        env_name = f'SuperMarioBros-{world}-{stage}-v0'
    else:
        env_name = f'SuperMarioBros-{world}-v0'
    
    env = gym_super_mario_bros.make(env_name)
    
    # 简化动作空间：从复杂的按钮组合简化为几个基本动作
    # SIMPLE_MOVEMENT = [['NOOP'], ['right'], ['right', 'A'], ['right', 'B'], ['right', 'A', 'B'], ['A'], ['left']]
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    
    # 应用我们的包装器
    env = MarioWrapper(env)
    
    return env


def preprocess_batch_frames(frames):
    """
    批量预处理帧（用于并行环境）
    
    Args:
        frames (list): 原始帧列表
        
    Returns:
        np.array: 预处理后的帧批次
    """
    processed_frames = []
    
    for frame in frames:
        processed_frame = preprocess_frame(frame)
        processed_frames.append(processed_frame)
    
    return np.array(processed_frames)


def frames_to_tensor(frames, device=None):
    """
    将帧数组转换为PyTorch张量
    
    Args:
        frames (np.array): 帧数组
        device (torch.device): 目标设备
        
    Returns:
        torch.Tensor: PyTorch张量
    """
    if device is None:
        device = Config.DEVICE
    
    # 确保数据类型正确
    if frames.dtype != np.float32:
        frames = frames.astype(np.float32)
    
    # 转换为张量并移到指定设备
    tensor = torch.from_numpy(frames).to(device)
    
    return tensor


def tensor_to_frames(tensor):
    """
    将PyTorch张量转换回numpy数组
    
    Args:
        tensor (torch.Tensor): 输入张量
        
    Returns:
        np.array: numpy数组
    """
    return tensor.cpu().numpy()