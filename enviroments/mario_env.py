"""
马里奥游戏环境模块
封装了马里奥游戏环境，提供标准化的接口
包括环境创建、状态预处理、奖励塑形等功能
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
from config import Config


class MarioEnvironment:
    """
    马里奥环境封装类
    
    提供统一的接口来创建和管理马里奥游戏环境
    支持不同的世界和关卡配置
    """
    
    def __init__(self, world='1-1', render_mode=None):
        """
        初始化马里奥环境
        
        Args:
            world (str): 世界-关卡，如 '1-1', '1-2', '2-1' 等
            render_mode (str): 渲染模式，None表示不渲染，'human'表示显示窗口
        """
        if not MARIO_AVAILABLE:
            raise ImportError("gym-super-mario-bros is required but not installed")
        
        self.world = world
        self.render_mode = render_mode
        
        # 创建环境
        self.env = self._create_env()
        
        # 环境信息
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        
        # 统计信息
        self.episode_count = 0
        self.total_steps = 0
        
    def _create_env(self):
        """
        创建马里奥游戏环境
        
        Returns:
            gym.Env: 配置好的马里奥环境
        """
        # 根据世界编号创建环境
        if '-' in self.world:
            world_num, level_num = self.world.split('-')
            env_name = f'SuperMarioBros-{world_num}-{level_num}-v0'
        else:
            env_name = f'SuperMarioBros-{self.world}-v0'
        
        # 创建基础环境
        try:
            env = gym_super_mario_bros.make(env_name)
        except Exception as e:
            print(f"Failed to create environment {env_name}, using default SuperMarioBros-v0")
            env = gym_super_mario_bros.make('SuperMarioBros-v0')
        
        # 复杂动作空间
        env = JoypadSpace(env, COMPLEX_MOVEMENT)
        
        # 应用我们的包装器进行预处理
        env = MarioWrapper(env)
        
        return env
    
    def reset(self):
        """
        重置环境
        
        Returns:
            np.array: 初始状态
        """
        state = self.env.reset()
        self.episode_count += 1
        return state
    
    def step(self, action):
        """
        执行一步动作
        
        Args:
            action (int): 要执行的动作
            
        Returns:
            tuple: (下一个状态, 奖励, 是否结束, 信息字典)
        """
        next_state, reward, done, info = self.env.step(action)
        self.total_steps += 1
        
        return next_state, reward, done, info
    
    def render(self, mode='human'):
        """
        渲染环境
        
        Args:
            mode (str): 渲染模式
        """
        return self.env.render(mode)
    
    def close(self):
        """
        关闭环境
        """
        self.env.close()
    
    def get_action_meanings(self):
        """
        获取动作的含义
        
        Returns:
            list: 动作含义列表
        """
        return [
            'NOOP',           # 0: 无操作
            'right',          # 1: 向右移动
            'right_A',        # 2: 向右跳跃
            'right_B',        # 3: 向右跑
            'right_A_B',      # 4: 向右跑跳
            'A',              # 5: 原地跳跃
            'left'            # 6: 向左移动
        ]
    
    def get_random_action(self):
        """
        获取随机动作
        
        Returns:
            int: 随机动作
        """
        return self.action_space.sample()
    
    def get_info(self):
        """
        获取环境信息
        
        Returns:
            dict: 环境信息
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
    多世界马里奥环境
    
    支持在不同关卡之间切换，增加训练的多样性
    """
    
    def __init__(self, worlds=['1-1', '1-2', '1-3'], render_mode=None, random_start=True):
        """
        初始化多世界环境
        
        Args:
            worlds (list): 可用的世界关卡列表
            render_mode (str): 渲染模式
            random_start (bool): 是否随机选择起始关卡
        """
        self.worlds = worlds
        self.current_world_idx = 0
        self.random_start = random_start
        
        # 如果随机开始，选择一个随机关卡
        if random_start:
            self.current_world_idx = np.random.randint(len(worlds))
        
        # 初始化当前世界
        super().__init__(worlds[self.current_world_idx], render_mode)
        
        # 世界切换统计
        self.world_episode_counts = {world: 0 for world in worlds}
        self.world_success_counts = {world: 0 for world in worlds}
    
    def switch_world(self, world_idx=None):
        """
        切换到指定世界
        
        Args:
            world_idx (int): 世界索引，None表示随机选择
        """
        if world_idx is None:
            world_idx = np.random.randint(len(self.worlds))
        
        if world_idx != self.current_world_idx:
            self.current_world_idx = world_idx
            self.world = self.worlds[world_idx]
            
            # 重新创建环境
            self.env.close()
            self.env = self._create_env()
            
            print(f"切换到世界: {self.world}")
    
    def reset(self, switch_world=None):
        """
        重置环境，可选择切换世界
        
        Args:
            switch_world (bool): 是否在重置时切换世界
            
        Returns:
            np.array: 初始状态
        """
        # 决定是否切换世界
        if switch_world is None:
            switch_world = self.random_start
        
        if switch_world and np.random.random() < 0.3:  # 30%概率切换世界
            self.switch_world()
        
        # 更新当前世界的回合计数
        self.world_episode_counts[self.world] += 1
        
        return super().reset()
    
    def step(self, action):
        """
        执行动作，记录成功统计
        
        Args:
            action (int): 动作
            
        Returns:
            tuple: 状态转移结果
        """
        next_state, reward, done, info = super().step(action)
        
        # 记录成功（通关）
        if done and info.get('flag_get', False):
            self.world_success_counts[self.world] += 1
        
        return next_state, reward, done, info
    
    def get_world_statistics(self):
        """
        获取各世界的统计信息
        
        Returns:
            dict: 世界统计信息
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
        获取多世界环境信息
        
        Returns:
            dict: 环境信息
        """
        info = super().get_info()
        info.update({
            'available_worlds': self.worlds,
            'current_world': self.world,
            'world_statistics': self.get_world_statistics(),
        })
        return info


def create_mario_environment(world='1-1', multi_world=False, worlds=None, render_mode=None):
    """
    马里奥环境创建工厂函数
    
    Args:
        world (str): 单一世界模式下的世界编号
        multi_world (bool): 是否使用多世界模式
        worlds (list): 多世界模式下的世界列表
        render_mode (str): 渲染模式
        
    Returns:
        MarioEnvironment: 创建的马里奥环境
    """
    if multi_world:
        if worlds is None:
            # 默认的多世界配置
            worlds = ['1-1', '1-2', '1-3', '1-4']
        return MultiWorldMarioEnvironment(worlds, render_mode)
    else:
        return MarioEnvironment(world, render_mode)


def test_mario_environment():
    """
    测试马里奥环境是否工作正常
    """
    print("测试马里奥环境...")
    
    if not MARIO_AVAILABLE:
        print("Mario environment not available, skipping test")
        return
    
    try:
        # 创建环境
        env = create_mario_environment('1-1')
        print(f"环境创建成功: {env.get_info()}")
        
        # 测试重置
        state = env.reset()
        print(f"重置成功，状态形状: {state.shape}")
        
        # 测试几步
        for i in range(5):
            action = env.get_random_action()
            next_state, reward, done, info = env.step(action)
            print(f"步骤 {i+1}: 动作={action}, 奖励={reward:.2f}, 结束={done}")
            
            if done:
                break
        
        # 关闭环境
        env.close()
        print("环境测试完成")
        
    except Exception as e:
        print(f"环境测试失败: {e}")


if __name__ == "__main__":
    test_mario_environment()