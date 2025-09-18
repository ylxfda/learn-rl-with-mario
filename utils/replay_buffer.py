"""
经验回放缓冲区模块
PPO算法需要收集一批经验后进行批量更新
这个模块负责存储和管理从环境中收集的经验数据
"""

import torch
import numpy as np
from typing import List, Tuple, Dict
from config import Config


class RolloutBuffer:
    """
    PPO算法使用的轨迹缓冲区
    
    与DQN等算法的经验回放不同，PPO使用on-policy学习，
    只使用当前策略收集的最新经验进行更新，不需要长期存储历史经验。
    
    这个缓冲区存储：
    - 状态 (states)
    - 动作 (actions) 
    - 奖励 (rewards)
    - 价值估计 (values)
    - 对数概率 (log_probs)
    - 回合结束标志 (dones)
    - 优势估计 (advantages) - 计算后添加
    - 回报 (returns) - 计算后添加
    """
    
    def __init__(self, buffer_size, num_envs, obs_shape, action_dim, device):
        """
        初始化轨迹缓冲区
        
        Args:
            buffer_size (int): 缓冲区大小（每个环境收集的步数）
            num_envs (int): 并行环境数量
            obs_shape (tuple): 观察空间形状
            action_dim (int): 动作空间维度
            device (torch.device): 存储设备
        """
        self.buffer_size = buffer_size
        self.num_envs = num_envs
        self.device = device
        
        # 总的数据点数量 = 缓冲区大小 × 环境数量
        self.total_size = buffer_size * num_envs
        
        # 初始化存储空间
        # 状态：形状为 (buffer_size, num_envs, *obs_shape)
        self.states = torch.zeros(
            (buffer_size, num_envs, *obs_shape), 
            dtype=torch.float32, 
            device=device
        )
        
        # 动作：离散动作空间
        self.actions = torch.zeros(
            (buffer_size, num_envs), 
            dtype=torch.long, 
            device=device
        )
        
        # 奖励
        self.rewards = torch.zeros(
            (buffer_size, num_envs), 
            dtype=torch.float32, 
            device=device
        )
        
        # 价值函数估计
        self.values = torch.zeros(
            (buffer_size, num_envs), 
            dtype=torch.float32, 
            device=device
        )
        
        # 动作的对数概率
        self.log_probs = torch.zeros(
            (buffer_size, num_envs), 
            dtype=torch.float32, 
            device=device
        )
        
        # 回合结束标志
        self.dones = torch.zeros(
            (buffer_size, num_envs), 
            dtype=torch.bool, 
            device=device
        )
        
        # 优势和回报（在收集完数据后计算）
        self.advantages = torch.zeros(
            (buffer_size, num_envs), 
            dtype=torch.float32, 
            device=device
        )
        
        self.returns = torch.zeros(
            (buffer_size, num_envs), 
            dtype=torch.float32, 
            device=device
        )
        
        # 当前存储位置
        self.pos = 0
        self.full = False  # 标记缓冲区是否已满
    
    def add(self, states, actions, rewards, values, log_probs, dones):
        """
        向缓冲区添加一步经验
        
        Args:
            states (torch.Tensor): 状态，形状 (num_envs, *obs_shape)
            actions (torch.Tensor): 动作，形状 (num_envs,)
            rewards (torch.Tensor): 奖励，形状 (num_envs,)
            values (torch.Tensor): 价值估计，形状 (num_envs,)
            log_probs (torch.Tensor): 对数概率，形状 (num_envs,)
            dones (torch.Tensor): 结束标志，形状 (num_envs,)
        """
        # 确保所有输入都在正确的设备上
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        values = values.to(self.device)
        log_probs = log_probs.to(self.device)
        dones = dones.to(self.device)
        
        # 存储数据
        self.states[self.pos] = states
        self.actions[self.pos] = actions
        self.rewards[self.pos] = rewards
        self.values[self.pos] = values
        self.log_probs[self.pos] = log_probs
        self.dones[self.pos] = dones
        
        # 更新位置
        self.pos += 1
        
        # 检查是否已满
        if self.pos >= self.buffer_size:
            self.full = True
    
    def compute_advantages_and_returns(self, next_values, gamma=0.99, gae_lambda=0.95):
        """
        计算优势函数和回报
        
        使用GAE (Generalized Advantage Estimation) 计算优势：
        优势函数衡量某个动作相比于平均水平有多好
        
        GAE公式：
        δ_t = r_t + γ * V(s_{t+1}) - V(s_t)  # TD误差
        A_t = δ_t + (γλ)δ_{t+1} + (γλ)²δ_{t+2} + ...  # GAE优势
        
        Args:
            next_values (torch.Tensor): 下一个状态的价值估计，形状 (num_envs,)
            gamma (float): 折扣因子
            gae_lambda (float): GAE参数，控制偏差-方差权衡
        """
        next_values = next_values.to(self.device)
        
        # 初始化
        advantages = torch.zeros_like(self.rewards)
        gae = 0
        
        # 从后向前计算GAE优势
        for t in reversed(range(self.buffer_size)):
            if t == self.buffer_size - 1:
                # 最后一步的下一个价值
                next_non_terminal = 1.0 - self.dones[t].float()
                next_value = next_values
            else:
                next_non_terminal = 1.0 - self.dones[t].float()
                next_value = self.values[t + 1]
            
            # 计算TD误差
            delta = (self.rewards[t] + 
                    gamma * next_value * next_non_terminal - 
                    self.values[t])
            
            # 计算GAE优势
            gae = (delta + 
                  gamma * gae_lambda * next_non_terminal * gae)
            
            advantages[t] = gae
        
        # 计算回报 = 优势 + 价值估计
        returns = advantages + self.values
        
        # 存储结果
        self.advantages = advantages
        self.returns = returns
        
        # 标准化优势（有助于训练稳定性）
        self.advantages = (self.advantages - self.advantages.mean()) / (
            self.advantages.std() + 1e-8)
    
    def get_batch_iterator(self, minibatch_size):
        """
        获取小批次数据迭代器，用于PPO的多轮更新
        
        Args:
            minibatch_size (int): 小批次大小
            
        Yields:
            dict: 包含一个小批次数据的字典
        """
        if not self.full:
            print("Warning: Buffer is not full, using partial data")
        
        # 将数据展平为 (total_size, ...)
        indices = torch.randperm(self.total_size, device=self.device)
        
        states = self.states.flatten(0, 1)  # (total_size, *obs_shape)
        actions = self.actions.flatten(0, 1)  # (total_size,)
        values = self.values.flatten(0, 1)
        log_probs = self.log_probs.flatten(0, 1)
        advantages = self.advantages.flatten(0, 1)
        returns = self.returns.flatten(0, 1)
        
        # 按小批次返回数据
        for start_idx in range(0, self.total_size, minibatch_size):
            end_idx = start_idx + minibatch_size
            batch_indices = indices[start_idx:end_idx]
            
            yield {
                'states': states[batch_indices],
                'actions': actions[batch_indices],
                'old_values': values[batch_indices],
                'old_log_probs': log_probs[batch_indices],
                'advantages': advantages[batch_indices],
                'returns': returns[batch_indices],
            }
    
    def reset(self):
        """
        重置缓冲区，准备收集新的轨迹
        """
        self.pos = 0
        self.full = False
        
        # 清零所有张量（可选，主要用于调试）
        self.states.zero_()
        self.actions.zero_()
        self.rewards.zero_()
        self.values.zero_()
        self.log_probs.zero_()
        self.dones.zero_()
        self.advantages.zero_()
        self.returns.zero_()
    
    def size(self):
        """
        返回当前缓冲区中的有效数据量
        
        Returns:
            int: 有效数据点数量
        """
        if self.full:
            return self.total_size
        else:
            return self.pos * self.num_envs
    
    def get_statistics(self):
        """
        获取缓冲区数据的统计信息，用于调试和监控
        
        Returns:
            dict: 统计信息字典
        """
        if self.size() == 0:
            return {}
        
        stats = {}
        
        # 奖励统计
        valid_rewards = self.rewards[:self.pos] if not self.full else self.rewards
        stats['reward_mean'] = float(valid_rewards.mean())
        stats['reward_std'] = float(valid_rewards.std())
        stats['reward_min'] = float(valid_rewards.min())
        stats['reward_max'] = float(valid_rewards.max())
        
        # 价值函数统计
        valid_values = self.values[:self.pos] if not self.full else self.values
        stats['value_mean'] = float(valid_values.mean())
        stats['value_std'] = float(valid_values.std())
        
        # 优势函数统计
        if hasattr(self, 'advantages') and self.advantages.numel() > 0:
            valid_advantages = self.advantages[:self.pos] if not self.full else self.advantages
            stats['advantage_mean'] = float(valid_advantages.mean())
            stats['advantage_std'] = float(valid_advantages.std())
        
        # 动作分布统计
        valid_actions = self.actions[:self.pos] if not self.full else self.actions
        action_counts = torch.bincount(valid_actions.flatten())
        total_actions = valid_actions.numel()
        
        for i, count in enumerate(action_counts):
            stats[f'action_{i}_ratio'] = float(count) / total_actions
        
        return stats
    
    def __len__(self):
        """
        返回缓冲区容量
        
        Returns:
            int: 缓冲区总容量
        """
        return self.total_size
    
    def __repr__(self):
        """
        返回缓冲区的字符串表示
        
        Returns:
            str: 缓冲区信息
        """
        return (f"RolloutBuffer(size={self.buffer_size}, "
                f"num_envs={self.num_envs}, "
                f"current_size={self.size()}, "
                f"full={self.full})")