"""
PPO (Proximal Policy Optimization) 算法实现

PPO是一种on-policy的策略梯度算法，通过裁剪目标函数来限制策略更新的幅度，
确保训练的稳定性。这是目前最流行和有效的强化学习算法之一。

主要特点：
1. 使用裁剪替代目标函数防止策略更新过大
2. 结合价值函数进行Actor-Critic训练
3. 使用GAE计算优势函数
4. 支持多轮更新提高样本效率
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from typing import Dict, Any, Tuple, List

from algorithms.base import BaseRLAlgorithm, ModelManager
from networks.networks import create_actor_critic_network
from utils.replay_buffer import RolloutBuffer
from config import Config


class PPOAlgorithm(BaseRLAlgorithm):
    """
    PPO算法实现类
    
    核心思想：
    - 使用重要性采样比率 r(θ) = π_θ(a|s) / π_θ_old(a|s)
    - 通过裁剪函数限制比率在 [1-ε, 1+ε] 范围内
    - 优化裁剪后的策略目标 + 价值函数损失 + 熵奖励
    """
    
    def __init__(self, 
                 observation_space, 
                 action_space,
                 device=None,
                 logger=None):
        """
        初始化PPO算法
        
        Args:
            observation_space: 观察空间
            action_space: 动作空间  
            device: 计算设备
            logger: 日志记录器
        """
        super().__init__(observation_space, action_space, device, logger)
        
        # 提取空间信息
        self.obs_shape = observation_space.shape
        self.action_dim = action_space.n
        
        # 创建Actor-Critic网络
        self.actor_critic = create_actor_critic_network(
            observation_shape=self.obs_shape,
            action_dim=self.action_dim,
            device=self.device
        )
        
        # 创建优化器
        self.optimizer = optim.Adam(
            self.actor_critic.parameters(),
            lr=Config.LEARNING_RATE,
            eps=1e-5  # 防止数值不稳定
        )
        
        # 学习率调度器（可选）
        self.lr_scheduler = optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=1000,  # 每1000次更新降低学习率
            gamma=0.9
        )
        
        # PPO超参数
        self.clip_epsilon = Config.CLIP_EPSILON
        self.ppo_epochs = Config.PPO_EPOCHS
        self.value_loss_coeff = Config.VALUE_LOSS_COEFF
        self.entropy_coeff = Config.ENTROPY_COEFF
        self.max_grad_norm = Config.MAX_GRAD_NORM
        
        # GAE参数
        self.gamma = Config.GAMMA
        self.gae_lambda = Config.GAE_LAMBDA
        
        # 存储网络和优化器引用
        self.networks = {'actor_critic': self.actor_critic}
        self.optimizers = {'main': self.optimizer}
        
        # 训练统计
        self.policy_losses = []
        self.value_losses = []
        self.entropies = []
        self.clip_fractions = []
        
        print(f"PPO算法初始化完成")
        print(f"网络参数总数: {self.actor_critic.count_parameters():,}")
        
    def act(self, observations, deterministic=False):
        """
        根据当前策略选择动作
        
        Args:
            observations (torch.Tensor): 观察，形状 (batch_size, *obs_shape)
            deterministic (bool): 是否选择确定性动作
            
        Returns:
            tuple: (动作, 额外信息字典)
        """
        with torch.no_grad():
            # 确保输入在正确设备上
            observations = observations.to(self.device)
            
            # 获取动作、对数概率和价值
            actions, log_probs, values = self.actor_critic.act(
                observations, 
                deterministic=deterministic
            )
            
            # 额外信息
            extra_info = {
                'log_probs': log_probs,
                'values': values
            }
            
            return actions, extra_info
    
    def compute_gae(self, rewards, values, dones, next_values):
        """
        计算GAE优势函数
        
        GAE (Generalized Advantage Estimation) 在偏差和方差之间取得平衡：
        - λ=0: 高偏差，低方差 (TD误差)
        - λ=1: 低偏差，高方差 (蒙特卡洛)
        
        Args:
            rewards (torch.Tensor): 奖励序列，形状 (T, N)
            values (torch.Tensor): 价值估计序列，形状 (T, N) 
            dones (torch.Tensor): 结束标志序列，形状 (T, N)
            next_values (torch.Tensor): 最后状态的价值估计，形状 (N,)
            
        Returns:
            tuple: (优势函数, 回报)
        """
        T, N = rewards.shape
        advantages = torch.zeros_like(rewards)
        
        # 从后往前计算GAE
        gae = 0
        for t in reversed(range(T)):
            if t == T - 1:
                # 最后一步的下一个价值
                next_non_terminal = 1.0 - dones[t].float()
                next_value = next_values
            else:
                next_non_terminal = 1.0 - dones[t].float()  
                next_value = values[t + 1]
            
            # TD误差: δ = r + γV(s') - V(s)
            delta = rewards[t] + self.gamma * next_value * next_non_terminal - values[t]
            
            # GAE: A = δ + γλ * next_non_terminal * gae
            gae = delta + self.gamma * self.gae_lambda * next_non_terminal * gae
            advantages[t] = gae
        
        # 计算回报: R = A + V
        returns = advantages + values
        
        return advantages, returns
    
    def update(self, rollout_buffer):
        """
        使用收集的轨迹数据更新策略和价值网络
        
        Args:
            rollout_buffer (RolloutBuffer): 包含轨迹数据的缓冲区
            
        Returns:
            dict: 更新统计信息
        """
        # 统计信息
        update_stats = {
            'policy_loss': 0.0,
            'value_loss': 0.0, 
            'entropy': 0.0,
            'total_loss': 0.0,
            'clip_fraction': 0.0,
            'kl_divergence': 0.0,
            'explained_variance': 0.0
        }
        
        # 进行多轮PPO更新
        for epoch in range(self.ppo_epochs):
            epoch_stats = {key: 0.0 for key in update_stats.keys()}
            batch_count = 0
            
            # 遍历小批次数据
            for batch in rollout_buffer.get_batch_iterator(Config.MINIBATCH_SIZE):
                batch_count += 1
                
                # 解包批次数据
                states = batch['states']
                actions = batch['actions'] 
                old_log_probs = batch['old_log_probs']
                old_values = batch['old_values']
                advantages = batch['advantages']
                returns = batch['returns']
                
                # 重新计算动作概率和价值
                new_log_probs, new_values, entropy = self.actor_critic.evaluate(
                    states, actions
                )
                
                # 计算重要性采样比率
                ratio = torch.exp(new_log_probs - old_log_probs)
                
                # PPO裁剪目标函数
                surr1 = ratio * advantages
                surr2 = torch.clamp(
                    ratio, 
                    1.0 - self.clip_epsilon, 
                    1.0 + self.clip_epsilon
                ) * advantages
                
                # 策略损失 = -min(surr1, surr2)
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # 价值函数损失 (可选择是否裁剪)
                if Config.CLIP_EPSILON > 0:
                    # 裁剪价值损失
                    value_pred_clipped = old_values + torch.clamp(
                        new_values - old_values,
                        -self.clip_epsilon,
                        self.clip_epsilon
                    )
                    value_loss_1 = (new_values - returns).pow(2)
                    value_loss_2 = (value_pred_clipped - returns).pow(2)
                    value_loss = 0.5 * torch.max(value_loss_1, value_loss_2).mean()
                else:
                    # 标准MSE损失
                    value_loss = 0.5 * (new_values - returns).pow(2).mean()
                
                # 熵损失（鼓励探索）
                entropy_loss = entropy.mean()
                
                # 总损失
                total_loss = (policy_loss + 
                             self.value_loss_coeff * value_loss - 
                             self.entropy_coeff * entropy_loss)
                
                # 反向传播
                self.optimizer.zero_grad()
                total_loss.backward()
                # print(f"debug: policy_loss={policy_loss.item()}, value_loss={value_loss.item()}, entropy_loss={entropy_loss.item()}, total_loss={total_loss.item()}")
                
                # 梯度裁剪（防止梯度爆炸）
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.actor_critic.parameters(), 
                    self.max_grad_norm
                )
                
                # 更新参数
                self.optimizer.step()
                
                # 统计信息
                with torch.no_grad():
                    # 计算裁剪比例
                    clipped = torch.abs(ratio - 1.0) > self.clip_epsilon
                    clip_fraction = clipped.float().mean()
                    
                    # 近似KL散度
                    kl_div = ((ratio - 1.0) - (new_log_probs - old_log_probs)).mean()
                    
                    # 解释方差 (explained variance)
                    y_true = returns
                    y_pred = new_values
                    var_y = y_true.var()
                    explained_var = 1 - (y_true - y_pred).var() / (var_y + 1e-8)
                
                # 累积统计
                epoch_stats['policy_loss'] += policy_loss.item()
                epoch_stats['value_loss'] += value_loss.item()
                epoch_stats['entropy'] += entropy_loss.item()
                epoch_stats['total_loss'] += total_loss.item()
                epoch_stats['clip_fraction'] += clip_fraction.item()
                epoch_stats['kl_divergence'] += kl_div.item()
                epoch_stats['explained_variance'] += explained_var.item()
            
            # 计算轮次平均值
            if batch_count > 0:
                for key in epoch_stats:
                    epoch_stats[key] /= batch_count
                    update_stats[key] += epoch_stats[key]
        
        # 计算最终平均值
        for key in update_stats:
            update_stats[key] /= self.ppo_epochs
        
        # 更新学习率
        self.lr_scheduler.step()
        update_stats['learning_rate'] = self.optimizer.param_groups[0]['lr']
        
        # 更新计数
        self.total_updates += 1
        
        # 记录统计信息
        if self.logger:
            self.logger.log_update(**update_stats)
        
        return update_stats
    
    def save_model(self, filepath):
        """
        保存PPO模型
        
        Args:
            filepath (str): 保存路径
        """
        checkpoint = self.create_checkpoint()
        
        # 添加PPO特定信息
        checkpoint.update({
            'ppo_config': {
                'clip_epsilon': self.clip_epsilon,
                'ppo_epochs': self.ppo_epochs,
                'value_loss_coeff': self.value_loss_coeff,
                'entropy_coeff': self.entropy_coeff,
                'gamma': self.gamma,
                'gae_lambda': self.gae_lambda,
            },
            'obs_shape': self.obs_shape,
            'action_dim': self.action_dim,
        })
        
        torch.save(checkpoint, filepath)
        print(f"PPO模型已保存到: {filepath}")
    
    def load_model(self, filepath):
        """
        加载PPO模型
        
        Args:
            filepath (str): 模型文件路径
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        
        # 验证模型兼容性
        if 'obs_shape' in checkpoint:
            if checkpoint['obs_shape'] != self.obs_shape:
                print(f"Warning: 观察空间不匹配. 期望 {self.obs_shape}, 得到 {checkpoint['obs_shape']}")
        
        if 'action_dim' in checkpoint:
            if checkpoint['action_dim'] != self.action_dim:
                print(f"Warning: 动作空间不匹配. 期望 {self.action_dim}, 得到 {checkpoint['action_dim']}")
        
        # 加载检查点
        self.load_checkpoint(checkpoint)
        
        # 恢复PPO特定配置
        if 'ppo_config' in checkpoint:
            ppo_config = checkpoint['ppo_config']
            self.clip_epsilon = ppo_config.get('clip_epsilon', self.clip_epsilon)
            self.ppo_epochs = ppo_config.get('ppo_epochs', self.ppo_epochs)
            self.value_loss_coeff = ppo_config.get('value_loss_coeff', self.value_loss_coeff)
            self.entropy_coeff = ppo_config.get('entropy_coeff', self.entropy_coeff)
            self.gamma = ppo_config.get('gamma', self.gamma)
            self.gae_lambda = ppo_config.get('gae_lambda', self.gae_lambda)
        
        print(f"PPO模型已从 {filepath} 加载")
    
    def get_action_probabilities(self, observations):
        """
        获取动作概率分布（用于分析）
        
        Args:
            observations (torch.Tensor): 观察
            
        Returns:
            torch.Tensor: 动作概率，形状 (batch_size, action_dim)
        """
        with torch.no_grad():
            observations = observations.to(self.device)
            action_dist, _ = self.actor_critic(observations)
            return action_dist.probs
    
    def compute_value(self, observations):
        """
        计算状态价值（用于分析）
        
        Args:
            observations (torch.Tensor): 观察
            
        Returns:
            torch.Tensor: 状态价值，形状 (batch_size,)
        """
        with torch.no_grad():
            observations = observations.to(self.device)
            values = self.actor_critic.get_value(observations)
            return values


def create_ppo_algorithm(observation_space, 
                        action_space, 
                        device=None, 
                        logger=None):
    """
    创建PPO算法的工厂函数
    
    Args:
        observation_space: 观察空间
        action_space: 动作空间
        device: 计算设备
        logger: 日志记录器
        
    Returns:
        PPOAlgorithm: PPO算法实例
    """
    return PPOAlgorithm(
        observation_space=observation_space,
        action_space=action_space,
        device=device,
        logger=logger
    )


def test_ppo_algorithm():
    """
    测试PPO算法的基本功能
    """
    print("测试PPO算法...")
    
    # 模拟环境空间
    import gym
    
    class MockObsSpace:
        def __init__(self):
            self.shape = (Config.FRAME_STACK, Config.FRAME_SIZE, Config.FRAME_SIZE)
    
    class MockActionSpace:
        def __init__(self):
            self.n = 7  # 马里奥游戏动作数
    
    obs_space = MockObsSpace()
    action_space = MockActionSpace()
    
    # 创建PPO算法
    ppo = create_ppo_algorithm(obs_space, action_space)
    
    # 创建测试数据
    batch_size = 4
    test_obs = torch.randn(batch_size, *obs_space.shape)
    
    print(f"测试观察形状: {test_obs.shape}")
    
    # 测试动作选择
    actions, extra_info = ppo.act(test_obs)
    print(f"选择的动作: {actions}")
    print(f"额外信息: {list(extra_info.keys())}")
    
    # 测试价值计算
    values = ppo.compute_value(test_obs)
    print(f"状态价值: {values}")
    
    # 测试概率分布
    probs = ppo.get_action_probabilities(test_obs)
    print(f"动作概率形状: {probs.shape}")
    
    print("PPO算法测试完成！")


if __name__ == "__main__":
    test_ppo_algorithm()