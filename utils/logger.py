"""
训练日志记录模块
负责记录训练过程中的各种指标，包括：
- 奖励曲线
- 损失函数值
- 训练统计信息
- TensorBoard可视化
"""

import os
import json
import time
import numpy as np
from collections import deque, defaultdict
from datetime import datetime

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    print("Warning: TensorBoard not available. Install with 'pip install tensorboard'")
    TENSORBOARD_AVAILABLE = False

from config import Config


class TrainingLogger:
    """
    训练日志记录器
    
    功能：
    1. 记录训练过程中的各种指标
    2. 计算滑动平均值
    3. 保存训练日志到文件
    4. TensorBoard可视化支持
    5. 性能统计
    """
    
    def __init__(self, log_dir=None, experiment_name=None):
        """
        初始化日志记录器
        
        Args:
            log_dir (str): 日志保存目录
            experiment_name (str): 实验名称
        """
        # 设置日志目录
        if log_dir is None:
            log_dir = Config.LOG_DIR
        
        # 创建实验特定的日志目录
        if experiment_name is None:
            experiment_name = f"mario_ppo_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        self.log_dir = os.path.join(log_dir, experiment_name)
        os.makedirs(self.log_dir, exist_ok=True)
        
        # 初始化TensorBoard写入器
        self.tensorboard_writer = None
        if Config.TENSORBOARD_LOG and TENSORBOARD_AVAILABLE:
            self.tensorboard_writer = SummaryWriter(self.log_dir)
        
        # 训练指标存储
        self.metrics = defaultdict(list)
        self.episode_metrics = defaultdict(list)
        
        # 滑动平均计算器（用于平滑曲线显示）
        self.running_averages = defaultdict(lambda: deque(maxlen=100))
        
        # 训练统计
        self.start_time = time.time()
        self.episode_count = 0
        self.step_count = 0
        self.update_count = 0
        
        # 最佳性能记录
        self.best_reward = float('-inf')
        self.best_episode = 0
        
        print(f"日志将保存到: {self.log_dir}")
        
    def log_episode(self, episode_reward, episode_length, info=None):
        """
        记录回合结束时的信息
        
        Args:
            episode_reward (float): 回合总奖励
            episode_length (int): 回合长度（步数）
            info (dict): 额外的游戏信息
        """
        self.episode_count += 1
        
        # 记录基本指标
        self.episode_metrics['reward'].append(episode_reward)
        self.episode_metrics['length'].append(episode_length)
        
        # 更新滑动平均
        self.running_averages['reward'].append(episode_reward)
        self.running_averages['length'].append(episode_length)
        
        # 记录游戏特定信息
        if info:
            for key, value in info.items():
                if isinstance(value, (int, float)):
                    self.episode_metrics[key].append(value)
                    self.running_averages[key].append(value)
        
        # 更新最佳性能
        if episode_reward > self.best_reward:
            self.best_reward = episode_reward
            self.best_episode = self.episode_count
        
        # TensorBoard记录
        if self.tensorboard_writer:
            self.tensorboard_writer.add_scalar('Episode/Reward', episode_reward, self.episode_count)
            self.tensorboard_writer.add_scalar('Episode/Length', episode_length, self.episode_count)
            self.tensorboard_writer.add_scalar('Episode/Reward_MA', np.mean(self.running_averages['reward']), self.episode_count)
            
            if info:
                for key, value in info.items():
                    if isinstance(value, (int, float)):
                        self.tensorboard_writer.add_scalar(f'Episode/{key}', value, self.episode_count)
    
    def log_training_step(self, **metrics):
        """
        记录训练步骤中的指标
        
        Args:
            **metrics: 各种训练指标（loss, learning_rate等）
        """
        self.step_count += 1
        
        for key, value in metrics.items():
            if isinstance(value, (int, float, np.floating, np.integer)):
                self.metrics[key].append(float(value))
                
                # TensorBoard记录
                if self.tensorboard_writer:
                    self.tensorboard_writer.add_scalar(f'Training/{key}', value, self.step_count)
    
    def log_update(self, **metrics):
        """
        记录PPO更新时的指标
        
        Args:
            **metrics: PPO相关指标（policy_loss, value_loss, entropy等）
        """
        self.update_count += 1
        
        for key, value in metrics.items():
            if isinstance(value, (int, float, np.floating, np.integer)):
                self.metrics[f'update_{key}'].append(float(value))
                
                # TensorBoard记录
                if self.tensorboard_writer:
                    self.tensorboard_writer.add_scalar(f'Update/{key}', value, self.update_count)
    
    def log_system_info(self, **info):
        """
        记录系统信息（内存使用、GPU利用率等）
        
        Args:
            **info: 系统信息字典
        """
        if self.tensorboard_writer:
            for key, value in info.items():
                if isinstance(value, (int, float, np.floating, np.integer)):
                    self.tensorboard_writer.add_scalar(f'System/{key}', value, self.step_count)
    
    def get_recent_average(self, metric_name, window=100):
        """
        获取指定指标的近期平均值
        
        Args:
            metric_name (str): 指标名称
            window (int): 平均窗口大小
            
        Returns:
            float: 平均值，如果数据不足则返回None
        """
        if metric_name in self.running_averages and len(self.running_averages[metric_name]) > 0:
            return np.mean(list(self.running_averages[metric_name])[-window:])
        return None
    
    def print_training_stats(self):
        """
        打印训练统计信息
        """
        if self.episode_count == 0:
            return
        
        current_time = time.time()
        elapsed_time = current_time - self.start_time
        
        # 计算各种平均值
        avg_reward = self.get_recent_average('reward', 100)
        avg_length = self.get_recent_average('length', 100)
        
        print(f"\n{'='*60}")
        print(f"训练统计 (回合 {self.episode_count})")
        print(f"{'='*60}")
        print(f"运行时间: {elapsed_time/3600:.2f} 小时")
        print(f"总步数: {self.step_count:,}")
        print(f"总更新次数: {self.update_count}")
        print(f"最近100回合平均奖励: {avg_reward:.2f}" if avg_reward else "平均奖励: N/A")
        print(f"最近100回合平均长度: {avg_length:.1f}" if avg_length else "平均长度: N/A")
        print(f"最佳奖励: {self.best_reward:.2f} (回合 {self.best_episode})")
        
        # 显示最近的损失信息
        recent_losses = ['update_policy_loss', 'update_value_loss', 'update_total_loss']
        for loss_name in recent_losses:
            if loss_name in self.metrics and self.metrics[loss_name]:
                recent_loss = self.metrics[loss_name][-1]
                display_name = loss_name.replace('update_', '').replace('_', ' ').title()
                print(f"{display_name}: {recent_loss:.4f}")
        
        print(f"{'='*60}\n")
    
    def save_training_log(self):
        """
        保存训练日志到JSON文件
        """
        log_data = {
            'experiment_info': {
                'start_time': datetime.fromtimestamp(self.start_time).isoformat(),
                'episode_count': self.episode_count,
                'step_count': self.step_count,
                'update_count': self.update_count,
                'best_reward': self.best_reward,
                'best_episode': self.best_episode,
            },
            'episode_metrics': dict(self.episode_metrics),
            'training_metrics': dict(self.metrics),
            'config': {
                'num_envs': Config.NUM_ENVS,
                'learning_rate': Config.LEARNING_RATE,
                'ppo_epochs': Config.PPO_EPOCHS,
                'clip_epsilon': Config.CLIP_EPSILON,
                'frame_stack': Config.FRAME_STACK,
            }
        }
        
        # 保存到JSON文件
        log_file = os.path.join(self.log_dir, 'training_log.json')
        with open(log_file, 'w') as f:
            json.dump(log_data, f, indent=2)
        
        print(f"训练日志已保存到: {log_file}")
    
    def close(self):
        """
        关闭日志记录器，清理资源
        """
        if self.tensorboard_writer:
            self.tensorboard_writer.close()
        
        # 保存最终日志
        self.save_training_log()
        
    def __del__(self):
        """
        析构函数，确保资源被正确清理
        """
        self.close()


class PerformanceMonitor:
    """
    性能监控器 - 监控训练过程中的系统资源使用情况
    """
    
    def __init__(self):
        """
        初始化性能监控器
        """
        self.gpu_available = False
        
        # 尝试导入GPU监控工具
        try:
            import torch
            if torch.cuda.is_available():
                self.gpu_available = True
                self.device_count = torch.cuda.device_count()
        except ImportError:
            pass
    
    def get_gpu_memory_usage(self):
        """
        获取GPU内存使用情况
        
        Returns:
            dict: GPU内存使用信息
        """
        if not self.gpu_available:
            return {}
        
        import torch
        gpu_info = {}
        
        for i in range(self.device_count):
            memory_allocated = torch.cuda.memory_allocated(i) / 1024**3  # GB
            memory_cached = torch.cuda.memory_reserved(i) / 1024**3     # GB
            
            gpu_info[f'gpu_{i}_memory_allocated'] = memory_allocated
            gpu_info[f'gpu_{i}_memory_cached'] = memory_cached
        
        return gpu_info
    
    def get_system_info(self):
        """
        获取系统信息
        
        Returns:
            dict: 系统资源使用信息
        """
        import psutil
        
        # CPU使用率
        cpu_percent = psutil.cpu_percent(interval=1)
        
        # 内存使用情况
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        memory_used_gb = memory.used / 1024**3
        
        system_info = {
            'cpu_percent': cpu_percent,
            'memory_percent': memory_percent,
            'memory_used_gb': memory_used_gb,
        }
        
        # 添加GPU信息
        system_info.update(self.get_gpu_memory_usage())
        
        return system_info


class ProgressTracker:
    """
    训练进度跟踪器 - 跟踪训练目标完成情况
    """
    
    def __init__(self, target_reward=3000, patience=100):
        """
        初始化进度跟踪器
        
        Args:
            target_reward (float): 目标奖励值
            patience (int): 早停耐心值（多少回合没有改进就停止）
        """
        self.target_reward = target_reward
        self.patience = patience
        
        self.best_avg_reward = float('-inf')
        self.episodes_without_improvement = 0
        self.target_achieved = False
        
        self.reward_history = deque(maxlen=100)  # 保存最近100回合的奖励
    
    def update(self, episode_reward):
        """
        更新进度跟踪
        
        Args:
            episode_reward (float): 当前回合奖励
            
        Returns:
            dict: 跟踪信息
        """
        self.reward_history.append(episode_reward)
        
        # 计算最近100回合的平均奖励
        if len(self.reward_history) >= 10:  # 至少10回合才开始计算
            avg_reward = np.mean(self.reward_history)
            
            # 检查是否有改进
            if avg_reward > self.best_avg_reward:
                self.best_avg_reward = avg_reward
                self.episodes_without_improvement = 0
            else:
                self.episodes_without_improvement += 1
            
            # 检查是否达到目标
            if avg_reward >= self.target_reward and not self.target_achieved:
                self.target_achieved = True
                print(f"\n🎉 目标达成！平均奖励 {avg_reward:.2f} 超过目标 {self.target_reward}")
        
        return {
            'avg_reward': np.mean(self.reward_history) if self.reward_history else 0,
            'best_avg_reward': self.best_avg_reward,
            'episodes_without_improvement': self.episodes_without_improvement,
            'target_achieved': self.target_achieved,
            'should_stop': self.episodes_without_improvement >= self.patience
        }