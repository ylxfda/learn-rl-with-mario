"""
并行环境管理模块
实现多个马里奥游戏环境的并行运行，加速数据收集
使用多进程或向量化环境来提高训练效率
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
    子进程向量化环境
    
    每个环境在独立的进程中运行，避免GIL限制，真正实现并行计算
    适合CPU密集型的环境（如马里奥游戏的图像处理）
    """
    
    def __init__(self, env_fns):
        """
        初始化子进程向量化环境
        
        Args:
            env_fns (list): 环境创建函数列表
        """
        self.waiting = False
        self.closed = False
        self.num_envs = len(env_fns)
        
        # 创建进程间通信的管道
        self.remotes, self.work_remotes = zip(*[mp.Pipe() for _ in range(self.num_envs)])
        
        # 启动子进程
        self.processes = []
        for work_remote, remote, env_fn in zip(self.work_remotes, self.remotes, env_fns):
            args = (work_remote, remote, env_fn)
            process = mp.Process(target=self._worker, args=args)
            process.daemon = True  # 主进程结束时子进程也结束
            process.start()
            self.processes.append(process)
            work_remote.close()  # 主进程中关闭工作端
        
        # 获取环境信息
        self.remotes[0].send(('get_spaces', None))
        observation_space, action_space = self.remotes[0].recv()
        self.observation_space = observation_space
        self.action_space = action_space
    
    @staticmethod
    def _worker(remote, parent_remote, env_fn):
        """
        子进程工作函数
        
        Args:
            remote: 子进程通信端
            parent_remote: 父进程通信端  
            env_fn: 环境创建函数
        """
        parent_remote.close()  # 子进程中关闭父进程端
        
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
                    # 执行动作
                    observation, reward, done, info = env.step(data)
                    if done:
                        # 如果回合结束，自动重置环境
                        observation = env.reset()
                    remote.send((observation, reward, done, info))
                
                elif cmd == 'reset':
                    # 重置环境
                    observation = env.reset()
                    remote.send(observation)
                
                elif cmd == 'close':
                    # 关闭环境
                    env.close()
                    remote.close()
                    break
                
                elif cmd == 'get_spaces':
                    # 获取空间信息
                    remote.send((env.observation_space, env.action_space))
                
                elif cmd == 'render':
                    # 渲染环境
                    img = env.render(mode=data)
                    remote.send(img)
                
                elif cmd == 'get_info':
                    # 获取环境信息
                    info = env.get_info() if hasattr(env, 'get_info') else {}
                    remote.send(info)
                
                elif cmd == 'set_world_weights':
                    # 更新多世界环境的采样权重
                    weights = data
                    if hasattr(env, 'set_world_weights'):
                        env.set_world_weights(weights)
                        remote.send('ok')
                    else:
                        remote.send('ignored')

                elif cmd == 'set_world':
                    # 重新配置到指定关卡（单世界环境）
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
        异步执行动作（非阻塞）
        
        Args:
            actions (list): 动作列表
        """
        if self.waiting:
            raise RuntimeError("Already waiting for step results")
        
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
        
        self.waiting = True

    def set_world_weights(self, weights):
        """
        设置所有子环境的关卡采样权重（仅多世界环境有效）
        
        Args:
            weights (dict|list): 权重配置
        """
        for remote in self.remotes:
            remote.send(('set_world_weights', weights))
        # 等待确认，避免队列堆积
        for remote in self.remotes:
            try:
                remote.recv()
            except Exception:
                pass

    def set_worlds(self, worlds):
        """
        为每个子环境设置固定关卡
        
        Args:
            worlds (list[str]): 长度等于 num_envs 的世界名称列表
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
        等待异步动作执行完成
        
        Returns:
            tuple: (观察, 奖励, 结束标志, 信息) 的批次
        """
        if not self.waiting:
            raise RuntimeError("Not waiting for step results")
        
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        
        # 检查错误
        for i, result in enumerate(results):
            if isinstance(result, tuple) and len(result) == 2 and result[0] == 'error':
                raise RuntimeError(f"Environment {i} error: {result[1]}")
        
        # 分离结果
        observations, rewards, dones, infos = zip(*results)
        
        return np.array(observations), np.array(rewards), np.array(dones), list(infos)
    
    def step(self, actions):
        """
        同步执行动作
        
        Args:
            actions (list): 动作列表
            
        Returns:
            tuple: (观察, 奖励, 结束标志, 信息) 的批次
        """
        self.step_async(actions)
        return self.step_wait()
    
    def reset(self):
        """
        重置所有环境
        
        Returns:
            np.array: 初始观察批次
        """
        for remote in self.remotes:
            remote.send(('reset', None))
        
        observations = [remote.recv() for remote in self.remotes]
        
        return np.array(observations)
    
    def close(self):
        """
        关闭所有环境和进程
        """
        if self.closed:
            return
        
        if self.waiting:
            # 等待当前操作完成
            for remote in self.remotes:
                remote.recv()
            self.waiting = False
        
        # 发送关闭命令
        for remote in self.remotes:
            remote.send(('close', None))
        
        # 等待进程结束
        for process in self.processes:
            process.join(timeout=5)
            if process.is_alive():
                process.terminate()  # 强制终止
        
        for remote in self.remotes:
            remote.close()
        
        self.closed = True
    
    def render(self, mode='human', env_id=0):
        """
        渲染指定环境
        
        Args:
            mode (str): 渲染模式
            env_id (int): 环境ID
            
        Returns:
            渲染结果
        """
        self.remotes[env_id].send(('render', mode))
        return self.remotes[env_id].recv()
    
    def get_env_info(self, env_id=0):
        """
        获取指定环境的信息
        
        Args:
            env_id (int): 环境ID
            
        Returns:
            dict: 环境信息
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
    虚拟向量化环境（单进程版本）
    
    所有环境在同一进程中运行，适合调试或者环境创建开销大的情况
    """
    
    def __init__(self, env_fns):
        """
        初始化虚拟向量化环境
        
        Args:
            env_fns (list): 环境创建函数列表
        """
        self.envs = [fn() for fn in env_fns]
        self.num_envs = len(self.envs)
        
        # 获取环境信息
        self.observation_space = self.envs[0].observation_space
        self.action_space = self.envs[0].action_space
        self.closed = False
    
    def step(self, actions):
        """
        执行动作
        
        Args:
            actions (list): 动作列表
            
        Returns:
            tuple: (观察, 奖励, 结束标志, 信息) 的批次
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
        重置所有环境
        
        Returns:
            np.array: 初始观察批次
        """
        observations = [env.reset() for env in self.envs]
        return np.array(observations)

    def set_world_weights(self, weights):
        """
        设置所有环境的关卡采样权重（仅多世界环境有效）
        """
        for env in self.envs:
            if hasattr(env, 'set_world_weights'):
                env.set_world_weights(weights)

    def set_worlds(self, worlds):
        """
        为每个环境设置固定关卡
        """
        if len(worlds) != self.num_envs:
            raise ValueError("worlds length must equal num_envs")
        for env, w in zip(self.envs, worlds):
            if hasattr(env, 'reconfigure_world'):
                env.reconfigure_world(w)
    
    def close(self):
        """
        关闭所有环境
        """
        if not self.closed:
            for env in self.envs:
                env.close()
            self.closed = True
    
    def render(self, mode='human', env_id=0):
        """
        渲染指定环境
        
        Args:
            mode (str): 渲染模式
            env_id (int): 环境ID
        """
        return self.envs[env_id].render(mode)
    
    def get_env_info(self, env_id=0):
        """
        获取指定环境的信息
        
        Args:
            env_id (int): 环境ID
            
        Returns:
            dict: 环境信息
        """
        return self.envs[env_id].get_info() if hasattr(self.envs[env_id], 'get_info') else {}
    
    def __len__(self):
        return self.num_envs
    
    def __del__(self):
        self.close()


class ParallelMarioEnvironments:
    """
    并行马里奥环境管理器
    
    封装了向量化环境的创建和管理，提供高级接口
    """
    
    def __init__(self, num_envs=16, worlds=None, use_subprocess=True, render_env_id=None):
        """
        初始化并行环境
        
        Args:
            num_envs (int): 环境数量
            worlds (list): 世界列表，None使用默认配置
            use_subprocess (bool): 是否使用子进程
            render_env_id (int): 需要渲染的环境ID，None表示不渲染
        """
        self.num_envs = num_envs
        self.use_subprocess = use_subprocess
        self.render_env_id = render_env_id
        
        if not MARIO_AVAILABLE:
            raise ImportError("Mario environment not available")
        
        # 默认世界配置
        if worlds is None:
            worlds = ['1-1', '1-2', '1-3', '1-4']
        
        # 创建环境函数列表
        env_fns = []
        use_dynamic = getattr(Config, 'DYNAMIC_WORLD_SAMPLING', False) and not getattr(Config, 'USE_DYNAMIC_WORLD_COUNTS', False) and len(worlds) > 1
        for i in range(num_envs):
            # 只有指定的环境才渲染
            render_mode = 'human' if i == render_env_id else None
            if use_dynamic:
                # 多世界环境：每回合按权重重采样关卡
                env_fn = lambda wlist=worlds, r=render_mode: create_mario_environment(
                    multi_world=True,
                    worlds=wlist,
                    render_mode=r
                )
            else:
                # 静态：固定到某个关卡
                world = worlds[i % len(worlds)]
                env_fn = lambda w=world, r=render_mode: create_mario_environment(
                    world=w,
                    render_mode=r
                )
            env_fns.append(env_fn)
        
        # 创建向量化环境
        if use_subprocess and num_envs > 1:
            self.vec_env = SubprocVecEnv(env_fns)
        else:
            self.vec_env = DummyVecEnv(env_fns)
        
        # 环境信息
        self.observation_space = self.vec_env.observation_space
        self.action_space = self.vec_env.action_space
        
        # 统计信息
        self.total_episodes = 0
        self.total_steps = 0
        self.episode_rewards = []
        self.episode_lengths = []
        
        print(f"创建了 {num_envs} 个并行马里奥环境")
        print(f"观察空间: {self.observation_space}")
        print(f"动作空间: {self.action_space}")
        if use_dynamic:
            print("已启用多关卡动态采样（按权重在reset时重采样关卡）")

        # 保存当前世界分配（用于后续动态调整）
        self.available_worlds = list(worlds)
        self.current_worlds = [worlds[i % len(worlds)] for i in range(num_envs)] if not use_dynamic else None
    
    def reset(self):
        """
        重置所有环境
        
        Returns:
            torch.Tensor: 初始状态批次，形状 (num_envs, *obs_shape)
        """
        observations = self.vec_env.reset()
        
        # 转换为张量
        return torch.FloatTensor(observations).to(Config.DEVICE)
    
    def step(self, actions):
        """
        并行执行动作
        
        Args:
            actions (torch.Tensor): 动作张量，形状 (num_envs,)
            
        Returns:
            tuple: (下一个状态, 奖励, 结束标志, 信息列表)
        """
        # 转换为numpy数组
        if isinstance(actions, torch.Tensor):
            actions = actions.cpu().numpy()
        
        # 执行动作
        observations, rewards, dones, infos = self.vec_env.step(actions)
        
        # 更新统计
        self.total_steps += self.num_envs
        
        for i, (reward, done, info) in enumerate(zip(rewards, dones, infos)):
            if done:
                self.total_episodes += 1
                # 记录回合统计（如果信息中有的话）
                if 'episode_reward' in info:
                    self.episode_rewards.append(info['episode_reward'])
                if 'episode_length' in info:
                    self.episode_lengths.append(info['episode_length'])
        
        # 转换为张量
        observations = torch.FloatTensor(observations).to(Config.DEVICE)
        rewards = torch.FloatTensor(rewards).to(Config.DEVICE)
        dones = torch.BoolTensor(dones).to(Config.DEVICE)
        
        return observations, rewards, dones, infos
    
    def render(self, env_id=None):
        """
        渲染指定环境
        
        Args:
            env_id (int): 环境ID，None使用默认渲染环境
        """
        if env_id is None:
            env_id = self.render_env_id
        
        if env_id is not None and env_id < self.num_envs:
            return self.vec_env.render(env_id=env_id)
    
    def set_world_weights(self, weights):
        """
        更新各关卡采样权重（仅在动态采样启用时生效）
        
        Args:
            weights (dict|list): {world: weight} 或按worlds顺序的列表
        """
        if hasattr(self.vec_env, 'set_world_weights'):
            self.vec_env.set_world_weights(weights)

    def set_world_allocation(self, weights_or_counts):
        """
        动态调整各关卡的子环境数量（每个子环境固定一个关卡）
        
        Args:
            weights_or_counts (dict|list):
                - dict {world: weight 或 count}
                - list 与 self.available_worlds 顺序一致的权重或数量
        """
        worlds = self.available_worlds
        n = self.num_envs
        # 解析输入为权重/数量数组
        if isinstance(weights_or_counts, dict):
            arr = np.array([float(weights_or_counts.get(w, 0.0)) for w in worlds], dtype=np.float64)
        else:
            arr = np.array(list(map(float, weights_or_counts)), dtype=np.float64)
            if arr.size != len(worlds):
                raise ValueError("weights_or_counts size must match number of available worlds")
        # 如果总和大于n，按比例缩放；如果小于等于1，按权重 * n 分配
        min_envs = int(getattr(Config, 'WORLD_MIN_ENVS_PER_WORLD', 1))
        if arr.sum() <= 1.0 + 1e-9:
            weights = arr / (arr.sum() + 1e-9)
            counts = np.floor(weights * n).astype(int)
        else:
            counts = arr.astype(int)
        # 至少 min_envs 个
        counts = np.maximum(counts, min_envs)
        # 调整总数为 n
        diff = counts.sum() - n
        if diff != 0:
            # 计算调整顺序：若需要减少，从平均奖励高（推测容易）的世界减；此处简单按当前 counts 大小调整
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
        # 展开为 per-env 世界列表
        new_assignment = []
        for w, c in zip(worlds, counts):
            new_assignment.extend([w] * c)
        # 若因四舍五入误差导致列表长度不等于 n，修正
        if len(new_assignment) > n:
            new_assignment = new_assignment[:n]
        elif len(new_assignment) < n:
            new_assignment.extend([worlds[0]] * (n - len(new_assignment)))
        # 下发到向量环境
        if hasattr(self.vec_env, 'set_worlds'):
            self.vec_env.set_worlds(new_assignment)
            self.current_worlds = list(new_assignment)
            print(f"已更新子环境关卡分配: {dict(zip(worlds, counts))}")
    
    def close(self):
        """
        关闭所有环境
        """
        self.vec_env.close()
    
    def get_statistics(self):
        """
        获取训练统计信息
        
        Returns:
            dict: 统计信息
        """
        stats = {
            'total_episodes': self.total_episodes,
            'total_steps': self.total_steps,
            'num_envs': self.num_envs,
        }
        
        if self.episode_rewards:
            stats.update({
                'avg_episode_reward': np.mean(self.episode_rewards[-100:]),  # 最近100回合
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
    创建并行马里奥环境的工厂函数
    
    Args:
        num_envs (int): 环境数量
        worlds (list): 世界列表
        use_subprocess (bool): 是否使用子进程
        render_env_id (int): 渲染环境ID
        
    Returns:
        ParallelMarioEnvironments: 并行环境管理器
    """
    return ParallelMarioEnvironments(
        num_envs=num_envs,
        worlds=worlds,
        use_subprocess=use_subprocess,
        render_env_id=render_env_id
    )


def test_parallel_environments():
    """
    测试并行环境功能
    """
    print("测试并行马里奥环境...")
    
    if not MARIO_AVAILABLE:
        print("Mario environment not available, skipping test")
        return
    
    try:
        # 创建少量环境进行测试
        envs = create_parallel_mario_envs(num_envs=2, use_subprocess=False)
        print(f"并行环境创建成功: {len(envs)} 个环境")
        
        # 重置环境
        states = envs.reset()
        print(f"重置成功，状态形状: {states.shape}")
        
        # 执行几步
        for i in range(3):
            actions = torch.randint(0, envs.action_space.n, (len(envs),))
            next_states, rewards, dones, infos = envs.step(actions)
            
            print(f"步骤 {i+1}:")
            print(f"  动作: {actions.tolist()}")
            print(f"  奖励: {rewards.tolist()}")
            print(f"  结束: {dones.tolist()}")
        
        # 获取统计信息
        stats = envs.get_statistics()
        print(f"统计信息: {stats}")
        
        # 关闭环境
        envs.close()
        print("并行环境测试完成")
        
    except Exception as e:
        print(f"并行环境测试失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_parallel_environments()
