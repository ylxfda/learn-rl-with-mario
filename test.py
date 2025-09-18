"""
PPO马里奥测试脚本

加载训练好的PPO模型并进行游戏测试，实时显示游戏画面
支持多种测试模式：单回合、多回合、不同关卡等

使用方法:
python test.py --model_path models/best_ppo_mario_model.pth --episodes 5
"""

import os
import time
import argparse
import numpy as np
import torch
from tqdm import tqdm

# 导入我们的模块
from config import Config
from enviroments.mario_env import create_mario_environment, MultiWorldMarioEnvironment
from algorithms.ppo import create_ppo_algorithm
from algorithms.base import ModelManager


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='Test trained PPO Mario model')
    
    # 模型参数
    parser.add_argument('--model_path', type=str, required=True,
                       help='训练好的模型文件路径')
    parser.add_argument('--load_best', action='store_true',
                       help='是否加载最佳模型')
    
    # 测试参数
    parser.add_argument('--episodes', type=int, default=5,
                       help='测试回合数')
    parser.add_argument('--world', type=str, default='1-1',
                       help='测试关卡 (如 1-1, 1-2 等)')
    parser.add_argument('--worlds', nargs='+', default=None,
                       help='多关卡测试')
    parser.add_argument('--deterministic', action='store_true',
                       help='是否使用确定性策略')
    parser.add_argument('--render', action='store_true', default=True,
                       help='是否渲染游戏画面')
    parser.add_argument('--render_mode', type=str, default='human',
                       choices=['human', 'rgb_array'],
                       help='渲染模式')
    
    # 分析参数
    parser.add_argument('--save_video', action='store_true',
                       help='是否保存游戏视频')
    parser.add_argument('--analyze_actions', action='store_true',
                       help='是否分析动作分布')
    parser.add_argument('--show_values', action='store_true',
                       help='是否显示状态价值')
    
    # 系统参数
    parser.add_argument('--device', type=str, default=None,
                       help='计算设备 (cuda/cpu)')
    parser.add_argument('--max_steps', type=int, default=10000,
                       help='每回合最大步数（防止卡死）')
    
    return parser.parse_args()


class PPOTester:
    """PPO测试器类"""
    
    def __init__(self, args):
        """
        初始化测试器
        
        Args:
            args: 命令行参数
        """
        self.args = args
        self.device = torch.device(args.device) if args.device else Config.DEVICE
        
        print("PPO马里奥测试器初始化...")
        print("=" * 50)
        print(f"模型路径: {args.model_path}")
        print(f"设备: {self.device}")
        print(f"测试回合: {args.episodes}")
        print(f"确定性策略: {args.deterministic}")
        print("=" * 50)
        
        # 加载模型
        self._load_model()
        
        # 创建环境
        self._create_environment()
        
        # 测试统计
        self.test_stats = []
        
    def _load_model(self):
        """加载训练好的模型"""
        print("加载PPO模型...")
        
        if not os.path.exists(self.args.model_path):
            raise FileNotFoundError(f"模型文件不存在: {self.args.model_path}")
        
        # 首先加载检查点获取模型信息
        checkpoint = torch.load(self.args.model_path, map_location=self.device)
        
        # 从检查点中获取观察空间和动作空间信息
        obs_shape = checkpoint.get('obs_shape', (Config.FRAME_STACK, Config.FRAME_SIZE, Config.FRAME_SIZE))
        action_dim = checkpoint.get('action_dim', 7)
        
        # 模拟空间对象
        class MockObsSpace:
            def __init__(self, shape):
                self.shape = shape
        
        class MockActionSpace:
            def __init__(self, n):
                self.n = n
        
        obs_space = MockObsSpace(obs_shape)
        action_space = MockActionSpace(action_dim)
        
        # 创建PPO算法
        self.ppo = create_ppo_algorithm(
            observation_space=obs_space,
            action_space=action_space,
            device=self.device
        )
        
        # 加载模型权重
        model_manager = ModelManager()
        try:
            model_manager.load_model(
                self.ppo, 
                filename=os.path.basename(self.args.model_path), 
                load_best=self.args.load_best
            )
        except:
            # 如果ModelManager加载失败，直接加载
            self.ppo.load_model(self.args.model_path)
        
        # 设置为评估模式
        self.ppo.eval()
        
        print("模型加载完成！")
        
        # 打印模型信息
        if 'total_steps' in checkpoint:
            print(f"训练步数: {checkpoint['total_steps']:,}")
        if 'total_episodes' in checkpoint:
            print(f"训练回合: {checkpoint['total_episodes']:,}")
        if 'best_reward' in checkpoint:
            print(f"最佳奖励: {checkpoint['best_reward']:.2f}")
    
    def _create_environment(self):
        """创建测试环境"""
        print("创建测试环境...")
        
        render_mode = self.args.render_mode if self.args.render else None
        
        if self.args.worlds:
            # 多关卡测试
            self.env = MultiWorldMarioEnvironment(
                worlds=self.args.worlds,
                render_mode=render_mode,
                random_start=True
            )
            print(f"多关卡环境: {self.args.worlds}")
        else:
            # 单关卡测试
            self.env = create_mario_environment(
                world=self.args.world,
                render_mode=render_mode
            )
            print(f"测试关卡: {self.args.world}")
        
        print("环境创建完成！")
    
    def test_episode(self, episode_num):
        """
        测试单个回合
        
        Args:
            episode_num (int): 回合编号
            
        Returns:
            dict: 回合统计信息
        """
        print(f"\n开始测试回合 {episode_num + 1}")
        
        # 重置环境
        observation = self.env.reset()
        observation = torch.FloatTensor(observation).unsqueeze(0).to(self.device)
        
        # 回合统计
        episode_reward = 0
        episode_length = 0
        actions_taken = []
        rewards_received = []
        values_estimated = []
        action_probs_history = []
        
        done = False
        start_time = time.time()
        
        while not done and episode_length < self.args.max_steps:
            # 获取动作
            with torch.no_grad():
                actions, extra_info = self.ppo.act(
                    observation, 
                    deterministic=self.args.deterministic
                )
                
                # 如果需要分析，记录额外信息
                if self.args.analyze_actions or self.args.show_values:
                    action_probs = self.ppo.get_action_probabilities(observation)
                    values = extra_info.get('values', torch.tensor([0.0]))
                    
                    action_probs_history.append(action_probs.cpu().numpy()[0])
                    values_estimated.append(values.cpu().numpy()[0] if hasattr(values, 'cpu') else values)
            
            action = actions.item()
            actions_taken.append(action)
            
            # 执行动作
            next_observation, reward, done, info = self.env.step(action)
            
            # 渲染
            if self.args.render:
                self.env.render()
                time.sleep(0.02)  # 稍微减慢游戏速度以便观察
            
            # 更新统计
            episode_reward += reward
            episode_length += 1
            rewards_received.append(reward)
            
            # 显示实时信息
            if episode_length % 100 == 0:
                print(f"  步数: {episode_length}, 奖励: {episode_reward:.2f}, 动作: {action}")
            
            # 准备下一步
            observation = torch.FloatTensor(next_observation).unsqueeze(0).to(self.device)
        
        # 回合结束统计
        episode_time = time.time() - start_time
        
        episode_stats = {
            'episode': episode_num + 1,
            'reward': episode_reward,
            'length': episode_length,
            'time': episode_time,
            'fps': episode_length / episode_time if episode_time > 0 else 0,
            'success': info.get('flag_get', False) if 'flag_get' in info else False,
            'x_pos': info.get('x_pos', 0) if 'x_pos' in info else 0,
            'world': getattr(self.env, 'world', self.args.world),
            'actions': actions_taken,
            'rewards': rewards_received,
        }
        
        # 添加分析数据
        if self.args.analyze_actions and action_probs_history:
            episode_stats['action_distribution'] = np.mean(action_probs_history, axis=0)
            episode_stats['action_entropy'] = np.mean([
                -np.sum(probs * np.log(probs + 1e-8)) for probs in action_probs_history
            ])
        
        if self.args.show_values and values_estimated:
            episode_stats['avg_value'] = np.mean(values_estimated)
            episode_stats['max_value'] = np.max(values_estimated)
            episode_stats['min_value'] = np.min(values_estimated)
        
        # 打印回合结果
        print(f"回合 {episode_num + 1} 完成:")
        print(f"  奖励: {episode_reward:.2f}")
        print(f"  步数: {episode_length}")
        print(f"  用时: {episode_time:.2f}s")
        print(f"  成功: {'是' if episode_stats['success'] else '否'}")
        print(f"  最终位置: {episode_stats['x_pos']:.1f}")
        
        return episode_stats
    
    def analyze_results(self):
        """分析测试结果"""
        if not self.test_stats:
            return
        
        print(f"\n{'='*60}")
        print("测试结果分析")
        print(f"{'='*60}")
        
        # 基础统计
        rewards = [stat['reward'] for stat in self.test_stats]
        lengths = [stat['length'] for stat in self.test_stats]
        times = [stat['time'] for stat in self.test_stats]
        successes = [stat['success'] for stat in self.test_stats]
        x_positions = [stat['x_pos'] for stat in self.test_stats]
        
        print(f"总回合数: {len(self.test_stats)}")
        print(f"平均奖励: {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
        print(f"最高奖励: {np.max(rewards):.2f}")
        print(f"最低奖励: {np.min(rewards):.2f}")
        print(f"平均步数: {np.mean(lengths):.1f} ± {np.std(lengths):.1f}")
        print(f"成功率: {np.mean(successes)*100:.1f}% ({sum(successes)}/{len(successes)})")
        print(f"平均最终位置: {np.mean(x_positions):.1f}")
        print(f"平均FPS: {np.mean([s['fps'] for s in self.test_stats]):.1f}")
        
        # 动作分析
        if self.args.analyze_actions:
            print(f"\n动作分析:")
            
            # 合并所有动作
            all_actions = []
            for stat in self.test_stats:
                all_actions.extend(stat['actions'])
            
            # 动作分布
            action_meanings = [
                'NOOP', 'right', 'right_A', 'right_B', 'right_A_B', 'A', 'left'
            ]
            
            action_counts = np.bincount(all_actions, minlength=7)
            action_ratios = action_counts / len(all_actions)
            
            for i, (meaning, ratio) in enumerate(zip(action_meanings, action_ratios)):
                print(f"  {meaning}: {ratio*100:.1f}% ({action_counts[i]}次)")
            
            # 平均动作熵
            avg_entropies = [stat.get('action_entropy', 0) for stat in self.test_stats]
            if any(avg_entropies):
                print(f"平均动作熵: {np.mean(avg_entropies):.3f}")
        
        # 价值函数分析
        if self.args.show_values:
            print(f"\n价值函数分析:")
            avg_values = [stat.get('avg_value', 0) for stat in self.test_stats]
            max_values = [stat.get('max_value', 0) for stat in self.test_stats]
            min_values = [stat.get('min_value', 0) for stat in self.test_stats]
            
            print(f"平均状态价值: {np.mean(avg_values):.2f} ± {np.std(avg_values):.2f}")
            print(f"最高状态价值: {np.mean(max_values):.2f}")
            print(f"最低状态价值: {np.mean(min_values):.2f}")
        
        # 关卡分析（如果是多关卡测试）
        if hasattr(self.env, 'worlds'):
            print(f"\n关卡分析:")
            world_stats = {}
            for stat in self.test_stats:
                world = stat['world']
                if world not in world_stats:
                    world_stats[world] = {'rewards': [], 'successes': []}
                world_stats[world]['rewards'].append(stat['reward'])
                world_stats[world]['successes'].append(stat['success'])
            
            for world, data in world_stats.items():
                avg_reward = np.mean(data['rewards'])
                success_rate = np.mean(data['successes']) * 100
                print(f"  {world}: 平均奖励={avg_reward:.2f}, 成功率={success_rate:.1f}%")
    
    def test(self):
        """执行完整测试"""
        print(f"\n开始测试 {self.args.episodes} 个回合...")
        
        try:
            # 执行测试回合
            for episode in tqdm(range(self.args.episodes), desc="测试进度"):
                episode_stats = self.test_episode(episode)
                self.test_stats.append(episode_stats)
            
            # 分析结果
            self.analyze_results()
            
            # 保存结果（可选）
            if self.args.save_video or len(self.args.model_path.split('/')) > 1:
                results_dir = os.path.dirname(self.args.model_path)
                results_file = os.path.join(results_dir, 'test_results.txt')
                
                with open(results_file, 'w', encoding='utf-8') as f:
                    f.write("PPO马里奥测试结果\n")
                    f.write("="*50 + "\n")
                    f.write(f"模型: {self.args.model_path}\n")
                    f.write(f"回合数: {self.args.episodes}\n")
                    f.write(f"确定性: {self.args.deterministic}\n")
                    f.write(f"测试关卡: {self.args.world if not self.args.worlds else self.args.worlds}\n")
                    f.write("\n详细结果:\n")
                    
                    for i, stat in enumerate(self.test_stats):
                        f.write(f"回合{i+1}: 奖励={stat['reward']:.2f}, "
                               f"步数={stat['length']}, 成功={stat['success']}\n")
                
                print(f"\n结果已保存到: {results_file}")
        
        except KeyboardInterrupt:
            print(f"\n测试被中断，已完成 {len(self.test_stats)} 个回合")
            if self.test_stats:
                self.analyze_results()
        
        except Exception as e:
            print(f"\n测试过程中出现错误: {e}")
            import traceback
            traceback.print_exc()
        
        finally:
            # 清理资源
            if hasattr(self, 'env'):
                self.env.close()
            
            print(f"\n测试结束。感谢使用PPO马里奥测试器！")

def main():
    """主函数"""
    # 解析命令行参数
    args = parse_args()
    
    print("PPO马里奥")
    # print("=" * 60)
    # print(f"设备: {torch.device(args.device) if args.device else Config.DEVICE}")
    # # print(f"并行环境数: {args.num_envs}")
    # print(f"训练关卡: {args.worlds}")
    # print(f"最大回合数: {args.max_episodes:,}")
    # print(f"最大步数: {args.max_steps:,}")
    # print("=" * 60)
    
    # 创建训练器
    tester = PPOTester(args)
    
    # 开始训练
    tester.test()


if __name__ == "__main__":
    main()