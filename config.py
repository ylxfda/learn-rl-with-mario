"""
配置文件 - 包含所有超参数和系统设置
这种集中式配置方便调试和后续添加其他算法
"""

import torch

class Config:
    # =============================================================================
    # 环境设置
    # =============================================================================
    
    # 游戏环境配置
    ENV_NAME = 'SuperMarioBros-v0'              # 环境名称
    WORLD_STAGE = ['1-1', '1-2', '1-3', '1-4']  # 继续训练的关卡列表

    # 并行环境数量 - 影响训练速度和样本多样性
    NUM_ENVS = 48
    
    # 图像预处理参数
    FRAME_SIZE = 84                 # 缩放后的图像尺寸 (84x84)
    FRAME_STACK = 4                 # 堆叠帧数，用于提供时序信息
    SKIP_FRAMES = 4                 # 动作重复次数，减少计算量
    
    # =============================================================================
    # PPO算法超参数
    # =============================================================================
    
    # 学习率
    LEARNING_RATE = 3.0e-4          # Adam优化器学习率
    
    # PPO核心参数
    PPO_EPOCHS = 10                 # 每次更新时的PPO迭代次数
    CLIP_EPSILON = 0.1              # PPO裁剪参数，控制策略更新幅度
    VALUE_LOSS_COEFF = 0.5          # 价值函数损失系数
    ENTROPY_COEFF = 0.1            # 熵奖励系数，鼓励探索
    
    # 折扣和优势计算
    GAMMA = 0.95                    # 折扣因子
    GAE_LAMBDA = 0.95               # GAE(Generalized Advantage Estimation)参数
    
    # 训练批次设置
    STEPS_PER_UPDATE = 2048         # 每次更新前收集的步数
    MINIBATCH_SIZE = 8192            # PPO更新时的小批次大小
    
    # 梯度裁剪
    MAX_GRAD_NORM = 0.8             # 梯度裁剪阈值，防止梯度爆炸

    # 若干回合没有改进就可以考虑停止
    PATIENCE = 1e10  
    
    # =============================================================================
    # 网络架构参数
    # =============================================================================
    
    # CNN特征提取器
    CNN_CHANNELS = [32, 64, 64]     # 卷积层通道数
    CNN_KERNELS = [8, 4, 3]         # 卷积核大小
    CNN_STRIDES = [4, 2, 1]         # 卷积步长
    
    # 全连接层
    HIDDEN_SIZE = 512               # 隐藏层大小
    
    # =============================================================================
    # 训练设置
    # =============================================================================
    
    # 训练轮次和保存
    MAX_EPISODES = 10000000000            # 最大训练回合数
    MAX_STEPS = 10000000000000            # 最大训练步数
    SAVE_FREQ = 100                 # 每多少次更新保存一次模型
    LOG_FREQ = 10                   # 每多少次更新记录一次日志
    
    # 早停条件 - 当平均奖励达到此值时认为训练成功
    TARGET_REWARD = 3000            # 马里奥游戏的目标奖励
    
    # =============================================================================
    # 系统设置
    # =============================================================================
    
    # 设备设置
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 随机种子
    SEED = 42
    
    # 文件路径
    MODEL_DIR = 'models'            # 模型保存目录
    LOG_DIR = 'logs'                # 日志保存目录
    
    # TensorBoard设置
    TENSORBOARD_LOG = True          # 是否使用TensorBoard记录
    
    # =============================================================================
    # 测试设置
    # =============================================================================
    
    # 测试时的渲染设置
    RENDER_MODE = 'human'           # 测试时的渲染模式
    TEST_EPISODES = 5               # 测试回合数
    
    # =============================================================================
    # 调试选项
    # =============================================================================
    
    # 详细输出
    VERBOSE = True                  # 是否显示详细训练信息
    
    # 性能监控
    PROFILE_TRAINING = False        # 是否进行性能分析
    
    @staticmethod
    def print_config():
        """打印当前配置"""
        print("=" * 60)
        print("PPO Mario Training Configuration")
        print("=" * 60)
        print(f"Device: {Config.DEVICE}")
        print(f"Environment: {Config.ENV_NAME}")
        print(f"Parallel Envs: {Config.NUM_ENVS}")
        print(f"Frame Stack: {Config.FRAME_STACK}")
        print(f"Learning Rate: {Config.LEARNING_RATE}")
        print(f"PPO Epochs: {Config.PPO_EPOCHS}")
        print(f"Clip Epsilon: {Config.CLIP_EPSILON}")
        print(f"Steps per Update: {Config.STEPS_PER_UPDATE}")
        print(f"Target Reward: {Config.TARGET_REWARD}")
        print("=" * 60)
