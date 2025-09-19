"""
Configuration file for hyperparameters and system settings.
Centralized config simplifies tuning and adding new algorithms.
"""

import torch

class Config:
    # =============================================================================
    # Environment settings
    # =============================================================================
    
    # Gym environment
    ENV_NAME = 'SuperMarioBros-v0'              # environment id
    WORLD_STAGE = ['1-1', '1-2', '1-3', '1-4']  # worlds to train/evaluate on
    
    # Multi-world dynamic sampling controls
    DYNAMIC_WORLD_SAMPLING = False   # re-sample world with weights inside single env
    USE_DYNAMIC_WORLD_COUNTS = True  # allocate per-world env counts dynamically
    WORLD_SAMPLING_MIN_WEIGHT = 0.05 # minimum sampling weight per world
    WORLD_SAMPLING_ALPHA = 1.0       # amplification factor (sensitivity to difficulty)
    WORLD_SWITCH_PROB = 1.0          # switch probability on reset (only when DYNAMIC_WORLD_SAMPLING=True)
    WORLD_MIN_ENVS_PER_WORLD = 1     # min sub-envs per world to avoid forgetting

    # Number of parallel envs — impacts speed/diversity
    NUM_ENVS = 48
    
    # Preprocessing
    FRAME_SIZE = 84                 # resized image (84x84)
    FRAME_STACK = 4                 # number of stacked frames (temporal info)
    SKIP_FRAMES = 4                 # action repeat to reduce compute
    
    # =============================================================================
    # PPO hyperparameters
    # =============================================================================
    
    # Learning rate
    LEARNING_RATE = 3.0e-4          # Adam LR
    
    # Core PPO params
    PPO_EPOCHS = 10                 # epochs per update
    CLIP_EPSILON = 0.1              # clipping for policy/value
    VALUE_LOSS_COEFF = 0.5          # value loss coefficient
    ENTROPY_COEFF = 0.1            # entropy coefficient (exploration)
    
    # Discount and advantages
    GAMMA = 0.95                    # discount factor
    GAE_LAMBDA = 0.95               # GAE lambda
    
    # Batch settings
    STEPS_PER_UPDATE = 2048         # steps collected per update
    MINIBATCH_SIZE = 8192           # minibatch size for PPO update
    
    # Gradient clipping
    MAX_GRAD_NORM = 0.8             # clip threshold (stability)

    # Early-stop patience (episodes without improvement)
    PATIENCE = 1e10  
    
    # =============================================================================
    # Network architecture
    # =============================================================================
    
    # CNN feature extractor
    CNN_CHANNELS = [32, 64, 64]     # channels
    CNN_KERNELS = [8, 4, 3]         # kernel sizes
    CNN_STRIDES = [4, 2, 1]         # strides
    
    # Fully-connected
    HIDDEN_SIZE = 512               # hidden size
    
    # =============================================================================
    # Training settings
    # =============================================================================
    
    # Limits and intervals
    MAX_EPISODES = 10000000000      # max episodes
    MAX_STEPS = 10000000000000      # max steps
    SAVE_FREQ = 100                 # save every N updates
    LOG_FREQ = 10                   # log every N updates
    
    # Target reward for early success
    TARGET_REWARD = 3000            # Mario target reward
    
    # =============================================================================
    # System settings
    # =============================================================================
    
    # Device
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Seed
    SEED = 42
    
    # Paths
    MODEL_DIR = 'models'            # model directory
    LOG_DIR = 'logs'                # log directory
    
    # TensorBoard
    TENSORBOARD_LOG = True          # enable TensorBoard logging
    
    # =============================================================================
    # Test settings
    # =============================================================================
    
    # Rendering during test
    RENDER_MODE = 'human'           # render mode for tests
    TEST_EPISODES = 5               # default test episodes
    
    # =============================================================================
    # Debug options
    # =============================================================================
    
    # Verbose output
    VERBOSE = True                  # print more training info
    
    # Profiling
    PROFILE_TRAINING = False        # enable training profiling
    
    @staticmethod
    def print_config():
        """Print current configuration"""
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
