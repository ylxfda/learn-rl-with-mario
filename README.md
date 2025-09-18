# Super Mario Bros PPO Training

## Overview
This project implements Proximal Policy Optimization (PPO) algorithm from scratch to train an agent to play Super Mario Bros. The implementation is modular and extensible for other RL algorithms.

## Requirements
```bash
pip install gym-super-mario-bros==7.4.0
pip install torch torchvision
pip install numpy matplotlib opencv-python
```

## Project Structure
- `environments/`: Environment wrappers and parallel environment manager
- `algorithms/`: PPO and base algorithm implementations
- `models/`: Neural network architectures
- `utils/`: Utilities for logging and data storage
- `train.py`: Main training script
- `test.py`: Testing and visualization script
- `config.py`: Hyperparameters and settings

## Usage

### Training
```bash
python train.py --num_envs 8 --episodes 10000
```

### Testing
```bash
python test.py --model_path checkpoints/best_model.pth
```

## Algorithm Details

### PPO (Proximal Policy Optimization)
PPO is an on-policy algorithm that:
1. Collects trajectories using current policy
2. Computes advantages using GAE (Generalized Advantage Estimation)
3. Updates policy multiple epochs with clipped objective
4. Updates value function to minimize MSE

Key features:
- Clipped surrogate objective prevents large policy updates
- Parallel environment collection for sample efficiency
- CNN-based actor-critic network for pixel observations

## Network Architecture
- 3 Convolutional layers (32, 64, 64 filters)
- 2 Fully connected layers (512 units)
- Separate heads for policy and value

## Hyperparameters
See `config.py` for all hyperparameters. Key ones:
- Learning rate: 2.5e-4
- Discount factor (γ): 0.99
- GAE λ: 0.95
- PPO clip range: 0.2
- Epochs per update: 4
- Minibatch size: 256