# PPO Usage Guide (Super Mario)

## Prerequisites
Install the required dependencies (or use `requirements.txt`):
```bash
pip install gym-super-mario-bros==7.4.0
pip install torch torchvision
pip install numpy matplotlib opencv-python
```

## Training
Run PPO training with configurable flags:
```bash
python train.py \
  --num_envs 8 \
  --max_episodes 10000 \
  --save_freq 50 \
  --log_freq 10
```
Common flags:
- `--max_steps`: Stop after a maximum number of environment steps
- `--learning_rate`, `--ppo_epochs`, `--clip_epsilon`, `--steps_per_update`
- `--device`: `cuda` or `cpu`
- `--seed`: Random seed
- `--resume`: Path to resume from a saved checkpoint

Checkpoints are written to `models/`:
- Regular: `ppo_mario_update_{N}.pth`
- Best: `best_ppo_mario_update_{N}.pth`

## Testing
Evaluate a trained PPO model and render gameplay:
```bash
python test.py \
  --model_path models/best_ppo_mario_update_100.pth \
  --episodes 5 \
  --render
```
Useful flags:
- `--world 1-1` or `--worlds 1-1 1-2 ...` for multi-world testing
- `--deterministic` for greedy policy
- `--render_mode human|rgb_array` and `--render_delay` to control playback speed

## PPO Details
High-level training loop:
1. Collect trajectories from parallel environments using the current policy
2. Compute advantages via GAE and returns for each transition
3. Optimize the clipped surrogate objective over multiple epochs
4. Update the value function and add entropy regularization

Key characteristics:
- Clipped objective bounds the policy update for stability
- Parallel environment rollout for better sample throughput
- CNN-based actor-critic network for pixel observations

## Network Architecture
- 3 convolutional layers (32, 64, 64 filters)
- 2 fully connected layers (512 units)
- Separate heads for policy and value

## Default Hyperparameters
See `config.py` for the complete list. Current defaults:
- Learning rate: 3.0e-4
- Discount factor (gamma): 0.95
- GAE lambda: 0.95
- PPO clip range: 0.1
- PPO epochs per update: 10
- Steps per update: 2048
- Minibatch size: 8192
- Max grad norm: 0.8
- Entropy coefficient: 0.1
- Value loss coefficient: 0.5

## Notes
- Environment wrappers and parallelization live in `enviroments/`
- Hyperparameters and world configurations are centralized in `config.py`
