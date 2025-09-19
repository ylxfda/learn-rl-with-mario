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
PPO (Proximal Policy Optimization) is an on-policy policy-gradient method that
stabilizes updates by clipping the importance ratio between the new and old
policies. It is typically combined with an Actor-Critic architecture and GAE.

Concepts at a glance:
- Policy gradient with importance ratio r(θ) = πθ(a|s) / πθ_old(a|s)
- Clipped surrogate: clip r(θ) into [1−ε, 1+ε] to avoid destructive updates
- Actor-Critic: policy (actor) and value function (critic)
- GAE: balances bias/variance when estimating advantages

High-level training loop:
1) Collect trajectories from parallel environments using the current policy
2) Compute advantages via GAE and returns
3) Optimize the clipped surrogate over multiple epochs and mini-batches
4) Update the critic (value function) and add entropy bonus for exploration

Key characteristics:
- Clipped objective bounds policy updates for stability
- Parallel environments improve sample throughput
- CNN-based actor-critic network processes pixel observations

Mathematical objective (intuition):
- Maximize min(r(θ)A, clip(r(θ), 1−ε, 1+ε)A) averaged over samples
- Add value loss (MSE, optionally clipped) and subtract entropy bonus

Code map (click to open):
- Parallel env setup: [train.py:126](../train.py#L126), [enviroments/parallel_envs.py:411](../enviroments/parallel_envs.py#L411), [enviroments/parallel_envs.py:659](../enviroments/parallel_envs.py#L659)
- Rollout collection: [train.py:243](../train.py#L243) (policy [algorithms/ppo.py:102](../algorithms/ppo.py#L102)), env step [enviroments/parallel_envs.py:497](../enviroments/parallel_envs.py#L497)
- Compute last values and GAE: last values [train.py:320](../train.py#L320), GAE/returns [utils/replay_buffer.py:142](../utils/replay_buffer.py#L142)
- PPO update loop: [algorithms/ppo.py:170](../algorithms/ppo.py#L170)
  - Ratio r(θ): [algorithms/ppo.py:214](../algorithms/ppo.py#L214)
  - Clipped objective: [algorithms/ppo.py:216](../algorithms/ppo.py#L216)
  - Policy loss: [algorithms/ppo.py:224](../algorithms/ppo.py#L224)
  - Value loss (clipped): [algorithms/ppo.py:227](../algorithms/ppo.py#L227)
  - Entropy bonus: [algorithms/ppo.py:242](../algorithms/ppo.py#L242)
  - Gradient clipping: [algorithms/ppo.py:255](../algorithms/ppo.py#L255)
- Actor-Critic forward/evaluate: [networks/networks.py:366](../networks/networks.py#L366), [networks/networks.py:399](../networks/networks.py#L399)
- Hyperparameters: [config.py:38](../config.py#L38), [config.py:41](../config.py#L41), [config.py:42](../config.py#L42), [config.py:47](../config.py#L47), [config.py:48](../config.py#L48), [config.py:55](../config.py#L55)
- Logging and saving: logger [utils/logger.py:27](../utils/logger.py#L27), model I/O [algorithms/base.py:310](../algorithms/base.py#L310)

Notes on implementation:
- Actor-Critic = CNN feature extractor + policy/value heads
  - Feature extractor: [networks/networks.py:16](../networks/networks.py#L16)
  - Policy head: [networks/networks.py:137](../networks/networks.py#L137)
  - Value head: [networks/networks.py:250](../networks/networks.py#L250)
- Action selection (sampling/argmax): [networks/networks.py:377](../networks/networks.py#L377)
- Evaluate log-probs/values for updates: [networks/networks.py:399](../networks/networks.py#L399)
- Trajectory buffer + advantages: [utils/replay_buffer.py:13](../utils/replay_buffer.py#L13), [utils/replay_buffer.py:142](../utils/replay_buffer.py#L142)

Tuning tips:
- Increase `PPO_EPOCHS` / adjust `MINIBATCH_SIZE` for update strength (mind stability)
- Tune `CLIP_EPSILON` for stability vs. learning speed
- Use `ENTROPY_COEFF` to prevent premature convergence
- `MAX_GRAD_NORM` helps with exploding gradients on pixel inputs

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
