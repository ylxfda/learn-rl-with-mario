# DreamerV3 训练流程重构计划

## 目标

将世界模型和 Actor-Critic 的训练方式从**分离训练**改为**耦合训练**：

### 当前方式（分离训练）
```python
收集经验
for i in range(train_ratio):
    采样批次_i → 训练世界模型
for j in range(train_ratio):
    采样起始状态_j → 想象轨迹 → 训练actor-critic
```

### 目标方式（耦合训练）
```python
收集经验
for i in range(train_ratio):
    采样批次_i → 训练世界模型 → 获取posterior states → 训练actor-critic
```

## 具体改动

### 1. 分析 world_model.compute_loss 的返回值
**目的**：确认是否已经返回了 posterior states (h, z)

**检查内容**：
- [ ] `compute_loss` 返回值中是否包含 h 和 z 序列
- [ ] 返回的状态维度格式：期望 `(B, T, ...)`
- [ ] 是否需要修改 `compute_loss` 来返回这些状态

**文件位置**：`algorithms/dreamer_v3/models/world_model.py`

---

### 2. 分析 actor-critic 训练的输入格式
**目的**：了解如何将序列状态用于训练

**检查内容**：
- [ ] 当前 `train_actor_critic` 如何使用 `sample_starts`
- [ ] `imagine` 函数的输入格式要求
- [ ] 如何从序列的最后一个状态 `(B, ...)` 开始想象

**文件位置**：`algorithms/dreamer_v3/training/trainer.py:315-496`

---

### 3. 修改 train_world_model 函数
**目的**：让它返回训练时得到的 posterior states

**修改内容**：
```python
def train_world_model(self, num_updates: int = 1):
    # ... 现有代码 ...

    # 修改为返回最后一个 update 的 posterior states
    return {
        'h': h_seq.detach(),      # (B, T, hidden_size)
        'z': z_seq.detach(),      # (B, T, stoch_size, discrete_size)
        'observations': observations,  # 可选，用于调试
    }
```

**注意事项**：
- 必须 `.detach()` 防止梯度泄露
- 只返回最后一次更新的状态
- 如果 `num_updates > 1`，只保留最后一个批次的结果

**文件位置**：`algorithms/dreamer_v3/training/trainer.py:239-309`

---

### 4. 修改 train_actor_critic 函数
**目的**：使用 world model 提供的 posterior states 而不是重新采样

**修改内容**：
```python
def train_actor_critic(self, posterior_states: Dict = None):
    """
    Args:
        posterior_states: Dict containing:
            - 'h': (B, T, hidden_size)
            - 'z': (B, T, stoch_size, discrete_size)
            如果为 None，则回退到原有的 sample_starts 方式
    """
    if posterior_states is not None:
        # 使用序列的最后一个状态作为起始点
        h_0 = posterior_states['h'][:, -1]  # (B, hidden_size)
        z_0 = posterior_states['z'][:, -1]  # (B, stoch_size, discrete_size)
    else:
        # 原有逻辑：sample_starts
        ...
```

**注意事项**：
- 保持向后兼容：如果 `posterior_states=None`，使用原有逻辑
- 从序列最后一个时间步提取状态：`[:, -1]`
- 确保 batch size 一致

**文件位置**：`algorithms/dreamer_v3/training/trainer.py:315-496`

---

### 5. 修改主训练循环 train()
**目的**：将两个独立的训练循环合并为一个

**修改内容**：
```python
# 当前代码（第 522-526 行）
# Phase 2: Train world model
self.train_world_model(num_updates=train_ratio)

# Phase 3: Train actor-critic
self.train_actor_critic(num_updates=train_ratio)

# 修改为
# Phase 2 & 3: Joint training
for _ in range(train_ratio):
    posterior_states = self.train_world_model(num_updates=1)
    self.train_actor_critic(posterior_states=posterior_states)
```

**注意事项**：
- `train_world_model(num_updates=1)` 每次只更新一次
- 立即使用返回的 `posterior_states` 训练 actor-critic
- 循环 `train_ratio` 次

**文件位置**：`algorithms/dreamer_v3/training/trainer.py:522-526`

---

### 6. 测试修改后的代码
**目的**：验证修改的正确性

**测试内容**：
- [ ] 代码可以正常运行
- [ ] 维度匹配正确（打印关键张量的 shape）
- [ ] 训练 loss 正常下降
- [ ] 没有梯度泄露（world model 参数不受 actor-critic 影响）
- [ ] 性能测试：训练速度是否有变化

---

## 实现约束

### 简化约束
- **不考虑**从序列的多个时间步采样起始状态
- **只使用**世界模型训练得到的序列的**最后一个状态** `(h[:, -1], z[:, -1])`

### 关键原则
1. **梯度隔离**：posterior states 必须 detach
2. **向后兼容**：保留原有的 sample_starts 逻辑作为备选
3. **代码清晰**：添加注释说明修改的原因

---

## 预期效果

### 优点
1. **数据效率更高**：重用世界模型训练时编码的真实状态，不需要重新采样和编码
2. **训练更紧密**：actor-critic 在世界模型刚更新的状态上训练，可能更一致
3. **减少采样开销**：省去 `sample_starts` 的 replay buffer 采样和编码步骤

### 潜在风险
1. **多样性降低**：actor-critic 只在固定的批次状态上训练，可能缺乏多样性
2. **batch size 耦合**：world model 和 actor-critic 的 batch size 必须兼容

---

## 文件清单

需要修改的文件：
1. `algorithms/dreamer_v3/training/trainer.py` - 主要修改文件
2. 可能需要查看：`algorithms/dreamer_v3/models/world_model.py` - 确认 compute_loss 返回值

---

## 执行顺序

1. ✅ 创建此计划文档
2. ⏳ 分析 world_model.compute_loss 返回值
3. ⏳ 分析 actor-critic 输入格式
4. ⏳ 修改 train_world_model 函数
5. ⏳ 修改 train_actor_critic 函数
6. ⏳ 修改主训练循环
7. ⏳ 测试代码

---

**文档创建时间**: 2025-10-14
**修改人**: Claude Code
**目标**: 提高训练数据效率，增强世界模型与策略学习的耦合性
