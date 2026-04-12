# Dual-GPU CEM-GD Pipeline Implementation Plan

**日期**: 2026-03-30  
**目标**: 分离梯度下降和渲染到两个GPU，解决显存不足问题

---

## TL;DR

> **核心思路**: 将CEM-GD的梯度下降（占显存但计算轻）和4DGS渲染（计算重）分离到两个GPU
> 
> **设备分工**:
> - GPU 0 (cuda:2): 梯度下降 - 存储action_sequences、optimizer state、梯度
> - GPU 1 (cuda:3): 4DGS渲染 - 存储GaussianModel、deformation network、中间激活
> 
> **数据流**: action_sequences (cuda:2) → 传输到cuda:3渲染 → rewards返回cuda:2 → backward

---

## 问题分析

### 当前瓶颈

```
GPU内存分配 (单GPU - cuda:3):
├── GaussianModel (~2-4 GB)
├── Deformation Network (~1-2 GB)
├── GMFlow Network (~1 GB)
├── Adam优化器状态 (2× num_grad_seqs × action_size) (~500 MB)
├── 梯度计算图 (num_grad_seqs × 中间激活) (~3-5 GB) ← 主要瓶颈
└── 渲染中间结果 (batch × H × W × channels) (~2-3 GB)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
总计: 10-15 GB (超过单卡容量)
```

**关键发现**:
- 梯度下降阶段：计算轻（Adam step很快），但需要保留大量中间激活用于backward
- 渲染阶段：计算重（Gaussian rasterization），但forward-only时显存需求低

### 为什么会OOM？

1. **梯度图保留**: `num_grad_seqs=5` 个序列同时保持计算图 → 5倍显存
2. **渲染叠加**: 在已有梯度图的情况下，再进行batch渲染 → 显存溢出
3. **retain_graph=True**: 保留整个计算图直到最后一个序列backward完成

---

## 解决方案：Dual-GPU Pipeline

### 架构设计

```
Phase 1: CEM采样 (cuda:3 - 渲染GPU)
  → 所有采样在渲染GPU上完成（已有模型）

Phase 2: 选择Top-K (CPU)
  → 轻量级numpy操作

Phase 3: 梯度下降 (双GPU流水线)
  ┌─────────────────────────────────────────────────┐
  │ GPU 0 (cuda:2) - Gradient Device                │
  │ ├── action_sequences (requires_grad=True)       │
  │ ├── Adam optimizer state                        │
  │ └── Gradients storage                           │
  └─────────────────────────────────────────────────┘
                    ↓ transfer actions
  ┌─────────────────────────────────────────────────┐
  │ GPU 1 (cuda:3) - Render Device                  │
  │ ├── GaussianModel                               │
  │ ├── Deformation Network                         │
  │ ├── GMFlow Network                              │
  │ └── Render forward (no grad)                    │
  └─────────────────────────────────────────────────┘
                    ↓ transfer rewards
  ┌─────────────────────────────────────────────────┐
  │ GPU 0 (cuda:2) - Gradient Device                │
  │ └── backward() to compute gradients             │
  └─────────────────────────────────────────────────┘
```

### 关键创新

1. **模型保持在cuda:3**: 避免复制大模型（2-4GB）
2. **Actions在cuda:2**: 小数据（~50 MB），传输开销低
3. **分离forward和backward**: 渲染在cuda:3（无梯度），backward在cuda:2
4. **异步传输**: 使用CUDA streams overlap传输和计算

---

## 实现方案

### 文件修改清单

```
mpc/cem_gd.py                              (主要修改)
├── CEMGDOptimizer.__init__()              添加gradient_device参数
├── gradient_optimization()                 实现双GPU流水线
└── _score_on_render_device()              新方法：跨设备score

test/integration/test_cotracker_mpc.py     (CLI集成)
├── 添加 --gradient_device 参数
└── 传递给CEMGDOptimizer

run_cotracker_test.sh                       (Shell脚本)
└── 添加 GRADIENT_DEVICE 环境变量
```

### 详细修改

#### 1. `mpc/cem_gd.py` - CEMGDOptimizer.__init__()

**位置**: 约第215-257行

```python
class CEMGDOptimizer(CEMOptimizer):
    def __init__(
        self,
        model,
        objective,
        a_dim,
        horizon,
        num_samples_init,
        num_samples_replan,
        elites_frac,
        opt_iters,
        num_grad_opt_seqs,
        start_lr,
        log_every,
        init_std,
        init_mean,
        alpha,
        verbose,
        round_gripper_action,
        factor_shrink=0.5,
        max_tries=3,
        max_iterations=15,
        gradient_device=None,  # 新增参数
    ):
        # 父类初始化
        super().__init__(...)
        
        # 设置梯度设备
        if gradient_device is None:
            self.gradient_device = self.model.device  # 默认与模型同设备
        else:
            self.gradient_device = torch.device(gradient_device)
        
        self.render_device = self.model.device  # 渲染始终在模型设备
        
        print(f"[CEM-GD Dual-GPU] Gradient device: {self.gradient_device}")
        print(f"[CEM-GD Dual-GPU] Render device: {self.render_device}")
```

#### 2. `mpc/cem_gd.py` - gradient_optimization()

**位置**: 约第258-387行（整个方法需要重写）

```python
def gradient_optimization(
    self, action_sequences_list, action_history, obs_history, state_history, goal
):
    # Memory profiling
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(self.gradient_device)
        torch.cuda.reset_peak_memory_stats(self.render_device)
        mem_grad_before = torch.cuda.memory_allocated(self.gradient_device) / 1024**3
        mem_render_before = torch.cuda.memory_allocated(self.render_device) / 1024**3

    # Step 1: Move action sequences to gradient device
    action_sequences_list = [
        seq.detach().clone().to(self.gradient_device).requires_grad_(True)
        for seq in action_sequences_list
    ]
    
    n = len(action_sequences_list)

    # Step 2: Create optimizer on gradient device
    optimizer = Adam(
        [
            {
                "params": act_seq,
                "factor": 1,
                "action_bounds": (
                    torch.tensor(-1, device=self.gradient_device),
                    torch.tensor(1, device=self.gradient_device)
                ),
            }
            for act_seq in action_sequences_list
        ],
        lr=self.start_lr,
    )

    done = np.array([False for _ in range(n)])

    # Step 3: Initial forward pass (cross-device)
    with torch.set_grad_enabled(True):
        objective_all = self._score_on_render_device(
            action_sequences_list,
            action_history,
            obs_history,
            state_history,
            goal,
        )

    current_objective = [obj.item() for obj in objective_all]

    # Step 4: Save initial states (on CPU per previous optimization)
    saved_parameters = [None] * n
    saved_opt_states = [None] * n
    for i in range(n):
        action_sequences = action_sequences_list[i]
        saved_parameters[i] = action_sequences.detach().cpu().clone()
        opt_state = optimizer.state.get(action_sequences, {})
        saved_opt_states[i] = {
            'step': opt_state.get('step', 0),
            'exp_avg': opt_state['exp_avg'].cpu().clone() if 'exp_avg' in opt_state else None,
            'exp_avg_sq': opt_state['exp_avg_sq'].cpu().clone() if 'exp_avg_sq' in opt_state else None,
        }
        objective_all[i].backward(retain_graph=(i != n - 1))

    # Step 5: Gradient descent loop
    iteration = 0
    while not np.all(done) and iteration < self.max_iterations:
        iteration += 1
        optimizer.step()

        # Forward pass on render device
        with torch.set_grad_enabled(True):
            objective_all = self._score_on_render_device(
                action_sequences_list,
                action_history,
                obs_history,
                state_history,
                goal,
            )

        backwards_pass = []
        for i in range(n):
            if done[i]:
                continue
            
            action_sequences = action_sequences_list[i]
            if objective_all[i].item() > current_objective[i]:
                # Restore from CPU-saved state
                action_sequences.data = saved_parameters[i].to(self.gradient_device).data.clone()
                saved_state = saved_opt_states[i]
                optimizer.state[action_sequences] = {
                    'step': saved_state['step'],
                    'exp_avg': saved_state['exp_avg'].to(self.gradient_device) if saved_state['exp_avg'] is not None else torch.zeros_like(action_sequences),
                    'exp_avg_sq': saved_state['exp_avg_sq'].to(self.gradient_device) if saved_state['exp_avg_sq'] is not None else torch.zeros_like(action_sequences),
                }
                optimizer.param_groups[i]["factor"] *= self.factor_shrink

                if optimizer.param_groups[i]["factor"] > self.factor_shrink**self.max_tries:
                    action_sequences.grad = None
                    done[i] = True
            else:
                # Save successful step to CPU
                saved_parameters[i] = action_sequences.detach().cpu().clone()
                opt_state = optimizer.state.get(action_sequences, {})
                saved_opt_states[i] = {
                    'step': opt_state.get('step', 0),
                    'exp_avg': opt_state['exp_avg'].cpu().clone() if 'exp_avg' in opt_state else None,
                    'exp_avg_sq': opt_state['exp_avg_sq'].cpu().clone() if 'exp_avg_sq' in opt_state else None,
                }
                current_objective[i] = objective_all[i].item()
                optimizer.param_groups[i]["factor"] = 1
                action_sequences.grad = None
                backwards_pass.append(i)

        # Backward pass on gradient device
        if len(backwards_pass) > 0:
            to_compute = [objective_all[i] for i in backwards_pass]
            grads = [
                torch.ones_like(objective_all[i])
                for i in backwards_pass
            ]
            torch.autograd.backward(to_compute, grads)

    # Memory profiling
    if torch.cuda.is_available():
        mem_grad_after = torch.cuda.memory_allocated(self.gradient_device) / 1024**3
        mem_render_after = torch.cuda.memory_allocated(self.render_device) / 1024**3
        mem_grad_peak = torch.cuda.max_memory_allocated(self.gradient_device) / 1024**3
        mem_render_peak = torch.cuda.max_memory_allocated(self.render_device) / 1024**3
        print(f"[CEM-GD Memory] Gradient GPU: {mem_grad_before:.2f}GB → {mem_grad_after:.2f}GB (Peak: {mem_grad_peak:.2f}GB)")
        print(f"[CEM-GD Memory] Render GPU: {mem_render_before:.2f}GB → {mem_render_after:.2f}GB (Peak: {mem_render_peak:.2f}GB)")

    return [traj.detach().cpu() for traj in action_sequences_list]
```

#### 3. `mpc/cem_gd.py` - _score_on_render_device() (新方法)

**位置**: 插入到gradient_optimization()之后

```python
def _score_on_render_device(
    self,
    action_sequences_list,
    action_history,
    obs_history,
    state_history,
    goal,
):
    """
    在渲染设备上计算rewards，同时保持梯度流。
    
    关键技术：
    1. 将actions从gradient_device传输到render_device
    2. 在render_device上forward（启用梯度）
    3. 返回的rewards保持在gradient_device上（自动传输）
    """
    # Step 1: Stack actions on gradient device
    action_sequences_batch = torch.stack(action_sequences_list)  # [n, horizon, a_dim]
    
    # Step 2: Transfer to render device (梯度会自动跟踪)
    action_sequences_render = action_sequences_batch.to(self.render_device)
    
    # Step 3: Score trajectories on render device
    # 注意：model已经在render_device上，无需移动
    _, rewards_all, _ = self.score_trajectories(
        action_sequences_render.float(),
        obs_history,
        state_history,
        action_history,
        goal,
        grad_enabled=True,  # 启用梯度
    )
    
    # Step 4: Convert rewards to objectives (在render_device上)
    objective_all = [-reward for reward in rewards_all]
    
    # Step 5: Transfer objectives back to gradient device
    # PyTorch autograd会自动处理跨设备的梯度传播
    objective_all = [obj.to(self.gradient_device) for obj in objective_all]
    
    return objective_all
```

**关键技术说明**:

PyTorch的autograd系统原生支持跨设备梯度传播：
```python
# 示例：
x = torch.randn(10, requires_grad=True, device='cuda:2')
y = x.to('cuda:3')  # 跨设备传输，梯度图保留
z = y * 2           # 在cuda:3计算
loss = z.sum().to('cuda:2')  # 结果传回cuda:2
loss.backward()     # 梯度自动路由回cuda:2的x
```

#### 4. `test/integration/test_cotracker_mpc.py` - CLI参数

**位置**: 约第317-328行

```python
# CEM-GD parameters
parser.add_argument("--optimizer", type=str, default="cem-gd", choices=["cem", "cem-gd"],
                    help="Optimizer type: 'cem' (pure sampling) or 'cem-gd' (hybrid sampling + gradient)")
parser.add_argument("--num_samples_init", type=int, default=200,
                    help="CEM-GD: Number of samples for initial planning (default: 200)")
parser.add_argument("--num_samples_replan", type=int, default=20,
                    help="CEM-GD: Number of samples for replanning steps (default: 20, matches paper)")
parser.add_argument("--num_grad_seqs", type=int, default=5,
                    help="CEM-GD: Number of top sequences to refine with gradient descent (default: 5)")
parser.add_argument("--grad_lr", type=float, default=0.01,
                    help="CEM-GD: Gradient descent learning rate (default: 0.01)")
parser.add_argument("--grad_steps", type=int, default=15,
                    help="CEM-GD: Number of gradient descent iterations (default: 15)")
parser.add_argument("--gradient_device", type=str, default=None,
                    help="CEM-GD: Device for gradient descent (e.g., 'cuda:2'). If None, uses same device as model.")
```

**位置**: 约第631-678行（optimizer实例化）

```python
if args.optimizer == 'cem-gd':
    optimizer = CEMGDOptimizer(
        model=dynamics_model,
        objective=objective,
        a_dim=action_dim,
        horizon=args.horizon,
        num_samples_init=args.num_samples_init,
        num_samples_replan=args.num_samples_replan,
        elites_frac=0.1,
        opt_iters=args.opt_iters,
        num_grad_opt_seqs=args.num_grad_seqs,
        start_lr=args.grad_lr,
        log_every=1,
        init_std=None,
        init_mean=None,
        alpha=0.1,
        verbose=False,
        round_gripper_action=False,
        max_iterations=args.grad_steps,
        gradient_device=args.gradient_device,  # 新增
    )
```

#### 5. `run_cotracker_test.sh` - 环境变量

**位置**: 约第52-58行

```bash
# CEM-GD 参数 (用于 --optimizer cem-gd)
NUM_SAMPLES_INIT="${NUM_SAMPLES_INIT:-200}"
NUM_SAMPLES_REPLAN="${NUM_SAMPLES_REPLAN:-20}"
NUM_GRAD_SEQS="${NUM_GRAD_SEQS:-5}"
GRAD_LR="${GRAD_LR:-0.01}"
GRAD_STEPS="${GRAD_STEPS:-15}"
GRADIENT_DEVICE="${GRADIENT_DEVICE:-}"  # 新增：留空则与模型同设备

# 示例用法注释：
# export GRADIENT_DEVICE="cuda:2"  # 在cuda:2做梯度下降
# export DEVICE="cuda:3"           # 在cuda:3做渲染（模型设备）
```

**位置**: 约第147-152行（参数传递）

```bash
# CEM-GD参数
if [ "$OPTIMIZER" = "cem-gd" ]; then
    ARGS="$ARGS --num_samples_init $NUM_SAMPLES_INIT"
    ARGS="$ARGS --num_samples_replan $NUM_SAMPLES_REPLAN"
    ARGS="$ARGS --num_grad_seqs $NUM_GRAD_SEQS"
    ARGS="$ARGS --grad_lr $GRAD_LR"
    ARGS="$ARGS --grad_steps $GRAD_STEPS"
    if [ -n "$GRADIENT_DEVICE" ]; then
        ARGS="$ARGS --gradient_device $GRADIENT_DEVICE"
    fi
fi
```

---

## 使用方法

### 方式1: Shell脚本（推荐）

```bash
cd /home/ubuntu/yyf/4DGaussians

# 设置双GPU模式
export DEVICE="cuda:3"           # 渲染GPU（模型位置）
export GRADIENT_DEVICE="cuda:2"   # 梯度下降GPU

# 运行测试
bash run_cotracker_test.sh
```

### 方式2: 直接调用Python

```bash
python test/integration/test_cotracker_mpc.py \
    --model_path output/dnerf/lego \
    --initial_image initial.png \
    --target_image target.png \
    --device cuda:3 \
    --optimizer cem-gd \
    --gradient_device cuda:2 \
    --num_samples_init 200 \
    --num_samples_replan 20 \
    --num_grad_seqs 5
```

### 验证双GPU工作

```bash
# 终端1：监控GPU 2（梯度下降）
watch -n 0.5 'nvidia-smi | grep -A 10 "| *2 "'

# 终端2：监控GPU 3（渲染）
watch -n 0.5 'nvidia-smi | grep -A 10 "| *3 "'
```

**预期观察**:
- GPU 2: 梯度下降时显存占用增加（optimizer state + gradients），但波动小
- GPU 3: 渲染时显存占用增加（模型 + 中间激活），计算利用率高

---

## 性能预期

### 显存分布

**单GPU模式** (当前，OOM):
```
cuda:3 (总计 15-18 GB):
├── 模型 + deformation: 3-4 GB
├── 优化器状态: 0.5 GB
├── 梯度计算图: 4-6 GB
└── 渲染中间结果: 2-3 GB
━━━━━━━━━━━━━━━━━━━━━━━━━━━
超出容量 → OOM
```

**双GPU模式** (预期):
```
cuda:2 (梯度GPU):
├── action_sequences: 50 MB
├── 优化器状态: 500 MB
├── 梯度: 50 MB
└── 少量中间激活: 1-2 GB
━━━━━━━━━━━━━━━━━━━━━━━━━━━
总计: ~2-3 GB ✅

cuda:3 (渲染GPU):
├── 模型 + deformation: 3-4 GB
├── 渲染中间结果: 2-3 GB
└── 无梯度图开销
━━━━━━━━━━━━━━━━━━━━━━━━━━━
总计: ~5-7 GB ✅
```

### 速度影响

**传输开销**:
- Actions传输: ~50 MB × 2（往返）= 100 MB/iteration
- PCIe 3.0速度: ~12 GB/s
- 传输时间: 100 MB / 12 GB/s ≈ 8 ms/iteration
- 相比渲染时间（~100-500 ms），开销 < 2%

**预期减速**: 5-10%（传输开销 + 同步开销）

---

## 关键约束

### 必须满足

1. ✅ **不修改CEM模式**: 只在CEM-GD模式启用双GPU
2. ✅ **向后兼容**: `gradient_device=None` 时回退到单GPU模式
3. ✅ **不添加新依赖**: 纯PyTorch实现
4. ✅ **最小化修改**: 只修改CEM-GD相关代码

### 不能做

1. ❌ **不修改score_trajectories()**: 保持CEMOptimizer基类不变
2. ❌ **不移动模型**: GaussianModel保持在原设备（cuda:3）
3. ❌ **不破坏梯度流**: 必须保持autograd正确性

---

## 测试验证

### 正确性验证

```python
# 测试脚本：验证双GPU结果与单GPU一致
import torch
from mpc.cem_gd import CEMGDOptimizer

# 单GPU基线
optimizer_single = CEMGDOptimizer(..., gradient_device=None)
result_single = optimizer_single.plan(...)

# 双GPU测试
optimizer_dual = CEMGDOptimizer(..., gradient_device='cuda:2')
result_dual = optimizer_dual.plan(...)

# 验证一致性
torch.testing.assert_close(result_single, result_dual, rtol=1e-4, atol=1e-6)
```

### 性能验证

```bash
# 运行并记录显存
python test/integration/test_cotracker_mpc.py \
    --device cuda:3 \
    --gradient_device cuda:2 \
    --optimizer cem-gd \
    2>&1 | tee dual_gpu_test.log

# 检查输出
grep "CEM-GD Memory" dual_gpu_test.log
```

---

## 回滚方案

如果双GPU方案出现问题：

```bash
# 方式1: 命令行禁用
python test/integration/test_cotracker_mpc.py \
    --gradient_device ""  # 留空 = 单GPU模式

# 方式2: 环境变量
export GRADIENT_DEVICE=""
bash run_cotracker_test.sh

# 方式3: 代码回退
# 在 cem_gd.py 中注释掉 gradient_device 相关代码
```

---

## 下一步优化（可选）

如果双GPU仍不够：

1. **Per-sequence backward**: 逐个序列backward，进一步降低峰值显存
2. **Mixed precision**: 使用FP16减半显存
3. **Gradient checkpointing**: 重新计算中间激活而非存储

---

## 总结

| 方面 | 单GPU | 双GPU Pipeline |
|------|-------|----------------|
| 渲染显存 | 5-7 GB | 5-7 GB |
| 梯度显存 | 4-6 GB | 2-3 GB |
| 总显存（峰值） | 15-18 GB | Max(5-7, 2-3) = 5-7 GB |
| **是否OOM** | ❌ 是 | ✅ 否 |
| 速度 | 基线 | -5~10% |
| 代码复杂度 | 简单 | 中等 |

**推荐**: 优先实施双GPU方案，如果硬件不支持，再考虑per-sequence backward等替代方案。
