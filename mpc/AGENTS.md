# MPC Module - Model Predictive Control

**Generated:** 2026-03-04  
**Updated:** 2026-03-09  
**Scope:** Control/planning for Gaussian manipulation

## OVERVIEW

Model Predictive Control (MPC) integration for robotic planning. Uses flow-guided objectives and learned Gaussian dynamics for trajectory optimization.

## CONTROL ACTION SPACE

### Robot Arm Control (15-dimensional vector)

The control action is a **15-dimensional vector** representing robot joint positions and gripper state:

```
u = [u₀, u₁, u₂, ..., u₁₄]  ∈ ℝ¹⁵
```

**Dimension breakdown:**

| Indices | Description | Range | Notes |
|---------|-------------|-------|-------|
| 0-1 | Joint 1: [sin(θ₁), cos(θ₁)] | [-1, 1] | Trigonometric encoding of joint angle |
| 2-3 | Joint 2: [sin(θ₂), cos(θ₂)] | [-1, 1] | Trigonometric encoding of joint angle |
| 4-5 | Joint 3: [sin(θ₃), cos(θ₃)] | [-1, 1] | Trigonometric encoding of joint angle |
| 6-7 | Joint 4: [sin(θ₄), cos(θ₄)] | [-1, 1] | Trigonometric encoding of joint angle |
| 8-9 | Joint 5: [sin(θ₅), cos(θ₅)] | [-1, 1] | Trigonometric encoding of joint angle |
| 10-11 | Joint 6: [sin(θ₆), cos(θ₆)] | [-1, 1] | Trigonometric encoding of joint angle |
| 12-14 | Gripper: [g₁, g₂, g₃] | [-1, 1] | Gripper state (3 DoF) |

**Why trigonometric encoding?**
- Avoids angle wrapping discontinuities (e.g., 359° → 0°)
- sin/cos representation is smooth and periodic
- Directly compatible with neural network inputs
- Preserves angular information without ambiguity

**Gripper dimensions:**
- Typically: [left_finger, right_finger, rotation] or [open/close, pitch, yaw]
- Exact interpretation depends on gripper hardware

**Example from transforms.json:**
```json
"joint_pos": [
    -0.998, 0.067,    // Joint 1: sin=-0.998, cos=0.067 ≈ θ₁=-86°
    -0.485, -0.875,   // Joint 2: sin=-0.485, cos=-0.875 ≈ θ₂=-151°
    0.838, -0.546,    // Joint 3
    -0.621, 0.784,    // Joint 4
    0.903, -0.429,    // Joint 5
    0.825, 0.566,     // Joint 6
    0.606, 0.593, 0.593  // Gripper state
]
```

**Action constraints in MPC:**
- Joint dimensions (0-11) represent absolute state in sin/cos and must not be clipped
- Gripper dimensions (12-14) are clipped to [-1, 1]
- Angular velocity is constrained via Δθ between timesteps (default 0.524 rad ≈ 30°)
- Use `mpc.constraint_utils.check_angular_velocity_constraint` for enforcement

## STRUCTURE

```
mpc/
├── agent.py                        # Planning agents (CEM-based)
├── cem.py                          # Cross-Entropy Method optimizer
├── cem_gd.py                       # CEM + gradient descent hybrid
├── gaussian_dynamics_model.py      # Dynamics model wrapper
├── flow_guided_gaussian_model.py   # Flow-guided dynamics variant
├── flow_objectives.py              # Flow-based objectives (44k LOC!)
├── objectives.py                   # General MPC objectives
├── flow_loss_utils.py              # Flow loss functions
├── perceptual_loss_utils.py        # VGG perceptual loss
├── important_gaussian_selector.py  # Sample relevant Gaussians
└── utils.py                        # MPC utilities
```

## KEY COMPONENTS

### Planning Agents (agent.py)
- **PlanningAgent** - Base CEM planner
- **RandomAgent** - Random baseline
- **SimplePlanningAgent** - Simplified planner

### Optimizers
- **CEM** (cem.py) - Cross-Entropy Method (sample-based)
- **CEM-GD** (cem_gd.py) - Hybrid sample + gradient (recommended for most use cases)
- **MPPI** (mppi.py) - Model Predictive Path Integral
- **L-BFGS** (lbfgs.py) - Second-order optimizer
  - Deprecated for joint-state planning (action clipping bug), use CEM/MPPI

#### CEM vs CEM-GD: When to Use Which

| Optimizer | Best For | Sample Efficiency | Convergence Speed | Computational Cost |
|-----------|----------|-------------------|-------------------|--------------------|
| **CEM** | Simple objectives, quick prototyping | Baseline (1×) | Baseline (5-7 iters) | Low |
| **CEM-GD** | Production use, complex objectives | 5-10× better | 3× faster (2-3 iters) | Medium (+30% overhead) |

**Recommendation**: Use **CEM-GD** by default. Switch to CEM only if:
- Objective is non-differentiable
- Gradient computation is prohibitively expensive
- Prototyping/debugging (CEM is simpler)

#### CEM-GD Hyperparameters

**Core Parameters**:
- `num_samples_init`: Initial planning samples (default: 200)
  - Use 200-300 for first planning step (more exploration needed)
  - Higher values improve initial solution quality
  
- `num_samples_replan`: Replanning samples (default: 100)
  - Use 100-150 for subsequent MPC steps (warm start from previous plan)
  - Can be lower than `num_samples_init` due to warm start
  
- `opt_iters`: CEM iterations before gradient descent (default: 2-3)
  - 2-3 iterations sufficient with gradient refinement
  - Pure CEM typically needs 5-7 iterations
  
- `num_grad_seqs`: Top-K sequences for gradient optimization (default: 5)
  - 5-10 sequences balance exploration vs computation
  - Higher values = more diverse gradient descent initializations
  
- `grad_lr`: Adam learning rate for gradient descent (default: 0.01)
  - 0.01 works well for most objectives
  - Lower (0.001-0.005) for sensitive objectives
  - Higher (0.02-0.05) for coarse objectives
  
- `grad_steps`: Gradient descent iterations (default: 15)
  - 10-20 steps typical for convergence
  - Monitor convergence in logs - may reduce for faster planning

**Advanced Parameters** (in `CEMGDOptimizer.__init__`):
- `elites_frac`: Top fraction for CEM refit (default: 0.1)
- `alpha`: CEM distribution smoothing (default: 0.1)
- `factor_shrink`: Learning rate decay factor (default: 0.5)
- `max_tries`: Max learning rate reduction attempts (default: 3)

**Example Configurations**:

```python
# Fast prototyping (lower quality, faster)
optimizer_type='cem-gd',
num_samples_init=100,
num_samples_replan=50,
opt_iters=2,
num_grad_seqs=3,
grad_lr=0.01,
grad_steps=10,

# Production (balanced quality/speed) - DEFAULT
optimizer_type='cem-gd',
num_samples_init=200,
num_samples_replan=100,
opt_iters=3,
num_grad_seqs=5,
grad_lr=0.01,
grad_steps=15,

# High quality (slower, best results)
optimizer_type='cem-gd',
num_samples_init=300,
num_samples_replan=150,
opt_iters=3,
num_grad_seqs=10,
grad_lr=0.005,
grad_steps=20,
```

### Dynamics Models
- **GaussianDynamicsModel** - Wraps GaussianModel for planning
- **FlowGuidedGaussianModel** - Adds GMFlow objectives

## TYPICAL WORKFLOW

1. **Load model:** `GaussianModel` + deformation from checkpoint
2. **Create dynamics:** `GaussianDynamicsModel(gaussian_model, ...)`
3. **Define objectives:** Flow goals, perceptual loss, regularization
4. **Plan:** `agent.plan(initial_state, objectives, horizon=H)`
5. **Execute:** Apply action sequence, render intermediate frames

## DEPENDENCIES

### External Models
- **GMFlow:** Optical flow network (expects checkpoint at `gmflow/checkpoints/gmflow_sintel-0c07dcb3.pth`)
- **VGG:** Perceptual loss (loaded from torchvision)

### Device Management
**CRITICAL:** Set `CUDA_VISIBLE_DEVICES` before importing torch
```python
# demo_flow_guided_mpc.py does this via parse_device_early()
os.environ['CUDA_VISIBLE_DEVICES'] = device_str
import torch  # Import AFTER setting env
```

## OBJECTIVES (flow_objectives.py)

### Flow-Based
- **FlowMatchingObjective** - Match GMFlow predictions
- **FlowGoalObjective** - Reach flow target points
- **FlowDirectionObjective** - Direction constraints

### Perceptual
- **PerceptualLossObjective** - VGG feature matching
- **ImageSimilarityObjective** - Pixel-level similarity

### Regularization
- **ActionRegularization** - Smooth actions
- **PhysicalConstraints** - Physical plausibility

## RUNNING MPC DEMOS

### Quick Demo
```bash
bash run_render_based_test.sh  # Uses defaults
```

### Manual Invocation

**CEM-GD Mode (Default, Recommended)**:
```bash
python demo_flow_guided_mpc.py \
  --model_path output/<exp>/ \
  --initial_image <initial.png> \
  --target_image <target.png> \
  --num_steps 20 \
  --horizon 10 \
  --optimizer cem-gd \
  --num_samples_init 200 \
  --num_samples_replan 100 \
  --num_grad_seqs 5 \
  --grad_lr 0.01 \
  --grad_steps 15 \
  --device cuda:0
```

**Pure CEM Mode (Backward Compatible)**:
```bash
python demo_flow_guided_mpc.py \
  --model_path output/<exp>/ \
  --initial_image <initial.png> \
  --target_image <target.png> \
  --num_steps 20 \
  --horizon 10 \
  --optimizer cem \
  --num_samples 1000 \
  --opt_iters 10 \
  --device cuda:0
```

**Benchmark CEM vs CEM-GD**:
```bash
python test/scripts/benchmark_cem_vs_cemgd.py \
  --model_path output/dnerf/lego \
  --initial initial.png \
  --target target.png \
  --output benchmark_results.json
```

### Config-Based (Hydra)
```bash
python scripts/run_control.py --config-name=<config>
```

## CONVENTIONS

### Rendering vs Control
- **Rendering code:** scene/, gaussian_renderer/
- **Control code:** mpc/
- **Bridge:** gaussian_dynamics_model.py wraps rendering for planning

### State Representation
- **State:** Gaussian parameters at timestep t
- **Action:** Deformation/control parameters
- **Dynamics:** state_{t+1} = f(state_t, action_t)

### Objective Composition
```python
objectives = [
    FlowGoalObjective(flow_targets, weight=1.0),
    PerceptualLossObjective(target_image, weight=0.5),
    ActionRegularization(weight=0.01)
]
```

## GOTCHAS

1. **GMFlow checkpoint required** - Download and place in `gmflow/checkpoints/`
2. **Device must be set early** - Before torch import
3. **Large files** - flow_objectives.py is 44k LOC (complex logic)
4. **Memory intensive** - MPC samples many trajectories on GPU

## WHERE TO LOOK

| Task | File |
|------|------|
| Modify planner | agent.py, cem.py |
| Add objective | flow_objectives.py or objectives.py |
| Change dynamics | gaussian_dynamics_model.py |
| Debug flow goals | flow_guided_gaussian_model.py |
| Run experiments | demo_flow_guided_mpc.py, run_control.py |

## ADDING NEW OBJECTIVE

1. Subclass `Objective` in objectives.py
2. Implement `__call__(state, action) → reward`
3. Add to objective list in demo/planning code
4. Tune weight via grid search

**When in doubt:** Check existing objectives in flow_objectives.py for patterns.
