# MPC Pipeline Architecture Analysis

**Generated:** 2026-03-10  
**Author:** Atlas (Sisyphus Work Orchestrator)  
**Purpose:** Compare user's expected pipeline with actual implementation

---

## Executive Summary

### 🔴 CRITICAL FINDINGS

1. **GMFlow/Motion Mask Sampling**: Called **ONCE** at initialization, NOT at each MPC step
2. **Point Re-sampling Strategy**: Tracking failure recovery does **NOT** use motion_mask method
3. **Point Tracking**: TAPIR incrementally tracks (current → next), NOT re-sampled from initial → target at each step

### ✅ CONFIRMED CORRECT

- CEM optimization loop structure matches expected pipeline
- Rendering happens inside CEM via Gaussian dynamics model
- Loss computation occurs after rendering
- CEM distribution updates work as expected
- Overall loop structure (steps 4-10) is correct

### 🟡 ARCHITECTURAL MISMATCH

User's expected pipeline assumes **motion-driven re-sampling at each step** (step 10).  
Actual implementation uses **incremental TAPIR tracking** with fallback to Shi-Tomasi (NOT motion_mask).

**Impact:** Moderate - Incremental tracking may accumulate drift over long horizons, but is computationally efficient and avoids repeated GMFlow calls.

---

## Pipeline Comparison: Expected vs Actual

| Step | User's Expected Pipeline (Chinese) | User's Expected Pipeline (English) | Actual Implementation | Status |
|------|-----------------------------------|-------------------------------------|----------------------|--------|
| 1 | GMFlow计算初始帧和目标帧的光流 | GMFlow computes optical flow between initial and target frame | ✅ **ONCE at initialization** (line 262) | ⚠️ **PARTIAL** |
| 2 | 分割出物体和机械臂 | Segment objects and robot arm | ✅ `compute_motion_mask()` returns boolean mask | ✅ **CORRECT** |
| 3 | 在物体和机械臂上获取跟踪点 | Sample tracking points on objects and robot arm | ✅ `sample_motion_driven_points()` at initialization | ✅ **CORRECT** |
| 4 | 通过CEM方法进行动作策略的评估 | Evaluate action policies via CEM | ✅ `agent.act()` → `optimizer.plan()` → `perform_cem()` (line 457) | ✅ **CORRECT** |
| 5 | 对策略通过高斯动力学模型渲染 | Render policies via Gaussian dynamics model | ✅ Inside CEM: `self.model(batch)` (line 248 in cem.py) | ✅ **CORRECT** |
| 6 | 渲染完成后计算loss | Compute loss after rendering | ✅ `self.obj_fn(predictions, goal_gpu)` (line 273 in cem.py) | ✅ **CORRECT** |
| 7 | 更新CEM分布 | Update CEM distribution | ✅ `self.update_dist()` (line 402 in cem.py) | ✅ **CORRECT** |
| 8 | 优化到规定步数后执行最优策略 | Execute optimal policy after optimization | ✅ Best action applied via `model.render_with_control()` (line 484) | ✅ **CORRECT** |
| 9 | 将最优策略执行后的帧作为现在帧 | Use executed frame as current frame | ✅ `current_image = next_image_np` (line 534) | ✅ **CORRECT** |
| 10 | 获取现在帧的跟踪点位置 | Get tracking point positions in current frame | ⚠️ **TAPIR incremental tracking** (line 495), NOT re-sampling from motion mask | 🔴 **MISMATCH** |
| 11 | 回到4开始循环4-10 | Loop back to step 4, repeat 4-10 | ✅ `for step in range(1, args.num_steps + 1)` (line 447) | ✅ **CORRECT** |
| 12 | 到达目标帧 | Reach target frame | ✅ Loop terminates after `num_steps` | ✅ **CORRECT** |

---

## Detailed Implementation Analysis

### Step 1: GMFlow Optical Flow Computation

**Expected:** GMFlow computes flow between initial and target at each step (or at least initially)

**Actual:**
```python
# test_cotracker_mpc.py:262-271
if args.sampling_method == "motion_mask":
    initial_points = point_sampling.sample_motion_driven_points(
        initial_image,    # FIXED: initial frame
        target_image,     # FIXED: target frame
        num_points=args.num_tracking_points,
        device=args.device,
        motion_ratio=0.7,
        save_diagnostics=True,
        output_dir=args.output_dir
    )
```

**Key Finding:** GMFlow is called **ONCE** during initialization phase, NOT inside the MPC loop.

**Location in Code:**
- **Initialization**: Line 262 (before MPC loop starts at line 447)
- **Inside MPC loop**: **NOT PRESENT**

**Implication:** 
- ✅ **Pro**: Computationally efficient (GMFlow is expensive ~100ms per call)
- ❌ **Con**: Motion mask based on initial → target flow, not updated as robot moves

---

### Steps 4-7: CEM Optimization Loop (INSIDE `optimizer.plan()`)

**Expected:** CEM samples actions → renders → computes loss → updates distribution

**Actual:** Confirmed correct structure in `mpc/cem.py:290-446`

```python
# mpc/cem.py:311-404 (perform_cem method)
for iter in range(self.opt_iters):
    # Step 4: Sample action candidates from distribution
    new_action_samples = self.sampler.sample_actions(
        self.num_samples, mu, np.sqrt(constrained_var)
    )
    
    # Step 5: Render trajectories via Gaussian dynamics
    # Step 6: Compute objectives/losses
    predictions, rewards, action_samples = self.score_trajectories(
        new_action_samples,
        obs_history,
        state_history,
        action_history,
        goal,
    )
    # Inside score_trajectories (line 197-288):
    #   - Prepare batch with action sequences (line 237-245)
    #   - Model prediction: self.model(batch) (line 248) → calls GaussianDynamicsModel.__call__
    #   - Objective computation: self.obj_fn(predictions, goal_gpu) (line 273)
    
    # Step 7: Update distribution based on elite samples
    mu, var = self.update_dist(
        action_samples[:, n_ctxt:], rewards, mu, var
    )
```

**Key Components:**

1. **Sampling:** `self.sampler.sample_actions()` - Samples action sequences from Gaussian distribution
2. **Rendering:** Inside `score_trajectories()` → `self.model(batch)` (line 248)
   - This calls `GaussianDynamicsModel.__call__()` (gaussian_dynamics_model.py:394-452)
   - Which internally calls `render_with_control()` for each timestep (line 435)
3. **Loss Computation:** `self.obj_fn(predictions, goal_gpu)` (line 273)
   - Objectives can be: FlowAlignmentObjective, VGGPerceptualObjective, ActionRegularization, etc.
   - Returns rewards (higher = better)
4. **Distribution Update:** `self.update_dist()` (line 170-190)
   - Selects elite samples (top 10-20% by reward)
   - Updates mean and variance toward elites

**Verification:** ✅ **FULLY CORRECT** - Matches user's expected steps 4-7

---

### Step 10: Get Tracking Points in Current Frame (🔴 KEY DIFFERENCE)

**User's Expected Behavior:**
```
Step 10: 获取现在帧的跟踪点位置
Translation: "Get tracking point positions in current frame"
```

**Interpretation:** Re-sample or re-detect tracking points based on motion in current frame.

**Actual Implementation:**
```python
# test_cotracker_mpc.py:492-497 (incremental TAPIR tracking)
video_tensor = torch.stack([
    torch.from_numpy(current_image).permute(2, 0, 1).float() / 255.0,
    torch.from_numpy(next_image_np).permute(2, 0, 1).float() / 255.0
], dim=0).unsqueeze(0).to(args.device)

tracks, visibles = tracker.track(video_tensor, current_tracked_points)
new_points = tracks[0, :, 1, :].cpu().numpy()  # Get points at t=1 (next frame)
```

**Key Difference:** TAPIR incrementally tracks existing points from current → next frame, **NOT** re-sampling from motion mask.

**Tracking Failure Recovery:**
```python
# test_cotracker_mpc.py:508-517
if failed:
    print(f"  ⚠️ Tracking failure detected: {failure_reason}")
    if args.sampling_method == "shi_tomasi":
        new_points = point_sampling.sample_shi_tomasi_points(next_image_np, ...)
    elif args.sampling_method == "combined":
        new_points = point_sampling.sample_combined(next_image_np, ...)
    # ... other methods ...
    # ⚠️ NOTICE: motion_mask case is MISSING!
```

**Critical Gap:** When tracking fails during MPC loop, the code does **NOT** call `sample_motion_driven_points(current_image, target_image)` to re-sample based on motion.

**Why This Matters:**
- **Incremental tracking** may accumulate drift over long horizons (20+ steps)
- **Motion-driven re-sampling** would refocus points on moving objects at each step
- **Trade-off:** Re-sampling is expensive (GMFlow + sampling ~150ms), incremental tracking is fast (~10ms)

---

## Architectural Evaluation

### Question 1: Which Pipeline is More Reasonable?

Both pipelines have merit depending on the use case:

#### **User's Expected Pipeline** (Motion re-sampling at each step)

**Pros:**
- ✅ **Robust to drift** - Points stay focused on moving objects throughout trajectory
- ✅ **Adaptive** - Motion mask recomputed based on current → target flow
- ✅ **Better for long horizons** - Less accumulation of tracking errors

**Cons:**
- ❌ **Computationally expensive** - GMFlow inference ~100ms per call × 20 steps = 2 seconds overhead
- ❌ **Discontinuous** - Point identity changes at each step (harder to track specific features)
- ❌ **May be overkill** - If tracking quality is already good, repeated re-sampling adds little value

#### **Actual Implementation** (Incremental TAPIR tracking)

**Pros:**
- ✅ **Fast** - TAPIR tracking ~10ms per step vs GMFlow ~100ms
- ✅ **Continuous** - Same points tracked throughout (better for objectives that care about specific features)
- ✅ **Sufficient for short horizons** - If MPC horizon is 5-10 steps, drift is minimal

**Cons:**
- ❌ **Drift accumulation** - Over 20+ steps, points may drift to background
- ❌ **No adaptation** - Motion mask computed once at start, not updated
- ❌ **Failure recovery incomplete** - Doesn't use motion_mask on tracking failure

---

## Final Verdict

### Is Current Code Correct?

**Mostly YES, with one critical gap:**

✅ **Correct:**
- CEM optimization loop (steps 4-7)
- Rendering integration (step 5)
- Loss computation (step 6)
- Distribution updates (step 7)
- Action execution (step 8)
- Loop structure (steps 9, 11, 12)

🔴 **Critical Gap:**
- Step 10: Tracking failure recovery missing `motion_mask` case
- No periodic re-sampling mechanism for long horizons

⚠️ **Design Choice (Not Wrong, But Different):**
- User expected: Motion re-sampling at each step
- Actual: Incremental TAPIR tracking (faster, may drift)

### Which Pipeline is More Reasonable?

**Both are reasonable depending on priorities:**

- **User's pipeline**: Better for long horizons, robust to drift, but slower
- **Current pipeline**: Faster, good for short horizons, but may drift

**Recommended:** Hybrid approach combining both strengths

---

**Document Version:** 1.0  
**Date:** 2026-03-10  
**Author:** Atlas (Sisyphus Work Orchestrator)
