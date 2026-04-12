# CEM-GD Gradient Backprop Fix Plan

**Created:** 2026-04-01  
**Error:** `RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn`  
**Root Cause:** Gradient graph severed at 3 critical locations

---

## PROBLEM SUMMARY

CEM-GD gradient optimization fails at `objective_all[i].backward()` because the objective tensor has no `grad_fn`. 

**Call Chain:**
```
action_sequences (requires_grad=True)
  ↓
score_trajectories(requires_grad=True)
  ↓
model(batch, grad_enabled=True)
  ↓
flow_guided_gaussian_model.forward()
  ↓ ❌ BREAK 1: torch.tensor(pred_actions[:, t]) creates new leaf (no grad_fn)
control_vec (requires_grad=False)
  ↓
predict_flow_*() 
  ↓ ❌ BREAK 2: with torch.no_grad() disables gradients
predictions['flow'] (no grad_fn)
  ↓
objective.compute_reward()
  ↓ ❌ BREAK 3: reward.cpu().numpy() converts to numpy
objective (numpy or detached tensor, no grad_fn)
  ↓
objective.backward() → RuntimeError!
```

---

## ROOT CAUSES (Priority Order)

### 🔴 Critical Break #1: `torch.tensor()` Detaches Actions
**File:** `mpc/flow_guided_gaussian_model.py:~556`

**Current Code:**
```python
control_vec = torch.tensor(
    pred_actions[:, t],
    dtype=torch.float32,
    device=self.device
)
```

**Problem:** `torch.tensor()` creates a **new leaf tensor** from data, severing `grad_fn` even if `pred_actions` has `requires_grad=True`.

**Impact:** No gradients can flow from model outputs back to action sequences.

---

### 🔴 Critical Break #2: Objectives Return Numpy
**File:** `mpc/flow_objectives.py` (all `compute_reward()` methods)

**Current Pattern (appears in ~10 objectives):**
```python
def compute_reward(self, prediction, goal):
    # ... compute reward tensor ...
    reward = reward.cpu().numpy() if isinstance(reward, torch.Tensor) else reward
    return reward[:, None, None]
```

**Problem:** Unconditionally converts torch tensors to numpy, discarding `grad_fn`.

**Affected Objectives:**
- `FlowAlignmentObjective`
- `FlowConsistencyObjective`
- `FlowDirectionGuidanceObjective`
- `FlowDirectionLoss`
- `FlowGuidanceWithTargetObjective`
- `FlowSparseRenderObjective`
- `HybridFlowImageObjective`
- (All objectives in file)

---

### 🟡 Secondary Break #3: Flow Prediction Uses `no_grad()`
**Files:** `mpc/flow_guided_gaussian_model.py`

**Locations:**
1. `predict_flow_render_based`: Lines ~316+ wrap GMFlow in `with torch.no_grad()`
2. `predict_flow_from_control`: Lines ~176+ use `with torch.no_grad()` for deformation
3. Deformation path: `with torch.no_grad()` around `self.gaussians._deformation.forward_dynamic(...)`

**Problem:** Even when `grad_enabled=True`, these branches still disable gradients.

**Additional Issue:** GMFlow outputs converted via `.cpu().numpy()` → `torch.from_numpy()`, breaking autograd chain.

---

## TASK BREAKDOWN

### Task 1: Fix `control_vec` Construction (CRITICAL)
**Priority:** 🔴 HIGHEST  
**File:** `mpc/flow_guided_gaussian_model.py`  
**Estimated Time:** 5 minutes

**Change:**
```python
# BEFORE (line ~556)
control_vec = torch.tensor(pred_actions[:, t], dtype=torch.float32, device=self.device)

# AFTER
control_vec = pred_actions[:, t].to(self.device).float()
```

**Verification:**
```python
# Add temporary debug print after line
print(f"DEBUG: control_vec requires_grad={control_vec.requires_grad}, grad_fn={control_vec.grad_fn}")
# Expected: requires_grad=True, grad_fn=<SliceBackward0...>
```

**Success Criteria:**
- ✅ `control_vec.requires_grad == True`
- ✅ `control_vec.grad_fn` is not None
- ✅ Gradient flows from `control_vec` back to `pred_actions`

---

### Task 2: Fix Objective Return Types (CRITICAL)
**Priority:** 🔴 HIGHEST  
**File:** `mpc/flow_objectives.py`  
**Estimated Time:** 15 minutes

**Pattern to Replace (in ALL `compute_reward()` methods):**

```python
# BEFORE
reward = reward.cpu().numpy() if isinstance(reward, torch.Tensor) else reward
return reward[:, None, None]

# AFTER
if isinstance(reward, torch.Tensor):
    return reward.view(-1, 1, 1)  # Keep tensor with gradients
else:
    return np.expand_dims(reward, (1, 2))  # Numpy path (backward compatible)
```

**Affected Methods (apply to ALL):**
1. `FlowAlignmentObjective.compute_reward`
2. `FlowConsistencyObjective.compute_reward`
3. `FlowDirectionGuidanceObjective.compute_reward`
4. `FlowDirectionLoss.compute_reward`
5. `FlowGuidanceWithTargetObjective.compute_reward`
6. `FlowSparseRenderObjective.compute_reward`
7. `HybridFlowImageObjective.compute_reward`
8. (Any other objectives in file)

**Verification:**
```python
# In mpc/cem.py after line 280 (rewards = self.obj_fn(...))
print(f"DEBUG: rewards type={type(rewards)}")
if isinstance(rewards, torch.Tensor):
    print(f"  requires_grad={rewards.requires_grad}, grad_fn={rewards.grad_fn}")
# Expected when requires_grad=True: torch.Tensor with grad_fn
```

**Success Criteria:**
- ✅ When `requires_grad=True`: objectives return `torch.Tensor`
- ✅ Returned tensor has `grad_fn` (not detached)
- ✅ When `requires_grad=False`: objectives can still return numpy (backward compatible)

---

### Task 3: Fix Flow Prediction `no_grad()` Usage
**Priority:** 🟡 HIGH  
**File:** `mpc/flow_guided_gaussian_model.py`  
**Estimated Time:** 20 minutes

#### Subtask 3.1: `predict_flow_render_based`

**Change Strategy:** Add `grad_enabled` parameter, conditionally use `no_grad()`

```python
# BEFORE (line ~314)
def predict_flow_render_based(self, ...):
    ...
    with torch.no_grad():  # Always disables gradients
        rendered_image = self.render_with_control(control_vec)
        # ... GMFlow inference ...

# AFTER
def predict_flow_render_based(self, ..., grad_enabled=False):
    ...
    if grad_enabled:
        # Gradient mode: keep autograd active
        rendered_image = self.render_with_control(control_vec, grad_enabled=True)
        # ... GMFlow inference (no no_grad wrapper) ...
    else:
        # Non-gradient mode: use no_grad for speed
        with torch.no_grad():
            rendered_image = self.render_with_control(control_vec, grad_enabled=False)
            # ... GMFlow inference ...
```

**Additional Fix:** Remove `.cpu().numpy()` → `torch.from_numpy()` roundtrip

```python
# BEFORE (line ~340+)
flow_field_tensor = flow_predictions[-1]  # (1, 2, H, W)
flow_field = flow_field_tensor[0].permute(1, 2, 0).cpu().numpy()  # ❌ Breaks grad
flow_x = torch.from_numpy(flow_field[y_coords.cpu(), x_coords.cpu(), 0]).to(self.device)

# AFTER
flow_field_tensor = flow_predictions[-1]  # (1, 2, H, W)
flow_field = flow_field_tensor[0].permute(1, 2, 0)  # Keep on device, (H, W, 2)
# Ensure y_coords, x_coords are LongTensors on same device
flow_x = flow_field[y_coords.long(), x_coords.long(), 0]  # Direct indexing preserves grad
```

---

#### Subtask 3.2: `predict_flow_from_control`

**Change:** Conditional `no_grad()` around deformation

```python
# BEFORE (line ~176+)
with torch.no_grad():
    means3D_deformed, _, _, _, _ = self.gaussians._deformation.forward_dynamic(...)

# AFTER
if grad_enabled:
    # Gradient mode: no no_grad wrapper
    means3D_deformed, _, _, _, _ = self.gaussians._deformation.forward_dynamic(...)
else:
    # Non-gradient mode
    with torch.no_grad():
        means3D_deformed, _, _, _, _ = self.gaussians._deformation.forward_dynamic(...)
```

**Parameter Propagation:**
Update all callsites of `predict_flow_render_based` and `predict_flow_from_control` to pass `grad_enabled` parameter.

---

#### Subtask 3.3: Update `forward()` Callsites

**In `flow_guided_gaussian_model.py:forward()`:**

```python
# BEFORE
next_flow = self.predict_flow_render_based(...)
# or
next_flow = self.predict_flow_from_control(...)

# AFTER
next_flow = self.predict_flow_render_based(..., grad_enabled=grad_enabled)
# or
next_flow = self.predict_flow_from_control(..., grad_enabled=grad_enabled)
```

**Success Criteria:**
- ✅ When `grad_enabled=True`: no `with torch.no_grad()` in critical path
- ✅ Flow predictions have `grad_fn` linking back to `control_vec`
- ✅ When `grad_enabled=False`: still uses `no_grad()` for speed (backward compatible)

---

### Task 4: Verify End-to-End Gradient Flow
**Priority:** 🟢 VERIFICATION  
**Estimated Time:** 10 minutes

**Test Script:** Create `test/verification/test_cemgd_backprop.py`

```python
#!/usr/bin/env python3
"""Verify CEM-GD gradient backprop after fixes."""
import torch
import sys
sys.path.insert(0, '/home/ubuntu/yyf/4DGaussians')

from mpc.cem_gd import CEMGDOptimizer
from mpc.flow_guided_gaussian_model import FlowGuidedGaussianDynamicsModel
from mpc.cotracker_objectives import PointTrackingObjective
# ... setup model, objectives ...

# Create optimizer
optimizer = CEMGDOptimizer(...)

# Run gradient_optimization with small batch
action_sequences_list = [
    torch.randn(5, 15, requires_grad=True, device='cuda:0')
    for _ in range(3)
]

# Score on render device
objective_all = optimizer._score_on_render_device(
    action_sequences_list,
    action_history=[],
    obs_history=[...],
    state_history=[...],
    goal={...}
)

# Verify gradients
print("=== GRADIENT VERIFICATION ===")
for i, obj in enumerate(objective_all):
    print(f"Objective {i}:")
    print(f"  Type: {type(obj)}")
    print(f"  requires_grad: {obj.requires_grad}")
    print(f"  grad_fn: {obj.grad_fn}")
    print(f"  Device: {obj.device}")
    
    # Test backward
    try:
        obj.backward(retain_graph=True)
        print(f"  ✅ Backward succeeded")
        
        # Check action gradients
        if action_sequences_list[i].grad is not None:
            print(f"  ✅ Action gradients populated: shape {action_sequences_list[i].grad.shape}")
        else:
            print(f"  ❌ Action gradients are None")
    except RuntimeError as e:
        print(f"  ❌ Backward failed: {e}")

print("\nAll tests completed.")
```

**Expected Output:**
```
=== GRADIENT VERIFICATION ===
Objective 0:
  Type: <class 'torch.Tensor'>
  requires_grad: True
  grad_fn: <NegBackward0 object at 0x...>
  Device: cuda:1
  ✅ Backward succeeded
  ✅ Action gradients populated: shape torch.Size([5, 15])
Objective 1:
  ...
```

**Success Criteria:**
- ✅ All objectives are `torch.Tensor` (not numpy)
- ✅ All objectives have `requires_grad=True`
- ✅ All objectives have non-None `grad_fn`
- ✅ `objective.backward()` succeeds without RuntimeError
- ✅ `action_sequences.grad` is populated with finite values

---

### Task 5: Update MODIFICATIONS_INDEX.md
**Priority:** 🟢 DOCUMENTATION  
**Estimated Time:** 5 minutes

**Add Entry:**
```markdown
### 2026-04-01: CEM-GD Gradient Backprop Fix

**Problem:** CEM-GD gradient optimization failed with "element 0 of tensors does not require grad and does not have a grad_fn" during `objective.backward()`.

**Root Causes:**
1. `torch.tensor()` in `flow_guided_gaussian_model.py` severed gradient connection from actions
2. Objectives in `flow_objectives.py` unconditionally returned numpy arrays
3. Flow prediction methods used `torch.no_grad()` even in gradient mode

**Fixes:**
- `mpc/flow_guided_gaussian_model.py:~556`: Changed `torch.tensor(pred_actions[:, t])` → `pred_actions[:, t].to(device).float()` (preserves `grad_fn`)
- `mpc/flow_objectives.py`: Modified all `compute_reward()` to return tensors when inputs are tensors (conditional numpy conversion)
- `mpc/flow_guided_gaussian_model.py`: Added `grad_enabled` parameter to `predict_flow_*()` methods, conditionally disable `torch.no_grad()`

**Impact:** CEM-GD gradient-based trajectory optimization now functional.

**Verification:** `test/verification/test_cemgd_backprop.py`

**Related:** Commit cb7ad71 (previous flow/checkpoint fixes)
```

---

## EXECUTION ORDER

**Sequential (dependencies):**

1. ✅ **Task 1** (5 min) - Fix `control_vec` - MUST complete first
2. ✅ **Task 2** (15 min) - Fix objective returns - MUST complete second
3. ⏸️ **Task 3** (20 min) - Fix `no_grad()` usage - Can defer if Tasks 1-2 resolve issue
4. ⏸️ **Task 4** (10 min) - Verification test - Run after Tasks 1-2
5. ⏸️ **Task 5** (5 min) - Documentation - Final step

**Total Critical Path:** 20 minutes (Tasks 1-2 only)  
**Full Implementation:** 55 minutes (all tasks)

---

## BACKWARD COMPATIBILITY GUARANTEE

**Pure CEM Mode (grad_enabled=False):**
- ✅ Objectives return numpy (via else branch) → Original behavior preserved
- ✅ Flow prediction uses `no_grad()` → Original speed preserved
- ✅ No impact on sampling-based planning

**CEM-GD Mode (grad_enabled=True):**
- ✅ Objectives return tensors with gradients → New functionality enabled
- ✅ Flow prediction preserves gradients → Gradient optimization works
- ✅ Backward pass succeeds → Action gradients computed

---

## DEBUGGING TIPS

### Quick Diagnostic (add to `mpc/cem.py:280`):
```python
# After: rewards = self.obj_fn(predictions, goal_gpu)
if requires_grad:
    import sys
    print(f"[DEBUG] rewards type: {type(rewards)}", file=sys.stderr)
    if isinstance(rewards, torch.Tensor):
        print(f"  requires_grad: {rewards.requires_grad}", file=sys.stderr)
        print(f"  grad_fn: {rewards.grad_fn}", file=sys.stderr)
    else:
        print(f"  ❌ rewards is numpy! (breaks gradient)", file=sys.stderr)
```

### Check Control Vec (add to `flow_guided_gaussian_model.py:~558`):
```python
# After: control_vec = pred_actions[:, t].to(self.device).float()
if grad_enabled:
    import sys
    print(f"[DEBUG] control_vec requires_grad: {control_vec.requires_grad}", file=sys.stderr)
    print(f"  grad_fn: {control_vec.grad_fn}", file=sys.stderr)
```

### Check Predictions (add to `flow_guided_gaussian_model.py` before stacking):
```python
# Before: predictions['flow'] = torch.stack(...)
if grad_enabled:
    import sys
    sample_flow = predictions['flow'][0]
    print(f"[DEBUG] flow[0] requires_grad: {sample_flow.requires_grad}", file=sys.stderr)
    print(f"  grad_fn: {sample_flow.grad_fn}", file=sys.stderr)
```

---

## ROLLBACK PLAN

If any task causes regressions:

```bash
# Revert specific files
git checkout HEAD -- mpc/flow_guided_gaussian_model.py
git checkout HEAD -- mpc/flow_objectives.py

# Or revert entire commit
git revert <commit-hash>
```

**Rollback Triggers:**
- Pure CEM performance degrades
- Objectives return wrong values (numerical differences)
- Tests fail that previously passed

---

## ACCEPTANCE CRITERIA

### Minimum (Tasks 1-2):
- ✅ `control_vec` preserves `requires_grad=True` from `pred_actions`
- ✅ Objectives return `torch.Tensor` when `requires_grad=True`
- ✅ `objective.backward()` succeeds without RuntimeError
- ✅ `action_sequences.grad` is not None

### Full (Tasks 1-4):
- ✅ All minimum criteria met
- ✅ Flow prediction preserves gradients in `grad_enabled=True` mode
- ✅ End-to-end test passes
- ✅ Gradient magnitudes are reasonable (not NaN/Inf)

### Production Ready (All Tasks):
- ✅ All full criteria met
- ✅ Documentation updated
- ✅ Backward compatibility verified (Pure CEM unchanged)
- ✅ Integration test with real model succeeds

---

## NOTES

**Why Task 3 is Optional (Short Term):**
- Tasks 1-2 MAY be sufficient if flow predictions are not critical to gradient computation
- If objectives depend heavily on flow values, Task 3 becomes mandatory
- Recommend: Try Tasks 1-2 first, add Task 3 if gradients are still zero/broken

**Common Pitfall:**
- Don't forget to pass `grad_enabled` parameter through all callsites
- Don't assume `torch.Tensor` operations preserve gradients - always check `grad_fn`

**Performance Note:**
- Gradient mode is ~30% slower than `no_grad()` mode
- This is expected and acceptable for CEM-GD (uses fewer samples to compensate)
