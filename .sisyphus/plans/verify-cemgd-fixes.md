# CEM-GD Gradient Mode Fix Verification Plan

**Created:** 2026-04-01  
**Status:** Ready for execution  
**Related Commit:** cb7ad71

## OVERVIEW

Verify that three critical fixes enable CEM-GD gradient mode:
1. ActionProcessor checkpoint compatibility
2. Flow prediction tensor/numpy handling
3. Point tracker gradient detachment

All tests must pass AND Pure CEM mode must remain functional (backward compatibility).

---

## CONTEXT

### What Was Fixed

**Fix 1: ActionProcessor Defaults** (scene/deformation_triplane.py:187-192)
```python
# BEFORE (wrong defaults causing checkpoint load errors)
action_use_pe = getattr(args, 'action_use_pe', True)         # Wrong: True
action_input_dim = getattr(args, 'action_input_dim', 6)      # Wrong: 6
action_output_dim = getattr(args, 'action_output_dim', 64)   # Wrong: 64

# AFTER (checkpoint-compatible defaults)
action_use_pe = getattr(args, 'action_use_pe', False)        # Correct: False
action_input_dim = getattr(args, 'action_input_dim', 15)     # Correct: 15
action_output_dim = getattr(args, 'action_output_dim', 32)   # Correct: 32
```

**Fix 2: Flow Append Logic** (mpc/flow_guided_gaussian_model.py:687-690)
```python
# BEFORE (unconditional numpy - broke gradient mode)
predictions['flow'].append(next_flow.cpu().numpy())

# AFTER (conditional - preserves tensors in gradient mode)
if grad_enabled:
    predictions['flow'].append(next_flow)  # Tensor for gradients
else:
    predictions['flow'].append(next_flow.cpu().numpy())  # Numpy for CEM
```

**Fix 3: Point Tracker Detach** (mpc/point_tracker.py:140)
```python
# BEFORE (failed on requires_grad tensors)
video_b = video_tensor[b].permute(0, 2, 3, 1).cpu().numpy()

# AFTER (detach before numpy conversion)
video_b = video_tensor[b].permute(0, 2, 3, 1).cpu().detach().numpy()
```

### Expected Outcomes

| Fix | Previous Error | Expected After Fix |
|-----|----------------|-------------------|
| ActionProcessor defaults | `RuntimeError: size mismatch for action_processor.mlp.0.weight` | Checkpoint loads with strict=True, zero missing/unexpected keys |
| Flow append logic | `TypeError: expected Tensor as element 0 in argument 0, but got numpy.ndarray` | Flow predictions succeed in gradient mode, return torch.Tensor |
| Point tracker detach | `RuntimeError: Can't call numpy() on Tensor that requires grad` | CoTracker processes video without error |

### Verification Checkpoint

**Location:** `outputs/dm_control_push8/point_cloud/iteration_20000/deformation.pth`

**Critical parameters:**
- `action_processor.mlp.0.weight`: `[128, 15]` → input_dim=15
- `action_processor.mlp.6.weight`: `[32, 128]` → output_dim=32
- No `freq_bands` key → use_pe=False

---

## TASK BREAKDOWN

### Task 1: Checkpoint Loading Verification
**Priority:** HIGH  
**Estimated Time:** 5 minutes  
**Dependencies:** None

**Objective:** Verify checkpoint loads successfully with fixed ActionProcessor defaults.

**Success Criteria:**
- ✅ Checkpoint file loads without exception
- ✅ Deformation network created successfully
- ✅ `load_state_dict(checkpoint, strict=True)` returns 0 missing keys
- ✅ `load_state_dict(checkpoint, strict=True)` returns 0 unexpected keys
- ✅ ActionProcessor shapes match checkpoint:
  - `mlp.0.weight`: `[128, 15]`
  - `mlp.6.weight`: `[32, 128]`

**Test Script:** `test/verification/test_checkpoint_loading.py`

**QA Scenarios:**

1. **Scenario: Load checkpoint with default config**
   - Input: `ModelParams()` (uses defaults from arguments/toyarm/triplane.py)
   - Expected: load_state_dict succeeds, all keys match
   - Failure mode: RuntimeError on shape mismatch → defaults still wrong

2. **Scenario: Verify ActionProcessor architecture**
   - Input: Created model
   - Expected: 
     - `model.action_processor.mlp[0].in_features == 15`
     - `model.action_processor.mlp[-1].out_features == 32`
     - No positional encoding layers (use_pe=False)
   - Failure mode: Wrong dimensions → config/defaults mismatch

3. **Scenario: Load with explicit config override**
   - Input: `ModelParams(action_input_dim=20)` (intentionally wrong)
   - Expected: RuntimeError on load_state_dict (dimension mismatch)
   - Failure mode: Loads successfully → checkpoint incompatible with overrides

**Evidence Requirements:**
- Console output showing "0 missing keys, 0 unexpected keys"
- Print of ActionProcessor shapes matching checkpoint
- Save to `.sisyphus/evidence/verify-task-1-checkpoint-loading.txt`

---

### Task 2: CEM-GD Gradient Mode Flow Prediction
**Priority:** HIGH  
**Estimated Time:** 10 minutes  
**Dependencies:** Task 1 (checkpoint must load)

**Objective:** Verify flow predictions work in gradient mode (grad_enabled=True).

**Success Criteria:**
- ✅ FlowGuidedGaussianModel.forward() completes without TypeError
- ✅ `predictions['flow']` is `torch.Tensor` (not numpy.ndarray)
- ✅ Flow tensor has `requires_grad=True` when grad_enabled=True
- ✅ Gradient computation succeeds (backward pass)
- ✅ Flow tensor shape correct: `(batch_size, horizon, H, W, 2)`

**Test Script:** `test/verification/test_cemgd_gradient_mode.py`

**QA Scenarios:**

1. **Scenario: Forward pass with grad_enabled=True**
   - Input: Batch of initial states, actions with requires_grad=True
   - Expected: 
     - No TypeError during flow prediction
     - `predictions['flow']` is torch.Tensor
     - `predictions['flow'].requires_grad == True`
   - Failure mode: TypeError at line 729 → flow append still broken

2. **Scenario: Backward pass on flow objective**
   - Input: Flow prediction from scenario 1
   - Expected:
     - `loss = flow_objective(predictions['flow'])` succeeds
     - `loss.backward()` completes without error
     - Action gradients populated (not None)
   - Failure mode: Runtime error on backward → tensor graph broken

3. **Scenario: Multi-step rollout**
   - Input: Horizon=5, batch_size=10
   - Expected:
     - All 5 timesteps produce tensor flow predictions
     - Stacking succeeds: `torch.stack(predictions['flow'], dim=1)`
     - Final shape: `(10, 5, H, W, 2)`
   - Failure mode: Mixed numpy/tensor list → stack fails

**Evidence Requirements:**
- Print flow tensor type, shape, requires_grad status
- Print action gradients after backward pass
- Save to `.sisyphus/evidence/verify-task-2-gradient-mode.txt`

---

### Task 3: Pure CEM Backward Compatibility
**Priority:** HIGH  
**Estimated Time:** 8 minutes  
**Dependencies:** Task 2 (gradient mode must work)

**Objective:** Verify Pure CEM mode (grad_enabled=False) still works identically to pre-fix behavior.

**Success Criteria:**
- ✅ FlowGuidedGaussianModel.forward() with grad_enabled=False completes
- ✅ `predictions['flow']` is `numpy.ndarray` (not torch.Tensor)
- ✅ Flow array has correct shape: `(batch_size, horizon, H, W, 2)`
- ✅ RGB predictions also numpy (existing behavior preserved)
- ✅ CEM optimizer runs without errors

**Test Script:** `test/verification/test_cem_backward_compatibility.py`

**QA Scenarios:**

1. **Scenario: Forward pass with grad_enabled=False**
   - Input: Same batch as Task 2, but grad_enabled=False
   - Expected:
     - `predictions['flow']` is numpy.ndarray
     - `predictions['rgb']` is numpy.ndarray
     - No gradient tracking (no autograd graph)
   - Failure mode: Still returns tensors → conditional broken

2. **Scenario: CEM planning loop**
   - Input: CEM optimizer, 100 samples, 3 iterations
   - Expected:
     - All samples scored successfully
     - CEM refit succeeds (no tensor/numpy mixing errors)
     - Returns valid action sequence
   - Failure mode: TypeError during scoring → numpy conversion broken

3. **Scenario: Compare outputs with baseline**
   - Input: Same random seed, same model, same initial state
   - Expected:
     - Flow predictions numerically identical to pre-fix behavior
     - Action sequence identical to pre-fix behavior
   - Failure mode: Different results → unintended behavior change

**Evidence Requirements:**
- Print flow/rgb types (should be numpy.ndarray)
- Print CEM planning success message
- Save to `.sisyphus/evidence/verify-task-3-backward-compatibility.txt`

---

### Task 4: Point Tracker Integration Test
**Priority:** MEDIUM  
**Estimated Time:** 12 minutes  
**Dependencies:** Task 2 (gradient mode must work)

**Objective:** Verify CoTracker point tracking works with gradient-mode video tensors.

**Success Criteria:**
- ✅ PointTracker.track() accepts video_tensor with requires_grad=True
- ✅ No RuntimeError on `.detach().numpy()` conversion
- ✅ Point tracking produces valid trajectories
- ✅ Tracked points have correct shape: `(batch_size, num_points, horizon, 2)`

**Test Script:** `test/verification/test_point_tracker_gradients.py`

**QA Scenarios:**

1. **Scenario: Track with gradient-enabled video**
   - Input: Video tensor from FlowGuidedGaussianModel (requires_grad=True)
   - Expected:
     - PointTracker.track() completes without error
     - Returns tracked_points (numpy array or detached tensor)
     - No "Can't call numpy() on Tensor that requires grad" error
   - Failure mode: RuntimeError at line 140 → detach() missing

2. **Scenario: Track with gradient-disabled video**
   - Input: Video tensor with requires_grad=False
   - Expected:
     - PointTracker.track() works identically
     - Same output format as scenario 1
   - Failure mode: Different behavior → detach() breaks non-gradient case

3. **Scenario: Integration with CEM-GD optimizer**
   - Input: Full CEM-GD loop with point tracking enabled
   - Expected:
     - Planning completes all iterations
     - Point-based objectives computed successfully
     - Action gradients populated
   - Failure mode: Error during objective computation → integration broken

**Evidence Requirements:**
- Print video tensor requires_grad status
- Print tracked_points shape and type
- Print CEM-GD iteration logs (if full integration)
- Save to `.sisyphus/evidence/verify-task-4-point-tracker.txt`

---

## INTEGRATION TEST (Task 5 - Optional)

### Task 5: End-to-End CEM-GD Planning
**Priority:** LOW (nice-to-have)  
**Estimated Time:** 15 minutes  
**Dependencies:** All previous tasks

**Objective:** Run full CEM-GD planning loop to verify all fixes work together.

**Success Criteria:**
- ✅ Load checkpoint successfully
- ✅ Initialize CEM-GD optimizer
- ✅ Plan trajectory for 3-5 MPC steps
- ✅ All gradient descents complete
- ✅ Final action sequence is valid (within constraints)
- ✅ Rendered images show reasonable motion

**Test Script:** `test/integration/test_cemgd_full_pipeline.py`

**QA Scenarios:**

1. **Scenario: Single MPC step**
   - Input: Initial state, target image, horizon=5
   - Expected:
     - CEM samples (100) + gradient seqs (5) both succeed
     - Best action sequence returned
     - Objective value improves during gradient descent
   - Failure mode: Any error → integration broken

2. **Scenario: Multi-step MPC loop**
   - Input: 3 MPC replanning steps
   - Expected:
     - All 3 steps complete
     - Actions remain within joint limits
     - Trajectory visualizes correctly
   - Failure mode: Error on step 2+ → replanning broken

3. **Scenario: Compare with Pure CEM**
   - Input: Same problem, run both CEM and CEM-GD
   - Expected:
     - CEM-GD achieves better objective (or equal)
     - CEM-GD uses fewer samples (100 vs 1000)
     - Both produce valid trajectories
   - Failure mode: CEM-GD worse quality → gradient optimization broken

**Evidence Requirements:**
- Objective value plot (CEM iterations + gradient descent)
- Final action sequence printout
- Rendered trajectory frames (optional)
- Save to `.sisyphus/evidence/verify-task-5-e2e-pipeline.txt`

---

## EXECUTION ORDER

**Sequential (no parallelization):**

1. **Task 1** (5 min) - Checkpoint loading MUST pass first
2. **Task 2** (10 min) - Gradient mode MUST work before testing compatibility
3. **Task 3** (8 min) - Backward compatibility verification
4. **Task 4** (12 min) - Point tracker integration
5. **Task 5** (15 min, optional) - Full pipeline test

**Total estimated time:** 35 minutes (50 min with Task 5)

**Stop conditions:**
- If Task 1 fails → defaults still wrong, abort and debug
- If Task 2 fails → flow append logic still broken, abort and debug
- If Task 3 fails → backward compatibility violated, abort and redesign
- If Task 4 fails → point tracker detach missing, abort and debug
- Task 5 failure → investigate but may be integration issue unrelated to fixes

---

## TOOLS & ARTIFACTS

### Test Scripts Location
```
test/verification/
├── test_checkpoint_loading.py           # Task 1
├── test_cemgd_gradient_mode.py          # Task 2
├── test_cem_backward_compatibility.py   # Task 3
└── test_point_tracker_gradients.py      # Task 4

test/integration/
└── test_cemgd_full_pipeline.py          # Task 5 (optional)
```

### Evidence Files
```
.sisyphus/evidence/
├── verify-task-1-checkpoint-loading.txt
├── verify-task-2-gradient-mode.txt
├── verify-task-3-backward-compatibility.txt
├── verify-task-4-point-tracker.txt
└── verify-task-5-e2e-pipeline.txt       # Optional
```

### Required Checkpoint
**Path:** `outputs/dm_control_push8/point_cloud/iteration_20000/deformation.pth`

**If missing:**
```bash
# User must train model first or provide checkpoint
python train.py -s data/dm_control_push8 \
  --configs arguments/toyarm/triplane.py \
  --expname "dm_control_push8"
```

---

## ACCEPTANCE CRITERIA

### Minimum (Tasks 1-3 MUST pass):
- ✅ Checkpoint loads without errors
- ✅ CEM-GD gradient mode produces tensors with gradients
- ✅ Pure CEM mode produces numpy arrays (unchanged behavior)

### Full Success (Tasks 1-4):
- ✅ All minimum criteria met
- ✅ Point tracker works with gradient-enabled videos
- ✅ All evidence files generated

### Excellent (Tasks 1-5):
- ✅ All full success criteria met
- ✅ End-to-end CEM-GD planning completes successfully
- ✅ CEM-GD outperforms Pure CEM (fewer samples, better quality)

---

## ROLLBACK PLAN

If any task fails:

1. **Revert commit:** `git revert cb7ad71`
2. **Restore original code:**
   - ActionProcessor defaults: 6, True, 64
   - Flow append: unconditional `.cpu().numpy()`
   - Point tracker: no `.detach()`
3. **Re-analyze root cause** with fresh evidence
4. **Create new fix plan** addressing failure mode

**Do NOT proceed with half-broken fixes.** All tasks must pass or rollback completely.

---

## NOTES

### Why This Verification Matters

1. **Checkpoint compatibility is critical** - Can't use trained models without correct defaults
2. **Gradient mode is the core feature** - CEM-GD useless if gradients don't flow
3. **Backward compatibility is a contract** - Pure CEM users must not break
4. **Integration testing catches edge cases** - Individual fixes may work but fail together

### Common Pitfalls

- **Don't skip Task 3** - Backward compatibility is as important as new features
- **Don't assume success** - Print actual types/shapes, don't trust intuition
- **Don't test with wrong checkpoint** - Must use dm_control_push8/iteration_20000

### Next Steps After Verification

If all tasks pass:
1. Update `.sisyphus/MODIFICATIONS_INDEX.md` with this fix
2. Close out analysis documents (mark complete)
3. User can proceed with CEM-GD experiments

If any task fails:
1. Collect failure evidence
2. Create new debug plan
3. Fix and re-verify (iterate)
