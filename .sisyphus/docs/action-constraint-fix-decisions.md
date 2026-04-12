# Action Constraint Bug Fix - User Decisions & Constraints

**Document Purpose**: Record all critical decisions made during planning phase to guide implementation and serve as reference for future maintenance.

**Plan Reference**: `.sisyphus/plans/fix-action-constraint-bug.md`  
**Created**: 2026-03-19  
**Status**: Decision Log (Wave 0 Task 0.1)

---

## Executive Summary

**Problem**: Joint states (sin/cos representations) are incorrectly clipped to [-1, 1], breaking unit circle constraint (sin²+cos²=1).

**Solution**: Remove state clipping, replace with angular velocity constraint (|Δθ| ≤ max_angular_velocity) between timesteps.

**Key Insight**: Variable named "action" is semantically a **joint STATE vector** [sin(θ₁), cos(θ₁), ..., sin(θ₆), cos(θ₆), grip₁, grip₂, grip₃], not action delta.

---

## User Decisions (5 Critical Questions)

### Decision 1: Angle Wrapping Strategy

**Question**: Should angular velocity computation account for angle wrapping (e.g., 170° → -170° is 20°, not 340°)?

**User Answer**: ✅ **YES** - Use atan2-based wrapping for shortest angular distance

**Implementation**:
```python
delta_angles = angles[..., 1:, :] - angles[..., :-1, :]
delta_angles_wrapped = np.arctan2(np.sin(delta_angles), np.cos(delta_angles))
# Ensures: 170° → -170° gives Δθ = 20° (not 340°)
```

**Rationale**: Prevents false positives when angle crosses ±180° boundary. Critical for continuous rotation trajectories.

**Reference**: Plan section "Task 1: Add Angular Velocity Constraint Functions"

---

### Decision 2: First Timestep Handling

**Question**: How should angular velocity constraint behave at t=0 when no previous state exists?

**User Answer**: ✅ **Skip constraint at t=0** (no previous state to compare)

**Implementation**:
```python
def check_angular_velocity_constraint(actions, action_t_prev=None, max_angular_velocity=0.524):
    if action_t_prev is None:
        # Skip constraint - first timestep has no predecessor
        num_samples = actions.shape[0]
        return (np.ones(num_samples, dtype=bool), np.zeros(num_samples))
    # ... rest of constraint check
```

**Rationale**: Initial state from `transforms.json` is ground truth - no constraint needed.

**Reference**: Plan section "Task 1: Add Angular Velocity Constraint Functions", User Decision 2

---

### Decision 3: LBFGS Optimizer Status

**Question**: Should LBFGS optimizer be fixed (has same clipping bug at line 134)?

**User Answer**: ✅ **Mark as DEPRECATED** - Do NOT fix, user not using it in experiments

**Implementation**:
- Add deprecation warning header in `mpc/lbfgs.py:1-10`
- Document known bug at line 134 (action clipping)
- Update `mpc/AGENTS.md` with deprecation notice
- Recommend CEM/MPPI as maintained alternatives

**Rationale**: LBFGS not used in active research experiments. Fixing would waste effort without user benefit.

**Reference**: Plan section "Task 0.3: Document LBFGS Deprecation"

---

### Decision 4: Angular Velocity Threshold (30°)

**Question**: Is 30° threshold a hardware constraint or experimental preference?

**User Answer**: ✅ **Experimental preference** (configurable parameter, not hardware limit)

**Implementation**:
```python
max_angular_velocity = 0.524  # 30° in radians (30 * π/180) - CONFIGURABLE
# Can be adjusted per experiment via config file
```

**Rationale**: User tested with 30° as reasonable constraint for smooth motion. Not a physical robot limit - can be tuned for different scenarios (e.g., faster motions use 60°, precise control uses 15°).

**Default Value**: `0.524 rad` (30°)  
**Configurable**: YES - expose in optimizer config classes

**Reference**: Plan section "Mathematical Constraints", User Decision 4

---

### Decision 5: Optical Flow Interaction

**Question**: Does angular velocity constraint interact with optical flow mask tracking system?

**User Answer**: ✅ **Independent systems** - Do NOT modify flow code, no cross-dependencies

**Implementation**:
- Keep `utils/flow_utils.py` unchanged
- Keep `external/gmflow/` unchanged
- Angular velocity constraint applies AFTER flow objectives computed
- Optical flow provides guidance, constraint provides feasibility check

**Constraint Application Order**:
```
Optical Flow Objectives → Action Samples → Angular Velocity Check → Gripper Clip → Render
                         (guidance)        (feasibility gate)
```

**Rationale**: Optical flow suggests directions for motion, angular velocity ensures physical feasibility. Systems operate at different stages of optimizer pipeline.

**Reference**: Plan section "Assumptions", Metis Review "Optical Flow Independence"

---

## Mathematical Constraints

### Unit Circle Constraint

**Equation**: sin²(θ) + cos²(θ) = 1

**Tolerance**: 1e-6 (numerical precision threshold)

**Enforcement**: Existing `project_joint_angles()` function in `mpc/constraint_utils.py:5, 22`

**Status**: ✅ Keep existing implementation (correct, just misplaced in pipeline)

**Verification**:
```python
# Check after every optimization step
pairs = actions[..., :12].reshape(..., -1, 2)
sin_vals, cos_vals = pairs[..., 0], pairs[..., 1]
unit_circle_error = np.abs(sin_vals**2 + cos_vals**2 - 1.0)
assert np.all(unit_circle_error < 1e-6), "Unit circle violated"
```

---

### Angular Velocity Constraint

**Equation**: |Δθ| ≤ max_angular_velocity (default: 0.524 rad = 30°)

**Per-Joint Constraint**: Each of 6 joints checked independently

**Temporal Constraint**: Between consecutive timesteps (t and t+1)

**Computation**:
```python
# Step 1: Extract sin/cos pairs (6 joints × 2 = 12 dims)
pairs = actions[..., :12].reshape(..., num_joints, 2)  # (..., 6, 2)
sin_vals, cos_vals = pairs[..., 0], pairs[..., 1]

# Step 2: Compute angles using atan2 (handles all quadrants)
angles = np.arctan2(sin_vals, cos_vals)  # Range: [-π, π]

# Step 3: Compute Δθ with wrapping
delta_angles = angles[..., 1:, :] - angles[..., :-1, :]  # Naive difference
delta_angles_wrapped = np.arctan2(np.sin(delta_angles), np.cos(delta_angles))  # Wrapped to [-π, π]

# Step 4: Check constraint
violations = np.abs(delta_angles_wrapped) > max_angular_velocity  # Per-joint boolean mask
```

**Edge Cases**:
- **Boundary crossing**: 179° → -179° gives Δθ = 2° (not 358°) ✅
- **Multiple wraps**: 170° → -150° → 160° all checked pairwise
- **First timestep**: Skipped (no previous state) ✅

---

### Angle Wrapping Formula

**Problem**: Naive angle difference gives wrong results near ±180° boundary

**Example**:
- Joint at 170° moves to -170°
- Naive: Δθ = -170° - 170° = -340° ❌ (WRONG - suggests huge rotation)
- Wrapped: Δθ = 20° ✅ (CORRECT - small clockwise rotation)

**Solution**: Use atan2 to map to shortest angular distance

**Formula**:
```python
delta_raw = angle_new - angle_old
delta_wrapped = np.arctan2(np.sin(delta_raw), np.cos(delta_raw))
```

**Mathematical Proof**:
```
Given: Δθ_raw = θ₂ - θ₁
Want: Δθ_wrapped ∈ [-π, π] representing shortest rotation

atan2(sin(Δθ_raw), cos(Δθ_raw)) maps any angle to [-π, π] via:
- sin(Δθ_raw): Projects vertical component (preserves direction)
- cos(Δθ_raw): Projects horizontal component (preserves magnitude)
- atan2(): Reconstructs angle in canonical range [-π, π]

Example: Δθ_raw = -340° = -340° + 360° = 20° (modulo 2π)
  sin(-340°) = sin(20°) = 0.342
  cos(-340°) = cos(20°) = 0.940
  atan2(0.342, 0.940) = 20° ✅
```

**Reference**: Plan section "Task 1", Decision 1 (Angle Wrapping)

---

## Action Space Structure

**Total Dimensions**: 15

### Dimensions 0-11: Joint States (NO CLIPPING)

**Structure**: [sin(θ₁), cos(θ₁), sin(θ₂), cos(θ₂), ..., sin(θ₆), cos(θ₆)]

**Semantic Meaning**: Current absolute joint angles (NOT deltas)

**Range**: sin ∈ [-1, 1], cos ∈ [-1, 1], but constrained by sin²+cos²=1

**Constraint**: Unit circle (enforced by `project_joint_angles()`)

**CRITICAL**: **NEVER CLIP** these values - clipping breaks unit circle invariant

**Example Corruption**:
```
Original: [sin=-0.9977839652403836, cos=0.06653689735159651]  # θ ≈ -86°
Clipped:  [sin=-0.8000,            cos=0.06653689735159651]  # INVALID: sin²+cos² = 0.644 ≠ 1
```

**Correct Handling**:
```python
# ❌ WRONG: Clips state values
actions = torch.clamp(actions, -1, 1)

# ✅ CORRECT: Project to unit circle (normalizes)
actions[..., :12] = project_joint_angles(actions[..., :12])
```

---

### Dimensions 12-14: Gripper Controls (CLIP TO [-1, 1])

**Structure**: [grip₁, grip₂, grip₃]

**Semantic Meaning**: Gripper actuator commands (deltas or absolute positions, depends on robot)

**Range**: [-1, 1] (normalized control signals)

**Constraint**: Hard clipping to [-1, 1] (actuator limits)

**Correct Handling**:
```python
# ✅ CORRECT: Clip ONLY gripper dimensions
actions[:, :, 12:15] = torch.clamp(actions[:, :, 12:15], -1, 1)  # CEM (torch)
actions[:, :, 12:15] = np.clip(actions[:, :, 12:15], -1, 1)      # MPPI (numpy)
```

**Rationale**: Gripper actuators have physical limits (fully open = 1, fully closed = -1). Exceeding these values is meaningless and can cause undefined behavior in robot controller.

---

## Implementation Checklist

### Wave 0: Clarification & Baseline (3 tasks)

- [x] **Task 0.1**: Document user decisions (THIS FILE)
- [ ] **Task 0.2**: Capture baseline metrics from buggy code
- [ ] **Task 0.3**: Document LBFGS deprecation

### Wave 1: Foundation Functions (2 tasks)

- [ ] **Task 1**: Add `compute_angular_velocity()`, `check_angular_velocity_constraint()` to `mpc/constraint_utils.py`
  - Implement atan2 wrapping (Decision 1) ✅
  - Handle `action_t_prev=None` (Decision 2) ✅
  - Support both NumPy and Torch backends
  - ~80 lines with docstrings

- [ ] **Task 2**: Create `test/unit/test_angular_velocity_constraint.py`
  - Test basic computation
  - Test wrap-around (170° → -170°)
  - Test first timestep skip
  - Test boundary cases (near ±180°)
  - Test torch/numpy equivalence
  - Test vectorized batch processing

### Wave 2: Core Optimizer Fixes (3 tasks)

- [ ] **Task 3**: Fix CEM (`mpc/cem.py`)
  - Delete: Lines 221, 231 (state clipping)
  - Add: Gripper-only clipping (Decision 5 - independent from flow) ✅
  - Add: Angular velocity constraint check

- [ ] **Task 4**: Fix MPPI (`mpc/mppi.py`)
  - Delete: Line 69 (state clipping)
  - Add: Gripper-only clipping
  - Add: Angular velocity constraint check

- [ ] **Task 5**: Update `mpc/flow_objectives.py`
  - Clarify ActionRegularizationObjective docstring
  - Document action semantics (state, not delta)
  - Reference new constraint functions

### Wave 3: Integration & Docs (3 tasks)

- [ ] **Task 6**: Integration test (`test/integration/test_cotracker_mpc.py`)
  - Verify joint_pos preservation (no -0.998 → -0.8 corruption)
  - Check angular velocities < 30°
  - Verify unit circle (sin²+cos² = 1 within 1e-6)

- [ ] **Task 7**: Update `mpc/AGENTS.md`
  - Document action space structure (Section above)
  - Document angular velocity constraint (Decision 4 - configurable) ✅
  - Add LBFGS deprecation notice (Decision 3) ✅
  - Reference new constraint functions

- [ ] **Task 8**: Create modification summary
  - `.sisyphus/modifications/2026-03-19_action_constraint_bug_fix.md`
  - Problem statement, solution, files changed, verification

### Wave FINAL: Verification (3 tasks)

- [ ] **Task V1**: End-to-end validation with real data
  - Load `/home/ubuntu/project/data/dm_control_push/transforms.json`
  - Verify exact preservation: `joint_pos=-0.9977839652403836` → `action=-0.9977839652403836`
  - Compute angular velocity stats (max, mean, violations)

- [ ] **Task V1b**: Failure case testing
  - Test large Δθ = 60° (should reject samples)
  - Test near-boundary: sin=0.9999, cos=0.0141
  - Test multi-wrap: crossing ±180° multiple times
  - Test optical flow independence (Decision 5): run with/without flow, verify constraint works in both modes ✅

- [ ] **Task V2**: User review checkpoint
  - Present before/after comparison (baseline vs fixed)
  - Show angular velocity statistics
  - Get explicit approval to proceed

---

## Constraint Enforcement Mechanism

### Where Constraints Apply

```
Pipeline Flow (AFTER FIX):

1. Initial Load (transforms.json)
   → No constraints (ground truth state)
   → Values: exact sin/cos from file

2. Optimizer Sampling (CEM/MPPI)
   → Generate action_samples (num_samples, horizon, 15)
   → Values: raw samples from Gaussian/distribution

3. Unit Circle Projection
   → project_joint_angles(actions[..., :12])
   → Ensures sin²+cos²=1 for all samples
   → Values: normalized sin/cos

4. Angular Velocity Check (NEW)
   → check_angular_velocity_constraint(actions, action_t_prev, max_angular_velocity)
   → Filters out samples with |Δθ| > threshold
   → Returns: (valid_mask, penalty_scores)

5. Gripper Clipping (NEW)
   → actions[:, :, 12:15] = torch.clamp(actions[:, :, 12:15], -1, 1)
   → Ensures actuator limits respected
   → Values: gripper dims in [-1, 1]

6. Render
   → render_with_control(actions[valid_mask])
   → Expects absolute state (sin/cos + gripper)
   → All constraints satisfied ✅
```

### Constraint Timing

| Constraint | Stage | Enforcement | Failure Mode |
|------------|-------|-------------|--------------|
| Unit Circle | After sampling | Soft (projection) | Normalize to valid state |
| Angular Velocity | After projection | Hard (rejection) | Discard invalid samples |
| Gripper Limits | After velocity check | Hard (clipping) | Clamp to [-1, 1] |

---

## References

### Plan Structure

- **Wave 0**: Clarification & Baseline (Tasks 0.1-0.3)
- **Wave 1**: Foundation Functions (Tasks 1-2)
- **Wave 2**: Core Optimizer Fixes (Tasks 3-5)
- **Wave 3**: Integration & Docs (Tasks 6-8)
- **Wave FINAL**: Verification (Tasks V1, V1b, V2)

### Related Documents

- **Work Plan**: `.sisyphus/plans/fix-action-constraint-bug.md` (1,276 lines)
- **Baseline Evidence**: `.sisyphus/evidence/action-constraint-fix/baseline-buggy-code.txt` (to be created)
- **Modification Summary**: `.sisyphus/modifications/2026-03-19_action_constraint_bug_fix.md` (to be created)
- **Code Files**: `mpc/constraint_utils.py`, `mpc/cem.py`, `mpc/mppi.py`, `mpc/lbfgs.py`, `mpc/flow_objectives.py`

### Background Sessions

- **Investigation**: Session `ses_2fa7d3c07ffeinY6gmha3iXMGK` (Sisyphus-Junior, 7m 24s)
- **Metis Review**: Session `ses_2fa6dfb82ffe40WT45R0s3ZTlX` (Metis, 2m 54s)

---

## Document Metadata

**Created By**: Sisyphus (Atlas Agent)  
**Session ID**: `ses_304b497d8ffeCUm8Id5SCvztTt`  
**Plan Reference**: `.sisyphus/plans/fix-action-constraint-bug.md`  
**Status**: ✅ COMPLETE (Wave 0 Task 0.1)  
**Next Task**: Task 0.2 (Capture baseline metrics)
