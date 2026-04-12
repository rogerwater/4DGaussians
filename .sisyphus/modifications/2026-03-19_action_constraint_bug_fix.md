# Action Constraint Bug Fix (2026-03-19)

## Summary

Joint state vectors (sin/cos) were clipped to [-1, 1] in CEM/MPPI before projection, which distorts angles and breaks the unit-circle invariant. The fix removes joint-state clipping and replaces it with angular-velocity constraints (|Δθ| ≤ 30°/step by default), while keeping gripper clipping on dims 12–14.

## Problem Statement

- The control vector is a **state** representation: `[sin(θ₁), cos(θ₁), ..., sin(θ₆), cos(θ₆), grip₁, grip₂, grip₃]`.
- Clipping the joint sin/cos entries changes the angle (even after projection), violating `sin²+cos²=1` and corrupting the state.
- Correct constraint is on **angular velocity** between timesteps (Δθ), not on the state magnitude.

## Solution

1. **Remove joint-state clipping** in CEM/MPPI sampling.
2. **Keep unit-circle projection** for joint pairs.
3. **Add angular-velocity constraint utilities** and reuse them for penalty computation.
4. **Clip gripper only** (dims 12–14).
5. **Deprecate LBFGS** (known clipping bug, not fixed per decision).

## Files Changed

- `mpc/constraint_utils.py`
  - Added `compute_angular_velocity(_torch)` and `check_angular_velocity_constraint(_torch)`.
  - First timestep is skipped when no previous state provided.

- `mpc/cem.py`
  - Removed `torch.clip/np.clip` over joint dims.
  - Added gripper-only clipping in both torch/numpy branches.
  - Reused `check_angular_velocity_constraint` to compute angular-velocity penalty.

- `mpc/mppi.py`
  - Removed joint-state clipping and added gripper-only clipping.

- `mpc/lbfgs.py`
  - Added runtime deprecation warning (action clipping bug, not maintained).

- `mpc/AGENTS.md`
  - Updated action constraint documentation (state semantics, angular velocity, LBFGS deprecation).

## Evidence & Verification

### Baseline Evidence

- File: `.sisyphus/evidence/action-constraint-fix/baseline-buggy-code.txt`
- Synthetic demo shows clip-before-project distorts angle:
  - Project-only angle: **+75.96°**
  - Clip→Project angle: **+73.30°**
  - Distortion: **2.66°**

### End-to-End Data Validation (data-only)

- Script: `.sisyphus/evidence/action-constraint-fix/end_to_end_validation.py`
- Output: `.sisyphus/evidence/action-constraint-fix/end_to_end_validation.txt`

Results (dm_control_push):
- Frames with `joint_pos`: **24000**
- First joint_pos[0]: **-0.9977839652403836**
- Unit-circle max error: **2.22e-16**
- Δθ max: **0.7680 rad**
- Δθ mean: **0.000961 rad**
- Violations (>30°): **8**

**Note**: CEM/MPPI execution not run due to missing numpy/torch in environment. Code-level changes verified by inspection.

### Failure Case Testing (data-only)

- Script: `.sisyphus/evidence/action-constraint-fix/failure_case_testing.py`
- Output: `.sisyphus/evidence/action-constraint-fix/failure_case_testing.txt`

Results:
- `large_delta_60deg`: **valid=False**, penalty **0.5236**
- `near_boundary_sincos`: **valid=True**, penalty **0.0**
- `multi_wrap_180`: **valid=True**, penalty **0.0**
- `optical_flow_independence`: **valid=True**, penalty **0.0**

## Notes / Limitations

- Full integration test `test/integration/test_cotracker_mpc.py` could not run (numpy missing; pip/ensurepip unavailable in this environment).
- Unit tests requiring numpy/torch also blocked by missing deps.
- CEM/MPPI logic changes are localized to sampling/pre-projection stage and gripper clipping.

## Next Steps

1. Install numpy/torch in a proper environment and rerun:
   - `python test/unit/test_angular_velocity_constraint.py`
   - `python test/integration/test_cotracker_mpc.py`
2. Run CEM planner to ensure no `joint_pos` corruption in rendered pipeline.
3. Confirm angular-velocity constraint thresholds match experiment config (default 30°).
