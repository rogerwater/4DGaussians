# Fix Critical Action Constraint Bug - Joint State Clipping

## TL;DR

> **Quick Summary**: Remove incorrect clipping of joint states (sin/cos values) and replace with angular velocity constraint (Δθ) between timesteps. Includes user decision clarifications, baseline capture, and comprehensive failure case testing.
> 
> **Deliverables**:
> - User decisions documented (angle wrapping, first timestep, LBFGS deprecation)
> - Fixed constraint logic in CEM/MPPI optimizers
> - Angular velocity constraint functions with atan2 wrapping
> - Updated ActionRegularizationObjective
> - Verification tests + failure case testing
> - Updated documentation (AGENTS.md, modification summary)
> 
> **Estimated Effort**: Medium (3-4 hours)
> **Parallel Execution**: YES - 4 waves (Wave 0, 1, 2, 3, FINAL)
> **Critical Path**: Wave 0 → Task 1 → Task 3-5 → Task 6 → V1/V1b → V2

---

## Context

### Original Request

User discovered critical bug where "actions" (actually **joint states**) are incorrectly clipped to [-1, 1], breaking unit circle constraints:

**Example from transforms.json**:
```json
"joint_pos": [-0.998, 0.067, -0.485, -0.875, ...]
```

**After processing** (WRONG):
```
Action (first 5 dims): [-0.8000, 0.0665, -0.4847, -0.8000, 0.8000...]
```

The sin=-0.998 got clipped to -0.8, violating sin²+cos²=1!

### User's Key Insight

**"action" is a misnomer** - it's actually a **state vector**:
- 15-dim: [sin(θ₁), cos(θ₁), sin(θ₂), cos(θ₂), ..., sin(θ₆), cos(θ₆), grip₁, grip₂, grip₃]
- Each state maps to specific deformation of 4D Gaussians
- State values (sin/cos) should NEVER be clipped
- Constraint should be on **Δθ between timesteps**, not state magnitude

**User's Example**:
- Joint1 current: [0.5, 0.747] → θ=30°
- Joint1 after action: [0.747, 0.5] → θ=60°
- Δθ = 30° → VALID (if threshold is 30°)
- sin/cos values unchanged, only angle difference checked

### Root Cause Analysis

**Current (Broken) Flow**:
```
transforms.json → initial_control (no clip ✅)
  ↓
CEM/MPPI sampling
  ↓
❌ np.clip(actions, -1, 1)  # WRONG: Breaks sin/cos pairs
  ↓
project_joint_angles()      # TOO LATE: Tries to fix damage
  ↓
render_with_control()       # Expects absolute state, gets corrupted values
```

**Correct Flow** (To Implement):
```
transforms.json → initial_control (preserve exact values)
  ↓
CEM/MPPI sampling
  ↓
project_joint_angles()      # Normalize to unit circle
  ↓
check_angular_velocity()    # NEW: Constrain Δθ ≤ threshold
  ↓
clip_gripper_only()         # NEW: Only clip dims 12-14
  ↓
render_with_control()       # Receives valid states
```

### Research Findings

**Code Analysis** (from investigation):
1. **CEM clipping**: `mpc/cem.py:221, 231` - clips to [-1, 1] BEFORE projection
2. **MPPI clipping**: `mpc/mppi.py:69` - clips to [-1, 1] BEFORE projection
3. **Initial load**: `test/integration/test_cotracker_mpc.py:560` - NO clipping ✅
4. **render_with_control**: `mpc/gaussian_dynamics_model.py:352` - expects **absolute state** [sin(θ), cos(θ), ...]
5. **Unit circle projection**: `mpc/constraint_utils.py:5, 22` - works correctly, but happens AFTER clipping (wrong order)

**Semantic Clarification** (from code comments):
- "action" terminology is misleading
- It's actually a **robot joint state vector**
- render_with_control docstring (line 353): "Control input [sin(θ1), cos(θ1), ..., sin(θ6), cos(θ6), grip1, grip2, grip3]"
- Deformation network expects absolute state, NOT delta

### Metis Review

**Status**: ✅ COMPLETED - 5 Priority 1 questions answered

**User Decisions** (critical for implementation):
1. **Angle wrapping**: YES - Use `atan2(sin(Δθ), cos(Δθ))` for shortest angular distance (prevents 170°→-170° false positives)
2. **First timestep**: Skip constraint at t=0 (no previous state available)
3. **LBFGS optimizer**: Mark as deprecated, do NOT fix (user not using it)
4. **30° threshold**: Experimental preference (configurable parameter, not hardware limit)
5. **Optical flow**: Independent systems (don't modify flow code, no cross-dependencies)

**Critical Gaps Identified** (all addressed in revised plan):
- ❌ Angle wrapping not addressed → ✅ FIXED: Task 1 implements atan2 wrapping
- ❌ First timestep unspecified → ✅ FIXED: Task 1 handles `action_t_prev=None`
- ❌ LBFGS omitted → ✅ FIXED: Wave 0 Task 0.3 documents deprecation
- ❌ Acceptance criteria not executable → ✅ FIXED: All tasks have concrete bash commands
- ❌ Failure cases missing → ✅ FIXED: Task V1b tests edge cases

**Recommendations Implemented**:
- ✅ Added Wave 0 (Clarification & Baseline) before Wave 1
- ✅ Explicit gripper clipping code in Tasks 3, 4 (`actions[:, :, 12:15] = torch.clamp(...)`)
- ✅ Baseline capture from buggy code (Task 0.2) for regression testing
- ✅ Failure case testing (Task V1b): large displacement, boundary states, wraparound trajectories
- ✅ All acceptance criteria agent-executable (grep commands, python imports, test runs)

---

## Work Objectives

### Core Objective

Fix action constraint logic to:
1. **Preserve joint state validity** - Never clip sin/cos values (they represent absolute angles)
2. **Add angular velocity constraint** - Limit |Δθ| between timesteps (e.g., ≤ 30° per step)
3. **Keep unit circle enforcement** - Maintain projection for numerical stability
4. **Preserve optical flow tracking** - No changes to bidirectional flow or motion-driven sampling

### Concrete Deliverables

- `mpc/constraint_utils.py`: Add `compute_angular_velocity()`, `check_angular_velocity_constraint()`
- `mpc/cem.py`: Remove state clipping (lines 221, 231), add angular velocity check
- `mpc/mppi.py`: Remove state clipping (line 69), add angular velocity check
- `mpc/flow_objectives.py`: Update ActionRegularizationObjective with angular velocity penalty
- `test/unit/test_angular_velocity_constraint.py`: New verification script
- Documentation updates: `mpc/AGENTS.md`, modification summary

### Definition of Done

- [ ] `python test/unit/test_angular_velocity_constraint.py` → ALL TESTS PASS
- [ ] transforms.json joint_pos loads without modification (verified in logs)
- [ ] Unit circle constraint maintained: sin²+cos²=1 (verified programmatically)
- [ ] Angular velocity constraint working: |Δθ| < 30° per joint (verified with test trajectories)
- [ ] Integration test runs: `bash run_cotracker_test.sh` → No crashes, valid renders
- [ ] Evidence captured: `.sisyphus/evidence/action-constraint-fix/`

### Must Have

- Angular velocity constraint function (Δθ computation using atan2)
- Remove ALL absolute value clipping of joint dimensions (indices 0-11)
- Keep gripper clipping (indices 12-14 should remain [-1, 1])
- Unit circle projection stays (correct and necessary for numerical stability)
- Test with real transforms.json data (not synthetic)

### Must NOT Have (Guardrails)

- ❌ No changes to optical flow mask tracking (bidirectional flow, motion sampling)
- ❌ No changes to point resampling logic
- ❌ No modification of transforms.json file itself
- ❌ No changes to render_with_control() interface
- ❌ No removal of unit circle projection (it's correct, just misplaced in current flow)
- ❌ No hardcoded angle limits (use configurable threshold parameter)

---

## Verification Strategy

> **ZERO HUMAN INTERVENTION** — ALL verification is agent-executed.

### Test Decision

- **Infrastructure exists**: YES (pytest, research-grade validation)
- **Automated tests**: Tests-after (implement functionality first, then unit tests)
- **Framework**: Python unittest/pytest
- **Agent QA**: MANDATORY for ALL tasks (see QA scenarios below)

### QA Policy

Every task includes agent-executed QA scenarios using:
- **Unit tests**: Bash (python test/unit/test_*.py)
- **Integration tests**: Bash (bash run_cotracker_test.sh)
- **Numerical validation**: Python REPL (import, call functions, assert values)
- **Log inspection**: Read output files, grep for key values

Evidence saved to `.sisyphus/evidence/action-constraint-fix/task-{N}-*.txt`

---

## Execution Strategy

### Parallel Execution Waves

```
Wave 0 (Clarification & Baseline - Start Immediately):
├── Task 0.1: Document user decisions [writing]
├── Task 0.2: Capture baseline metrics from buggy code [quick]
└── Task 0.3: Document LBFGS deprecation [writing]

Wave 1 (Foundation - After Wave 0):
├── Task 1: Add angular velocity functions with atan2 wrapping [quick]
└── Task 2: Create unit test for angular velocity constraint [quick]

Wave 2 (Core Fixes - After Wave 1, MAX PARALLEL):
├── Task 3: Fix CEM optimizer constraint logic [unspecified-high]
├── Task 4: Fix MPPI optimizer constraint logic [unspecified-high]
└── Task 5: Update ActionRegularizationObjective penalty [unspecified-high]

Wave 3 (Integration & Docs - After Wave 2):
├── Task 6: Run integration test and verify [unspecified-high]
├── Task 7: Update mpc/AGENTS.md documentation [writing]
└── Task 8: Create modification summary document [writing]

Wave FINAL (Verification - After ALL tasks):
├── Task V1: End-to-end validation with real data [deep]
├── Task V1b: Failure case testing (edge cases, boundary states) [unspecified-high]
└── Task V2: User review checkpoint [manual]

Critical Path: Wave 0 → Task 1 → Task 3-5 → Task 6 → V1/V1b → V2
Parallel Speedup: ~60% faster than sequential
Max Concurrent: 3 (Waves 0, 2, 3)
Total Tasks: 11 implementation + 3 verification = 14 tasks
```

### Dependency Matrix

| Task  | Depends On   | Blocks        | Wave  |
|-------|--------------|---------------|-------|
| 0.1   | —            | 1             | 0     |
| 0.2   | —            | V1, V1b       | 0     |
| 0.3   | —            | —             | 0     |
| 1     | 0.1          | 2, 3, 4, 5    | 1     |
| 2     | 1            | V1, V1b       | 1     |
| 3     | 1            | 6, V1, V1b    | 2     |
| 4     | 1            | 6, V1, V1b    | 2     |
| 5     | 1            | 6, V1, V1b    | 2     |
| 6     | 3, 4, 5      | V1, V1b       | 3     |
| 7     | —            | —             | 3     |
| 8     | 3, 4, 5      | —             | 3     |
| V1    | 1-8, 0.2     | V2            | FINAL |
| V1b   | 1-8, 0.2     | V2            | FINAL |
| V2    | V1, V1b      | —             | FINAL |

### Agent Dispatch Summary

- **Wave 0**: 3 tasks → Task 0.1, 0.3 (`writing`), Task 0.2 (`quick`)
- **Wave 1**: 2 tasks → Task 1, 2 (`quick`)
- **Wave 2**: 3 tasks → Task 3-5 (`unspecified-high`)
- **Wave 3**: 3 tasks → Task 6 (`unspecified-high`), Task 7-8 (`writing`)
- **Wave FINAL**: 3 tasks → V1 (`deep`), V1b (`unspecified-high`), V2 (user review)
- **Wave FINAL**: 2 tasks → V1 (`deep`), V2 (user review)

---

## TODOs

### Wave 0: Clarification & Baseline (Foundation)

> **Purpose**: Document user decisions, capture baseline metrics from buggy code, clarify LBFGS status
> **Runs before any code changes** - establishes ground truth

- [ ] 0.1. **Document user decisions and system constraints**

  **What to do**:
  - Create file `.sisyphus/docs/action-constraint-fix-decisions.md`
  - Document user's 5 clarification answers:
    1. **Angle wrapping**: YES - Use atan2(sin(Δθ), cos(Δθ)) for shortest angular distance
    2. **First timestep**: Skip constraint at t=0 (no previous state to compare)
    3. **LBFGS optimizer**: Mark as deprecated (do not fix, document in AGENTS.md)
    4. **30° threshold**: Experimental preference (configurable parameter, not hardware limit)
    5. **Optical flow**: Independent systems (don't modify flow code)
  - Document mathematical constraints:
    - Unit circle: sin²(θ) + cos²(θ) = 1 (must be preserved)
    - Angular velocity: |Δθ| ≤ max_angular_velocity (default 0.524 rad = 30°)
    - Angle wrapping: Δθ ∈ [-π, π] via atan2
  - Document action space structure:
    - Dims 0-11: [sin(θ₁), cos(θ₁), ..., sin(θ₆), cos(θ₆)] - joint states (NO CLIPPING)
    - Dims 12-14: [grip₁, grip₂, grip₃] - gripper controls (CLIP to [-1, 1])

  **Must NOT do**:
  - Do not make code changes yet
  - Do not modify any Python files

  **Recommended Agent Profile**:
  - **Category**: `writing`
    - Reason: Pure documentation task
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES (with 0.2, 0.3)
  - **Parallel Group**: Wave 0
  - **Blocks**: All subsequent waves (provides context)
  - **Blocked By**: None

  **References**:
  
  **User's original request** (Chinese):
  - Original issue description about joint_pos clipping
  - Key insight: "action信息是不能对其进行限制的" (state values cannot be clipped)
  
  **Metis review findings**:
  - Session ID: `ses_2fa6dfb82ffe40WT45R0s3ZTlX`
  - 6 critical gaps identified
  - 5 Priority 1 questions (now answered)

  **Acceptance Criteria**:
  - [ ] File exists: `.sisyphus/docs/action-constraint-fix-decisions.md`
  - [ ] All 5 user decisions documented with rationale
  - [ ] Mathematical constraints section present (unit circle, angular velocity, wrapping)
  - [ ] Action space structure clearly defined (dims 0-11 vs 12-14)
  - [ ] grep "atan2" .sisyphus/docs/action-constraint-fix-decisions.md → finds angle wrapping formula

  **QA Scenarios**:
  ```
  Scenario: Decision document completeness
    Tool: Bash (grep)
    Preconditions: Document created
    Steps:
      1. cd /home/ubuntu/yyf/4DGaussians
      2. grep -E "(angle wrapping|first timestep|LBFGS|30°|optical flow)" .sisyphus/docs/action-constraint-fix-decisions.md | wc -l
      3. [ $(wc -l < result) -ge 5 ] # All 5 decisions mentioned
      4. grep "atan2" .sisyphus/docs/action-constraint-fix-decisions.md
    Expected Result: At least 5 matches for decision topics, plus atan2 formula present
    Failure Indicators: Missing any of the 5 decision topics
    Evidence: .sisyphus/evidence/action-constraint-fix/task-0.1-decisions-doc.txt
  ```

  **Commit**: NO (group with 0.2, 0.3)

---

- [ ] 0.2. **Capture baseline metrics from buggy code**

  **What to do**:
  - Run integration test with CURRENT (buggy) code
  - Capture metrics: unit circle violations, clipped values, rendered outputs
  - Save baseline to `.sisyphus/evidence/action-constraint-fix/baseline-buggy-code.txt`
  - Specific metrics to capture:
    1. Initial state from transforms.json: joint_pos values
    2. After clipping: how many values changed (e.g., -0.998 → -0.800)
    3. Unit circle violations: max |sin²+cos²-1| after clipping
    4. Rendered frame quality: PSNR, SSIM (if available)
  - This establishes "before" state for regression testing

  **Must NOT do**:
  - Do not modify any code yet
  - Do not run long experiments (just 1-2 timesteps sufficient)

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Simple script execution and data capture
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES (with 0.1, 0.3)
  - **Parallel Group**: Wave 0
  - **Blocks**: Task V1 (need baseline for comparison)
  - **Blocked By**: None

  **References**:
  
  **Integration test**:
  - `test/integration/test_cotracker_mpc.py:560` - Initial control loading (correct, no clip)
  - `test/integration/test_cotracker_mpc.py:~700-800` - MPC optimization loop
  
  **Data file**:
  - `/home/ubuntu/project/data/dm_control_push/transforms.json` - Real robot states
  
  **Clipping locations** (from background agent):
  - `mpc/cem.py:221, 231` - CEM clipping
  - `mpc/mppi.py:69` - MPPI clipping

  **Acceptance Criteria**:
  - [ ] Baseline file exists: `.sisyphus/evidence/action-constraint-fix/baseline-buggy-code.txt`
  - [ ] File contains: original joint_pos from transforms.json (15 values)
  - [ ] File contains: clipped values after CEM/MPPI processing
  - [ ] File contains: unit circle violation metric (|sin²+cos²-1|)
  - [ ] python -c "import json; d=json.load(open('.sisyphus/evidence/action-constraint-fix/baseline-buggy-code.txt')); assert 'joint_pos_original' in d"

  **QA Scenarios**:
  ```
  Scenario: Baseline capture shows clipping damage
    Tool: Bash (Python snippet)
    Preconditions: Integration test runs with current code
    Steps:
      1. cd /home/ubuntu/yyf/4DGaussians
      2. python test/integration/test_cotracker_mpc.py --steps 2 --save-baseline .sisyphus/evidence/action-constraint-fix/baseline-buggy-code.txt
      3. python -c "import json; d=json.load(open('.sisyphus/evidence/action-constraint-fix/baseline-buggy-code.txt')); print('Max unit circle violation:', d['unit_circle_violation'])"
      4. Expect violation > 0.01 (proves clipping breaks unit circle)
    Expected Result: unit_circle_violation > 0.01, at least 1 joint_pos value clipped
    Failure Indicators: No violations found (means test didn't capture clipping)
    Evidence: .sisyphus/evidence/action-constraint-fix/task-0.2-baseline-capture.txt
  ```

  **Commit**: NO (group with 0.1, 0.3)

---

- [ ] 0.3. **Document LBFGS deprecation decision**

  **What to do**:
  - Update `mpc/AGENTS.md` with LBFGS status
  - Add deprecation notice:
    ```markdown
    ### LBFGS Optimizer (DEPRECATED - DO NOT USE)
    
    **Status**: Not maintained, known bugs (state clipping at line 134)
    
    **Recommendation**: Use CEM or MPPI optimizers instead
    
    **Reason**: User confirmed LBFGS is not used in experiments. Bug fix deferred.
    
    See: .sisyphus/docs/action-constraint-fix-decisions.md for details
    ```
  - Add comment in `mpc/lbfgs.py:1-10` (file header):
    ```python
    """
    DEPRECATED: This optimizer is not actively maintained.
    Known issue: Line 134 clips joint states incorrectly.
    Use CEM (mpc/cem.py) or MPPI (mpc/mppi.py) instead.
    """
    ```

  **Must NOT do**:
  - Do not delete mpc/lbfgs.py (may break imports)
  - Do not fix the clipping bug (user chose deprecation, not fix)

  **Recommended Agent Profile**:
  - **Category**: `writing`
    - Reason: Documentation updates only
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES (with 0.1, 0.2)
  - **Parallel Group**: Wave 0
  - **Blocks**: None (informational only)
  - **Blocked By**: None

  **References**:
  
  **Files to update**:
  - `mpc/AGENTS.md:~100-200` - Optimizer comparison section
  - `mpc/lbfgs.py:1-10` - File header docstring
  
  **LBFGS clipping location** (from background agent):
  - `mpc/lbfgs.py:134` - Same clipping bug as CEM/MPPI

  **Acceptance Criteria**:
  - [ ] `mpc/AGENTS.md` contains "DEPRECATED" section for LBFGS
  - [ ] `mpc/lbfgs.py` has deprecation warning in docstring (lines 1-10)
  - [ ] grep -i "deprecated" mpc/lbfgs.py → finds warning
  - [ ] grep -i "lbfgs.*deprecated" mpc/AGENTS.md → finds deprecation notice

  **QA Scenarios**:
  ```
  Scenario: LBFGS deprecation is visible to users
    Tool: Bash (grep)
    Preconditions: Files updated with deprecation warnings
    Steps:
      1. cd /home/ubuntu/yyf/4DGaussians
      2. grep -n -i "deprecated" mpc/lbfgs.py | head -n 1
      3. line_num=$(extract line number from grep output)
      4. [ $line_num -le 10 ] # Warning must be in first 10 lines (file header)
      5. grep -A 3 "LBFGS.*DEPRECATED" mpc/AGENTS.md
    Expected Result: Deprecation warning in lbfgs.py:1-10, section in AGENTS.md with reason
    Failure Indicators: No "DEPRECATED" found, or warning buried deep in file
    Evidence: .sisyphus/evidence/action-constraint-fix/task-0.3-lbfgs-deprecation.txt
  ```

  **Commit**: YES
  - Message: `docs(mpc): Document decisions and deprecate LBFGS optimizer`
  - Files: `.sisyphus/docs/action-constraint-fix-decisions.md`, `.sisyphus/evidence/action-constraint-fix/baseline-buggy-code.txt`, `mpc/AGENTS.md`, `mpc/lbfgs.py`
  - Pre-commit: `grep -i "deprecated" mpc/lbfgs.py && grep "atan2" .sisyphus/docs/action-constraint-fix-decisions.md`

---

### Wave 1: Foundation Functions (After Wave 0)

- [ ] 1. **Add angular velocity constraint functions to constraint_utils.py**

  **What to do**:
  - Add `compute_angular_velocity(actions, start_idx=0, end_idx=12)` function
    - Extract sin/cos pairs from action tensor: `pairs = actions[..., start_idx:end_idx].reshape(..., -1, 2)`
    - Compute θ = atan2(sin, cos) for each joint at each timestep
    - Compute Δθ with **atan2 wrapping** (user decision 1):
      ```python
      delta_angles = angles[..., 1:, :] - angles[..., :-1, :]
      # Wrap to [-π, π] using: atan2(sin(Δθ), cos(Δθ))
      delta_angles_wrapped = np.arctan2(np.sin(delta_angles), np.cos(delta_angles))
      ```
    - Return shape: (num_samples, horizon-1, num_joints) or (horizon-1, num_joints)
  - Add `check_angular_velocity_constraint(actions, action_t_prev=None, max_angular_velocity=0.524, start_idx=0, end_idx=12)` function
    - **First timestep handling** (user decision 2):
      ```python
      if action_t_prev is None:
          # Skip constraint at t=0, return all valid
          return (np.ones(num_samples, dtype=bool), np.zeros(num_samples))
      ```
    - Prepend action_t_prev to actions: `actions_full = np.concatenate([action_t_prev[None], actions], axis=1)`
    - Call compute_angular_velocity(actions_full)
    - Check if max |Δθ| per sample exceeds threshold
    - Return (valid_mask, max_violations) tuple
  - Make max_angular_velocity a **configurable parameter** (user decision 4: experimental preference, not hardware limit)

  **Must NOT do**:
  - Do not modify existing project_joint_angles functions (they are correct)
  - Do not add clipping inside these functions
  - Do not hardcode the angular velocity threshold (accept as parameter with default=0.524)
  - Do not use naive Δθ = θ₂ - θ₁ without wrapping (will fail at ±180° boundary)

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Self-contained utility functions, no complex dependencies
  - **Skills**: []
    - No specialized skills needed for basic trigonometry

  **Parallelization**:
  - **Can Run In Parallel**: YES (with Task 2)
  - **Parallel Group**: Wave 1
  - **Blocks**: Tasks 3, 4, 5
  - **Blocked By**: Wave 0 (needs decision document for context)

  **References**:
  
  **User decisions** (CRITICAL - read first):
  - `.sisyphus/docs/action-constraint-fix-decisions.md` - All 5 clarification answers
    - Decision 1: Use atan2 wrapping for shortest angular distance
    - Decision 2: Skip constraint at t=0 (action_t_prev=None)
    - Decision 4: 30° (0.524 rad) is experimental preference, make it configurable
  
  **Existing code to preserve**:
  - `mpc/constraint_utils.py:1-36` - Keep existing project_joint_angles functions unchanged
  
  **Mathematical reference**:
  - Angular velocity formula: Δθ = atan2(sin(θ_{t+1} - θ_t), cos(θ_{t+1} - θ_t))
  - This wraps angle differences to [-π, π] range automatically
  - Ensures 170° → -170° gives Δθ=20°, not 340°
  
  **Similar implementation** (for reference, from background agent findings):
  - `mpc/cem.py:143` - Existing delta angle computation (can reference pattern)
  - Uses atan2 for angle extraction and diff for velocity

  **Acceptance Criteria**:
  - [ ] File `mpc/constraint_utils.py` has new functions (lines ~40-120)
  - [ ] Functions handle both 2D (horizon, action_dim) and 3D (num_samples, horizon, action_dim) inputs
  - [ ] Angle wrapping works correctly: 170° → -170° gives Δθ=20°, not 340°
  - [ ] Python import check: `python -c "from mpc.constraint_utils import compute_angular_velocity, check_angular_velocity_constraint; print('✓ Import successful')"`

  **QA Scenarios**:
  ```
  Scenario: Angular velocity computation correctness
    Tool: Python REPL
    Preconditions: constraint_utils.py updated with new functions
    Steps:
      1. python -c "import numpy as np; from mpc.constraint_utils import compute_angular_velocity"
      2. Create test: actions = np.array([[[0.5, 0.866, 0, 0, ...], [0.866, 0.5, 0, 0, ...]]])  # 30°→60°
      3. delta = compute_angular_velocity(actions, start_idx=0, end_idx=2)
      4. expected_delta = 30° * (π/180) = 0.524 rad
      5. assert abs(delta[0,0,0] - 0.524) < 0.01, f"Got {delta[0,0,0]}, expected 0.524"
    Expected Result: Δθ ≈ 0.524 rad (30°), tolerance ±0.01
    Evidence: .sisyphus/evidence/action-constraint-fix/task-1-angular-velocity-basic.txt
  
  Scenario: Angle wrapping validation
    Tool: Python REPL
    Preconditions: Functions implemented
    Steps:
      1. Test case: 170° → -170° (should give +20°, not -340°)
      2. actions = [[sin(170°), cos(170°)], [sin(-170°), cos(-170°)]]
      3. delta = compute_angular_velocity(actions)
      4. assert abs(delta[0] - 0.349) < 0.01, "Wrapping failed"  # 0.349 = 20° in radians
    Expected Result: Δθ = 0.349 rad (20°), NOT -5.93 rad (-340°)
    Evidence: .sisyphus/evidence/action-constraint-fix/task-1-angular-velocity-wrapping.txt
  ```

  **Commit**: NO (group with Task 2)

---

- [ ] 2. **Create unit test for angular velocity constraint**

  **What to do**:
  - Create file `test/unit/test_angular_velocity_constraint.py`
  - Test 1: `test_compute_angular_velocity_basic()` - Simple 30° → 60° case
  - Test 2: `test_angular_velocity_wrapping()` - 170° → -170° edge case
  - Test 3: `test_check_constraint_passes()` - Trajectory within limits
  - Test 4: `test_check_constraint_fails()` - Trajectory exceeds limits
  - Test 5: `test_multiple_joints()` - All 6 joints with different velocities
  - Use unittest.TestCase or simple assert statements

  **Must NOT do**:
  - Do not test the projection functions (already tested elsewhere)
  - Do not test CEM/MPPI integration (that's Task 6)

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Standard unit test writing, no complex setup
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: NO (depends on Task 1)
  - **Parallel Group**: Wave 1
  - **Blocks**: Task V1
  - **Blocked By**: Task 1

  **References**:
  
  **Test file pattern**:
  - `test/unit/test_biflow_functions.py` - Example unittest structure
  - `test/unit/test_joint_angle_projection.py` - Example for constraint testing
  
  **Functions to test**:
  - `mpc/constraint_utils.py:~40` - compute_angular_velocity (new)
  - `mpc/constraint_utils.py:~80` - check_angular_velocity_constraint (new)

  **Acceptance Criteria**:
  - [ ] File exists: `test/unit/test_angular_velocity_constraint.py`
  - [ ] All 5 test cases implemented
  - [ ] Run test: `python test/unit/test_angular_velocity_constraint.py` → exit 0
  - [ ] Test output shows "5 tests passed" or similar success message

  **QA Scenarios**:
  ```
  Scenario: All unit tests pass
    Tool: Bash
    Preconditions: test file created, constraint_utils.py has new functions
    Steps:
      1. cd /home/ubuntu/yyf/4DGaussians
      2. python test/unit/test_angular_velocity_constraint.py 2>&1 | tee test_output.txt
      3. grep -E "(OK|PASSED|5.*pass)" test_output.txt
      4. echo $? # Should be 0 (grep found match)
    Expected Result: "5 tests passed" or ".....OK" (unittest format)
    Failure Indicators: "FAILED", "ERROR", "AssertionError" in output
    Evidence: .sisyphus/evidence/action-constraint-fix/task-2-unit-test-results.txt
  ```

  **Commit**: YES
  - Message: `feat(mpc): Add angular velocity constraint functions and tests`
  - Files: `mpc/constraint_utils.py`, `test/unit/test_angular_velocity_constraint.py`
  - Pre-commit: `python test/unit/test_angular_velocity_constraint.py`

---

###Wave 2: Core Optimizer Fixes (Parallel execution)

- [ ] 3. **Fix CEM optimizer constraint logic**

  **What to do**:
  - **Remove full-spectrum clipping** (WRONG):
    - Line 221: Delete or comment `new_action_samples = torch.clip(new_action_samples, -1, 1)`
    - Line 231: Delete or comment `new_action_samples = np.clip(new_action_samples, -1, 1)`
  - **Keep unit circle projection** (CORRECT - preserve these):
    - Line 222: `project_joint_angles_torch(new_action_samples, ...)` - KEEP UNCHANGED
    - Line 234: `project_joint_angles(new_action_samples, ...)` - KEEP UNCHANGED
  - **Add gripper-only clipping** (NEW - user decision 5):
    - After line 222 (torch branch): `new_action_samples[:, :, 12:15] = torch.clamp(new_action_samples[:, :, 12:15], -1, 1)`
    - After line 234 (numpy branch): `new_action_samples[:, :, 12:15] = np.clip(new_action_samples[:, :, 12:15], -1, 1)`
    - **Gripper dims 12-14**: User decision 5 confirmed these are independent from joint angles
  - **Verify order**: project → gripper_clip, not clip → project
  - **Optional**: Add angular velocity constraint check after projection (penalty, not hard reject)

  **Must NOT do**:
  - Do not remove project_joint_angles calls (lines 222, 234) - they are CORRECT
  - Do not change the CEM algorithm itself (elite selection, mean/variance updates)
  - Do not modify optical flow objectives or sampling (user decision 5: independent systems)

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
    - Reason: Core optimizer logic, requires careful understanding of CEM workflow
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES (with Tasks 4, 5)
  - **Parallel Group**: Wave 2
  - **Blocks**: Task 6, V1
  - **Blocked By**: Task 1

  **References**:
  
  **Critical lines to modify**:
  - `mpc/cem.py:221` - `torch.clip(new_action_samples, -1, 1)` → DELETE
  - `mpc/cem.py:231` - `np.clip(new_action_samples, -1, 1)` → DELETE
  - `mpc/cem.py:46` - `lower_bound, upper_bound = -1, 1` → Document as "gripper bounds only" or remove
  - `mpc/cem.py:327` - Variance constraint may need adjustment
  
  **Keep unchanged**:
  - `mpc/cem.py:222` - `project_joint_angles_torch(...)` - KEEP THIS
  - `mpc/cem.py:234` - `project_joint_angles(...)` - KEEP THIS
  - `mpc/cem.py:143` - Angular velocity penalty computation - KEEP (already exists)
  - `mpc/cem.py:282` - Apply penalty - KEEP
  
  **Background agent findings** (critical context):
  - CEM already has angular velocity penalty at lines 143, 282 (good!)
  - Clipping happens BEFORE projection (wrong order)
  - Fix order: sample → project → (optional) gripper clip

  **Acceptance Criteria**:
  - [ ] Lines 221, 231 deleted or replaced with gripper-only clipping
  - [ ] Unit circle projection calls (222, 234) unchanged
  - [ ] Python syntax check: `python -c "import mpc.cem; print('✓ Import successful')"`
  - [ ] Grep check: `grep -n "clip.*-1.*1" mpc/cem.py` shows only gripper-specific clipping (or none for joints)

  **QA Scenarios**:
  ```
  Scenario: Joint states not clipped during sampling
    Tool: Bash + Python REPL
    Preconditions: cem.py modified, test environment set up
    Steps:
      1. Create minimal CEM test: sample actions with values outside [-0.8, 0.8]
      2. python -c "from mpc.cem import CrossEntropyMethod; import numpy as np; cem = CrossEntropyMethod(...); samples = cem._sample(...)"
      3. Check: assert (samples[:, :, 0:12].max() > 0.9 or samples[:, :, 0:12].min() < -0.9), "Joint states still clipped"
      4. Check: assert samples[:, :, 12:].max() <= 1.0 and samples[:, :, 12:].min() >= -1.0, "Gripper should be clipped"
    Expected Result: Joint dims can exceed [-0.8, 0.8], gripper dims stay within [-1, 1]
    Evidence: .sisyphus/evidence/action-constraint-fix/task-3-cem-no-clipping.txt
  
  Scenario: Unit circle projection still active
    Tool: Python REPL
    Preconditions: cem.py modified
    Steps:
      1. Verify project_joint_angles_torch call exists at line ~222
      2. Verify project_joint_angles call exists at line ~234
      3. python -c "import ast; code=open('mpc/cem.py').read(); assert 'project_joint_angles' in code"
    Expected Result: Both projection calls present, no syntax errors
    Evidence: .sisyphus/evidence/action-constraint-fix/task-3-cem-projection-intact.txt
  ```

  **Commit**: YES
  - Message: `fix(mpc): Remove incorrect joint state clipping from CEM optimizer`
  - Files: `mpc/cem.py`
  - Pre-commit: `python -c "import mpc.cem"`

---

- [ ] 4. **Fix MPPI optimizer constraint logic**

  **What to do**:
  - **Remove full-spectrum clipping** (WRONG):
    - Line 69: Delete or comment `new_action_samples = np.clip(new_action_samples, -1, 1)`
  - **Keep unit circle projection** (CORRECT - preserve this):
    - Line 70: `project_joint_angles(new_action_samples, ...)` - KEEP UNCHANGED
  - **Add gripper-only clipping** (NEW):
    - After line 70: `new_action_samples[:, :, 12:15] = np.clip(new_action_samples[:, :, 12:15], -1, 1)`
    - **Exact array slicing**: `12:15` for gripper dims (3 values)
  - **Verify**: No other clipping locations in mpc/mppi.py (background agent found only line 69)

  **Must NOT do**:
  - Do not remove project_joint_angles call (line 70) - it is CORRECT
  - Do not change MPPI temperature or sampling logic
  - Do not modify optical flow code (user decision 5: independent)

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
    - Reason: Similar to Task 3, core optimizer modification
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES (with Tasks 3, 5)
  - **Parallel Group**: Wave 2
  - **Blocks**: Task 6, V1
  - **Blocked By**: Task 1

  **References**:
  
  **Critical line to modify**:
  - `mpc/mppi.py:69` - `np.clip(new_action_samples, -1, 1)` → DELETE
  
  **Keep unchanged**:
  - `mpc/mppi.py:70` - `project_joint_angles(new_action_samples, ...)` - KEEP THIS
  
  **Similarity to Task 3**:
  - Same pattern as CEM fix
  - MPPI is simpler (only one clipping location found)

  **Acceptance Criteria**:
  - [ ] Line 69 deleted or replaced with gripper-only clipping
  - [ ] Unit circle projection call (line 70) unchanged
  - [ ] Python syntax check: `python -c "import mpc.mppi; print('✓ Import successful')"`
  - [ ] Grep check: `grep -n "clip.*-1.*1" mpc/mppi.py` shows no results or only gripper clipping

  **QA Scenarios**:
  ```
  Scenario: MPPI joint states not clipped
    Tool: Python REPL
    Preconditions: mppi.py modified
    Steps:
      1. python -c "from mpc.mppi import MPPI; import numpy as np; mppi = MPPI(...); samples = mppi._sample(...)"
      2. Check joint dims: assert (samples[:, :, 0:12].max() > 0.9 or samples[:, :, 0:12].min() < -0.9)
      3. Check gripper dims: assert samples[:, :, 12:].max() <= 1.0 and samples[:, :, 12:].min() >= -1.0
    Expected Result: Joint values unrestricted, gripper values in [-1, 1]
    Evidence: .sisyphus/evidence/action-constraint-fix/task-4-mppi-no-clipping.txt
  ```

  **Commit**: YES
  - Message: `fix(mpc): Remove incorrect joint state clipping from MPPI optimizer`
  - Files: `mpc/mppi.py`
  - Pre-commit: `python -c "import mpc.mppi"`

---

- [ ] 5. **Update ActionRegularizationObjective with angular velocity emphasis**

  **What to do**:
  - Review ActionRegularizationObjective class (line 828)
  - Verify it already uses delta-based penalty (line 982 background findings)
  - Add documentation clarifying that "action" means "joint state"
  - Ensure unit circle penalty (line 1040) weight remains at 10.0
  - If magnitude penalty exists (line 1004), document that it should NOT apply to joint dims
  - Add comment: "Joint dimensions (0-11) are sin/cos states, not action deltas. Only penalize gripper magnitude if needed."

  **Must NOT do**:
  - Do not remove existing delta penalty logic
  - Do not add new state magnitude clipping
  - Do not change penalty weights without user approval

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
    - Reason: Complex objective function with 200+ lines, requires careful understanding
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES (with Tasks 3, 4)
  - **Parallel Group**: Wave 2
  - **Blocks**: Task 6, V1
  - **Blocked By**: Task 1 (for understanding angular velocity)

  **References**:
  
  **Critical sections**:
  - `mpc/flow_objectives.py:828-867` - Class definition and __init__
  - `mpc/flow_objectives.py:868-927` - compute_reward method signature
  - `mpc/flow_objectives.py:982` - Delta penalty computation (keep this!)
  - `mpc/flow_objectives.py:1004` - Magnitude penalty start (may need gripper-only guard)
  - `mpc/flow_objectives.py:1040-1045` - Unit circle penalty (keep unchanged)
  
  **Background agent findings**:
  - Delta threshold used at line 982 (correct approach)
  - Unit circle penalty at line 1040 (correct)
  - Magnitude penalty at line 1004 (may need adjustment)
  
  **Desired behavior**:
  - `penalty_type='delta'`: Only penalize Δθ between steps ✓ (already exists)
  - `penalty_type='magnitude'`: Only apply to gripper dims, not joints
  - `penalty_type='both'`: Delta for joints, magnitude for gripper only

  **Acceptance Criteria**:
  - [ ] Documentation added clarifying joint state vs action semantics
  - [ ] Magnitude penalty (if active) only applies to dims 12-14
  - [ ] Unit circle penalty unchanged (weight=10.0)
  - [ ] Delta penalty logic unchanged
  - [ ] Python syntax check: `python -c "from mpc.flow_objectives import ActionRegularizationObjective; print('✓')"`

  **QA Scenarios**:
  ```
  Scenario: Objective still computes delta penalty correctly
    Tool: Python REPL
    Preconditions: flow_objectives.py updated
    Steps:
      1. from mpc.flow_objectives import ActionRegularizationObjective
      2. obj = ActionRegularizationObjective(penalty_type='delta', max_delta=0.5)
      3. Create test prediction: actions with large deltas
      4. reward = obj.compute_reward(prediction, goal)
      5. assert reward < 0, "Large deltas should give negative reward"
    Expected Result: Delta penalty active, reward decreases with larger Δθ
    Evidence: .sisyphus/evidence/action-constraint-fix/task-5-objective-delta-penalty.txt
  
  Scenario: Unit circle penalty still active
    Tool: Python REPL
    Preconditions: flow_objectives.py updated
    Steps:
      1. obj = ActionRegularizationObjective(unit_circle_penalty_weight=10.0)
      2. Create actions violating unit circle: sin=0.8, cos=0.8 (sin²+cos²=1.28≠1)
      3. reward = obj.compute_reward(prediction, goal)
      4. assert reward contains unit circle penalty term
    Expected Result: Penalty increases when sin²+cos²≠1
    Evidence: .sisyphus/evidence/action-constraint-fix/task-5-objective-unit-circle.txt
  ```

  **Commit**: YES
  - Message: `feat(mpc): Clarify ActionRegularizationObjective joint state semantics`
  - Files: `mpc/flow_objectives.py`
  - Pre-commit: `python -c "from mpc.flow_objectives import ActionRegularizationObjective"`

---

- [ ] 6. **Run integration test and verify end-to-end**

  **What to do**:
  - Run `bash run_cotracker_test.sh --num_mpc_steps 5 --horizon 3`
  - Capture full log output to file
  - Verify "Initial control u loaded" message shows original values (not clipped)
  - Verify "Action (first 5 dims)" in subsequent steps shows values outside [-0.8, 0.8] if expected
  - Check that rendering succeeds without NaN/Inf errors
  - Verify output images generated in expected location
  - Compare initial frame joint_pos from transforms.json with logged values

  **Must NOT do**:
  - Do not modify test script itself (unless path/argument errors)
  - Do not skip verification steps to "just run it"
  - Do not ignore warning messages without investigating

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
    - Reason: Complex integration test with many moving parts
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: NO (depends on Tasks 3, 4, 5)
  - **Parallel Group**: Wave 3
  - **Blocks**: V1
  - **Blocked By**: Tasks 3, 4, 5

  **References**:
  
  **Test script**:
  - `run_cotracker_test.sh` - Integration test runner
  - Calls `test/integration/test_cotracker_mpc.py`
  
  **Expected log format**:
  - Initial load: "Initial control u loaded from <frame>"
  - Control values: "Control values (first 5): [-0.998, 0.067, ...]"
  - MPC steps: "Action (first 5 dims): [...]"
  
  **Transforms data location**:
  - `/home/ubuntu/project/data/dm_control_push/transforms.json`
  - Or path specified in test script

  **Acceptance Criteria**:
  - [ ] Test runs without crashes: `bash run_cotracker_test.sh` → exit 0
  - [ ] Log captured: `.sisyphus/evidence/action-constraint-fix/task-6-integration-test-log.txt`
  - [ ] Grep "Initial control u loaded" → values match transforms.json
  - [ ] Output images exist: `ls output/*.png | wc -l` > 0
  - [ ] No NaN/Inf errors in log: `grep -E "(NaN|Inf|nan|inf)" log.txt` → no critical errors

  **QA Scenarios**:
  ```
  Scenario: Integration test passes without errors
    Tool: Bash
    Preconditions: Tasks 3, 4, 5 complete, test environment ready
    Steps:
      1. cd /home/ubuntu/yyf/4DGaussians
      2. bash run_cotracker_test.sh --num_mpc_steps 5 --horizon 3 2>&1 | tee integration_test.log
      3. echo "Exit code: $?"
      4. grep "ERROR" integration_test.log | wc -l
      5. ls -lh output/*.png | head -5
    Expected Result: Exit code 0, no ERROR lines, at least 5 output images
    Failure Indicators: Non-zero exit, "ERROR", "Traceback", no images generated
    Evidence: .sisyphus/evidence/action-constraint-fix/task-6-integration-test-log.txt
  
  Scenario: Joint values preserved from JSON to execution
    Tool: Bash + Python
    Preconditions: Integration test completed
    Steps:
      1. Extract from JSON: python -c "import json; d=json.load(open('transforms.json')); print(d['frames']['frame_00001']['joint_pos'][:5])"
      2. Extract from log: grep "Control values (first 5)" integration_test.log
      3. Compare values: should be IDENTICAL (no -0.998→-0.8 corruption)
      4. python verify_value_preservation.py integration_test.log transforms.json
    Expected Result: Values match exactly (±1e-6 for float precision)
    Evidence: .sisyphus/evidence/action-constraint-fix/task-6-value-preservation.txt
  ```

  **Commit**: NO (verification only)

---

- [ ] 7. **Update mpc/AGENTS.md documentation**

  **What to do**:
  - Update "CONTROL ACTION SPACE" section (lines ~20-70)
  - Clarify that 15-dim vector represents **joint states**, not action commands
  - Update table: Change "Control input" to "Robot State Vector"
  - Add note: "Historical naming: called 'action' in code, but actually represents absolute joint state"
  - Update constraint description: Remove mention of "clipped to [-action_limit, action_limit]"
  - Add angular velocity constraint documentation: "Δθ ≤ threshold between consecutive states"
  - Update example interpretation: Clarify sin/cos → angle conversion
  - Keep gripper documentation as-is (gripper values still clipped to [-1, 1])

  **Must NOT do**:
  - Do not rewrite entire AGENTS.md file
  - Do not change sections unrelated to action/state semantics
  - Do not remove gripper constraint documentation

  **Recommended Agent Profile**:
  - **Category**: `writing`
    - Reason: Documentation update, no code changes
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES (with Task 8, independent of Task 6)
  - **Parallel Group**: Wave 3
  - **Blocks**: None
  - **Blocked By**: None (can start after understanding from Tasks 3-5)

  **References**:
  
  **File to update**:
  - `mpc/AGENTS.md:20-70` - CONTROL ACTION SPACE section
  
  **Current (incorrect) statement** (from background findings):
  - Line 57: "Actions are clipped to [-action_limit, action_limit] per dimension"
  
  **Correct statement** (to replace with):
  - "Joint states (dims 0-11) are sin/cos pairs representing absolute angles, not deltas. No clipping applied."
  - "Gripper values (dims 12-14) are clipped to [-1, 1]."
  - "Angular velocity constraint: |Δθ| ≤ threshold (e.g., 30°/step) between consecutive states."

  **Acceptance Criteria**:
  - [ ] Section "CONTROL ACTION SPACE" updated with correct semantics
  - [ ] No mention of action_limit clipping for joint states
  - [ ] Angular velocity constraint documented
  - [ ] Example updated: sin=-0.998, cos=0.067 → θ≈-86° (keep this pattern)
  - [ ] Markdown syntax valid: `markdown-lint mpc/AGENTS.md` or manual review

  **QA Scenarios**:
  ```
  Scenario: Documentation accurately reflects implementation
    Tool: Bash (grep)
    Preconditions: AGENTS.md updated
    Steps:
      1. grep -i "action_limit" mpc/AGENTS.md | grep -v "gripper" | wc -l
      2. Result should be 0 (no action_limit mentions except for gripper context)
      3. grep "angular velocity" mpc/AGENTS.md
      4. Should find documentation of Δθ constraint
    Expected Result: No misleading action_limit references, angular velocity documented
    Evidence: .sisyphus/evidence/action-constraint-fix/task-7-docs-updated.txt
  ```

  **Commit**: NO (group with Task 8)

---

- [ ] 8. **Create modification summary document**

  **What to do**:
  - Create `.sisyphus/modifications/2026-03-19_action_constraint_bug_fix.md`
  - Document the bug: incorrect state clipping breaking unit circle
  - List all files changed: constraint_utils.py, cem.py, mppi.py, flow_objectives.py, AGENTS.md
  - Explain the fix: removed state clipping, added angular velocity constraint
  - Include before/after examples showing value preservation
  - Add verification evidence summary
  - Link to this plan: `.sisyphus/plans/fix-action-constraint-bug.md`

  **Must NOT do**:
  - Do not duplicate plan content (reference it instead)
  - Do not include entire code diffs (summary only)

  **Recommended Agent Profile**:
  - **Category**: `writing`
    - Reason: Documentation, summary writing
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES (with Task 7, after Tasks 3-5)
  - **Parallel Group**: Wave 3
  - **Blocks**: None
  - **Blocked By**: Tasks 3, 4, 5 (needs to know what was changed)

  **References**:
  
  **Template** (existing modification summaries):
  - `.sisyphus/modifications/2026-03-19_mpc_improvements_and_cudnn_fix.md` - Format reference
  - `.sisyphus/modifications/2026-03-12_bidirectional_flow_tracking_update.md` - Example
  
  **Content to include**:
  - Date: 2026-03-19
  - Issue: Critical bug - joint state clipping
  - Root cause: Semantic confusion (action vs state)
  - Solution: Remove clipping, add angular velocity constraint
  - Files modified: (list with line numbers)
  - Evidence: (list evidence files)
  - Related plan: `.sisyphus/plans/fix-action-constraint-bug.md`

  **Acceptance Criteria**:
  - [ ] File created: `.sisyphus/modifications/2026-03-19_action_constraint_bug_fix.md`
  - [ ] All 5 modified files documented
  - [ ] Before/after example included (transforms.json values)
  - [ ] Evidence files referenced
  - [ ] Markdown valid, readable

  **QA Scenarios**:
  ```
  Scenario: Modification summary complete and accurate
    Tool: Bash (read + grep)
    Preconditions: Summary file created
    Steps:
      1. cat .sisyphus/modifications/2026-03-19_action_constraint_bug_fix.md
      2. grep "constraint_utils.py" <file>
      3. grep "cem.py" <file>
      4. grep "mppi.py" <file>
      5. grep "flow_objectives.py" <file>
      6. grep "AGENTS.md" <file>
      7. wc -l <file>  # Should be substantial (50-100 lines)
    Expected Result: All 5 files mentioned, summary is comprehensive
    Evidence: .sisyphus/evidence/action-constraint-fix/task-8-modification-summary.txt
  ```

  **Commit**: YES
  - Message: `docs(mpc): Document action constraint bug fix and angular velocity constraints`
  - Files: `mpc/AGENTS.md`, `.sisyphus/modifications/2026-03-19_action_constraint_bug_fix.md`
  - Pre-commit: None (documentation only)

---

## Final Verification Wave

- [ ] **V1. End-to-End Validation** — `deep`

  **What to do**:
  - Load real transforms.json from `/home/ubuntu/project/data/dm_control_push/transforms.json`
  - Extract initial joint_pos values
  - Run MPC optimization with new constraints (10 steps, horizon=5)
  - Verify sin/cos values preserved throughout pipeline
  - Verify angular velocities all < 30°
  - Capture log evidence showing constraint satisfaction

  **QA Scenarios**:
  ```
  Scenario: Real data preservation check
    Tool: Python REPL + Bash
    Preconditions: transforms.json exists, test data available
    Steps:
      1. python -c "import json; data=json.load(open('transforms.json')); print(data['frames'][0]['joint_pos'][:5])"
      2. bash run_cotracker_test.sh --num_mpc_steps 5 --horizon 3 2>&1 | tee test.log
      3. grep "Initial control u loaded" test.log
      4. grep "Action (first 5 dims)" test.log
      5. python verify_no_clipping.py test.log transforms.json
    Expected Result: First 5 values match EXACTLY between JSON and log (no -0.998→-0.8 corruption)
    Evidence: .sisyphus/evidence/action-constraint-fix/task-v1-real-data-validation.txt
  
  Scenario: Angular velocity constraint enforcement
    Tool: Python REPL
    Preconditions: constraint_utils.py updated, test trajectory generated
    Steps:
      1. Generate test trajectory: actions = np.random.randn(10, 5, 15) * 0.1
      2. Clip to [-1, 1], project to unit circle
      3. valid, violations = check_angular_velocity_constraint(actions, max_angular_velocity=0.524)
      4. assert violations.max() < 0.524, "Angular velocity constraint violated"
      5. Compute Δθ manually and cross-check
    Expected Result: All Δθ values < 30° (0.524 rad), violations array all zeros
    Evidence: .sisyphus/evidence/action-constraint-fix/task-v1-angular-velocity-check.txt
  ```

  **Commit**: NO (verification only)

---

- [ ] **V1b. Failure Case Testing** — `unspecified-high`

  **What to do**:
  - Test edge cases and failure scenarios that SHOULD be rejected or handled gracefully
  - Ensure constraints correctly identify violations
  - Verify system doesn't accept physically impossible motions

  **QA Scenarios**:
  ```
  Scenario: Large angular displacement rejected
    Tool: Python REPL
    Preconditions: check_angular_velocity_constraint() implemented
    Steps:
      1. Create test trajectory with 60° jump in one step (exceeds 30° limit)
      2. actions = create_test_trajectory(theta_start=0, theta_end=60, steps=1)  # Δθ=60°
      3. valid, violations = check_angular_velocity_constraint(actions, max_angular_velocity=0.524)
      4. assert not valid.all(), "Should reject large angular displacement"
      5. assert violations.max() > 0.524, f"Expected violation >30°, got {violations.max()}"
    Expected Result: valid=False, violations ≈1.047 rad (60°)
    Failure Indicators: Large jump accepted (valid=True)
    Evidence: .sisyphus/evidence/action-constraint-fix/task-v1b-large-displacement-rejected.txt
  
  Scenario: Near-boundary state handling
    Tool: Python REPL
    Preconditions: project_joint_angles() working
    Steps:
      1. Create near-boundary state: sin=0.9999, cos=0.0141 (very close to 90°)
      2. Project to unit circle: projected = project_joint_angles(state)
      3. Verify unit circle: error = abs(projected[0]**2 + projected[1]**2 - 1.0)
      4. assert error < 1e-6, f"Unit circle violated: {error}"
      5. Verify angle preserved: theta = atan2(projected[0], projected[1]) ≈ 1.5567 rad (89.2°)
    Expected Result: sin²+cos²=1 within 1e-6, angle ≈90° preserved
    Failure Indicators: Unit circle error >1e-4, angle significantly changed
    Evidence: .sisyphus/evidence/action-constraint-fix/task-v1b-boundary-state-handling.txt
  
  Scenario: Multiple wrap-around trajectory
    Tool: Python REPL
    Preconditions: compute_angular_velocity() with wrapping
    Steps:
      1. Create trajectory crossing ±180° boundary multiple times:
         angles = [160°, -170°, 150°, -160°, 140°]  # Each step ~20° (within limit)
      2. delta = compute_angular_velocity(angles_to_actions(angles))
      3. Verify each Δθ wrapped correctly: all |delta| < 25° (should be ~20° each)
      4. assert (abs(delta) < 0.436).all(), "Wrapping failed, got naive differences"
    Expected Result: All Δθ ≈0.349 rad (20°), none >180°
    Failure Indicators: Any Δθ >π (unwrapped differences like 330°)
    Evidence: .sisyphus/evidence/action-constraint-fix/task-v1b-wraparound-trajectory.txt
  
  Scenario: Optical flow independence verification
    Tool: Bash + Python
    Preconditions: All fixes applied, optical flow code untouched
    Steps:
      1. Run MPC with optical flow tracking enabled
      2. Check that flow_utils.py, external/gmflow/ unchanged (git diff)
      3. Verify constraint checks don't call flow functions
      4. grep -r "flow_utils" mpc/constraint_utils.py mpc/cem.py mpc/mppi.py
      5. Should return no results (no cross-dependency)
    Expected Result: Flow code untouched, no imports in constraint code
    Failure Indicators: Imports found, flow files modified
    Evidence: .sisyphus/evidence/action-constraint-fix/task-v1b-optical-flow-independence.txt
  ```

  **Commit**: NO (verification only)

---

- [ ] **V2. User Review Checkpoint** — (manual)

  **What to do**:
  - Present consolidated results to user
  - Show before/after comparison of joint value preservation
  - Show angular velocity statistics
  - Show integration test output
  - Get explicit "okay" to proceed or feedback for iteration

  **Must show**:
  - transforms.json joint_pos: [-0.998, ...] → Processed action: [-0.998, ...] (SAME)
  - Max angular velocity across 100 test trajectories: X° (should be < 30°)
  - Unit circle error: max |sin²+cos²-1| = Y (should be < 1e-4)
  - Integration test status: PASS/FAIL

  **Commit**: NO (user review)

---

## Commit Strategy

- **Commit 0**: `docs(mpc): Document decisions and deprecate LBFGS optimizer` — .sisyphus/docs/, .sisyphus/evidence/, mpc/AGENTS.md, mpc/lbfgs.py
- **Commit 1**: `feat(mpc): Add angular velocity constraint functions with atan2 wrapping` — mpc/constraint_utils.py, test/unit/test_angular_velocity_constraint.py
- **Commit 2**: `fix(mpc): Remove incorrect state clipping from CEM optimizer` — mpc/cem.py
- **Commit 3**: `fix(mpc): Remove incorrect state clipping from MPPI optimizer` — mpc/mppi.py
- **Commit 4**: `feat(mpc): Clarify ActionRegularizationObjective joint state semantics` — mpc/flow_objectives.py
- **Commit 5**: `docs(mpc): Document action constraint bug fix and angular velocity constraints` — mpc/AGENTS.md, .sisyphus/modifications/

---

## Success Criteria

### Verification Commands

```bash
# Unit tests pass
python test/unit/test_angular_velocity_constraint.py  # Expected: ALL TESTS PASSED

# Integration test runs without crashes
bash run_cotracker_test.sh --num_mpc_steps 5  # Expected: exit 0, images generated

# Value preservation check
python -c "
import json
import numpy as np
data = json.load(open('/home/ubuntu/project/data/dm_control_push/transforms.json'))
joint_pos = np.array(data['frames']['frame_00001']['joint_pos'])
print(f'Original range: [{joint_pos.min():.4f}, {joint_pos.max():.4f}]')
assert joint_pos.min() >= -1.0 and joint_pos.max() <= 1.0
assert any(abs(joint_pos) > 0.9), 'Should have values outside [-0.8, 0.8]'
print('✓ Joint states preserve full [-1, 1] range')
"
# Expected: ✓ Joint states preserve full [-1, 1] range
```

### Final Checklist

- [ ] All "Must Have" implemented
- [ ] All "Must NOT Have" avoided
- [ ] transforms.json joint_pos values unchanged from load to render
- [ ] Unit circle constraint maintained (sin²+cos²=1 within tolerance)
- [ ] Angular velocity constraint enforced (|Δθ| < threshold)
- [ ] No clipping of joint dimensions (indices 0-11)
- [ ] Gripper dimensions still clipped (indices 12-14)
- [ ] Integration test passes with real data
- [ ] Documentation updated
- [ ] User explicitly approves results
