# MPC Planning System Improvements

## TL;DR

> **Quick Summary**: Fix three critical MPC planning issues: (1) Invalid robot joint angles violating sin²+cos²=1, (2) Tracking points clustering on low-motion objects instead of robot arm, (3) Insufficient optimization iterations causing local optima.
> 
> **Deliverables**:
> - Joint angle constraint enforcement via projection + penalty
> - Flow-magnitude-weighted point sampling (1-line bug fix)
> - Increased MPC iterations from 5 → 10
> - Configuration class for DM Control MPC parameters
> - Verification scripts for all three improvements
> 
> **Estimated Effort**: Medium (3-4 hours implementation + 1 hour testing)
> **Parallel Execution**: YES - 3 waves (Task 3 quick → Task 2 medium → Task 1 complex)
> **Critical Path**: Task 3 (5 min) → Task 2 (30 min) → Task 1 (2 hours) → Integration (30 min)

---

## Context

### Original Request

**User's Goal** (Chinese → English):
1. **Problem 1 - Invalid Actions**: MPC produces actions like `[-0.8, -0.8, -0.8, -0.8, 0.8...]` that violate sin²+cos²=1 constraint for joint angle representations. Need to add constraints in `arguments/planning_dmcontrol.py` to enforce unit circle constraint with large penalty.

2. **Problem 2 - Point Distribution**: Optical flow mask is correct, but tracking points cluster on low-motion object (block) instead of high-motion region (robot arm). Need to sample more points in regions with larger flow magnitude.

3. **Problem 3 - Iteration Count**: 5 MPC iterations is insufficient (causing local optima). Increase default to 10.

### Interview Summary

**Key Discussions**:
- **5 parallel background agents** completed comprehensive research (all successful)
- **Critical bug discovered**: `sample_motion_driven_points()` computes flow magnitude weights but doesn't use them (line 488-495)
- **Existing helper found**: `demo_flow_guided_mpc.py` has `normalize_sincos_control()` that can be adapted
- **Academic guidance**: TD-CD-MPPI and BC-MPPI papers recommend hybrid projection + penalty approach

**Research Findings**:
- **Librarian (joint constraints)**: PyTorch F.normalize, projection methods, penalty approaches
- **Librarian (flow sampling)**: 2D importance sampling, temperature parameters, CoTracker3 initialization patterns
- **Explore (code locations)**: Exact file/line numbers for all modifications
- **Explore (MPC parameters)**: All `opt_iters` locations identified
- **Explore (optimization patterns)**: CEM/MPPI sampling pipeline mapped

### Metis Review

**Identified Gaps** (addressed):
- ❓ Constraint tolerance → **RESOLVED: 1e-6** (user confirmed)
- ❓ Penalty weight → **RESOLVED: 10.0** (user confirmed)
- ❓ Normalization order → **RESOLVED: After clipping** (user confirmed)
- ❓ Temperature configuration → **RESOLVED: Configurable parameter** (user confirmed)
- ❓ Config directory contents → **RESOLVED: Create Python config class** (user confirmed)
- ❓ Performance requirements → **RESOLVED: No hard constraints** (user confirmed)

**Guardrails Applied**:
- ✅ No abstract base classes (ConstraintHandler, SamplingStrategy)
- ✅ No extensive logging/visualization beyond verification
- ✅ No refactoring of unrelated code
- ✅ Research code style (minimal comments, 4-space indent)
- ✅ Verification scripts instead of unit tests

---

## Work Objectives

### Core Objective

Implement three independent improvements to the 4DGaussians MPC planning system to fix invalid robot actions, improve tracking point distribution, and reduce local optima in optimization.

### Concrete Deliverables

- **File**: `arguments/planning_dmcontrol/__init__.py` - Configuration class with all parameters
- **File**: `mpc/cem.py` - Projection logic in `score_trajectories()` (~15 lines)
- **File**: `mpc/mppi.py` - Projection logic in `plan()` (~15 lines)
- **File**: `mpc/flow_objectives.py` - Unit circle penalty in `ActionRegularizationObjective` (~30 lines)
- **File**: `mpc/point_sampling.py` - Weighted sampling fix lines 492-500, 1051-1062 (~20 lines)
- **File**: `test_cotracker_mpc.py` - Fix initialization sampling + opt_iters default (~10 lines)
- **File**: `demo_flow_guided_mpc.py` - Update opt_iters defaults (~5 lines)
- **File**: `demo_cotracker_mpc.py` - Update opt_iters default (~2 lines)
- **File**: `verify_constraints.py` - Verification script for Task 1
- **File**: `verify_flow_sampling.py` - Verification script for Task 2
- **File**: `verify_iterations.py` - Verification script for Task 3

### Definition of Done

- [ ] All sin/cos pairs satisfy sin²(θ) + cos²(θ) ≈ 1.0 within 1e-6 tolerance
- [ ] Tracking points concentrated on high-flow regions (robot arm) not block
- [ ] MPC runs 10 iterations per planning step (default, overridable)
- [ ] All verification scripts pass (`python verify_*.py` → exit code 0)
- [ ] Config class `PlanningDMControlParams` successfully loaded
- [ ] No performance regression >20% (smoke test on bouncingballs scene)

### Must Have

- Hard projection of sin/cos pairs to unit circle (after clipping, before scoring)
- Soft penalty term for constraint violations (weight=10.0)
- Weighted sampling with `p=probabilities` in all 3 code paths
- Configurable temperature parameter (`flow_magnitude_exponent`)
- Zero-flow fallback to uniform sampling
- opt_iters default=10 in all demo files
- Verification scripts with specific assertions

### Must NOT Have (Guardrails)

- ❌ Abstract constraint handling framework (ConstraintHandler base class)
- ❌ Alternative sampling strategies beyond weighted random choice
- ❌ Adaptive temperature scheduling
- ❌ Extensive logging/visualization infrastructure
- ❌ Type hints beyond existing minimal usage
- ❌ Refactoring of unmodified code
- ❌ Unit test infrastructure (use verification scripts)
- ❌ Docstrings (follow research code style: inline comments only)

---

## Verification Strategy (MANDATORY)

> **ZERO HUMAN INTERVENTION** — ALL verification is agent-executed via Python scripts.
> Acceptance criteria requiring "user manually tests/confirms" are FORBIDDEN.

### Test Decision

- **Infrastructure exists**: NO (research codebase, no pytest/unittest)
- **Automated tests**: **Verification scripts** (NOT unit tests)
- **Framework**: Python scripts that exit with code 0 on pass, non-zero on failure
- **Validation method**: Direct Python imports + assertions + argparse --help checking

### QA Policy

Every task MUST include verification script with:
- Concrete imports and function calls
- Specific numerical assertions (not "verify it works")
- Exit code 0 on success, non-zero on failure
- Evidence output (print statements showing pass/fail)

---

## Execution Strategy

### Parallel Execution Waves

> Tasks are independent but recommended sequential order minimizes debugging complexity.
> Task 3 is simplest (parameter change), Task 1 is most complex (constraint enforcement).

```
Wave 1 (Quick wins - can start immediately):
└── Task 3: Increase opt_iters default 5→10 [quick] (5 min)

Wave 2 (After Wave 1 - bug fix):
└── Task 2: Flow-weighted sampling [quick] (30 min)

Wave 3 (After Wave 2 - complex constraint logic):
├── Task 1a: Create config class [quick] (15 min)
├── Task 1b: Implement projection [deep] (45 min)
├── Task 1c: Add penalty term [unspecified-high] (30 min)
└── Task 1d: Integrate projection into CEM/MPPI [deep] (30 min)

Wave 4 (After Wave 3 - verification):
├── Task V1: Create verify_iterations.py [quick] (10 min)
├── Task V2: Create verify_flow_sampling.py [unspecified-high] (20 min)
└── Task V3: Create verify_constraints.py [deep] (30 min)

Wave FINAL (After all tasks - integration test):
└── Task F1: Run full MPC pipeline smoke test [unspecified-high] (30 min)

Critical Path: Task 3 → Task 2 → Task 1a → Task 1b → Task 1c → Task 1d → Task V3 → F1
Parallel Speedup: Minimal (most tasks sequential due to integration dependencies)
Max Concurrent: 1 (recommended sequential for easier debugging)
```

### Dependency Matrix

- **Task 3**: — — Task V1
- **Task 2**: — — Task V2
- **Task 1a**: — — Task 1b, 1c
- **Task 1b**: Task 1a — Task 1d
- **Task 1c**: Task 1a — Task 1d
- **Task 1d**: Task 1b, 1c — Task V3
- **Task V1**: Task 3 — F1
- **Task V2**: Task 2 — F1
- **Task V3**: Task 1d — F1
- **F1**: V1, V2, V3 —

### Agent Dispatch Summary

- **Wave 1**: 1 task — Task 3 → `quick`
- **Wave 2**: 1 task — Task 2 → `quick`
- **Wave 3**: 4 tasks — Task 1a → `quick`, Task 1b → `deep`, Task 1c → `unspecified-high`, Task 1d → `deep`
- **Wave 4**: 3 tasks — V1 → `quick`, V2 → `unspecified-high`, V3 → `deep`
- **Wave FINAL**: 1 task — F1 → `unspecified-high`

---

## TODOs

> Implementation + Verification = ONE Task group per wave.
> EVERY task MUST have: Recommended Agent Profile + QA Scenarios + Explicit file/line references.
> **A task WITHOUT QA Scenarios is INCOMPLETE. No exceptions.**

- [ ] 3. **Increase MPC Iteration Count from 5 to 10**

  **What to do**:
  - Change argparse default in `test_cotracker_mpc.py` line 256: `default=5` → `default=10`
  - Change argparse default in `demo_flow_guided_mpc.py` (search for `add_argument.*opt_iters`)
  - Change function parameter default in `demo_cotracker_mpc.py`: `opt_iters: int = 5` → `10`
  - Grep for any shell scripts with hardcoded `--opt_iters 5` (unlikely but check)

  **Must NOT do**:
  - Add iteration scheduling (warmup, annealing)
  - Make iterations dataset-dependent
  - Add convergence-based early stopping
  - Refactor argparse into config classes

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Trivial parameter change, 3 files, search-and-replace task
  - **Skills**: []
    - No special skills needed
  
  **Parallelization**:
  - **Can Run In Parallel**: YES (independent of other tasks)
  - **Parallel Group**: Wave 1 (quickest task, run first)
  - **Blocks**: Task V1 (verification for this task)
  - **Blocked By**: None (can start immediately)

  **References**:

  **Pattern References** (existing code to follow):
  - `test_cotracker_mpc.py:256` - Current argparse pattern: `parser.add_argument("--opt_iters", type=int, default=5, help="CEM optimization iterations")`
  - `demo_flow_guided_mpc.py` - Multiple locations with opt_iters defaults (search with grep)
  - `demo_cotracker_mpc.py` - Function parameter: `def run_cotracker_mpc(..., opt_iters: int = 5)`

  **Why These References Matter**:
  - Shows existing pattern: simply change `5` → `10` in default values
  - Three independent locations to update (no shared constant)
  - Argparse help text should remain unchanged

  **Acceptance Criteria**:

  > **AGENT-EXECUTABLE VERIFICATION ONLY**

  - [ ] `python test_cotracker_mpc.py --help` contains `--opt_iters ... default: 10`
  - [ ] `python demo_flow_guided_mpc.py --help` contains `--opt_iters ... default: 10`
  - [ ] Grep shows no remaining `default=5` for opt_iters in modified files
  - [ ] Smoke test: `python test_cotracker_mpc.py --scene bouncingballs` runs without crash

  **QA Scenarios (MANDATORY)**:

  ```
  Scenario: Verify argparse default changed to 10
    Tool: Bash (grep + --help)
    Preconditions: Files modified
    Steps:
      1. python test_cotracker_mpc.py --help | grep -A1 "opt_iters"
      2. Assert output contains "default: 10"
      3. python demo_flow_guided_mpc.py --help | grep -A1 "opt_iters"
      4. Assert output contains "default: 10"
    Expected Result: Both scripts show default: 10 in help text
    Failure Indicators: "default: 5" still present, or missing opt_iters entirely
    Evidence: .sisyphus/evidence/task-3-argparse-defaults.txt

  Scenario: Verify no hardcoded --opt_iters 5 in shell scripts
    Tool: Bash (grep)
    Preconditions: Repository searched
    Steps:
      1. grep -r "--opt_iters 5" scripts/ *.sh 2>/dev/null || echo "None found"
      2. Assert: No matches found (or manually verify false positives)
    Expected Result: No shell scripts pass --opt_iters 5 explicitly
    Evidence: .sisyphus/evidence/task-3-shell-scripts-check.txt
  ```

  **Evidence to Capture**:
  - [ ] Terminal output of `--help` showing `default: 10` for both scripts
  - [ ] Grep results showing no hardcoded 5 values in scripts

  **Commit**: YES
  - Message: `feat(mpc): Increase default MPC optimization iterations from 5 to 10`
  - Files: `test_cotracker_mpc.py`, `demo_flow_guided_mpc.py`, `demo_cotracker_mpc.py`
  - Pre-commit: `python test_cotracker_mpc.py --help && python demo_flow_guided_mpc.py --help`

- [ ] 2. **Implement Flow-Magnitude-Weighted Point Sampling**

  **What to do**:
  - Fix `mpc/point_sampling.py` line 492-500 in `sample_motion_driven_points()`:
    - Add: `probabilities = weights / weights.sum()` before `np.random.choice`
    - Change: `indices = np.random.choice(..., replace=False, p=probabilities)`
  - Fix `mpc/point_sampling.py` lines 1051-1062 in `update_tracking_points_dynamic()`:
    - Same pattern: compute probabilities from flow_magnitude, add `p=probabilities`
  - Fix `test_cotracker_mpc.py` lines 381-383 (initialization sampling):
    - Replace uniform sampling with weighted sampling or call `sample_motion_driven_points()`
  - Add **zero-flow fallback**: If `weights.sum() == 0`, use uniform sampling
  - Add **configurable temperature**: `weights = flow_magnitude ** (1.0 / temperature)`

  **Must NOT do**:
  - Refactor sampling into separate utility module
  - Add alternative sampling strategies (Gumbel, stratified, blue noise)
  - Implement adaptive temperature scheduling
  - Add sampling diagnostics/visualization beyond verification
  - Create SamplingStrategy abstraction

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Bug fix, 1-line change per location (add `p=probabilities`), weights already computed
  - **Skills**: []
    - No special skills needed
  
  **Parallelization**:
  - **Can Run In Parallel**: YES (independent of Task 1 and 3)
  - **Parallel Group**: Wave 2 (after Task 3 for easier sequential debugging)
  - **Blocks**: Task V2 (verification for this task)
  - **Blocked By**: None (but recommend after Task 3)

  **References**:

  **Pattern References** (existing code):
  - `mpc/point_sampling.py:486-495` - Bug location: weights computed but NOT used
    ```python
    # Line 486-488: weights ARE computed
    weights = flow_magnitude[motion_y, motion_x]
    
    # Line 492-495: BUT NOT USED (BUG!)
    indices = np.random.choice(len(motion_coords), size=motion_points, replace=False)
    # ^^^ MISSING: p=probabilities parameter
    ```
  - `mpc/point_sampling.py:1051-1062` - Second location with same pattern (uniform sampling)
  - `test_cotracker_mpc.py:381-384` - Third location (initialization): `motion_indices = np.random.choice(len(motion_points_candidates), size=num_motion, replace=False)`

  **API/Type References**:
  - NumPy `random.choice` documentation: `p` parameter must sum to 1.0, shape matches choices
  - Flow magnitude shape: `(H, W)` float array from `np.linalg.norm(flow_field, axis=-1)`

  **External References**:
  - Importance sampling algorithm (from librarian research): flatten → normalize → sample → unravel
  - Temperature parameter: exponent to emphasize high-magnitude regions (0.5-2.0 range)

  **Why These References Matter**:
  - Line 488 shows weights ARE already computed correctly - just need to USE them!
  - This is a **bug fix**, not a new feature implementation
  - Same fix applies to 3 independent locations

  **Acceptance Criteria**:

  > **AGENT-EXECUTABLE VERIFICATION ONLY**

  - [ ] `mpc/point_sampling.py` line ~493 contains `p=probabilities` or `p=probs`
  - [ ] `mpc/point_sampling.py` line ~1055 contains `p=probabilities` or `p=probs`
  - [ ] `test_cotracker_mpc.py` line ~382 uses weighted sampling (either inline or via function call)
  - [ ] Zero-flow fallback added: `if weights.sum() == 0: probabilities = uniform`
  - [ ] Temperature parameter added (configurable `flow_magnitude_exponent`)

  **QA Scenarios (MANDATORY)**:

  ```
  Scenario: Verify weighted sampling concentrates on high-flow regions
    Tool: Python script (verify_flow_sampling.py)
    Preconditions: Code modified, synthetic flow map created
    Steps:
      1. Create synthetic flow map: 512x512, one 100x100 hot region (flow=10.0), background (flow=0.1)
      2. Create motion mask covering entire image
      3. Call sample_motion_driven_points() with num_points=1000
      4. Count how many points fall in hot region (10000 pixels out of 262144 total = 3.8% area)
      5. Assert: >30% of points in hot region (if working correctly ~97% due to 100x magnitude difference)
    Expected Result: High-flow region sampled significantly more than uniform (>30% vs 3.8% expected)
    Failure Indicators: ~3.8% (means still uniform sampling), crash on zero flow
    Evidence: .sisyphus/evidence/task-2-weighted-sampling.png (visualization)

  Scenario: Verify zero-flow fallback doesn't crash
    Tool: Python script (verify_flow_sampling.py)
    Preconditions: Code modified
    Steps:
      1. Create all-zero flow map: np.zeros((512, 512, 2))
      2. Create motion mask with some True values
      3. Call sample_motion_driven_points() with num_points=100
      4. Assert: No crash, returns 100 valid points
      5. Check points are uniformly distributed (no clustering)
    Expected Result: Fallback to uniform sampling, no NaN/crash
    Failure Indicators: Crash with "probabilities do not sum to 1" or division by zero
    Evidence: .sisyphus/evidence/task-2-zero-flow-fallback.txt

  Scenario: Verify temperature parameter controls emphasis
    Tool: Python script (verify_flow_sampling.py)
    Preconditions: Code modified with configurable temperature
    Steps:
      1. Create flow map with gradient (0.1 to 10.0)
      2. Sample 1000 points with temperature=0.5 (high emphasis)
      3. Sample 1000 points with temperature=2.0 (low emphasis)
      4. Compute mean flow magnitude at sampled points for each
      5. Assert: mean(temp=0.5) > mean(temp=2.0) * 1.2 (at least 20% higher)
    Expected Result: Lower temperature concentrates on high-flow regions
    Evidence: .sisyphus/evidence/task-2-temperature-tuning.txt
  ```

  **Evidence to Capture**:
  - [ ] Visualization showing point distribution on synthetic flow map
  - [ ] Terminal output showing percentage in high-flow region
  - [ ] Zero-flow test completion without crash
  - [ ] Temperature parameter comparison results

  **Commit**: YES
  - Message: `fix(mpc): Add flow-magnitude-weighted sampling to point selection
    
    Fixes bug where flow magnitude weights were computed but not used in
    np.random.choice. Adds zero-flow fallback and configurable temperature.
    
    Modified 3 locations: sample_motion_driven_points, update_tracking_points_dynamic,
    test_cotracker_mpc initialization.`
  - Files: `mpc/point_sampling.py`, `test_cotracker_mpc.py`
  - Pre-commit: `python verify_flow_sampling.py`

- [ ] 1a. **Create DM Control Configuration Class**

  **What to do**:
  - Create directory `arguments/planning_dmcontrol/`
  - Create file `arguments/planning_dmcontrol/__init__.py` with class `PlanningDMControlParams`
  - Add parameters:
    - `constraint_tolerance: float = 1e-6` - Max violation tolerance for sin²+cos²=1
    - `unit_circle_penalty_weight: float = 10.0` - Penalty weight for constraint violations
    - `enable_projection: bool = True` - Hard projection after clipping
    - `enable_penalty: bool = True` - Soft penalty in objective
    - `flow_magnitude_exponent: float = 1.0` - Temperature for flow-weighted sampling
  - Follow existing config pattern from `arguments/__init__.py` (ParamGroup inheritance)

  **Must NOT do**:
  - Add validation logic in config class (keep it pure data)
  - Create separate configs for each robot (use one generic config)
  - Add complex inheritance hierarchies
  - Add setter/getter methods

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Straightforward config class creation following existing pattern
  - **Skills**: []
    - No special skills needed

  **Parallelization**:
  - **Can Run In Parallel**: YES (independent file creation)
  - **Parallel Group**: Wave 3a (first task in constraint enforcement group)
  - **Blocks**: Task 1b, 1c (both need config parameters)
  - **Blocked By**: None

  **References**:

  **Pattern References** (existing code to follow):
  - `arguments/__init__.py:ModelParams` - ParamGroup inheritance pattern
    ```python
    class ModelParams(ParamGroup):
        def __init__(self, parser, sentinel=False):
            self.parameter_name = default_value
            super().__init__(parser, "Group Name", sentinel)
    ```
  - `arguments/dnerf/__init__.py` - Dataset-specific config example
  - `arguments/__init__.py:OptimizationParams` - Multiple parameter class in one file

  **Why These References Matter**:
  - ParamGroup inheritance required for integration with argparse system
  - `sentinel` parameter pattern must be preserved
  - Parameters are simple attributes, no complex logic

  **Acceptance Criteria**:

  > **AGENT-EXECUTABLE VERIFICATION ONLY**

  - [ ] File `arguments/planning_dmcontrol/__init__.py` exists
  - [ ] `from arguments.planning_dmcontrol import PlanningDMControlParams` succeeds
  - [ ] Class has all 5 parameters with correct defaults
  - [ ] Instantiation succeeds: `params = PlanningDMControlParams(parser=None, sentinel=True)`

  **QA Scenarios (MANDATORY)**:

  ```
  Scenario: Verify config class creation and parameter defaults
    Tool: Python script (verify_constraints.py section)
    Preconditions: File created
    Steps:
      1. import sys; sys.path.insert(0, '.')
      2. from arguments.planning_dmcontrol import PlanningDMControlParams
      3. params = PlanningDMControlParams(parser=None, sentinel=True)
      4. assert params.constraint_tolerance == 1e-6
      5. assert params.unit_circle_penalty_weight == 10.0
      6. assert params.enable_projection == True
      7. assert params.enable_penalty == True
      8. assert params.flow_magnitude_exponent == 1.0
    Expected Result: All assertions pass, no import errors
    Failure Indicators: ImportError, AttributeError, wrong default values
    Evidence: .sisyphus/evidence/task-1a-config-class.txt
  ```

  **Evidence to Capture**:
  - [ ] Terminal output showing successful import and parameter values

  **Commit**: NO (group with 1b, 1c, 1d)

- [ ] 1b. **Implement Joint Angle Projection Logic**

  **What to do**:
  - Create function `project_joint_angles()` in new file `mpc/constraint_utils.py`
  - Function signature: `project_joint_angles(actions: np.ndarray, start_idx=0, end_idx=12) -> np.ndarray`
  - Logic:
    1. Extract sin/cos pairs: `pairs = actions[..., start_idx:end_idx].reshape(..., (end_idx-start_idx)//2, 2)`
    2. Compute norms: `norms = np.linalg.norm(pairs, axis=-1, keepdims=True)`
    3. Normalize: `pairs_normalized = pairs / np.maximum(norms, 1e-6)`
    4. Reshape back: `actions[..., start_idx:end_idx] = pairs_normalized.reshape(..., end_idx-start_idx)`
  - Add PyTorch version: `project_joint_angles_torch()` (same logic, torch.norm)
  - Support both (batch_size, action_dim) and (batch_size, horizon, action_dim) shapes

  **Must NOT do**:
  - Create ConstraintHandler base class
  - Add gradient computation (this is hard projection, no gradients needed)
  - Implement soft normalization (use hard L2 normalization)
  - Add per-joint weighting
  - Refactor into multiple utility functions

  **Recommended Agent Profile**:
  - **Category**: `deep`
    - Reason: Requires careful handling of tensor shapes, numpy/torch dual implementation
  - **Skills**: []
    - No special skills needed

  **Parallelization**:
  - **Can Run In Parallel**: NO (depends on 1a for config parameters)
  - **Parallel Group**: Wave 3b (after config creation)
  - **Blocks**: Task 1d (integration needs this function)
  - **Blocked By**: Task 1a (needs config class)

  **References**:

  **Pattern References** (existing code to adapt):
  - `demo_flow_guided_mpc.py:233-242` - `normalize_sincos_control()` helper function
    ```python
    def normalize_sincos_control(control_vec):
        # Extract sin/cos, compute angles via atan2, reconstruct normalized pairs
        for i in range(6):
            sin_val = control_vec[i*2]
            cos_val = control_vec[i*2+1]
            angle = np.arctan2(sin_val, cos_val)
            control_vec[i*2] = np.sin(angle)
            control_vec[i*2+1] = np.cos(angle)
    ```
    - **NOTE**: This uses atan2 approach. Our approach is simpler (direct L2 normalization)

  **API/Type References**:
  - `np.linalg.norm(x, axis=-1, keepdims=True)` - Compute L2 norm along last dimension
  - `torch.norm(x, p=2, dim=-1, keepdim=True)` - PyTorch equivalent
  - Action tensor shapes: `(num_samples, horizon, 15)` in CEM, `(num_rollouts, horizon, 15)` in MPPI

  **External References**:
  - PyTorch F.normalize: `F.normalize(pairs, p=2, dim=-1, eps=1e-12)` - Alternative implementation
  - TD-CD-MPPI paper section 3.2: Projection onto constraint manifolds

  **Why These References Matter**:
  - Existing `normalize_sincos_control()` shows problem is understood, but uses atan2 (slower)
  - Direct L2 normalization is faster and equivalent for unit circle
  - Must handle both numpy (MPPI uses numpy rollouts) and torch (CEM uses torch tensors)
  - Shape handling critical: function must work on 2D and 3D tensors

  **Acceptance Criteria**:

  > **AGENT-EXECUTABLE VERIFICATION ONLY**

  - [ ] File `mpc/constraint_utils.py` exists with both functions
  - [ ] Both functions handle 2D input: `(batch_size, 15)` → `(batch_size, 15)`
  - [ ] Both functions handle 3D input: `(batch_size, horizon, 15)` → `(batch_size, horizon, 15)`
  - [ ] Output satisfies: `sin[i]² + cos[i]² ≈ 1.0` within 1e-6 for all pairs
  - [ ] Indices 12-14 (gripper) unchanged

  **QA Scenarios (MANDATORY)**:

  ```
  Scenario: Verify projection enforces unit circle constraint (NumPy)
    Tool: Python script (verify_constraints.py)
    Preconditions: mpc/constraint_utils.py created
    Steps:
      1. import numpy as np; from mpc.constraint_utils import project_joint_angles
      2. Create invalid action: actions = np.array([[-0.8, -0.8, -0.8, -0.8, 0.8, 0.8, -0.6, -0.6, 0.5, 0.5, 0.9, 0.9, 0.0, 0.0, 0.0]])
      3. projected = project_joint_angles(actions.copy())
      4. For each pair (i=0,2,4,6,8,10): assert abs(projected[0,i]**2 + projected[0,i+1]**2 - 1.0) < 1e-6
      5. assert np.allclose(projected[0, 12:15], actions[0, 12:15])  # Gripper unchanged
    Expected Result: All 6 sin/cos pairs normalized, max violation < 1e-6, gripper unchanged
    Failure Indicators: Violation > 1e-6, gripper modified, shape mismatch
    Evidence: .sisyphus/evidence/task-1b-projection-numpy.txt

  Scenario: Verify projection handles batch + horizon shape (PyTorch)
    Tool: Python script (verify_constraints.py)
    Preconditions: mpc/constraint_utils.py created
    Steps:
      1. import torch; from mpc.constraint_utils import project_joint_angles_torch
      2. Create batch: actions = torch.randn(32, 10, 15) * 2.0  # 32 samples, 10 horizon
      3. projected = project_joint_angles_torch(actions.clone())
      4. pairs = projected[..., :12].view(32, 10, 6, 2)
      5. norms = (pairs**2).sum(dim=-1)  # Should be all 1.0
      6. assert torch.allclose(norms, torch.ones_like(norms), atol=1e-6)
      7. assert projected.shape == (32, 10, 15)
    Expected Result: All pairs normalized, shape preserved
    Evidence: .sisyphus/evidence/task-1b-projection-torch.txt

  Scenario: Verify edge case - zero vector input
    Tool: Python script (verify_constraints.py)
    Preconditions: mpc/constraint_utils.py created
    Steps:
      1. from mpc.constraint_utils import project_joint_angles
      2. actions = np.zeros((1, 15))  # All zeros
      3. projected = project_joint_angles(actions.copy())
      4. Assert no NaN: assert not np.any(np.isnan(projected))
      5. Assert no Inf: assert not np.any(np.isinf(projected))
    Expected Result: No crash, output has no NaN/Inf (epsilon prevents division by zero)
    Failure Indicators: NaN, Inf, or crash
    Evidence: .sisyphus/evidence/task-1b-projection-edge.txt
  ```

  **Evidence to Capture**:
  - [ ] Max constraint violation across 1000 random samples
  - [ ] Shape verification output
  - [ ] Edge case handling confirmation

  **Commit**: NO (group with 1a, 1c, 1d)

- [ ] 1c. **Add Unit Circle Penalty to Action Regularization**

  **What to do**:
  - Modify `mpc/flow_objectives.py` class `ActionRegularizationObjective` (line 828)
  - In `__call__` method, after existing regularization, add unit circle penalty:
    ```python
    # After line ~840: existing action regularization
    if hasattr(self, 'unit_circle_penalty_weight') and self.unit_circle_penalty_weight > 0:
        sin_vals = actions[..., 0:12:2]  # Extract sin (indices 0,2,4,6,8,10)
        cos_vals = actions[..., 1:12:2]  # Extract cos (indices 1,3,5,7,9,11)
        unit_error = (sin_vals**2 + cos_vals**2 - 1.0)**2  # Squared error
        penalty = unit_error.sum(dim=[-1, -2])  # Sum over time & joints
        reward = reward - self.unit_circle_penalty_weight * penalty
    ```
  - Add `unit_circle_penalty_weight` parameter to `__init__` (default 10.0)
  - Load weight from `PlanningDMControlParams` config when available

  **Must NOT do**:
  - Create separate ConstraintPenalty class
  - Add per-joint penalty weights
  - Implement adaptive penalty weight scheduling
  - Add penalty visualization/logging beyond verification

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
    - Reason: Requires understanding objective composition, reward vs penalty direction
  - **Skills**: []
    - No special skills needed

  **Parallelization**:
  - **Can Run In Parallel**: YES (with Task 1b - they modify different files)
  - **Parallel Group**: Wave 3b (after config creation)
  - **Blocks**: Task 1d (integration needs penalty active)
  - **Blocked By**: Task 1a (needs config class)

  **References**:

  **Pattern References** (existing code to extend):
  - `mpc/flow_objectives.py:828-850` - `ActionRegularizationObjective.__call__`
    ```python
    def __call__(self, actions, observations, info_dict):
        # Line ~835: action_limit clipping check
        # Line ~840: L2 regularization on actions
        reward = -self.weight * (actions**2).sum(dim=[-1, -2])
        return reward  # Negative because it's a penalty
    ```
  - `mpc/flow_objectives.py:810-827` - `__init__` method shows how to add parameters

  **API/Type References**:
  - Actions shape: `(num_samples, horizon, action_dim)` - penalty must sum over last 2 dims
  - Reward sign: Penalty → subtract from reward (reward = reward - penalty)
  - Config loading pattern: Check if `PlanningDMControlParams` exists in config

  **Why These References Matter**:
  - Existing regularization shows correct tensor dimension handling
  - Reward is NEGATIVE of penalty (more negative = worse)
  - Must preserve existing behavior when penalty weight is 0
  - Sum over both time (horizon) and joint dimensions

  **Acceptance Criteria**:

  > **AGENT-EXECUTABLE VERIFICATION ONLY**

  - [ ] `ActionRegularizationObjective.__init__` has `unit_circle_penalty_weight` parameter
  - [ ] `ActionRegularizationObjective.__call__` computes unit circle penalty
  - [ ] Penalty is 0 when weight=0
  - [ ] Penalty is >0 when actions violate constraint
  - [ ] Existing action regularization still works (backward compatible)

  **QA Scenarios (MANDATORY)**:

  ```
  Scenario: Verify penalty is zero for valid actions
    Tool: Python script (verify_constraints.py)
    Preconditions: flow_objectives.py modified
    Steps:
      1. from mpc.flow_objectives import ActionRegularizationObjective
      2. obj = ActionRegularizationObjective(weight=0.0, unit_circle_penalty_weight=10.0)
      3. Create valid actions: sin/cos pairs all normalized (sin²+cos²=1)
      4. reward = obj(valid_actions, observations=None, info_dict={})
      5. Assert penalty contribution ≈ 0 (within 1e-5)
    Expected Result: No penalty for valid actions
    Failure Indicators: Non-zero penalty for valid actions
    Evidence: .sisyphus/evidence/task-1c-penalty-valid.txt

  Scenario: Verify penalty increases with constraint violation
    Tool: Python script (verify_constraints.py)
    Preconditions: flow_objectives.py modified
    Steps:
      1. from mpc.flow_objectives import ActionRegularizationObjective
      2. obj = ActionRegularizationObjective(weight=0.0, unit_circle_penalty_weight=10.0)
      3. Create invalid actions: [0.5, 0.5, ...] repeated (sin²+cos²=0.5, error=0.5)
      4. reward = obj(invalid_actions, observations=None, info_dict={})
      5. Compute expected penalty: 6 joints * horizon * 0.5² * 10.0
      6. Assert abs(reward - (-expected_penalty)) < 0.01
    Expected Result: Penalty proportional to violation magnitude
    Failure Indicators: Penalty is 0, or wrong magnitude
    Evidence: .sisyphus/evidence/task-1c-penalty-invalid.txt

  Scenario: Verify backward compatibility (weight=0 → no penalty)
    Tool: Python script (verify_constraints.py)
    Preconditions: flow_objectives.py modified
    Steps:
      1. from mpc.flow_objectives import ActionRegularizationObjective
      2. obj = ActionRegularizationObjective(weight=0.1, unit_circle_penalty_weight=0.0)
      3. Create invalid actions (violate constraint)
      4. reward = obj(invalid_actions, observations=None, info_dict={})
      5. Assert reward only from L2 regularization, no constraint penalty
    Expected Result: Penalty disabled when weight=0
    Evidence: .sisyphus/evidence/task-1c-penalty-disabled.txt
  ```

  **Evidence to Capture**:
  - [ ] Penalty values for valid vs invalid actions
  - [ ] Backward compatibility confirmation

  **Commit**: NO (group with 1a, 1b, 1d)

- [ ] 1d. **Integrate Projection into CEM and MPPI Optimizers**

  **What to do**:
  - Modify `mpc/cem.py` in `score_trajectories()` method (~line 340):
    - **AFTER** clipping actions to [-1, 1] (line ~338)
    - **BEFORE** calling model (line ~350)
    - Add: `from mpc.constraint_utils import project_joint_angles_torch`
    - Add: `actions = project_joint_angles_torch(actions, start_idx=0, end_idx=12)`
  - Modify `mpc/mppi.py` in `plan()` method (similar location):
    - **AFTER** clipping actions
    - **BEFORE** scoring trajectories
    - Add: `from mpc.constraint_utils import project_joint_angles` (NumPy version)
    - Add: `actions = project_joint_angles(actions, start_idx=0, end_idx=12)`
  - Load config: Check if scene uses `planning_dmcontrol` config, only apply if enabled

  **Must NOT do**:
  - Refactor clipping logic into separate function
  - Add projection to sampler classes (projection happens after sampling+clipping)
  - Implement gradient-aware projection (this is hard projection, no gradients)
  - Add per-step projection inside the optimization loop (once per scoring is enough)

  **Recommended Agent Profile**:
  - **Category**: `deep`
    - Reason: Requires understanding optimizer flow, correct insertion point, numpy vs torch
  - **Skills**: []
    - No special skills needed

  **Parallelization**:
  - **Can Run In Parallel**: NO (depends on 1b projection function)
  - **Parallel Group**: Wave 3c (after projection implementation)
  - **Blocks**: Task V3 (verification for this task group)
  - **Blocked By**: Task 1b (needs projection function), Task 1c (penalty should be active)

  **References**:

  **Pattern References** (exact insertion points):
  - `mpc/cem.py:330-350` - `score_trajectories()` method
    ```python
    # Line ~338: Clipping happens here
    actions = torch.clamp(actions, -self.action_limit, self.action_limit)
    
    # INSERT PROJECTION HERE (line ~340)
    # actions = project_joint_angles_torch(actions)
    
    # Line ~350: Model call happens here
    predicted_states = self.model(...)
    ```
  - `mpc/mppi.py` - Similar pattern in `plan()` method (find clipping, insert after)
  - `demo_flow_guided_mpc.py:284-295` - Example of post-clipping processing

  **API/Type References**:
  - CEM uses **torch.Tensor**: Shape `(num_samples, horizon, action_dim)`
  - MPPI uses **np.ndarray**: Shape `(num_rollouts, horizon, action_dim)`
  - Action limit: `self.action_limit` (usually 1.0)

  **Config Loading Pattern**:
  - Check scene config: `if hasattr(args, 'planning_dmcontrol_params') and args.planning_dmcontrol_params.enable_projection:`

  **Why These References Matter**:
  - Insertion point CRITICAL: Must be after clipping (to respect bounds) but before model (to ensure valid inputs)
  - NumPy vs PyTorch: CEM optimizer uses PyTorch, MPPI uses NumPy
  - Config check prevents applying constraints to non-robot scenes

  **Acceptance Criteria**:

  > **AGENT-EXECUTABLE VERIFICATION ONLY**

  - [ ] `mpc/cem.py` calls `project_joint_angles_torch()` after clipping, before model
  - [ ] `mpc/mppi.py` calls `project_joint_angles()` after clipping, before scoring
  - [ ] Import statements added at top of both files
  - [ ] Config check added (only apply if planning_dmcontrol config active)
  - [ ] Smoke test: `python test_cotracker_mpc.py --scene bouncingballs` runs without crash

  **QA Scenarios (MANDATORY)**:

  ```
  Scenario: Verify CEM produces valid actions after projection
    Tool: Python script (verify_constraints.py)
    Preconditions: mpc/cem.py modified, constraint_utils.py exists
    Steps:
      1. Set up minimal CEM optimizer with planning_dmcontrol config
      2. Run one planning step: action_seq = cem.plan(obs, horizon=5)
      3. Extract all actions from returned sequence
      4. For each sin/cos pair: assert sin²+cos² within 1e-6 of 1.0
      5. Check 100 random actions from the sequence
    Expected Result: All actions satisfy constraint
    Failure Indicators: Any action violates constraint, crash during planning
    Evidence: .sisyphus/evidence/task-1d-cem-validation.txt

  Scenario: Verify MPPI produces valid actions after projection  
    Tool: Python script (verify_constraints.py)
    Preconditions: mpc/mppi.py modified, constraint_utils.py exists
    Steps:
      1. Set up minimal MPPI optimizer with planning_dmcontrol config
      2. Run one planning step: action_seq = mppi.plan(obs, horizon=5)
      3. Extract all actions from returned sequence
      4. For each sin/cos pair: assert sin²+cos² within 1e-6 of 1.0
      5. Check 100 random actions from the sequence
    Expected Result: All actions satisfy constraint
    Failure Indicators: Any action violates constraint, crash during planning
    Evidence: .sisyphus/evidence/task-1d-mppi-validation.txt

  Scenario: Verify non-planning_dmcontrol scenes unaffected
    Tool: Python script (verify_constraints.py)
    Preconditions: mpc/cem.py modified with config check
    Steps:
      1. Run bouncingballs scene (does not use planning_dmcontrol config)
      2. Verify no projection applied (actions can violate constraint)
      3. Verify no crash, normal execution
    Expected Result: Non-robot scenes work as before
    Failure Indicators: Crash, unexpected projection applied
    Evidence: .sisyphus/evidence/task-1d-backward-compat.txt
  ```

  **Evidence to Capture**:
  - [ ] Constraint satisfaction statistics for CEM
  - [ ] Constraint satisfaction statistics for MPPI
  - [ ] Backward compatibility test output

  **Commit**: YES (group 1a-1d together)
  - Message: `feat(mpc): Add joint angle constraint enforcement for DM Control scenes
    
    Implements hybrid projection + penalty approach:
    - Hard projection: Normalize sin/cos pairs to unit circle after clipping
    - Soft penalty: Add constraint violation penalty (weight=10.0) to objectives
    - Config class: PlanningDMControlParams with all constraint parameters
    
    Modified files:
    - arguments/planning_dmcontrol/__init__.py (new config class)
    - mpc/constraint_utils.py (projection functions)
    - mpc/flow_objectives.py (penalty term in ActionRegularizationObjective)
    - mpc/cem.py (integrate projection in score_trajectories)
    - mpc/mppi.py (integrate projection in plan)
    
    Resolves invalid robot actions violating sin²+cos²=1 constraint.`
  - Files: `arguments/planning_dmcontrol/__init__.py`, `mpc/constraint_utils.py`, `mpc/flow_objectives.py`, `mpc/cem.py`, `mpc/mppi.py`
  - Pre-commit: `python verify_constraints.py`

- [ ] V1. **Create verify_iterations.py Script**

  **What to do**:
  - Create Python script `verify_iterations.py` in project root
  - Test argparse defaults:
    - Import argparse from test_cotracker_mpc and demo_flow_guided_mpc
    - Use `--help` output parsing to verify `default: 10`
  - Test function defaults:
    - Import demo_cotracker_mpc functions, check signature default values
  - Exit code 0 on pass, 1 on failure
  - Print clear PASS/FAIL message

  **Must NOT do**:
  - Use unittest/pytest framework (standalone script)
  - Add extensive test infrastructure
  - Test beyond iteration count (no integration testing here)

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Simple script, subprocess --help checks
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES (with V2, V3 - independent verification)
  - **Parallel Group**: Wave 4 (verification wave)
  - **Blocks**: Task F1 (final integration test)
  - **Blocked By**: Task 3 (must have modified files)

  **References**:

  **Pattern References**:
  - Subprocess pattern: `subprocess.run(["python", "test_cotracker_mpc.py", "--help"], capture_output=True, text=True)`
  - Argparse help parsing: `if "default: 10" in result.stdout:`

  **Why These References Matter**:
  - Simple pattern: run --help, parse output, check for "default: 10"
  - No need to actually run MPC (too slow for verification script)

  **Acceptance Criteria**:

  - [ ] File `verify_iterations.py` exists and is executable
  - [ ] `python verify_iterations.py` exits with code 0 if all defaults are 10
  - [ ] Script prints which files pass/fail
  - [ ] Script checks all 3 files: test_cotracker_mpc.py, demo_flow_guided_mpc.py, demo_cotracker_mpc.py

  **QA Scenarios (MANDATORY)**:

  ```
  Scenario: Verify script detects correct defaults
    Tool: Bash
    Preconditions: verify_iterations.py created, Task 3 completed
    Steps:
      1. python verify_iterations.py
      2. Check exit code: echo $?
      3. Assert exit code == 0
      4. Check output contains "PASS" for all 3 files
    Expected Result: Exit 0, "PASS" messages
    Failure Indicators: Exit 1, "FAIL" messages, crashes
    Evidence: .sisyphus/evidence/task-v1-verify-iterations.txt
  ```

  **Evidence to Capture**:
  - [ ] Script output showing PASS for all files

  **Commit**: NO (verification scripts committed separately)

- [ ] V2. **Create verify_flow_sampling.py Script**

  **What to do**:
  - Create Python script `verify_flow_sampling.py` in project root
  - Implement 3 test scenarios:
    1. **Weighted sampling test**: Create synthetic flow (hot region 100x100 with flow=10.0, rest=0.1), sample 1000 points, assert >30% in hot region
    2. **Zero-flow fallback**: Create all-zero flow, verify no crash, returns valid points
    3. **Temperature parameter**: Test with temp=0.5 vs temp=2.0, verify emphasis difference >20%
  - Use actual `sample_motion_driven_points()` function from `mpc/point_sampling.py`
  - Exit code 0 on all pass, 1 on any failure
  - Print clear PASS/FAIL for each scenario

  **Must NOT do**:
  - Mock/stub the sampling function (test the real implementation)
  - Add visualization beyond saving evidence files
  - Test unrelated sampling logic

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
    - Reason: Requires synthetic data generation, statistical testing
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES (with V1, V3 - independent verification)
  - **Parallel Group**: Wave 4 (verification wave)
  - **Blocks**: Task F1 (final integration test)
  - **Blocked By**: Task 2 (must have modified sampling code)

  **References**:

  **Pattern References**:
  - `mpc/point_sampling.py:sample_motion_driven_points()` - Function to test
  - Synthetic flow creation: `flow_field = np.ones((512, 512, 2)) * 0.1; flow_field[y1:y2, x1:x2] = 10.0`
  - Motion mask: `motion_mask = np.ones((512, 512), dtype=bool)`

  **Why These References Matter**:
  - Must test actual function, not reimplementation
  - Synthetic flow allows controlled testing (known ground truth)
  - Statistical test: >30% in hot region vs 3.8% expected uniform

  **Acceptance Criteria**:

  - [ ] File `verify_flow_sampling.py` exists and is executable
  - [ ] `python verify_flow_sampling.py` exits with code 0 if all tests pass
  - [ ] Script tests weighted sampling with high-flow region
  - [ ] Script tests zero-flow fallback (no crash)
  - [ ] Script tests temperature parameter effect

  **QA Scenarios (MANDATORY)**:

  ```
  Scenario: Verify script detects weighted sampling behavior
    Tool: Bash
    Preconditions: verify_flow_sampling.py created, Task 2 completed
    Steps:
      1. python verify_flow_sampling.py
      2. Check exit code: echo $?
      3. Assert exit code == 0
      4. Check output contains "PASS: Weighted sampling" (>30% in hot region)
      5. Check output contains "PASS: Zero-flow fallback"
      6. Check output contains "PASS: Temperature parameter"
    Expected Result: Exit 0, all 3 scenarios pass
    Failure Indicators: Exit 1, any "FAIL" message, crash
    Evidence: .sisyphus/evidence/task-v2-verify-flow-sampling.txt
  ```

  **Evidence to Capture**:
  - [ ] Script output showing all scenarios pass
  - [ ] Percentage of points in hot region (should be >30%)

  **Commit**: NO (verification scripts committed separately)

- [ ] V3. **Create verify_constraints.py Script**

  **What to do**:
  - Create Python script `verify_constraints.py` in project root
  - Implement test scenarios from Tasks 1a-1d:
    1. **Config class**: Import and instantiate `PlanningDMControlParams`, verify parameters
    2. **Projection (NumPy)**: Test with invalid actions, verify sin²+cos²≈1 after projection
    3. **Projection (PyTorch)**: Test with batch+horizon shape, verify all pairs normalized
    4. **Projection edge case**: Test zero vector, verify no NaN/Inf
    5. **Penalty (valid)**: Test ActionRegularizationObjective with valid actions, penalty≈0
    6. **Penalty (invalid)**: Test with invalid actions, verify penalty >0
    7. **Penalty disabled**: Test with weight=0, verify no penalty
    8. **CEM integration**: Run minimal CEM plan, verify actions valid
    9. **MPPI integration**: Run minimal MPPI plan, verify actions valid
  - Use actual implementations from mpc/ modules
  - Exit code 0 on all pass, 1 on any failure
  - Print max violation found, expected <1e-6

  **Must NOT do**:
  - Full MPC integration test (minimal optimizer setup only)
  - Visual inspection tests (all assertions must be code-executable)
  - Test unrelated constraint types

  **Recommended Agent Profile**:
  - **Category**: `deep`
    - Reason: Complex multi-component verification, requires understanding all 4 sub-tasks
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES (with V1, V2 - independent verification)
  - **Parallel Group**: Wave 4 (verification wave)
  - **Blocks**: Task F1 (final integration test)
  - **Blocked By**: Task 1d (must have all constraint components)

  **References**:

  **Pattern References**:
  - Import pattern: `from arguments.planning_dmcontrol import PlanningDMControlParams`
  - Test projection: `from mpc.constraint_utils import project_joint_angles, project_joint_angles_torch`
  - Test penalty: `from mpc.flow_objectives import ActionRegularizationObjective`
  - Minimal optimizer: Create with dummy model, single planning step

  **Why These References Matter**:
  - Must test actual implementations end-to-end
  - Constraint checking: `abs(sin**2 + cos**2 - 1.0) < 1e-6`
  - Each sub-task (1a-1d) has specific verification requirements from QA scenarios

  **Acceptance Criteria**:

  - [ ] File `verify_constraints.py` exists and is executable
  - [ ] `python verify_constraints.py` exits with code 0 if all tests pass
  - [ ] Script tests all 9 scenarios listed above
  - [ ] Script prints max constraint violation (<1e-6 expected)
  - [ ] Script works without GPU (use CPU tensors for portability)

  **QA Scenarios (MANDATORY)**:

  ```
  Scenario: Verify script detects constraint enforcement
    Tool: Bash
    Preconditions: verify_constraints.py created, Task 1a-1d completed
    Steps:
      1. python verify_constraints.py
      2. Check exit code: echo $?
      3. Assert exit code == 0
      4. Check output contains "Max violation: X.XXe-7" where X < 1.0
      5. Check output contains "PASS" for all 9 test scenarios
    Expected Result: Exit 0, max violation <1e-6, all scenarios pass
    Failure Indicators: Exit 1, violation >1e-6, any "FAIL" message
    Evidence: .sisyphus/evidence/task-v3-verify-constraints.txt
  ```

  **Evidence to Capture**:
  - [ ] Script output showing all scenarios pass
  - [ ] Max constraint violation value

  **Commit**: YES (verification scripts together)
  - Message: `test(mpc): Add verification scripts for three MPC improvements
    
    Three standalone scripts verify each improvement:
    - verify_iterations.py: Check opt_iters defaults are 10
    - verify_flow_sampling.py: Test weighted sampling on synthetic flow
    - verify_constraints.py: Test joint angle constraint enforcement
    
    All scripts exit 0 on pass, 1 on failure. No test framework required.`
  - Files: `verify_iterations.py`, `verify_flow_sampling.py`, `verify_constraints.py`
  - Pre-commit: `python verify_iterations.py && python verify_flow_sampling.py && python verify_constraints.py`

---

## Final Verification Wave (MANDATORY — after ALL implementation tasks)

- [ ] F1. **Full MPC Pipeline Integration Test** — `unspecified-high`

  Run complete MPC planning pipeline with all three improvements enabled on bouncingballs scene. Verify: (1) No crashes, (2) Tracking points on robot arm, (3) Valid actions produced, (4) 10 iterations executed, (5) Metrics within 10% of baseline.
  
  **QA Scenario**:
  ```
  Tool: Bash (run full demo script)
  Preconditions: All three improvements implemented and verified individually
  Steps:
    1. cd /home/ubuntu/yyf/4DGaussians
    2. python demo_flow_guided_mpc.py --scene bouncingballs --horizon 10 --num_tracking_points 100
    3. Check terminal output for: "Iter 10/10" (confirms 10 iterations)
    4. Check for crash/exception (exit code must be 0)
    5. Check printed action values: assert sin²+cos² ≈ 1.0 for first action
  Expected Result: Script completes successfully, 10 iterations run, actions valid
  Evidence: Terminal output showing completion + action validation
  ```
  
  **Commit**: YES
  - Message: `feat(mpc): Implement three MPC planning improvements
    
    - Add joint angle constraint enforcement (projection + penalty)
    - Fix flow-weighted sampling (add p=probabilities)
    - Increase default MPC iterations from 5 to 10
    
    Resolves invalid robot actions, improves tracking point distribution,
    reduces local optima in optimization.`
  - Files: All modified files
  - Pre-commit: `python verify_constraints.py && python verify_flow_sampling.py && python verify_iterations.py`

---

## Commit Strategy

- **Commit 1**: Task 3 (opt_iters defaults)
- **Commit 2**: Task 2 (flow-weighted sampling)
- **Commit 3**: Tasks 1a-1d + verification (constraint enforcement)
- **Commit 4**: Integration test pass

---

## Success Criteria

### Verification Commands

```bash
# Task 1: Constraint Enforcement
python verify_constraints.py --num_samples 1000
# Expected: "PASS: Max constraint violation: X.XXe-7 (tolerance: 1e-6)"

# Task 2: Flow-Weighted Sampling
python verify_flow_sampling.py --synthetic_flow
# Expected: "PASS: High-flow region sampled X% (threshold: >70%)"

# Task 3: Iteration Count
python demo_flow_guided_mpc.py --help | grep "opt_iters"
# Expected: "--opt_iters ... default: 10"

# Integration Test
python demo_flow_guided_mpc.py --scene bouncingballs --horizon 10
# Expected: No crash, "Iter 10/10" in output, valid actions printed
```

### Final Checklist

- [ ] All "Must Have" items implemented
- [ ] All "Must NOT Have" items avoided
- [ ] All verification scripts pass (exit code 0)
- [ ] Integration test passes on bouncingballs scene
- [ ] Config class `PlanningDMControlParams` loads without errors
- [ ] Performance within acceptable bounds (no >20% regression)
