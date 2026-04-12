# MPC Improvements + CoTracker Test Debug Summary

**Date**: 2026-03-19  
**Status**: ✅ Completed  
**Scope**: MPC planning improvements, motion-mask sampling fix, joint constraint enforcement, CoTracker test script fixes, cuDNN GPU selection fix  

---

## Summary of Modifications

### 1) MPC Planning Improvements (Core)

- **CEM optimization iterations**: Increased default `opt_iters` from **5 → 10** across demos/integration scripts.  
  **Why**: Stabilize planning and improve convergence.

- **Flow-weighted sampling bug fix**: Added `p=probabilities` in all motion-weighted sampling calls.  
  **Why**: Previously computed flow magnitude weights were ignored; sampling was effectively uniform.

- **Joint angle constraint enforcement**: Added projection + penalty for sin/cos joint pairs.  
  **Why**: Keep each joint’s sin/cos pair on the unit circle for valid angles.
  
  **Implementation**:
  - Hard projection after clipping in CEM/MPPI
  - Soft penalty term (weight=10.0) in ActionRegularizationObjective

- **Planning config for dm_control**: Added `arguments/planning_dmcontrol/` parameter group.  
  **Why**: Make joint constraints and planner parameters explicit for dm_control planning.

### 2) CoTracker + Planning Integration Updates

- **Motion mask initialization** upgraded to use **bidirectional flow with consistency check**.
- **Dynamic tracking point update** (per step) is enabled for motion mask mode.
- **Per-step motion mask resampling** supported to maintain dense, reliable tracking points.

### 3) Test Script & Execution Fixes

- `run_cotracker_test.sh` now calls **`test/integration/test_cotracker_mpc.py`** (not the demo).  
- Shell script arguments updated to **match test script argparse**:
  - `--output_dir` replaces `--log_dir`
  - `--sampling_method motion_mask` used
  - Removed unsupported args (`--vgg_weight`, `--direction_weight`, etc.)

### 4) cuDNN Initialization Failure Fix

**Issue**: `RuntimeError: cuDNN error: CUDNN_STATUS_NOT_INITIALIZED` during GMFlow forward pass.  
**Root cause**: default device selected GPU with 96% memory usage.  

**Fix** in `test/integration/test_cotracker_mpc.py`:
- Auto-select least-busy GPU if `--device` not given (via `nvidia-smi`).
- Respect pre-set `CUDA_VISIBLE_DEVICES`.
- Move `torch` import **after** device env setup.

**Verification**:
- GMFlow BiFlow computation succeeds on free GPU.
- `01_biflow_initialization.png` and step renders generated.

---

## Modified Files (Key)

### Core MPC / Planning
- `mpc/point_sampling.py`  
  - Motion-mask sampling fix (probability weighting)
  - Bidirectional flow + consistency functions
  - Dynamic tracking point update utilities

- `mpc/flow_objectives.py`  
  - Action constraint penalty (unit circle)

- `mpc/cem.py`, `mpc/mppi.py`  
  - Hard projection of sin/cos pairs after clipping

### Planning Configuration
- `arguments/planning_dmcontrol/__init__.py`
- `arguments/__init__.py`

### Demos & Integration
- `demo_cotracker_mpc.py` (opt_iters=10)
- `demo_flow_guided_mpc.py` (opt_iters=10)
- `test/integration/test_cotracker_mpc.py`  
  - Bidirectional flow init + dynamic update
  - Auto GPU selection for cuDNN reliability

### Script Fix
- `run_cotracker_test.sh`
  - Correct script target + argument set

---

## Verification Artifacts

Saved under `.sisyphus/evidence/`:

- `task-v1-iterations.txt` (opt_iters = 10 everywhere)
- `task-v2-flow-sampling.txt` (probability sampling verification)
- `task-v3-constraints.txt` (unit-circle projection + penalty)
- `task-f1-integration-lightweight.txt` (integration sanity check)
- `cudnn-fix-verification.txt` (GMFlow + cuDNN fix proof)

---

## Current Planning Pipeline (High-Level)

1. **Load model + set device** (auto GPU selection if not specified)
2. **Load initial/target images** (resize to 480x480)
3. **Sample tracking points**:
   - Motion mask via bidirectional flow + consistency
   - 70% motion + 30% corners (weighted by flow magnitude)
4. **TAPIR/CoTracker** computes target point locations
5. **Load initial joint control** from transforms.json
6. **Build MPC stack**:
   - Dynamics model (FlowGuidedGaussianDynamicsModel)
   - Objective (PointTrackingObjective + regularization)
   - Optimizer (CEM, opt_iters=10)
   - Action constraints (projection + penalty)
7. **MPC loop**:
   - (Optional) per-step motion mask resampling
   - Plan action → render next frame
   - Track updated points
   - Save step images / debug diagnostics

---

## Notes for Next Agent

- `test/` directory is gitignored — changes there are **not committed**.
- `run_cotracker_test.sh` is the authoritative entrypoint for integration test runs.
- cuDNN error was caused by **GPU memory pressure**, not code logic.
- GMFlow checkpoint must exist at `gmflow/checkpoints/`.
