# Motion-Driven Mask Sampling for Point Tracking

## TL;DR

> **Quick Summary**: Fix TAPIR tracking accuracy by only sampling points on moving objects (robot arm + cube), excluding static background.
> 
> **Deliverables**:
> - New function: `compute_motion_mask()` - GMFlow-based motion detection
> - New function: `sample_motion_driven_points()` - Sample only in motion regions
> - CLI integration: `--sampling_method motion_mask`
> - Diagnostic visualizations: flow magnitude, motion mask overlay
> 
> **Estimated Effort**: Short (2-3 hours)
> **Parallel Execution**: NO - sequential (depends on GMFlow setup)
> **Critical Path**: Task 1 → Task 2 → Task 3 → Task 4

---

## Context

### Original Request
**User reported**: 
> "现在捕获到的跟踪点是不正确的，在初始图上的跟踪点能附着在移动的目标（机械臂/方块）上，但是目标图像无法获取所需要的跟踪点在目标图上的正确位置"

Translation: Points are correctly placed on robot arm/cube in initial image, but tracked positions in target image are incorrect.

### Root Cause
- TAPIR tracks from frame 1 → frame 18 (large time gap, ~0.5-1 second)
- Current sampling methods (Sobel, Shi-Tomasi) sample across entire image
- Many points land on **static background** (floor, walls, table)
- Static background points don't provide useful signal for MPC
- Moving object points get mixed with background noise → poor control

### Interview Summary
**User's choice**: Motion-driven mask sampling (方案1)
- Use GMFlow optical flow to identify motion regions
- Only sample tracking points within motion mask
- Exclude static background automatically

### Metis Review Findings
**CRITICAL**: Metis identified several gaps:

1. **Adaptive thresholding required** - Fixed pixel threshold fails across scenes
   - Solution: Use percentile-based threshold (follow `demo_flow_guided_mpc.py`)

2. **Motion mask post-processing** - Raw threshold creates noisy masks
   - Solution: Morphological operations (close → open)

3. **Coverage validation** - Need guardrails for edge cases
   - Camera motion → coverage > 80% → fallback to Shi-Tomasi
   - Static scene → coverage < 1% → fallback to uniform grid

4. **Diagnostic outputs** - Need verifiable evidence for QA
   - Flow magnitude heatmap
   - Motion mask overlay
   - Point distribution histogram

5. **GMFlow device handling** - Must cleanup GPU memory after flow computation
   - Solution: Explicit `del flownet` + `torch.cuda.empty_cache()`

---

## Work Objectives

### Core Objective
Implement motion-driven point sampling that focuses tracking points on moving objects (robot arm + cube), improving TAPIR tracking accuracy from initial to target frames.

### Concrete Deliverables
- `mpc/point_sampling.py`:
  - New function: `compute_motion_mask()` (100 lines)
  - New function: `sample_motion_driven_points()` (80 lines)
- `test_cotracker_mpc.py`:
  - Add `motion_mask` to `--sampling_method` choices
  - Wire motion mask sampling in point setup section
- Diagnostic outputs in `outputs/{exp}/`:
  - `flow_magnitude_heatmap.png`
  - `motion_mask_overlay.png`
  - `point_distribution_histogram.png`

### Definition of Done
- [ ] `python test_cotracker_mpc.py --sampling_method motion_mask` runs successfully
- [ ] Motion mask covers 5-80% of image (validated via assertion)
- [ ] >50% of sampled points fall within motion regions (validated)
- [ ] Diagnostic visualizations saved automatically
- [ ] Tracking accuracy on target frame visibly improved (manual QA)

### Must Have
- Adaptive percentile-based threshold (no fixed pixel values)
- Morphological post-processing (close + open)
- Coverage guardrails (fallback to other methods)
- GPU memory cleanup (del + empty_cache)
- Diagnostic visualizations (flow, mask, distribution)

### Must NOT Have (Guardrails)
- **Fixed pixel thresholds** (e.g., > 5.0 pixels) → Use percentile instead
- **One-size-fits-all parameters** → Must validate coverage and adapt
- **Memory leaks** → Must cleanup GMFlow after use
- **Unverified outputs** → Must save diagnostic images for every run
- **Bidirectional flow** (v1 scope) → Document as future improvement, use one-way + percentile

---

## Verification Strategy

> **ZERO HUMAN INTERVENTION** — ALL verification is agent-executed.

### Test Decision
- **Infrastructure exists**: YES (GMFlow in `gmflow/`)
- **Automated tests**: NO (research codebase, manual QA via visualization)
- **Framework**: N/A

### QA Policy
Every task includes agent-executed QA scenarios with diagnostic file checks.
Evidence saved to `.sisyphus/evidence/motion-mask-{scenario}.{ext}`.

---

## Execution Strategy

### Parallel Execution Waves

```
Wave 1 (Foundation - 1 task):
└── Task 1: Implement compute_motion_mask() [deep]

Wave 2 (Sampling - 1 task):
└── Task 2: Implement sample_motion_driven_points() [quick]

Wave 3 (Integration - 1 task):
└── Task 3: Integrate into test_cotracker_mpc.py [quick]

Wave 4 (Testing - 1 task):
└── Task 4: Test and validate with real data [unspecified-high]

Critical Path: Task 1 → Task 2 → Task 3 → Task 4
Parallel Speedup: None (sequential dependencies)
Max Concurrent: 1
```

### Dependency Matrix

- **1**: — — 2
- **2**: 1 — 3
- **3**: 2 — 4
- **4**: 3 — —

### Agent Dispatch Summary

- **Wave 1**: 1 task → T1: `deep`
- **Wave 2**: 1 task → T2: `quick`
- **Wave 3**: 1 task → T3: `quick`
- **Wave 4**: 1 task → T4: `unspecified-high`

---

## TODOs

- [ ] 1. Implement compute_motion_mask() function

  **What to do**:
  - Add `compute_motion_mask()` to `mpc/point_sampling.py`
  - Initialize GMFlow model (follow `demo_flow_guided_mpc.py:271-288`)
  - Compute forward optical flow (initial → target)
  - Calculate flow magnitude: `np.linalg.norm(flow_field, axis=-1)`
  - Apply adaptive threshold: `max(np.percentile(magnitude, 70), 0.5)`
  - Post-process with morphological ops (close + open, kernel 5x5)
  - Validate coverage: assert 0.01 < coverage < 0.8
  - Cleanup GPU memory: `del flownet` + `torch.cuda.empty_cache()`
  - Save diagnostic: flow magnitude heatmap

  **Must NOT do**:
  - Use fixed pixel threshold (e.g., > 5.0)
  - Skip morphological post-processing
  - Leave GMFlow model in GPU memory
  - Implement bidirectional flow (out of scope for v1)

  **Recommended Agent Profile**:
  - **Category**: `deep`
    - Reason: Complex integration - requires GMFlow initialization, device handling, GPU memory management, adaptive thresholding
  - **Skills**: []
    - Reason: No specialized skills needed, core PyTorch/CV operations

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Sequential only (foundation task)
  - **Blocks**: Task 2 (sampling depends on mask computation)
  - **Blocked By**: None (can start immediately)

  **References**:

  **Pattern References** (GMFlow initialization):
  - `demo_flow_guided_mpc.py:271-288` - GMFlow initialization pattern (config, checkpoint loading, device setup)
  - `demo_flow_guided_mpc.py:291-305` - Flow computation pattern (image preprocessing, forward pass, output format)
  - `demo_flow_guided_mpc.py:312` - Adaptive threshold pattern (percentile-based)

  **API/Type References**:
  - `gmflow/config.py:get_gmflow_cfg()` - Returns GMFlow config with checkpoint path
  - `gmflow/gmflow.py:GMFlow` - Main flow model class
  - Input format: (B, 3, H, W) torch.Tensor, float
  - Output format: List of flow predictions, take `[-1][0]` for final flow

  **Codebase References** (morphological ops):
  - `test_cotracker_mpc.py:64-82` - `detect_object_regions()` function shows morphological closing pattern

  **External References**:
  - Official GMFlow: https://github.com/haofeixu/gmflow
  - UnFlow algorithm (ICCV 2017): Adaptive flow thresholding

  **WHY Each Reference Matters**:
  - `demo_flow_guided_mpc.py` contains working GMFlow code - copy this pattern exactly to avoid device/checkpoint issues
  - `detect_object_regions()` shows existing morphological ops in codebase - follow same kernel size and operations
  - Percentile threshold (line 312) is production-validated - don't reinvent

  **Acceptance Criteria**:

  > **AGENT-EXECUTABLE VERIFICATION ONLY**

  - [ ] Function signature exists: `compute_motion_mask(img1, img2, device='cuda:0', percentile=70, min_magnitude=0.5)`
  - [ ] Returns tuple: `(motion_mask: np.ndarray bool, flow_magnitude: np.ndarray float)`
  - [ ] Test call: `mask, flow_mag = compute_motion_mask(initial_img, target_img)`
  - [ ] Coverage assertion passes: `assert 0.01 < mask.sum()/mask.size < 0.8`

  **QA Scenarios**:

  ```
  Scenario: Motion mask computation with real robot data
    Tool: Bash (Python REPL)
    Preconditions: GMFlow checkpoint at gmflow/checkpoints/gmflow_sintel-0c07dcb3.pth
    Steps:
      1. Import: `from mpc.point_sampling import compute_motion_mask`
      2. Load images: `img1 = load_image('assets/start-end/cam5_sample1_frame_00001.jpg', (512, 512))`
      3. Load images: `img2 = load_image('assets/start-end/cam5_sample1_frame_00018.jpg', (512, 512))`
      4. Compute mask: `mask, flow_mag = compute_motion_mask(img1, img2)`
      5. Check shape: `assert mask.shape == (512, 512)`
      6. Check dtype: `assert mask.dtype == bool`
      7. Check coverage: `coverage = mask.sum() / mask.size; assert 0.01 < coverage < 0.8`
      8. Check flow magnitude non-negative: `assert (flow_mag >= 0).all()`
    Expected Result: Function returns valid mask with 1-80% coverage
    Failure Indicators: ImportError, shape mismatch, coverage out of range, negative flow values
    Evidence: .sisyphus/evidence/task-1-motion-mask-computation.txt (print shapes and coverage)

  Scenario: GMFlow memory cleanup verification
    Tool: Bash (nvidia-smi before/after)
    Preconditions: CUDA available
    Steps:
      1. Check initial GPU memory: `nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits`
      2. Run mask computation (above scenario)
      3. Check final GPU memory: `nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits`
      4. Assert memory released: final_mem < initial_mem + 500MB
    Expected Result: GPU memory returns to near-baseline after function call
    Failure Indicators: Memory leak (>500MB increase persists)
    Evidence: .sisyphus/evidence/task-1-gpu-memory-cleanup.txt

  Scenario: Diagnostic visualization saved
    Tool: Bash (ls + file check)
    Preconditions: Output directory exists
    Steps:
      1. Run compute_motion_mask() with save_diagnostics=True
      2. Check file exists: `ls outputs/test_motion/flow_magnitude_heatmap.png`
      3. Check file size: file size > 10KB (valid PNG)
      4. Check file type: `file outputs/test_motion/flow_magnitude_heatmap.png` contains "PNG"
    Expected Result: Flow magnitude heatmap saved as valid PNG
    Evidence: .sisyphus/evidence/task-1-diagnostic-visualization.png (copy of heatmap)
  ```

  **Evidence to Capture**:
  - [ ] Mask shape, dtype, coverage percentage (task-1-motion-mask-computation.txt)
  - [ ] GPU memory before/after (task-1-gpu-memory-cleanup.txt)
  - [ ] Flow magnitude heatmap (task-1-diagnostic-visualization.png)

  **Commit**: NO (group with Task 2)

---

- [ ] 2. Implement sample_motion_driven_points() function

  **What to do**:
  - Add `sample_motion_driven_points()` to `mpc/point_sampling.py`
  - Call `compute_motion_mask()` to get motion regions
  - Check coverage: if < 1%, fallback to `sample_uniform_grid()`
  - Check coverage: if > 80%, fallback to `sample_shi_tomasi_points()`
  - Sample points in motion mask using combined strategy:
    - 70% from motion regions (reuse `sample_combined()` with mask constraint)
    - 30% from Shi-Tomasi corners (for texture quality)
  - Apply spatial NMS (radius=8) for diversity
  - Save diagnostic: motion mask overlay with sampled points

  **Must NOT do**:
  - Sample all points from motion mask without fallback validation
  - Skip NMS (causes point clustering at boundaries)
  - Ignore texture quality (use Shi-Tomasi for 30% of points)

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Straightforward - combines existing functions with simple logic
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Sequential (depends on Task 1)
  - **Blocks**: Task 3 (integration needs both functions)
  - **Blocked By**: Task 1 (needs compute_motion_mask)

  **References**:

  **Pattern References**:
  - `mpc/point_sampling.py:152-230` - `sample_combined()` function shows hybrid sampling pattern
  - `mpc/point_sampling.py:236-260` - `apply_nms()` function for spatial filtering
  - `test_cotracker_mpc.py:54-130` - `sample_object_focused_points()` shows mask-constrained sampling

  **API References**:
  - `compute_motion_mask()` (Task 1): Returns (motion_mask, flow_magnitude)
  - `sample_shi_tomasi_points()`: Returns (N, 2) points
  - `apply_nms()`: Takes points, returns filtered points

  **WHY Each Reference Matters**:
  - `sample_combined()` already implements weighted hybrid sampling - adapt this for motion mask
  - `apply_nms()` prevents point clustering - MUST use for motion boundaries
  - `sample_object_focused_points()` shows how to apply mask constraint to cv2.goodFeaturesToTrack

  **Acceptance Criteria**:

  - [ ] Function signature: `sample_motion_driven_points(img1, img2, num_points=384, device='cuda:0')`
  - [ ] Returns: (N, 2) numpy array of [x, y] coordinates
  - [ ] Test call: `points = sample_motion_driven_points(initial_img, target_img, 384)`
  - [ ] Point count: `len(points) >= 0.8 * 384` (allow slight under-sampling)
  - [ ] Points within image bounds: `assert (0 <= points).all() and (points[:, 0] < W).all() and (points[:, 1] < H).all()`

  **QA Scenarios**:

  ```
  Scenario: Motion-driven sampling with robot movement
    Tool: Bash (Python REPL)
    Preconditions: Initial and target images with robot arm movement
    Steps:
      1. Import: `from mpc.point_sampling import sample_motion_driven_points, compute_motion_mask`
      2. Load images: img1, img2 (512x512)
      3. Sample points: `points = sample_motion_driven_points(img1, img2, num_points=384)`
      4. Compute motion mask: `mask, _ = compute_motion_mask(img1, img2)`
      5. Check points in motion: `points_in_motion = mask[points[:, 1].astype(int), points[:, 0].astype(int)]`
      6. Calculate ratio: `motion_ratio = points_in_motion.sum() / len(points)`
      7. Assert: `assert motion_ratio > 0.5, f"Expected >50% in motion, got {motion_ratio:.1%}"`
    Expected Result: >50% of points fall in motion regions
    Failure Indicators: Low motion ratio (<50%), points out of bounds
    Evidence: .sisyphus/evidence/task-2-point-distribution.txt (motion_ratio, point stats)

  Scenario: Fallback to Shi-Tomasi for high camera motion
    Tool: Bash (Python REPL)
    Preconditions: Images with camera motion (coverage > 80%)
    Steps:
      1. Mock scenario: Set coverage to 85% artificially
      2. Call sample_motion_driven_points()
      3. Capture logs: check for "High coverage" warning
      4. Verify fallback: points should match Shi-Tomasi distribution (not uniform across image)
    Expected Result: Function falls back to Shi-Tomasi, logs warning
    Evidence: .sisyphus/evidence/task-2-camera-motion-fallback.txt (logs + point distribution)

  Scenario: Fallback to uniform grid for static scene
    Tool: Bash (Python REPL)
    Preconditions: Two identical images (no motion)
    Steps:
      1. Mock: img1 = img2 (static scene)
      2. Call sample_motion_driven_points()
      3. Capture logs: check for "Low coverage" warning
      4. Verify: points should be uniformly distributed (grid pattern)
    Expected Result: Falls back to uniform grid
    Evidence: .sisyphus/evidence/task-2-static-scene-fallback.txt
  ```

  **Evidence to Capture**:
  - [ ] Motion ratio percentage (task-2-point-distribution.txt)
  - [ ] Fallback logs (task-2-camera-motion-fallback.txt, task-2-static-scene-fallback.txt)
  - [ ] Motion mask overlay with points (task-2-mask-overlay.png)

  **Commit**: YES (group with Task 1)
  - Message: `feat(mpc): add motion-driven mask sampling for improved tracking`
  - Files: `mpc/point_sampling.py`
  - Pre-commit: `python -c "from mpc.point_sampling import compute_motion_mask, sample_motion_driven_points; print('✓ Import OK')"`

---

- [ ] 3. Integrate motion_mask into test_cotracker_mpc.py

  **What to do**:
  - Add `motion_mask` to `--sampling_method` choices (line 178)
  - Add elif branch in sampling logic (around line 240):
    ```python
    elif args.sampling_method == "motion_mask":
        initial_points = point_sampling.sample_motion_driven_points(
            initial_image, target_image, 
            num_points=args.num_tracking_points,
            device=args.device
        )
        sampling_desc = "Motion-driven (GMFlow mask)"
    ```
  - Update help text to mention motion_mask option

  **Must NOT do**:
  - Forget to pass `target_image` as second argument (required for flow)
  - Skip updating help text (users need to know new option exists)

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Trivial integration - add 1 choice + 1 elif branch
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Sequential (depends on Task 2)
  - **Blocks**: Task 4 (testing needs integration)
  - **Blocked By**: Task 2 (needs sample_motion_driven_points)

  **References**:

  **Pattern References**:
  - `test_cotracker_mpc.py:178` - `--sampling_method` argparse with choices list
  - `test_cotracker_mpc.py:223-255` - Existing sampling logic with if/elif/else branches

  **WHY Each Reference Matters**:
  - Exact line numbers where changes needed - copy existing pattern for consistency

  **Acceptance Criteria**:

  - [ ] `--sampling_method motion_mask` accepted by argparse
  - [ ] Test: `python test_cotracker_mpc.py --sampling_method motion_mask --help` (no error)
  - [ ] Code inspection: `motion_mask` in choices list
  - [ ] Code inspection: elif branch exists with `sample_motion_driven_points()` call

  **QA Scenarios**:

  ```
  Scenario: CLI argument validation
    Tool: Bash
    Preconditions: None
    Steps:
      1. Run: `python test_cotracker_mpc.py --sampling_method motion_mask --help`
      2. Check exit code: `echo $?` should be 0
      3. Check help text contains "motion_mask"
    Expected Result: Help displays successfully, motion_mask listed as option
    Failure Indicators: argparse error, motion_mask not in help text
    Evidence: .sisyphus/evidence/task-3-cli-validation.txt (help output)

  Scenario: Motion mask sampling branch execution
    Tool: Bash
    Preconditions: Tasks 1 and 2 complete
    Steps:
      1. Run: `python test_cotracker_mpc.py --sampling_method motion_mask --num_steps 1`
      2. Check logs: should see "Using sampling method: motion_mask"
      3. Check logs: should see "Sampled X tracking points (Motion-driven)"
      4. Check file exists: outputs/*/01_initial_with_points.png
    Expected Result: Sampling executes without error, visualization saved
    Failure Indicators: ImportError, AttributeError, no visualization
    Evidence: .sisyphus/evidence/task-3-sampling-execution.txt (logs)
  ```

  **Evidence to Capture**:
  - [ ] Help text output (task-3-cli-validation.txt)
  - [ ] Sampling execution logs (task-3-sampling-execution.txt)

  **Commit**: YES
  - Message: `feat(test): integrate motion-driven sampling into test_cotracker_mpc`
  - Files: `test_cotracker_mpc.py`
  - Pre-commit: `python test_cotracker_mpc.py --sampling_method motion_mask --help`

---

- [ ] 4. Test and validate with real robot data

  **What to do**:
  - Run full MPC test with motion mask sampling:
    ```bash
    python test_cotracker_mpc.py \
      --sampling_method motion_mask \
      --image_height 512 \
      --image_width 512 \
      --num_tracking_points 384 \
      --num_steps 3 \
      --output_dir outputs/test_motion_mask
    ```
  - Compare results with baseline (sobel_hybrid) and other methods:
    - Visual inspection: Are points focused on robot arm + cube?
    - Quantitative: Check "Visible target points" ratio in logs
    - Final distance: Compare "Avg Distance to Target" at last step
  - Inspect diagnostic visualizations:
    - `flow_magnitude_heatmap.png` - Should show high magnitude on moving objects
    - `motion_mask_overlay.png` - Mask should cover robot arm + cube, not background
    - `01_initial_with_points.png` - Points should be on robot/cube, not floor/walls
    - `01_target_with_points.png` - **CRITICAL**: Target points should be on correct positions
  - Document findings in `.sisyphus/evidence/task-4-comparison-report.txt`

  **Must NOT do**:
  - Skip comparison with baseline (need to prove improvement)
  - Accept results without visual inspection of target points
  - Ignore diagnostic visualizations (they reveal if mask/flow is correct)

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
    - Reason: Requires testing, comparison, visual QA, and documentation
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Sequential (final validation)
  - **Blocks**: None (final task)
  - **Blocked By**: Task 3 (needs integration complete)

  **References**:

  **Test Data**:
  - Initial: `assets/start-end/cam5_sample1_frame_00001.jpg`
  - Target: `assets/start-end/cam5_sample1_frame_00018.jpg`
  - Transforms: `assets/example_transforms.json`
  - Model: `outputs/dm_control_push_test_flow2/point_cloud/iteration_12000`

  **Baseline Results** (for comparison):
  - Sobel @ 480x480: 332/384 visible (86.5%), 187.5px final distance
  - Shi-Tomasi @ 512x512: 350/384 visible (91.1%), 244.8px final distance
  - Combined @ 512x512: 274/300 visible (91.3%), 231.3px final distance

  **WHY Each Reference Matters**:
  - Test data is user's actual robot task - must use same data for valid comparison
  - Baseline results provide quantitative benchmark - motion mask should achieve ≥90% visibility

  **Acceptance Criteria**:

  - [ ] Test completes without errors (exit code 0)
  - [ ] Visible points ratio: ≥90% (match or exceed current best methods)
  - [ ] Motion mask overlay shows robot arm + cube (visual QA)
  - [ ] Target points are on robot/cube, not background (visual QA of 01_target_with_points.png)
  - [ ] Diagnostic files exist: flow_magnitude_heatmap.png, motion_mask_overlay.png

  **QA Scenarios**:

  ```
  Scenario: Full MPC test with motion mask sampling
    Tool: Bash
    Preconditions: Tasks 1-3 complete, GMFlow checkpoint exists
    Steps:
      1. Run: `python test_cotracker_mpc.py --sampling_method motion_mask --num_steps 3 --output_dir outputs/test_motion_mask 2>&1 | tee test_motion_mask.log`
      2. Check exit code: `echo $?` should be 0
      3. Parse logs: grep "Visible target points:" | extract ratio
      4. Assert: visible_ratio >= 0.90
      5. Check files exist: ls outputs/test_motion_mask/{flow_magnitude_heatmap,motion_mask_overlay,01_initial_with_points,01_target_with_points}.png
    Expected Result: Test runs successfully, ≥90% points visible, all diagnostic files saved
    Failure Indicators: Non-zero exit, visible ratio <90%, missing diagnostic files
    Evidence: .sisyphus/evidence/task-4-full-test-execution.txt (full logs)

  Scenario: Visual QA - Target points on robot/cube
    Tool: Look_at (image analysis)
    Preconditions: Test complete, 01_target_with_points.png exists
    Steps:
      1. Read image: outputs/test_motion_mask/01_target_with_points.png
      2. Analyze: "Are the green tracking points concentrated on the robot arm and cube, or are they scattered on the background (floor/walls)?"
      3. Expected: "Points are clearly focused on robot arm joints and cube edges"
      4. If points on background: FAIL - mask not working correctly
    Expected Result: Visual confirmation that target points are correctly positioned on moving objects
    Failure Indicators: Points scattered on floor, walls, or table
    Evidence: .sisyphus/evidence/task-4-target-points-visual-qa.png (annotated screenshot)

  Scenario: Motion mask quality verification
    Tool: Look_at (image analysis)
    Preconditions: motion_mask_overlay.png exists
    Steps:
      1. Read: outputs/test_motion_mask/motion_mask_overlay.png
      2. Analyze: "Does the motion mask (highlighted region) cover the robot arm and cube while excluding the static background?"
      3. Expected: "Mask tightly bounds robot arm and cube, background is excluded"
      4. If mask covers >50% of image: FAIL - likely camera motion
      5. If mask covers <5% of image: FAIL - threshold too high
    Expected Result: Motion mask accurately segments moving objects
    Failure Indicators: Mask too large (>50%), too small (<5%), or excludes robot arm
    Evidence: .sisyphus/evidence/task-4-motion-mask-quality.png

  Scenario: Quantitative comparison with baseline
    Tool: Bash (log parsing + Python analysis)
    Preconditions: Baseline logs exist (sobel, shi_tomasi, combined)
    Steps:
      1. Parse all logs: extract "Visible target points" and "Avg Distance to Target"
      2. Create comparison table:
         | Method | Visible Ratio | Final Distance |
         |--------|---------------|----------------|
         | Sobel  | 86.5%         | 187.5 px       |
         | Shi-Tomasi | 91.1%    | 244.8 px       |
         | Combined | 91.3%       | 231.3 px       |
         | Motion Mask | ?%       | ? px           |
      3. Assert: motion_mask visible_ratio >= 0.90
      4. Document: If final distance lower than baseline → improvement!
    Expected Result: Motion mask achieves ≥90% visibility, competitive or better final distance
    Evidence: .sisyphus/evidence/task-4-comparison-report.txt (table + analysis)
  ```

  **Evidence to Capture**:
  - [ ] Full test logs (task-4-full-test-execution.txt)
  - [ ] Target points visual QA (task-4-target-points-visual-qa.png)
  - [ ] Motion mask quality check (task-4-motion-mask-quality.png)
  - [ ] Quantitative comparison table (task-4-comparison-report.txt)

  **Commit**: YES
  - Message: `test(motion-mask): validate motion-driven sampling with robot data`
  - Files: `.sisyphus/evidence/task-4-*.{txt,png}`, `outputs/test_motion_mask/` (add to .gitignore)
  - Pre-commit: N/A (evidence files)

---

## Final Verification Wave

*No separate final verification wave - Task 4 includes comprehensive validation.*

---

## Commit Strategy

- **Task 1-2**: Combined commit after both complete
  - `feat(mpc): add motion-driven mask sampling for improved tracking`
  - Files: `mpc/point_sampling.py`

- **Task 3**: Separate commit
  - `feat(test): integrate motion-driven sampling into test_cotracker_mpc`
  - Files: `test_cotracker_mpc.py`

- **Task 4**: Evidence commit
  - `test(motion-mask): validate motion-driven sampling with robot data`
  - Files: Evidence and output files

---

## Success Criteria

### Verification Commands
```bash
# Test CLI integration
python test_cotracker_mpc.py --sampling_method motion_mask --help

# Run full test
python test_cotracker_mpc.py \
  --sampling_method motion_mask \
  --image_height 512 \
  --image_width 512 \
  --num_tracking_points 384 \
  --num_steps 3 \
  --output_dir outputs/test_motion_mask

# Check visible ratio in logs
grep "Visible target points:" outputs/test_motion_mask/*.log
# Expected: ≥90% (e.g., "350/384")

# Visual inspection
ls outputs/test_motion_mask/{flow_magnitude_heatmap,motion_mask_overlay,01_initial_with_points,01_target_with_points}.png
# All files should exist
```

### Final Checklist
- [ ] Motion mask coverage is 5-80% (validated via assertion)
- [ ] >50% of sampled points fall in motion regions
- [ ] ≥90% of target points remain visible (TAPIR tracking)
- [ ] Target points visually on robot/cube (not background) - **CRITICAL USER REQUIREMENT**
- [ ] Diagnostic visualizations saved (flow, mask, point distribution)
- [ ] Quantitative comparison shows improvement over baseline

---

## Notes

### Research Sources (from Metis)
- **GMFlow official**: https://github.com/haofeixu/gmflow
- **UnFlow (ICCV 2017)**: Adaptive flow thresholding for motion detection
- **RoboTAP (DeepMind 2024)**: Motion clustering for dense tracking
- **Rerender_A_Video**: Production use of bidirectional flow consistency

### Future Improvements (Out of Scope v1)
- **Bidirectional flow consistency**: Compute forward + backward, check consistency
- **Occlusion-aware sampling**: Filter occluded regions at boundaries
- **Temporal consistency**: Track motion mask across multiple frames in MPC loop
- **SAM integration**: Use Segment Anything for precise object masks

### Known Limitations
- **One-way flow only**: v1 uses forward flow (initial → target), not bidirectional
  - Consequence: Occluded regions may get sampled (mitigated by Shi-Tomasi texture filtering)
- **Static threshold strategy**: Uses percentile, not adaptive alpha-beta from UnFlow
  - Consequence: May need manual tuning for drastically different scenes
- **No temporal propagation**: Computes fresh mask every time, doesn't track across frames
  - Consequence: Higher computational cost in MPC loop

### User Requirement Verification
**Original problem**: "目标图像无法获取所需要的跟踪点在目标图上的正确位置"
**Translation**: Target image cannot obtain correct tracking point positions

**How this plan solves it**:
1. Motion mask ensures points are sampled **only on moving objects** (robot arm + cube)
2. Static background (floor, walls) is **explicitly excluded** via flow thresholding
3. Target points **inherit from initial sampling** → if initial points are on robot/cube, target points will track those same objects
4. Visual QA explicitly checks target points are on robot/cube (**CRITICAL acceptance criterion**)

**Risk if motion mask is wrong**:
- If motion mask covers background → same problem persists
- Mitigation: Task 4 includes motion mask overlay visual QA to catch this

---

**Plan complete. Ready for execution via `/start-work`.**
