# Motion Mask Per-Step Resampling for MPC

## TL;DR

> **Quick Summary**: Implement Solution A - resample motion-driven tracking points at every MPC step to provide dense, accurate loss signals for CEM optimization, preventing first-step failures.
> 
> **Deliverables**: 
> - Modified MPC loop with per-step motion mask resampling
> - Updated failure recovery to use motion_mask method
> - Optional CLI flag for backward compatibility
> - Diagnostic outputs per step
> 
> **Estimated Effort**: Medium  
> **Parallel Execution**: NO - sequential modifications to single file  
> **Critical Path**: Task 1 → Task 2 → Task 3 → Task 4

---

## Context

### Original Request (Chinese + Translation)

**User's Problem Statement:**
> "但是实际情况是，在此cem策略评估的流程中，由于渲染需要的时间更长，所以在每一步之间插入一个gmflow的消耗实际是不大的，我现在需要的就是一个足够稠密的loss来完成规划任务，之前的你给出的规划方法由于1.追踪效果不好2.评估的动作不合理，这导致了渲染的图像出现了很大的问题，直接导致此规划结果从第一步开始就变成不可用的了
所以我现在需要的是，更好的追踪效果（通过光流方法将运动物体mask，在物体上进行追踪点获取），让策略得到更加合理的动作，这是否可行？"

**Translation:**
> "The actual situation is: in the CEM policy evaluation process, because rendering takes longer, inserting a GMFlow computation between steps doesn't add much overhead. What I need now is a sufficiently dense loss signal to complete the planning task. The previous planning method had problems due to: 1. Poor tracking quality 2. Unreasonable action evaluation, which caused major issues with rendered images, making the planning result unusable from the first step.
So what I need now is: better tracking quality (using optical flow to mask moving objects, sampling tracking points on objects), so that the policy gets more reasonable actions. Is this feasible?"

**Root Cause Analysis:**
```
Bad tracking points → Inaccurate loss signal → CEM optimizes wrong direction → 
Rendered images collapse → First step fails → Entire planning unusable
```

**Why Current Implementation Fails:**
1. **Tracking points drift to static background** - Loss doesn't reflect true motion objectives
2. **Loss signal too sparse/noisy** - CEM cannot find reasonable actions
3. **First step already wrong** - All subsequent steps fail

**User's Insight:**
- CEM rendering is already slow (100-500ms per sample)
- GMFlow overhead (~100ms per step) is **relatively small** (<10% of total time)
- **Dense, accurate loss signal is worth the cost**

### Pipeline Analysis Summary (from previous analysis)

**Current Implementation (Incremental TAPIR):**
- GMFlow called **ONCE** at initialization (line 262)
- TAPIR incrementally tracks current → next (line 492-497)
- **Problem**: Points drift to background over 20+ steps

**User's Expected Pipeline (Motion Re-sampling):**
- GMFlow should be called **at each MPC step** (current → target)
- Re-sample tracking points on moving objects
- **Benefit**: Points always focused on robot arm + cube

---

## Work Objectives

### Core Objective

Implement per-step motion mask resampling in the MPC outer loop to ensure tracking points always focus on moving objects, providing dense and accurate loss signals for CEM optimization.

### Concrete Deliverables

1. **Modified MPC Loop** (`test_cotracker_mpc.py` lines 447-543)
   - Add motion mask resampling at the beginning of each step
   - Update `current_tracked_points` with newly sampled points
   - Compute `target_points` via TAPIR tracking (current → target)

2. **Updated Failure Recovery** (`test_cotracker_mpc.py` lines 504-519)
   - Add `motion_mask` case to failure recovery block
   - Ensure consistency with per-step resampling

3. **Backward Compatibility**
   - Add CLI flag `--resample_motion_mask_per_step` (default: True for motion_mask, False for others)
   - Preserve original incremental tracking behavior for other sampling methods

4. **Diagnostic Outputs**
   - Save per-step motion mask visualizations
   - Save per-step point distributions
   - Log GMFlow computation times

### Definition of Done

- [ ] Motion mask resampling occurs at every MPC step (when `--sampling_method=motion_mask`)
- [ ] Tracking failure recovery includes motion_mask case
- [ ] Backward compatibility preserved via CLI flag
- [ ] Diagnostic outputs saved to `{output_dir}/step_{N:03d}/`
- [ ] Test run completes without errors on robot manipulation dataset
- [ ] Visual inspection shows points stay on robot arm + cube throughout trajectory

### Must Have

- Per-step `sample_motion_driven_points(current_image, target_image)` call
- TAPIR offline tracking (current → target) to get target_points
- Updated agent.set_goal() with fresh current/target points
- Diagnostic file outputs per step

### Must NOT Have (Guardrails)

- ❌ **No changes to CEM internal logic** - Keep changes isolated to MPC outer loop
- ❌ **No breaking changes to other sampling methods** - Only affect motion_mask behavior
- ❌ **No removal of incremental TAPIR tracking** - Make it optional, not deleted
- ❌ **No excessive logging** - Only essential diagnostics (user mentioned rendering is slow)
- ❌ **No hardcoded paths** - Use args.output_dir for all outputs

---

## Verification Strategy

### Test Decision
- **Infrastructure exists**: NO (research codebase, no formal tests)
- **Automated tests**: None (verification via manual inspection + metrics)
- **Framework**: N/A

### QA Policy

Every task MUST include agent-executed QA scenarios.  
Evidence saved to `.sisyphus/evidence/task-{N}-{scenario-slug}.{ext}`.

- **Execution verification**: Run `python test_cotracker_mpc.py` with motion_mask sampling
- **Visual inspection**: Check diagnostic images show points on moving objects
- **Quantitative check**: Verify visibility ratio stays >80% throughout trajectory
- **Loss monitoring**: Confirm CEM rewards improve (not diverge)

---

## Execution Strategy

### Parallel Execution Waves

**Sequential Only** - All tasks modify the same file (`test_cotracker_mpc.py`)

```
Wave 1 (Foundation):
├── Task 1: Add CLI flag for per-step resampling [quick]

Wave 2 (Core Implementation):
├── Task 2: Implement per-step motion mask resampling in MPC loop [unspecified-high]

Wave 3 (Failure Handling):
├── Task 3: Add motion_mask case to failure recovery [quick]

Wave 4 (Verification):
├── Task 4: Test on robot manipulation dataset [unspecified-high]

Wave FINAL (Review):
├── Task F1: Manual QA - Visual inspection [unspecified-high]
└── Task F2: Quantitative metrics check [unspecified-high]
```

### Dependency Matrix

- **Task 1**: — → Task 2
- **Task 2**: Task 1 → Task 3
- **Task 3**: Task 2 → Task 4
- **Task 4**: Task 3 → F1, F2
- **F1, F2**: Task 4 → —

### Agent Dispatch Summary

- **Wave 1**: 1 task → `quick`
- **Wave 2**: 1 task → `unspecified-high`
- **Wave 3**: 1 task → `quick`
- **Wave 4**: 1 task → `unspecified-high`
- **FINAL**: 2 tasks → `unspecified-high`

---

## TODOs

- [ ] 1. Add CLI flag for per-step motion mask resampling

  **What to do**:
  - Add `--resample_motion_mask_per_step` argument to argparse
  - Default: True if `args.sampling_method == "motion_mask"`, else False
  - Add help text explaining the feature
  - Validate flag is only effective when `sampling_method == "motion_mask"`
  
  **Must NOT do**:
  - Change default behavior for other sampling methods
  - Add complex conditional logic (keep it simple)
  
  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Simple argparse addition, ~10 lines of code
  - **Skills**: []
    - No special skills needed
  - **Skills Evaluated but Omitted**:
    - N/A
  
  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Sequential
  - **Blocks**: Task 2
  - **Blocked By**: None (can start immediately)
  
  **References**:
  
  **Code Patterns**:
  - `test_cotracker_mpc.py:175-177` - Existing sampling_method argument
    ```python
    parser.add_argument("--sampling_method", type=str, 
                        choices=["shi_tomasi", "combined", "texture", "sobel_hybrid", "grid", "motion_mask"],
                        default="motion_mask", help="...")
    ```
  - Follow this pattern for the new flag
  
  **Implementation Location**:
  - `test_cotracker_mpc.py:~178` - Add after existing sampling_method argument
  
  **Expected Output**:
  ```python
  parser.add_argument("--resample_motion_mask_per_step", action="store_true",
                      default=None,  # Will be set based on sampling_method
                      help="Re-sample motion mask at every MPC step (only effective with --sampling_method=motion_mask). "
                           "Default: True for motion_mask, False for others.")
  ```
  
  **Post-parsing Logic**:
  ```python
  # After args = parser.parse_args()
  if args.resample_motion_mask_per_step is None:
      args.resample_motion_mask_per_step = (args.sampling_method == "motion_mask")
  ```
  
  **Acceptance Criteria**:
  
  - [ ] Argument `--resample_motion_mask_per_step` added to parser
  - [ ] Default value set correctly based on sampling_method
  - [ ] Help text is clear and mentions motion_mask requirement
  - [ ] Code runs without errors: `python test_cotracker_mpc.py --help`
  
  **QA Scenarios (MANDATORY)**:
  
  ```
  Scenario: Verify CLI flag is recognized
    Tool: Bash
    Preconditions: None
    Steps:
      1. Run: python test_cotracker_mpc.py --help
      2. Grep output for "resample_motion_mask_per_step"
      3. Verify help text contains "motion_mask" keyword
    Expected Result: Help text displays new flag with clear description
    Failure Indicators: Flag not in help output, or unclear description
    Evidence: .sisyphus/evidence/task-1-cli-help.txt
  
  Scenario: Test default behavior (motion_mask → True)
    Tool: Bash (python script with args inspection)
    Preconditions: Task 1 implementation complete
    Steps:
      1. Create test script: print(args.resample_motion_mask_per_step)
      2. Run with --sampling_method=motion_mask
      3. Verify output is True
    Expected Result: Default is True when sampling_method=motion_mask
    Evidence: .sisyphus/evidence/task-1-default-true.txt
  
  Scenario: Test default behavior (shi_tomasi → False)
    Tool: Bash
    Preconditions: Task 1 implementation complete
    Steps:
      1. Run with --sampling_method=shi_tomasi
      2. Verify resample_motion_mask_per_step is False
    Expected Result: Default is False for non-motion_mask methods
    Evidence: .sisyphus/evidence/task-1-default-false.txt
  ```
  
  **Evidence to Capture**:
  - [ ] Help text output showing new flag
  - [ ] Test script output confirming default values
  
  **Commit**: YES
  - Message: `feat(mpc): add CLI flag for per-step motion mask resampling`
  - Files: `test_cotracker_mpc.py`
  - Pre-commit: `python test_cotracker_mpc.py --help | grep resample_motion_mask_per_step`

---

- [ ] 2. Implement per-step motion mask resampling in MPC loop

  **What to do**:
  - Modify MPC loop (lines 447-543) to add motion mask resampling at the beginning of each step
  - Create per-step output directory: `{output_dir}/step_{step:03d}/`
  - Call `sample_motion_driven_points(current_image, target_image, save_diagnostics=True, output_dir=step_dir)`
  - Use TAPIR to track current → target to get target_points
  - Update `agent.set_goal()` with fresh current_tracked_points and target_points
  - Add timing logs for GMFlow computation
  - Make resampling conditional on `args.resample_motion_mask_per_step`
  
  **Implementation Structure**:
  ```python
  for step in range(1, args.num_steps + 1):
      print(f"\n--- Step {step}/{args.num_steps} ---")
      
      # 🆕 NEW: Per-step motion mask resampling
      if args.resample_motion_mask_per_step and args.sampling_method == "motion_mask":
          step_dir = os.path.join(args.output_dir, f"step_{step:03d}")
          os.makedirs(step_dir, exist_ok=True)
          
          print(f"  🔄 Re-computing motion mask: current frame → target frame")
          start_time = time.time()
          
          current_tracked_points = point_sampling.sample_motion_driven_points(
              current_image,      # Current frame (not initial!)
              target_image,       # Target frame (fixed)
              num_points=args.num_tracking_points,
              device=args.device,
              motion_ratio=0.7,
              save_diagnostics=True,
              output_dir=step_dir
          )
          
          gmflow_time = time.time() - start_time
          print(f"    ✓ Sampled {len(current_tracked_points)} points on moving objects ({gmflow_time:.2f}s)")
          
          # Compute target points via TAPIR (current → target)
          print(f"  📍 Computing target point positions via TAPIR...")
          video_tensor_to_target = torch.stack([
              torch.from_numpy(current_image).permute(2, 0, 1).float() / 255.0,
              torch.from_numpy(target_image).permute(2, 0, 1).float() / 255.0
          ], dim=0).unsqueeze(0).to(args.device)
          
          tracks_to_target, visibles_to_target = tracker.track(
              video_tensor_to_target,
              current_tracked_points
          )
          target_points = tracks_to_target[0, :, 1, :].cpu().numpy()
          
          # Visualize resampled points
          vis_current_points = visualize_points(current_image, current_tracked_points, color=(255, 255, 0), radius=2)
          Image.fromarray((vis_current_points).astype(np.uint8)).save(
              os.path.join(step_dir, f"current_with_resampled_points.png"))
      
      # Set goal with (potentially) updated points
      agent.set_goal({
          'target_points': target_points,
          'current_tracked_points': current_tracked_points
      })
      
      # ... rest of existing MPC loop logic (CEM planning, rendering, etc.) ...
  ```
  
  **Must NOT do**:
  - Modify incremental TAPIR tracking logic (lines 492-497) - keep it for non-motion_mask methods
  - Change CEM internal logic
  - Remove existing diagnostic outputs
  
  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
    - Reason: Moderate complexity, modifying critical MPC loop, requires careful integration
  - **Skills**: []
    - No specialized skills needed (standard Python/PyTorch)
  - **Skills Evaluated but Omitted**:
    - `playwright`: Not applicable (no browser interaction)
    - `git-master`: Not needed at implementation stage
  
  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Sequential (after Task 1)
  - **Blocks**: Task 3, Task 4
  - **Blocked By**: Task 1 (requires CLI flag)
  
  **References**:
  
  **Existing Motion Mask Sampling** (initialization phase):
  - `test_cotracker_mpc.py:262-271` - Initial motion mask sampling
    - Use same function call pattern
    - Change: `initial_image` → `current_image`
    - Change: Save diagnostics to per-step directory
  
  **TAPIR Offline Tracking** (initial → target):
  - `test_cotracker_mpc.py:284-296` - Offline TAPIR tracking pattern
    - Reuse this pattern for current → target tracking
  
  **MPC Loop Structure**:
  - `test_cotracker_mpc.py:447-543` - Main MPC loop
    - Insert resampling logic at the beginning (after line 448)
  
  **Diagnostic Output Pattern**:
  - `test_cotracker_mpc.py:488-489` - Saving rendered images
    - Follow similar pattern for per-step diagnostics
  
  **Point Visualization**:
  - `test_cotracker_mpc.py:280-282` - Visualize points on image
    - Reuse `visualize_points()` function
  
  **External References**:
  - `mpc/point_sampling.py:431-561` - `sample_motion_driven_points()` implementation
    - Understand parameters: `save_diagnostics`, `output_dir`
    - Expected outputs: motion_mask.png, flow_magnitude_heatmap.png, etc.
  
  **Acceptance Criteria**:
  
  - [ ] Motion mask resampling occurs at step 1, 2, ..., num_steps
  - [ ] Per-step directories created: `{output_dir}/step_{001,002,...}/`
  - [ ] GMFlow computation time logged per step
  - [ ] Current and target points updated before agent.set_goal()
  - [ ] Resampling only triggers when `args.resample_motion_mask_per_step == True`
  - [ ] Diagnostic images saved per step: motion_mask_with_points.png, flow_magnitude_heatmap.png
  - [ ] Code runs without errors on test dataset
  
  **QA Scenarios (MANDATORY)**:
  
  ```
  Scenario: Verify per-step resampling triggers
    Tool: Bash
    Preconditions: Task 1 complete, test dataset available
    Steps:
      1. Run: python test_cotracker_mpc.py --sampling_method motion_mask --num_steps 3 \
              --initial_image assets/start-end/cam5_sample1_frame_00001.jpg \
              --target_image assets/start-end/cam5_sample1_frame_00018.jpg \
              --output_dir /tmp/test_resample --resample_motion_mask_per_step
      2. Check output contains "🔄 Re-computing motion mask" for steps 1, 2, 3
      3. Verify directories exist: /tmp/test_resample/step_001/, step_002/, step_003/
      4. Count GMFlow time logs (should be 3)
    Expected Result: Resampling occurs at every step, directories created
    Failure Indicators: Resampling not triggered, missing directories, no time logs
    Evidence: .sisyphus/evidence/task-2-resampling-triggered.txt
  
  Scenario: Verify diagnostic outputs per step
    Tool: Bash (ls + file inspection)
    Preconditions: Scenario 1 passed
    Steps:
      1. List files in /tmp/test_resample/step_001/
      2. Verify existence: motion_mask_with_points.png, flow_magnitude_heatmap.png
      3. Check file sizes > 10KB (non-empty images)
      4. Repeat for step_002/ and step_003/
    Expected Result: All diagnostic files present and non-empty
    Failure Indicators: Missing files, 0-byte files
    Evidence: .sisyphus/evidence/task-2-diagnostics.txt
  
  Scenario: Verify backward compatibility (non-motion_mask method)
    Tool: Bash
    Preconditions: Task 2 complete
    Steps:
      1. Run: python test_cotracker_mpc.py --sampling_method shi_tomasi --num_steps 3 \
              --initial_image assets/start-end/cam5_sample1_frame_00001.jpg \
              --target_image assets/start-end/cam5_sample1_frame_00018.jpg \
              --output_dir /tmp/test_shi_tomasi
      2. Check output does NOT contain "🔄 Re-computing motion mask"
      3. Verify incremental TAPIR tracking still works
    Expected Result: No per-step resampling for shi_tomasi, original behavior preserved
    Failure Indicators: Motion mask triggered, or tracking failed
    Evidence: .sisyphus/evidence/task-2-backward-compat.txt
  ```
  
  **Evidence to Capture**:
  - [ ] Terminal output showing per-step resampling logs
  - [ ] Directory listing showing per-step folders
  - [ ] File listing showing diagnostic outputs per step
  - [ ] Screenshot or file inspection of motion_mask_with_points.png
  
  **Commit**: YES
  - Message: `feat(mpc): implement per-step motion mask resampling in MPC loop`
  - Files: `test_cotracker_mpc.py`
  - Pre-commit: `python test_cotracker_mpc.py --help`  # Ensure no syntax errors

---

- [ ] 3. Add motion_mask case to failure recovery block

  **What to do**:
  - Modify failure recovery block (lines 504-519) to add motion_mask case
  - Call `sample_motion_driven_points(next_image_np, target_image)` when tracking fails
  - Use same pattern as per-step resampling (no diagnostics to save time)
  - Log re-sampling event
  
  **Implementation**:
  ```python
  # test_cotracker_mpc.py:504-519
  if failed:
      print(f"  ⚠️ Tracking failure detected: {failure_reason}")
      print(f"     Re-sampling points using {args.sampling_method} method...")
      
      if args.sampling_method == "motion_mask":
          # 🆕 NEW: Motion mask re-sampling on failure
          new_points = point_sampling.sample_motion_driven_points(
              next_image_np,
              target_image,  # Use original target from initialization
              num_points=args.num_tracking_points,
              device=args.device,
              motion_ratio=0.7,
              save_diagnostics=False,  # Don't save diagnostics on failure (save time)
              output_dir=args.output_dir
          )
          print(f"     Re-sampled {len(new_points)} points using motion mask")
      elif args.sampling_method == "sobel_hybrid":
          new_points = sample_object_focused_points(next_image_np, num_points=args.num_tracking_points, object_ratio=0.7)
      # ... rest of existing cases ...
  ```
  
  **Must NOT do**:
  - Remove existing failure recovery cases
  - Change failure detection logic
  
  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Simple addition of one elif branch, ~10 lines
  - **Skills**: []
  - **Skills Evaluated but Omitted**:
    - N/A
  
  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Sequential (after Task 2)
  - **Blocks**: Task 4
  - **Blocked By**: Task 2 (requires per-step resampling implementation)
  
  **References**:
  
  **Existing Failure Recovery Block**:
  - `test_cotracker_mpc.py:504-519` - Current failure recovery logic
    - Add motion_mask case at line ~508
  
  **Motion Mask Sampling Function**:
  - `mpc/point_sampling.py:431-561` - `sample_motion_driven_points()`
    - Use same function, set `save_diagnostics=False`
  
  **Failure Detection**:
  - `test_cotracker_mpc.py:499-502` - `detect_tracking_failure()` call
    - Don't modify this
  
  **Acceptance Criteria**:
  
  - [ ] `motion_mask` case added to failure recovery if-elif chain
  - [ ] Function call matches per-step resampling pattern (except save_diagnostics=False)
  - [ ] Log message indicates motion mask re-sampling
  - [ ] Code runs without errors when tracking fails
  
  **QA Scenarios (MANDATORY)**:
  
  ```
  Scenario: Verify motion_mask failure recovery triggers
    Tool: Bash (manual failure injection)
    Preconditions: Task 2, Task 3 complete
    Steps:
      1. Modify detect_tracking_failure() temporarily to always return (True, "test")
      2. Run: python test_cotracker_mpc.py --sampling_method motion_mask --num_steps 2 ...
      3. Check output contains "Re-sampling points using motion_mask method"
      4. Verify output contains "Re-sampled N points using motion mask"
      5. Restore original detect_tracking_failure()
    Expected Result: Motion mask re-sampling triggered on failure
    Failure Indicators: No re-sampling log, or wrong sampling method used
    Evidence: .sisyphus/evidence/task-3-failure-recovery.txt
  
  Scenario: Verify no diagnostics saved on failure (performance)
    Tool: Bash (file system check)
    Preconditions: Scenario 1 passed
    Steps:
      1. After triggering failure recovery, check {output_dir}
      2. Verify NO new motion_mask.png or flow_magnitude_heatmap.png in root output_dir
      3. Confirm save_diagnostics=False is used
    Expected Result: No diagnostic files saved on failure (faster recovery)
    Evidence: .sisyphus/evidence/task-3-no-diagnostics.txt
  ```
  
  **Evidence to Capture**:
  - [ ] Terminal output showing motion_mask failure recovery
  - [ ] File system check confirming no extra diagnostics
  
  **Commit**: YES
  - Message: `fix(mpc): add motion_mask case to tracking failure recovery`
  - Files: `test_cotracker_mpc.py`
  - Pre-commit: `python test_cotracker_mpc.py --help`

---

- [ ] 4. End-to-end test on robot manipulation dataset

  **What to do**:
  - Run full MPC test with per-step motion mask resampling
  - Use robot manipulation test data: cam5_sample1 (frame 1 → 18)
  - Verify all steps complete without errors
  - Check visibility ratio stays >80% throughout
  - Inspect rendered images for quality (no collapse)
  - Monitor CEM reward trends (should not diverge)
  - Collect performance metrics: GMFlow time per step, total MPC time
  
  **Test Command**:
  ```bash
  python test_cotracker_mpc.py \
    --sampling_method motion_mask \
    --resample_motion_mask_per_step \
    --num_steps 10 \
    --initial_image assets/start-end/cam5_sample1_frame_00001.jpg \
    --target_image assets/start-end/cam5_sample1_frame_00018.jpg \
    --model_path assets \
    --output_dir outputs/test_per_step_resample \
    --num_tracking_points 384 \
    --device cuda:0 \
    --horizon 5 \
    --cem_iterations 3 \
    --num_samples 10
  ```
  
  **Must NOT do**:
  - Modify code during testing (testing only, no fixes)
  - Skip error analysis if failures occur
  
  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
    - Reason: End-to-end testing with analysis, requires understanding MPC metrics
  - **Skills**: []
  - **Skills Evaluated but Omitted**:
    - `playwright`: Not applicable
  
  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Sequential (after Task 3)
  - **Blocks**: F1, F2
  - **Blocked By**: Task 3
  
  **References**:
  
  **Test Dataset**:
  - `assets/start-end/cam5_sample1_frame_00001.jpg` - Initial frame (512×512)
  - `assets/start-end/cam5_sample1_frame_00018.jpg` - Target frame (18 frames apart)
  
  **Expected Outputs**:
  - `outputs/test_per_step_resample/step_{001..010}/` - Per-step diagnostics
  - `outputs/test_per_step_resample/step_{0001..0010}_rendered.png` - Rendered frames
  - `outputs/test_per_step_resample/step_{0001..0010}_with_points.png` - Point visualizations
  - Terminal logs with CEM rewards, visibility ratios, GMFlow times
  
  **Success Indicators**:
  - All 10 steps complete without exceptions
  - Visibility ratio >80% at each step
  - CEM rewards improve (or stay stable, not diverge)
  - Rendered images show coherent robot motion (no artifacts/collapse)
  - GMFlow overhead <15% of total MPC time
  
  **Acceptance Criteria**:
  
  - [ ] Test completes all 10 steps without errors
  - [ ] Per-step directories created (step_001 through step_010)
  - [ ] Visibility ratio >80% at each step (check terminal logs)
  - [ ] CEM rewards do not diverge (best reward stays positive or increases)
  - [ ] Rendered images visually coherent (manual inspection in Task F1)
  - [ ] GMFlow time per step logged (check overhead is acceptable)
  
  **QA Scenarios (MANDATORY)**:
  
  ```
  Scenario: Full MPC run with per-step resampling
    Tool: Bash
    Preconditions: Tasks 1-3 complete, GPU available
    Steps:
      1. Run command above (10 steps, motion_mask, per-step resampling)
      2. Monitor terminal output for errors
      3. Check final line indicates completion
      4. Verify exit code is 0
    Expected Result: Script completes successfully
    Failure Indicators: Exception raised, non-zero exit code, hung process
    Evidence: .sisyphus/evidence/task-4-full-run.txt
  
  Scenario: Verify per-step output structure
    Tool: Bash (directory inspection)
    Preconditions: Scenario 1 passed
    Steps:
      1. Run: ls -R outputs/test_per_step_resample/
      2. Count step_* directories (should be 10)
      3. Check each step_* contains motion_mask_with_points.png, flow_magnitude_heatmap.png
      4. Check root contains step_{0001..0010}_rendered.png
    Expected Result: All expected files present
    Failure Indicators: Missing directories, missing files
    Evidence: .sisyphus/evidence/task-4-output-structure.txt
  
  Scenario: Verify visibility ratio stays high
    Tool: Bash (grep terminal output)
    Preconditions: Scenario 1 passed
    Steps:
      1. Grep terminal output for "Tracking status: OK"
      2. Extract visibility percentages from each step
      3. Verify all steps have >80% visibility
    Expected Result: All steps have >80% visible points
    Failure Indicators: Any step <80% visibility
    Evidence: .sisyphus/evidence/task-4-visibility.txt
  
  Scenario: Monitor CEM reward trends
    Tool: Bash (grep terminal output)
    Preconditions: Scenario 1 passed
    Steps:
      1. Grep terminal output for "Best Reward:"
      2. Extract best reward values for steps 1-10
      3. Check rewards are not all negative, and trend is stable/improving
    Expected Result: Rewards are reasonable (not diverging to -infinity)
    Failure Indicators: All rewards negative and decreasing
    Evidence: .sisyphus/evidence/task-4-rewards.txt
  
  Scenario: Measure GMFlow overhead
    Tool: Bash (grep terminal output)
    Preconditions: Scenario 1 passed
    Steps:
      1. Grep for GMFlow time logs: "Sampled N points ... (X.XXs)"
      2. Sum GMFlow times across all steps
      3. Compare to total MPC time (from final log)
      4. Calculate overhead percentage
    Expected Result: GMFlow overhead <15% of total time (as user predicted)
    Evidence: .sisyphus/evidence/task-4-timing.txt
  ```
  
  **Evidence to Capture**:
  - [ ] Full terminal output (saved to .txt)
  - [ ] Directory structure listing
  - [ ] Visibility ratios extracted from logs
  - [ ] CEM reward values per step
  - [ ] Timing breakdown (GMFlow vs total)
  
  **Commit**: NO (testing only, no code changes)

---

## Final Verification Wave (MANDATORY)

- [ ] F1. Manual QA - Visual Inspection

  **What to do**:
  - Open all rendered images (`step_{0001..0010}_rendered.png`) in sequence
  - Verify robot arm and cube are visible and moving coherently
  - Check for visual artifacts: black screens, extreme blur, distortions
  - Open point visualizations (`step_{0001..0010}_with_points.png`)
  - Verify tracking points stay on robot arm + cube (not drifting to background)
  - Compare to previous implementation (if available) to confirm improvement
  
  **Manual Inspection Checklist**:
  - [ ] Rendered images show coherent robot motion (not collapsed)
  - [ ] No black screens or extreme artifacts
  - [ ] Tracking points visually on moving objects (robot + cube)
  - [ ] Points NOT drifting to static background (floor/walls)
  - [ ] Motion mask visualizations show correct segmentation
  
  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
    - Reason: Requires visual judgment and comparison
  - **Skills**: []
  
  **Parallelization**:
  - **Can Run In Parallel**: YES (with F2)
  - **Parallel Group**: FINAL wave (after Task 4)
  - **Blocks**: None
  - **Blocked By**: Task 4
  
  **References**:
  - `outputs/test_per_step_resample/step_*_rendered.png` - Rendered images
  - `outputs/test_per_step_resample/step_*_with_points.png` - Point visualizations
  - `outputs/test_per_step_resample/step_*/motion_mask_with_points.png` - Motion masks
  
  **Acceptance Criteria**:
  
  - [ ] All rendered images visually coherent (no collapse)
  - [ ] Tracking points stay on moving objects throughout trajectory
  - [ ] Motion masks correctly segment robot arm + cube
  - [ ] Visual quality comparable or better than previous implementation
  
  **QA Scenarios (MANDATORY)**:
  
  ```
  Scenario: Visual inspection of rendered frames
    Tool: Bash (image viewer + manual inspection)
    Preconditions: Task 4 complete
    Steps:
      1. Open step_0001_rendered.png through step_0010_rendered.png in sequence
      2. Check each image for: coherent robot pose, visible cube, no artifacts
      3. Note any frames with issues (frame number + description)
    Expected Result: All frames show coherent motion, no collapsed images
    Failure Indicators: Black screens, extreme blur, distorted geometry
    Evidence: .sisyphus/evidence/task-f1-rendered-inspection.txt (manual notes)
  
  Scenario: Tracking point distribution inspection
    Tool: Bash (image viewer + manual inspection)
    Preconditions: Task 4 complete
    Steps:
      1. Open step_0001_with_points.png through step_0010_with_points.png
      2. For each image, visually check if points are on robot arm + cube
      3. Count approximate percentage of points on moving objects vs background
    Expected Result: >70% of points on moving objects throughout trajectory
    Failure Indicators: Most points on static background (floor/walls)
    Evidence: .sisyphus/evidence/task-f1-points-inspection.txt (manual notes)
  
  Scenario: Motion mask quality inspection
    Tool: Bash (image viewer + manual inspection)
    Preconditions: Task 4 complete
    Steps:
      1. Open step_001/motion_mask_with_points.png through step_010/motion_mask_with_points.png
      2. Check if motion mask (white regions) covers robot arm + cube
      3. Verify points are sampled within motion regions
    Expected Result: Motion masks correctly segment moving objects
    Failure Indicators: Motion mask empty, or covers wrong regions
    Evidence: .sisyphus/evidence/task-f1-mask-inspection.txt (manual notes)
  ```
  
  **Evidence to Capture**:
  - [ ] Manual notes on rendered image quality
  - [ ] Manual notes on point distribution
  - [ ] Manual notes on motion mask quality
  
  **Commit**: NO (verification only)

---

- [ ] F2. Quantitative Metrics Check

  **What to do**:
  - Extract quantitative metrics from terminal logs and output files
  - Compute summary statistics: mean/std of visibility ratios, CEM rewards, GMFlow times
  - Compare to baseline (if available) or expected values
  - Identify any anomalies or failure modes
  - Generate summary report
  
  **Metrics to Extract**:
  1. **Visibility Ratios** (per step)
     - Mean, min, max across all steps
     - Number of steps with <80% visibility
  2. **CEM Rewards** (per step)
     - Best reward per step
     - Mean reward per step
     - Trend (improving/stable/degrading)
  3. **GMFlow Timing**
     - Time per step
     - Total GMFlow time
     - Percentage of total MPC time
  4. **Tracking Failure Events**
     - Number of failures across all steps
     - Steps where failure occurred
  
  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
    - Reason: Requires parsing logs and statistical analysis
  - **Skills**: []
  
  **Parallelization**:
  - **Can Run In Parallel**: YES (with F1)
  - **Parallel Group**: FINAL wave (after Task 4)
  - **Blocks**: None
  - **Blocked By**: Task 4
  
  **References**:
  - Terminal output from Task 4 (saved to .txt)
  - Output files in `outputs/test_per_step_resample/`
  
  **Acceptance Criteria**:
  
  - [ ] Visibility ratio mean >80%
  - [ ] CEM rewards do not diverge (not all negative)
  - [ ] GMFlow overhead <20% of total time (acceptable per user's insight)
  - [ ] Tracking failures <3 across all 10 steps
  - [ ] Summary report generated
  
  **QA Scenarios (MANDATORY)**:
  
  ```
  Scenario: Extract and analyze visibility ratios
    Tool: Bash (grep + awk/python)
    Preconditions: Task 4 complete
    Steps:
      1. Parse terminal output for "Tracking status: OK (X/Y visible = Z%)"
      2. Extract visibility percentages for all steps
      3. Compute mean, min, max
      4. Count steps with <80% visibility
    Expected Result: Mean visibility >80%, min >70%
    Failure Indicators: Mean <80%, or multiple steps <70%
    Evidence: .sisyphus/evidence/task-f2-visibility-stats.txt
  
  Scenario: Analyze CEM reward trends
    Tool: Bash (grep + awk/python)
    Preconditions: Task 4 complete
    Steps:
      1. Parse terminal output for "Best Reward: X"
      2. Extract best rewards for all steps
      3. Compute mean, check for negative trend
      4. Verify rewards are reasonable (not -infinity)
    Expected Result: Mean reward >0 (or at least stable, not diverging to -inf)
    Failure Indicators: All rewards negative and decreasing
    Evidence: .sisyphus/evidence/task-f2-reward-stats.txt
  
  Scenario: Measure GMFlow timing overhead
    Tool: Bash (grep + awk/python)
    Preconditions: Task 4 complete
    Steps:
      1. Parse terminal output for GMFlow times: "Sampled ... (X.XXs)"
      2. Sum GMFlow times across all steps
      3. Extract total MPC time from final summary
      4. Compute overhead percentage: (GMFlow_total / MPC_total) * 100
    Expected Result: Overhead <20% (user expects <10%, allow some margin)
    Failure Indicators: Overhead >30%
    Evidence: .sisyphus/evidence/task-f2-timing-stats.txt
  
  Scenario: Count tracking failure events
    Tool: Bash (grep)
    Preconditions: Task 4 complete
    Steps:
      1. Grep terminal output for "⚠️ Tracking failure detected"
      2. Count number of occurrences
      3. Extract step numbers where failures occurred
    Expected Result: <3 failures across 10 steps
    Failure Indicators: >5 failures
    Evidence: .sisyphus/evidence/task-f2-failure-count.txt
  ```
  
  **Evidence to Capture**:
  - [ ] Visibility statistics summary
  - [ ] CEM reward statistics summary
  - [ ] Timing breakdown and overhead percentage
  - [ ] Tracking failure count and locations
  
  **Commit**: NO (verification only)

---

## Commit Strategy

- **Task 1**: `feat(mpc): add CLI flag for per-step motion mask resampling` - test_cotracker_mpc.py
- **Task 2**: `feat(mpc): implement per-step motion mask resampling in MPC loop` - test_cotracker_mpc.py
- **Task 3**: `fix(mpc): add motion_mask case to tracking failure recovery` - test_cotracker_mpc.py

**Pre-commit checks:**
- `python test_cotracker_mpc.py --help` (verify syntax)
- `git diff test_cotracker_mpc.py` (review changes)

---

## Success Criteria

### Verification Commands

```bash
# 1. Verify CLI flag exists
python test_cotracker_mpc.py --help | grep resample_motion_mask_per_step

# 2. Run end-to-end test
python test_cotracker_mpc.py \
  --sampling_method motion_mask \
  --resample_motion_mask_per_step \
  --num_steps 10 \
  --initial_image assets/start-end/cam5_sample1_frame_00001.jpg \
  --target_image assets/start-end/cam5_sample1_frame_00018.jpg \
  --model_path assets \
  --output_dir outputs/test_per_step_resample \
  --device cuda:0

# Expected: Completes successfully, per-step directories created

# 3. Verify per-step diagnostics
ls -la outputs/test_per_step_resample/step_001/
# Expected: motion_mask_with_points.png, flow_magnitude_heatmap.png

# 4. Check visibility ratio
grep "Tracking status: OK" outputs/test_per_step_resample/*.txt
# Expected: >80% visibility at each step

# 5. Check CEM rewards
grep "Best Reward:" outputs/test_per_step_resample/*.txt
# Expected: Rewards are not all negative, stable or improving
```

### Final Checklist

**Functional Requirements:**
- [ ] Per-step motion mask resampling implemented
- [ ] TAPIR offline tracking (current → target) per step
- [ ] agent.set_goal() updated with fresh points per step
- [ ] Failure recovery includes motion_mask case
- [ ] Backward compatibility preserved for other sampling methods

**Quality Requirements:**
- [ ] Visibility ratio >80% throughout trajectory
- [ ] CEM rewards do not diverge
- [ ] Tracking points stay on moving objects (visual inspection)
- [ ] Rendered images coherent (no collapse)

**Performance Requirements:**
- [ ] GMFlow overhead <20% of total MPC time
- [ ] Per-step diagnostic outputs saved correctly
- [ ] No significant slowdown compared to user's expectations

**Code Quality:**
- [ ] Clean git commits with descriptive messages
- [ ] No commented-out code or debug prints (except essential logs)
- [ ] Code follows existing style (4-space indent, docstrings)

---

**Plan Version:** 1.0  
**Created:** 2026-03-10  
**Author:** Prometheus (Strategic Planning Consultant)
