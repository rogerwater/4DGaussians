# Checkpoint Interface Alignment Plan

## TL;DR

> **Quick Summary**: Fix 3 wrong `getattr()` defaults in `deformation_triplane.py` that cause checkpoint loading failures.
> 
> **Deliverables**: 
> - Updated `scene/deformation_triplane.py` with correct defaults
> - Verification test confirming checkpoint loads successfully
> 
> **Estimated Effort**: Quick (5 min code change + 2 min test)
> **Parallel Execution**: NO - Single file change, sequential
> **Critical Path**: Fix defaults → Test loading → Done

---

## Context

### Original Request
用户报告 checkpoint 加载失败，要求全面分析 checkpoint 参数并生成对齐计划。

### Issue Summary
Checkpoint 加载时报错：
```
RuntimeError: Error(s) in loading state_dict:
  Missing key(s): "deformation_net.action_processor.freq_bands"
  size mismatch for mlp.0.weight: [128, 15] vs [128, 54]
  size mismatch for mlp.6.weight: [32, 128] vs [64, 128]
```

### Root Cause Identified
✅ **Checkpoint analysis complete** - 详细报告见 `.sisyphus/evidence/checkpoint-config-analysis.md`

**问题根源:**
- Config 文件 `arguments/toyarm/triplane.py` **配置正确** (15, False, 32)
- 代码 `scene/deformation_triplane.py` 的 `getattr()` **默认值错误** (6, True, 64)
- 当 config 未加载或属性缺失时，使用错误默认值导致模型结构不匹配

---

## Work Objectives

### Core Objective
修复 `scene/deformation_triplane.py` 中的 3 个 `getattr()` 默认值，使其与训练 checkpoint 的配置一致。

### Concrete Deliverables
- `scene/deformation_triplane.py` (lines 187-192) - 修复 3 个默认值
- `.sisyphus/evidence/checkpoint-load-test.txt` - 加载测试成功的证据

### Definition of Done
- [ ] `action_input_dim` default: 6 → **15**
- [ ] `action_use_pe` default: True → **False**
- [ ] `action_output_dim` default: 64 → **32**
- [ ] Checkpoint 成功加载，无 size mismatch 错误
- [ ] `load_state_dict()` 返回空的 `missing_keys` 和 `unexpected_keys`

### Must Have
- 保持向后兼容性：已训练的 checkpoint 必须能加载
- 代码注释说明为何使用这些默认值

### Must NOT Have (Guardrails)
- ❌ 不修改 config 文件 (已经是正确的)
- ❌ 不修改 checkpoint 文件本身
- ❌ 不添加 key remapping 逻辑
- ❌ 不改变任何模型架构或功能
- ❌ 不影响其他数据集配置

---

## Verification Strategy

### Test Decision
- **Infrastructure exists**: YES (Python unittest, pytest)
- **Automated tests**: None (manual verification only)
- **Framework**: Manual Python script

### QA Policy
Every task includes agent-executed verification via Python script.
Evidence saved to `.sisyphus/evidence/task-{N}-{scenario}.txt`.

---

## Execution Strategy

### Sequential Execution (No Parallelization)

**Task 1** → **Task 2** (Sequential dependency)

```
Task 1: Fix getattr() defaults
  ↓
Task 2: Verify checkpoint loading
```

**Why Sequential:**
- Only 2 tasks, Task 2 depends on Task 1 completion
- No benefit from parallelization

---

## TODOs

- [ ] 1. Fix getattr() default values in deformation_triplane.py

  **What to do**:
  - Open `scene/deformation_triplane.py`
  - Locate lines 187-192 (ActionProcessor initialization)
  - Change 3 default values in `getattr()` calls:
    - Line 187: `getattr(args, 'action_use_pe', True)` → `getattr(args, 'action_use_pe', False)`
    - Line 189: `getattr(args, 'action_input_dim', 6)` → `getattr(args, 'action_input_dim', 15)`
    - Line 192: `getattr(args, 'action_output_dim', 64)` → `getattr(args, 'action_output_dim', 32)`
  - Add inline comment explaining these defaults match trained checkpoints:
    ```python
    # Defaults match dm_control_push8 checkpoint (iteration_20000)
    # action_input_dim=15 (joint state dim), use_pe=False, output_dim=32 (FiLM input)
    ```

  **Must NOT do**:
  - Modify any other lines
  - Change variable names
  - Add new parameters
  - Modify logic beyond default values

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Simple 3-line edit with clear specification
  - **Skills**: None
  - **Skills Evaluated but Omitted**:
    - `git-master`: Not needed - simple edit, no complex git operations

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Sequential (Task 1 must complete first)
  - **Blocks**: Task 2 (verification depends on this fix)
  - **Blocked By**: None (can start immediately)

  **References**:

  **Pattern References**:
  - `.sisyphus/evidence/checkpoint-config-analysis.md` - Complete analysis showing checkpoint uses (15, False, 32)
  - `scene/deformation_triplane.py:187-192` - Current code with wrong defaults

  **API/Type References**:
  - None (simple value changes)

  **External References**:
  - Checkpoint file: `/home/ubuntu/yyf/4DGaussians/outputs/dm_control_push8/point_cloud/iteration_20000/deformation.pth`
    - Shows `action_processor.mlp.0.weight: [128, 15]` → input_dim=15
    - Shows `action_processor.mlp.6.weight: [32, 128]` → output_dim=32
    - Missing `freq_bands` key → use_pe=False

  **Acceptance Criteria**:

  - [ ] `scene/deformation_triplane.py` line 187 reads: `getattr(args, 'action_use_pe', False)`
  - [ ] `scene/deformation_triplane.py` line 189 reads: `getattr(args, 'action_input_dim', 15)`
  - [ ] `scene/deformation_triplane.py` line 192 reads: `getattr(args, 'action_output_dim', 32)`
  - [ ] Inline comment added explaining why these are the defaults

  **QA Scenarios (MANDATORY)**:

  ```
  Scenario: Code modification verified
    Tool: Read
    Preconditions: File edited
    Steps:
      1. Read scene/deformation_triplane.py lines 187-192
      2. Verify line 187 contains: "getattr(args, 'action_use_pe', False)"
      3. Verify line 189 contains: "getattr(args, 'action_input_dim', 15)"
      4. Verify line 192 contains: "getattr(args, 'action_output_dim', 32)"
      5. Verify comment mentions checkpoint compatibility
    Expected Result: All 3 defaults changed to (False, 15, 32), comment present
    Failure Indicators: Any default still shows old value (True, 6, 64)
    Evidence: .sisyphus/evidence/task-1-code-verify.txt

  Scenario: No unintended changes
    Tool: Bash (git diff)
    Preconditions: File edited
    Steps:
      1. Run: git diff scene/deformation_triplane.py
      2. Count changed lines (should be 3-4: three getattr + one comment)
      3. Verify no changes outside lines 187-192
    Expected Result: Only 3-4 lines changed, all within expected range
    Failure Indicators: Changes to other lines, modifications to logic
    Evidence: .sisyphus/evidence/task-1-diff.txt
  ```

  **Evidence to Capture:**
  - [ ] task-1-code-verify.txt (Read output showing new defaults)
  - [ ] task-1-diff.txt (git diff showing minimal changes)

  **Commit**: YES
  - Message: `fix(scene): update ActionProcessor defaults to match checkpoint config`
  - Files: `scene/deformation_triplane.py`
  - Pre-commit: None (no tests to run)

---

- [ ] 2. Verify checkpoint loads successfully

  **What to do**:
  - Create Python script to test checkpoint loading
  - Load checkpoint at `/home/ubuntu/yyf/4DGaussians/outputs/dm_control_push8/point_cloud/iteration_20000/deformation.pth`
  - Import deformation network with fixed defaults
  - Call `load_state_dict()` and capture result
  - Verify no missing keys, no size mismatches
  - Save test output to evidence file

  **Must NOT do**:
  - Modify checkpoint file
  - Modify model code (only use fixed code from Task 1)
  - Add workarounds or key remapping

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Simple verification script, straightforward test
  - **Skills**: None
  - **Skills Evaluated but Omitted**:
    - `playwright`: Not needed - no browser interaction
    - `git-master`: Not needed - simple test script

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Sequential (after Task 1)
  - **Blocks**: None (final task)
  - **Blocked By**: Task 1 (needs fixed defaults to work)

  **References**:

  **Pattern References**:
  - `.sisyphus/evidence/checkpoint-config-analysis.md` - Expected checkpoint structure
  - `mpc/gaussian_dynamics_model.py:233` - Checkpoint loading pattern (for reference)

  **API/Type References**:
  - PyTorch `torch.load()` - https://pytorch.org/docs/stable/generated/torch.load.html
  - PyTorch `load_state_dict()` - https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.load_state_dict

  **Test References**:
  - None (new test script)

  **External References**:
  - Checkpoint: `/home/ubuntu/yyf/4DGaussians/outputs/dm_control_push8/point_cloud/iteration_20000/deformation.pth`

  **Acceptance Criteria**:

  - [ ] Python test script created and executed
  - [ ] `torch.load()` successfully loads checkpoint
  - [ ] Deformation network instantiated with args from `arguments/toyarm/triplane.py`
  - [ ] `load_state_dict()` returns with `missing_keys=[]` and `unexpected_keys=[]`
  - [ ] No `RuntimeError` about size mismatch
  - [ ] Evidence file saved showing successful load

  **QA Scenarios (MANDATORY)**:

  ```
  Scenario: Checkpoint loads without errors
    Tool: Bash (Python script)
    Preconditions: Task 1 completed, defaults fixed
    Steps:
      1. Create test script:
         ```python
         import torch
         from scene.deformation_factory import create_deform_network
         from arguments.toyarm.triplane import ModelHiddenParams
         
         # Load checkpoint
         ckpt = torch.load('/home/ubuntu/yyf/4DGaussians/outputs/dm_control_push8/point_cloud/iteration_20000/deformation.pth')
         
         # Create args object
         class Args:
             pass
         args = Args()
         for k, v in ModelHiddenParams.items():
             setattr(args, k, v)
         
         # Create deformation network
         deform_net = create_deform_network(args)
         
         # Load state dict
         result = deform_net.load_state_dict(ckpt, strict=True)
         
         print(f"Missing keys: {result.missing_keys}")
         print(f"Unexpected keys: {result.unexpected_keys}")
         print("✅ Checkpoint loaded successfully!")
         ```
      2. Run: python test_checkpoint_load.py
      3. Verify output shows empty missing/unexpected keys
      4. Verify "✅ Checkpoint loaded successfully!" printed
    Expected Result: Script runs without errors, all keys match, success message printed
    Failure Indicators: RuntimeError, missing_keys not empty, size mismatch errors
    Evidence: .sisyphus/evidence/task-2-load-success.txt

  Scenario: Verify ActionProcessor structure matches
    Tool: Bash (Python inspection)
    Preconditions: Checkpoint loaded in previous scenario
    Steps:
      1. Extend test script to print ActionProcessor layer shapes:
         ```python
         print("ActionProcessor structure:")
         for name, param in deform_net.action_processor.named_parameters():
             print(f"  {name}: {list(param.shape)}")
         ```
      2. Verify mlp.0.weight is [128, 15]
      3. Verify mlp.6.weight is [32, 128]
      4. Verify no freq_bands parameter exists
    Expected Result: All shapes match checkpoint exactly
    Failure Indicators: Shape mismatches, unexpected freq_bands parameter
    Evidence: .sisyphus/evidence/task-2-structure-verify.txt
  ```

  **Evidence to Capture:**
  - [ ] task-2-load-success.txt (Test script output showing successful load)
  - [ ] task-2-structure-verify.txt (ActionProcessor structure inspection)

  **Commit**: NO (test script is temporary verification, not production code)

---

## Final Verification Wave

> **All implementation complete** - No additional review needed for this simple fix.
> User can manually verify by running their original test script.

---

## Commit Strategy

- **Task 1**: `fix(scene): update ActionProcessor defaults to match checkpoint config` — `scene/deformation_triplane.py`

---

## Success Criteria

### Verification Commands
```bash
# Load checkpoint and verify no errors
python -c "
import torch
from scene.deformation_triplane import deform_network_triplane
from arguments.toyarm import triplane

class Args: pass
args = Args()
for k, v in triplane.ModelHiddenParams.items():
    setattr(args, k, v)

ckpt = torch.load('outputs/dm_control_push8/point_cloud/iteration_20000/deformation.pth')
net = deform_network_triplane(args)
result = net.load_state_dict(ckpt, strict=True)
assert len(result.missing_keys) == 0, f'Missing: {result.missing_keys}'
assert len(result.unexpected_keys) == 0, f'Unexpected: {result.unexpected_keys}'
print('✅ SUCCESS: Checkpoint loaded with zero mismatches')
"
# Expected: ✅ SUCCESS message, no errors
```

### Final Checklist
- [ ] All "Must Have" present:
  - [ ] Default values updated (15, False, 32)
  - [ ] Inline comment added explaining checkpoint compatibility
- [ ] All "Must NOT Have" absent:
  - [ ] Config file unchanged
  - [ ] Checkpoint file unchanged
  - [ ] No key remapping logic added
  - [ ] Model architecture unchanged
- [ ] Checkpoint loads successfully with `strict=True`
- [ ] No missing or unexpected keys
- [ ] User's original test script (`test_cotracker_mpc.py`) runs without RuntimeError
