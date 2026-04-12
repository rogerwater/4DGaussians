# Fix Flow Numpy-Tensor Type Mismatch

## TL;DR

> **Quick Summary**: Fix unconditional numpy conversion in flow prediction to respect `grad_enabled` flag for CEM-GD gradient computation.
> 
> **Deliverables**: 
> - Updated `mpc/flow_guided_gaussian_model.py` line 687 with grad-aware type handling
> - Test verification for both gradient and non-gradient modes
> 
> **Estimated Effort**: Quick (5 min fix + 10 min test)
> **Parallel Execution**: NO - Single file change, sequential
> **Critical Path**: Fix line 687 → Test CEM-GD → Test pure CEM → Done

---

## Context

### Original Error
```python
TypeError: expected Tensor as element 0 in argument 0, but got numpy.ndarray
  at flow_guided_gaussian_model.py:729 in torch.stack(predictions['flow'], dim=1)
```

### User Question
> "能否解释下两者的关系并且写出新问题的修改计划"
> (Can you explain the relationship between this bug and the previous modification, and write a fix plan?)

### Root Cause Analysis Complete
详细分析见 `.sisyphus/evidence/numpy-tensor-type-mismatch-analysis.md`

**问题根源:**
- Line 687: `predictions['flow'].append(next_flow.cpu().numpy())` **无条件转numpy**
- Line 729: `torch.stack(predictions['flow'], dim=1)` 期望 tensor（当 `grad_enabled=True` 时）
- RGB 处理已修复（lines 706-719），但 flow 处理被遗漏

**与之前修改的关系:**
- ❌ 不是 ActionProcessor 重命名导致的（不同模块）
- ✅ 是同样的 numpy/tensor 冲突模式（CEM-GD 引入梯度计算时的遗留bug）
- ✅ RGB 已在 commit 4ef1191 中修复，但 flow 被遗漏

---

## Work Objectives

### Core Objective
修复 `mpc/flow_guided_gaussian_model.py` line 687，使 flow 预测根据 `grad_enabled` 选择正确的数据类型。

### Concrete Deliverables
- `mpc/flow_guided_gaussian_model.py` (line 687) - 添加 grad_enabled 条件分支
- Test evidence 验证两种模式均正常工作

### Definition of Done
- [ ] Line 687 根据 `grad_enabled` 分支处理
- [ ] `grad_enabled=True`: append torch.Tensor（保留梯度）
- [ ] `grad_enabled=False`: append numpy.ndarray（原有行为）
- [ ] CEM-GD 测试通过（不再报 TypeError）
- [ ] Pure CEM 测试通过（向后兼容）

### Must Have
- 精确匹配 RGB 处理模式（已验证正确）
- 向后兼容：非梯度模式行为不变
- 保留梯度：梯度模式下 tensor 保持在 GPU

### Must NOT Have (Guardrails)
- ❌ 不修改 stacking 逻辑（lines 724-729 已经正确）
- ❌ 不修改 RGB 处理（已经正确）
- ❌ 不破坏非梯度模式（纯 CEM 仍需工作）
- ❌ 不添加不必要的类型转换（避免性能损失）

---

## Verification Strategy

### Test Decision
- **Infrastructure exists**: YES (pytest, manual scripts)
- **Automated tests**: Manual verification via test_cotracker_mpc.py
- **Framework**: Python test scripts

### QA Policy
Every task includes agent-executed QA scenarios.
Evidence saved to `.sisyphus/evidence/task-{N}-{scenario}.txt`.

---

## Execution Strategy

### Sequential Execution (No Parallelization)

**Task 1** → **Task 2** (Sequential dependency)

```
Task 1: Fix grad-aware flow append logic
  ↓
Task 2: Verify both gradient and non-gradient modes
```

**Why Sequential:**
- Only 2 tasks, Task 2 验证 Task 1 的修复
- 简单修改，不需要并行

---

## TODOs

- [ ] 1. Fix flow append to respect grad_enabled flag

  **What to do**:
  - Open `mpc/flow_guided_gaussian_model.py`
  - Locate line 687: `predictions['flow'].append(next_flow.cpu().numpy())`
  - Replace with conditional logic matching RGB pattern (lines 714-719):
    ```python
    # Line 687 (BEFORE)
    predictions['flow'].append(next_flow.cpu().numpy())
    
    # Line 687-691 (AFTER)
    if grad_enabled:
        # 保留tensor用于梯度计算
        predictions['flow'].append(next_flow)
    else:
        # 转换为numpy（原始行为）
        predictions['flow'].append(next_flow.cpu().numpy())
    ```
  - Add inline comment explaining the purpose (match RGB pattern at lines 706-711)

  **Must NOT do**:
  - Modify lines 724-729 (stacking logic - already correct)
  - Modify RGB handling (lines 706-719 - already correct)
  - Change any other logic in the forward() method

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Simple 5-line edit with clear pattern to follow
  - **Skills**: None
  - **Skills Evaluated but Omitted**:
    - `git-master`: Not needed - simple edit

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Sequential (Task 1 must complete first)
  - **Blocks**: Task 2 (verification depends on this fix)
  - **Blocked By**: None (can start immediately)

  **References**:

  **Pattern References**:
  - `.sisyphus/evidence/numpy-tensor-type-mismatch-analysis.md` - Complete bug analysis
  - `mpc/flow_guided_gaussian_model.py:706-719` - RGB handling pattern (CORRECT implementation to copy)
    ```python
    # Lines 706-711 (RGB pattern to copy for flow)
    if grad_enabled:
        # 保留tensor用于梯度计算
        full_rgb_hwc = full_rgb.permute(1, 2, 0)  # (H, W, 3) tensor
    else:
        # 转换为numpy（原始行为）
        full_rgb_hwc = full_rgb.permute(1, 2, 0).cpu().numpy()  # (H, W, 3)
    ```

  **API/Type References**:
  - `torch.Tensor.cpu()` - Move tensor to CPU
  - `torch.Tensor.numpy()` - Convert tensor to numpy array
  - PyTorch autograd - Requires tensors to maintain gradient graph

  **External References**:
  - PyTorch autograd mechanics: https://pytorch.org/docs/stable/notes/autograd.html

  **Acceptance Criteria**:

  - [ ] Line 687-691 contains conditional logic: `if grad_enabled: ... else: ...`
  - [ ] When `grad_enabled=True`: appends `next_flow` (tensor)
  - [ ] When `grad_enabled=False`: appends `next_flow.cpu().numpy()` (numpy)
  - [ ] Inline comment added explaining grad-aware handling
  - [ ] Code style matches RGB pattern (lines 706-711)

  **QA Scenarios (MANDATORY)**:

  ```
  Scenario: Code modification verified
    Tool: Read
    Preconditions: File edited
    Steps:
      1. Read mpc/flow_guided_gaussian_model.py lines 685-695
      2. Verify line 687-691 contains: if grad_enabled: predictions['flow'].append(next_flow)
      3. Verify else branch: predictions['flow'].append(next_flow.cpu().numpy())
      4. Verify inline comment mentions gradient preservation
      5. Compare pattern to lines 706-711 (should be structurally identical)
    Expected Result: Flow append logic matches RGB pattern exactly
    Failure Indicators: Unconditional .cpu().numpy(), missing else branch, wrong indentation
    Evidence: .sisyphus/evidence/task-1-code-verify.txt

  Scenario: No unintended changes
    Tool: Bash (git diff)
    Preconditions: File edited
    Steps:
      1. Run: git diff mpc/flow_guided_gaussian_model.py
      2. Count changed lines (should be 5-6: replace 1 line with if/else + comment)
      3. Verify only line 687 region modified
      4. Verify no changes to lines 724-729 (stacking logic)
      5. Verify no changes to lines 706-719 (RGB logic)
    Expected Result: Only lines 687-691 changed, rest untouched
    Failure Indicators: Changes outside expected range, RGB logic modified, stacking modified
    Evidence: .sisyphus/evidence/task-1-diff.txt
  ```

  **Evidence to Capture:**
  - [ ] task-1-code-verify.txt (Read output showing new conditional logic)
  - [ ] task-1-diff.txt (git diff showing minimal changes)

  **Commit**: YES
  - Message: `fix(mpc): respect grad_enabled in flow prediction append`
  - Files: `mpc/flow_guided_gaussian_model.py`
  - Pre-commit: None (no unit tests for this file)

---

- [ ] 2. Verify both gradient and non-gradient modes work

  **What to do**:
  - Create Python test script to verify both modes:
    1. **Gradient mode test** (CEM-GD use case):
       - Call `model(batch, grad_enabled=True)`
       - Verify `predictions['flow']` is torch.Tensor
       - Verify `.requires_grad=True`
       - Verify torch.stack() succeeds
    2. **Non-gradient mode test** (Pure CEM use case):
       - Call `model(batch, grad_enabled=False)`
       - Verify `predictions['flow']` is numpy.ndarray
       - Verify np.stack() succeeds (backward compatibility)
  - Save test outputs to evidence files

  **Must NOT do**:
  - Run full MPC pipeline (too expensive for quick test)
  - Modify any model code
  - Add workarounds if tests fail

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Simple verification script, straightforward test
  - **Skills**: None
  - **Skills Evaluated but Omitted**:
    - `playwright`: Not needed - no browser interaction

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Sequential (after Task 1)
  - **Blocks**: None (final task)
  - **Blocked By**: Task 1 (needs fixed code to test)

  **References**:

  **Pattern References**:
  - `.sisyphus/evidence/numpy-tensor-type-mismatch-analysis.md` - Test case specifications
  - `test/integration/test_cotracker_mpc.py` - Example model invocation patterns

  **API/Type References**:
  - `isinstance(obj, torch.Tensor)` - Check if object is tensor
  - `isinstance(obj, np.ndarray)` - Check if object is numpy array
  - `tensor.requires_grad` - Check if tensor tracks gradients

  **External References**:
  - PyTorch tensor API: https://pytorch.org/docs/stable/tensors.html

  **Acceptance Criteria**:

  - [ ] Test script created and executed
  - [ ] **Gradient mode test**: `predictions['flow']` is torch.Tensor with requires_grad=True
  - [ ] **Gradient mode test**: torch.stack() succeeds without TypeError
  - [ ] **Non-gradient mode test**: `predictions['flow']` is numpy.ndarray
  - [ ] **Non-gradient mode test**: np.stack() succeeds (backward compatibility)
  - [ ] Evidence files saved showing test results

  **QA Scenarios (MANDATORY)**:

  ```
  Scenario: Gradient mode test (CEM-GD use case)
    Tool: Bash (Python script)
    Preconditions: Task 1 completed, code fixed
    Steps:
      1. Create test_gradient_mode.py:
         ```python
         import torch
         import numpy as np
         from mpc.flow_guided_gaussian_model import FlowGuidedGaussianModel
         
         # Setup minimal test (mock if needed)
         # ... (simplified model initialization)
         
         # Test gradient mode
         predictions = model(batch, grad_enabled=True)
         
         flow = predictions['flow']
         print(f"Type: {type(flow)}")
         print(f"Is tensor: {isinstance(flow, torch.Tensor)}")
         if isinstance(flow, torch.Tensor):
             print(f"Requires grad: {flow.requires_grad}")
             print(f"Device: {flow.device}")
         
         # Try stacking (this was failing before)
         try:
             stacked = torch.stack([flow], dim=0)
             print("✅ torch.stack() succeeded!")
         except TypeError as e:
             print(f"❌ torch.stack() failed: {e}")
         ```
      2. Run: python test_gradient_mode.py
      3. Verify output shows: torch.Tensor, requires_grad=True, stack succeeded
    Expected Result: Flow is tensor with gradients, stacking works
    Failure Indicators: numpy.ndarray type, requires_grad=False, TypeError
    Evidence: .sisyphus/evidence/task-2-gradient-mode-test.txt

  Scenario: Non-gradient mode test (backward compatibility)
    Tool: Bash (Python script)
    Preconditions: Task 1 completed, code fixed
    Steps:
      1. Create test_non_gradient_mode.py:
         ```python
         import torch
         import numpy as np
         from mpc.flow_guided_gaussian_model import FlowGuidedGaussianModel
         
         # Setup minimal test (mock if needed)
         # ... (simplified model initialization)
         
         # Test non-gradient mode
         predictions = model(batch, grad_enabled=False)
         
         flow = predictions['flow']
         print(f"Type: {type(flow)}")
         print(f"Is numpy: {isinstance(flow, np.ndarray)}")
         if isinstance(flow, np.ndarray):
             print(f"Shape: {flow.shape}")
             print(f"Dtype: {flow.dtype}")
         
         # Try stacking (original behavior)
         try:
             stacked = np.stack([flow], axis=0)
             print("✅ np.stack() succeeded!")
         except Exception as e:
             print(f"❌ np.stack() failed: {e}")
         ```
      2. Run: python test_non_gradient_mode.py
      3. Verify output shows: numpy.ndarray, stacking works
    Expected Result: Flow is numpy array, backward compatible
    Failure Indicators: torch.Tensor type, np.stack() fails
    Evidence: .sisyphus/evidence/task-2-non-gradient-mode-test.txt

  Scenario: CEM-GD integration test (full pipeline)
    Tool: Bash (existing test script)
    Preconditions: Tasks 1-2 passed, trained model available
    Steps:
      1. Check if trained model exists: ls outputs/dm_control_push8/
      2. If exists, run: python test/integration/test_cotracker_mpc.py \
           --model_path outputs/dm_control_push8/ \
           --optimizer cem-gd \
           --num_steps 3 \
           --num_grad_seqs 2
      3. Verify no TypeError about numpy/tensor mismatch
      4. Check logs for "gradient_optimization" phase completion
    Expected Result: CEM-GD runs without TypeError, gradient phase completes
    Failure Indicators: TypeError at line 729, gradient computation fails
    Evidence: .sisyphus/evidence/task-2-cemgd-integration.txt
  ```

  **Evidence to Capture:**
  - [ ] task-2-gradient-mode-test.txt (Gradient mode test output)
  - [ ] task-2-non-gradient-mode-test.txt (Non-gradient mode test output)
  - [ ] task-2-cemgd-integration.txt (Optional: full CEM-GD test if model available)

  **Commit**: NO (test scripts are temporary verification, not production code)

---

## Final Verification Wave

> **Simple fix** - No additional review wave needed.
> User can verify by running their original failing command after fix.

---

## Commit Strategy

- **Task 1**: `fix(mpc): respect grad_enabled in flow prediction append` — `mpc/flow_guided_gaussian_model.py`

---

## Success Criteria

### Verification Commands
```bash
# Gradient mode verification (should not raise TypeError)
python -c "
import torch
from mpc.flow_guided_gaussian_model import FlowGuidedGaussianModel
# ... (minimal test setup)
# predictions = model(batch, grad_enabled=True)
# assert isinstance(predictions['flow'], torch.Tensor)
# torch.stack([predictions['flow']], dim=0)  # Should succeed
print('✅ Gradient mode: PASS')
"

# Non-gradient mode verification (backward compatibility)
python -c "
import numpy as np
from mpc.flow_guided_gaussian_model import FlowGuidedGaussianModel
# ... (minimal test setup)
# predictions = model(batch, grad_enabled=False)
# assert isinstance(predictions['flow'], np.ndarray)
# np.stack([predictions['flow']], axis=0)  # Should succeed
print('✅ Non-gradient mode: PASS')
"

# Full CEM-GD test (if model available)
python test/integration/test_cotracker_mpc.py \
    --model_path outputs/dm_control_push8/ \
    --optimizer cem-gd \
    --num_steps 5
# Expected: No TypeError, gradient optimization completes
```

### Final Checklist
- [ ] All "Must Have" present:
  - [ ] Conditional logic matches RGB pattern
  - [ ] Gradient mode: appends torch.Tensor
  - [ ] Non-gradient mode: appends numpy.ndarray
  - [ ] Inline comment explains purpose
- [ ] All "Must NOT Have" absent:
  - [ ] Stacking logic unchanged (lines 724-729)
  - [ ] RGB logic unchanged (lines 706-719)
  - [ ] Pure CEM mode still works
- [ ] Both test modes pass:
  - [ ] Gradient mode: torch.stack() succeeds
  - [ ] Non-gradient mode: np.stack() succeeds
- [ ] User's original error (`TypeError` at line 729) resolved
