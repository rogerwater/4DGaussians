# CEM → CEM-GD 迁移工作计划

**Plan Version**: 1.0  
**Date**: March 24, 2026  
**Status**: 🟢 Ready for Execution

---

## TL;DR

> **Quick Summary**: 将 4DGaussians MPC 的默认优化器从纯 CEM 升级为 CEM-GD (混合采样-梯度优化)，通过修复梯度流阻塞点、更新 demo 脚本、添加配置切换选项实现。
> 
> **Deliverables**:
> - 修复 `gaussian_dynamics_model.py` 的梯度流阻塞 (2 处)
> - 更新 `demo_flow_guided_mpc.py` 使用 CEM-GD
> - 添加 `--optimizer` 命令行参数支持 CEM/CEM-GD 切换
> - 创建梯度流验证测试脚本
> - 性能基准测试 (CEM vs CEM-GD)
> 
> **Estimated Effort**: Medium (1-2 天)
> **Parallel Execution**: YES - 2 waves
> **Critical Path**: Task 1 → Task 3 → Task 5 → Task 7 → F1-F4

---

## Context

### Original Request
基于竞争优势分析 (`cem-gd-flow-competitive-analysis.md`)，实施 CEM → CEM-GD 迁移，以获得：
- **5-10× 样本效率提升** (150-300 次评估 vs 1000 次)
- **3× 收敛速度提升** (2-3 轮迭代 vs 5-7 轮)
- **保持高鲁棒性** (90% 成功率，CEM 初始化避免局部最优)

### Research Findings
**From gradient-based-mpc-analysis.md (Section 11)**:
- ✅ CEM-GD 已实现: `mpc/cem_gd.py` (lines 206-407)
- ⚠️ **梯度流阻塞点 1**: `render_with_control()` 使用 `torch.no_grad()` (line 374)
- ⚠️ **梯度流阻塞点 2**: `__call__()` 转换为 numpy 破坏计算图 (line 443)
- ✅ 修复方案已提供: 10 行代码补丁

### Key Dependencies
- `mpc/cem_gd.py` - CEMGDOptimizer 类 (继承 CEMOptimizer)
- `mpc/gaussian_dynamics_model.py` - 动力学模型 (需修复)
- `demo_flow_guided_mpc.py` - 主演示脚本 (需更新)
- `mpc/flow_objectives.py` - 光流目标函数 (已支持梯度)

---

## Work Objectives

### Core Objective
将 4DGaussians MPC 默认优化器从纯 CEM 升级为 CEM-GD 混合优化，实现样本效率 5-10× 提升，同时保持后向兼容性。

### Concrete Deliverables
1. **修复的 `gaussian_dynamics_model.py`** - 支持 `grad_enabled=True` 时保留梯度流
2. **更新的 `demo_flow_guided_mpc.py`** - 默认使用 CEM-GD，支持 `--optimizer` 切换
3. **梯度流测试脚本** - `test/unit/test_gradient_flow_mpc.py`
4. **基准测试脚本** - `test/scripts/benchmark_cem_vs_cemgd.py`
5. **配置更新** - 新增 CEM-GD 超参数选项

### Definition of Done
- [ ] `python test/unit/test_gradient_flow_mpc.py` 通过 (梯度流验证)
- [ ] `python demo_flow_guided_mpc.py --optimizer cem-gd` 正常运行
- [ ] `python demo_flow_guided_mpc.py --optimizer cem` 保持兼容
- [ ] CEM-GD 样本效率 ≥ 5× (vs 纯 CEM)

### Must Have
- 梯度流修复 (阻塞点 1 和 2)
- CEM-GD 与现有 Flow Objectives 集成
- 后向兼容性 (纯 CEM 仍可用)
- 基本测试覆盖

### Must NOT Have (Guardrails)
- ❌ **不修改 `mpc/cem_gd.py`** - 已有实现经过验证，直接使用
- ❌ **不破坏现有 CEM 功能** - 保持 `--optimizer cem` 正常工作
- ❌ **不添加新依赖** - 使用现有 PyTorch/numpy
- ❌ **不过度工程** - 最小化修改，专注核心功能

---

## Verification Strategy

> **ZERO HUMAN INTERVENTION** — ALL verification is agent-executed. No exceptions.

### Test Decision
- **Infrastructure exists**: YES (`test/` 目录已存在)
- **Automated tests**: YES (Tests-after)
- **Framework**: pytest (Python 标准)

### QA Policy
Every task MUST include agent-executed QA scenarios.
Evidence saved to `.sisyphus/evidence/task-{N}-{scenario-slug}.{ext}`.

- **Python 测试**: Use Bash (pytest) — Run test file, assert exit code 0
- **MPC 集成测试**: Use Bash (python demo) — Run demo, check output
- **梯度流验证**: Use Bash (python test) — Run gradient test, check output contains "PASSED"

---

## Execution Strategy

### Parallel Execution Waves

```
Wave 1 (Start Immediately — foundation fixes):
├── Task 1: Fix gradient flow in gaussian_dynamics_model.py [quick]
├── Task 2: Create gradient flow test script [quick]
└── Task 3: Update render_with_control() signature [quick]

Wave 2 (After Wave 1 — integration):
├── Task 4: Update demo_flow_guided_mpc.py with CEM-GD [unspecified-high]
├── Task 5: Add --optimizer argument parsing [quick]
└── Task 6: Create benchmark script [unspecified-high]

Wave 3 (After Wave 2 — validation):
├── Task 7: Run full integration test [deep]
└── Task 8: Document changes in mpc/AGENTS.md [writing]

Wave FINAL (After ALL tasks — 4 parallel reviews, then user okay):
├── Task F1: Plan compliance audit (oracle)
├── Task F2: Code quality review (unspecified-high)
├── Task F3: Real manual QA (unspecified-high)
└── Task F4: Scope fidelity check (deep)
-> Present results -> Get explicit user okay

Critical Path: Task 1 → Task 3 → Task 4 → Task 7 → F1-F4 → user okay
Parallel Speedup: ~50% faster than sequential
Max Concurrent: 3 (Wave 1)
```

### Dependency Matrix

| Task | Depends On | Blocks | Wave |
|------|------------|--------|------|
| 1 | — | 3, 4, 5, 7 | 1 |
| 2 | — | 7 | 1 |
| 3 | 1 | 4, 7 | 1 |
| 4 | 1, 3 | 7 | 2 |
| 5 | 1 | 4 | 2 |
| 6 | — | 7 | 2 |
| 7 | 1, 2, 3, 4, 5, 6 | F1-F4 | 3 |
| 8 | 4 | F1 | 3 |

### Agent Dispatch Summary

- **Wave 1**: **3 tasks** — T1 → `quick`, T2 → `quick`, T3 → `quick`
- **Wave 2**: **3 tasks** — T4 → `unspecified-high`, T5 → `quick`, T6 → `unspecified-high`
- **Wave 3**: **2 tasks** — T7 → `deep`, T8 → `writing`
- **FINAL**: **4 tasks** — F1 → `oracle`, F2 → `unspecified-high`, F3 → `unspecified-high`, F4 → `deep`

---

## TODOs

---

- [ ] 1. Fix gradient flow blocker 1: render_with_control()

  **What to do**:
  - 修改 `mpc/gaussian_dynamics_model.py` 的 `render_with_control()` 方法
  - 添加 `grad_enabled` 参数 (默认 False)
  - 当 `grad_enabled=True` 时，移除 `torch.no_grad()` 包装

  **Must NOT do**:
  - 不改变现有 API 签名的默认行为 (保持 `grad_enabled=False` 为默认)
  - 不修改渲染逻辑本身

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: 单文件修改，逻辑简单，约 10 行代码变更
  - **Skills**: `[]`
    - 无需特殊 skill

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with Tasks 2, 3)
  - **Blocks**: Tasks 3, 4, 5, 7
  - **Blocked By**: None (can start immediately)

  **References**:

  **Pattern References**:
  - `mpc/gaussian_dynamics_model.py:372-385` — 当前 `render_with_control()` 实现，需修改
  - `.sisyphus/docs/gradient-based-mpc-analysis.md:1279-1316` — 修复代码模板

  **API/Type References**:
  - `gaussian_renderer/__init__.py:render()` — 渲染函数签名，确认无需修改

  **WHY Each Reference Matters**:
  - `gaussian_dynamics_model.py:372-385`: 这是阻塞点 1 的确切位置，`with torch.no_grad():` 在 line 374
  - `gradient-based-mpc-analysis.md:1279-1316`: 提供了经过验证的修复代码模板

  **Acceptance Criteria**:

  **QA Scenarios (MANDATORY):**

  ```
  Scenario: [Happy path — grad_enabled=True preserves gradients]
    Tool: Bash (python)
    Preconditions: Model loaded, control_vec with requires_grad=True
    Steps:
      1. python -c "
import torch
import sys
sys.path.insert(0, '.')
from mpc.gaussian_dynamics_model import GaussianDynamicsModel
# Create minimal test
control = torch.randn(15, requires_grad=True, device='cuda')
print('grad_enabled param exists:', 'grad_enabled' in str(GaussianDynamicsModel.render_with_control.__code__.co_varnames))
"
      2. Assert output contains "grad_enabled param exists: True"
    Expected Result: Method signature includes grad_enabled parameter
    Failure Indicators: "grad_enabled param exists: False" or import error
    Evidence: .sisyphus/evidence/task-1-signature-check.txt

  Scenario: [Backward compat — grad_enabled=False default behavior unchanged]
    Tool: Bash (python)
    Preconditions: Existing code calling render_with_control() without args
    Steps:
      1. grep -n "render_with_control(" mpc/gaussian_dynamics_model.py | head -5
      2. Verify default parameter is grad_enabled=False
    Expected Result: Method works without explicit grad_enabled argument
    Evidence: .sisyphus/evidence/task-1-default-check.txt
  ```

  **Commit**: YES (group with Task 3)
  - Message: `fix(mpc): enable gradient flow in render_with_control`
  - Files: `mpc/gaussian_dynamics_model.py`
  - Pre-commit: N/A (no tests yet)

---

- [ ] 2. Create gradient flow test script

  **What to do**:
  - 创建 `test/unit/test_gradient_flow_mpc.py`
  - 测试梯度从 loss 流回 control_vec
  - 使用 minimal mock 或实际模型加载

  **Must NOT do**:
  - 不创建复杂的测试 fixture
  - 不依赖外部数据文件

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: 创建单个测试文件，逻辑已在 gradient-based-mpc-analysis.md 提供
  - **Skills**: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with Tasks 1, 3)
  - **Blocks**: Task 7
  - **Blocked By**: None (can start immediately)

  **References**:

  **Pattern References**:
  - `.sisyphus/docs/gradient-based-mpc-analysis.md:1366-1405` — 完整测试脚本模板

  **Test References**:
  - `test/unit/test_biflow_functions.py` — 现有单元测试格式参考

  **WHY Each Reference Matters**:
  - `gradient-based-mpc-analysis.md:1366-1405`: 提供完整的测试代码，只需复制并适配
  - `test_biflow_functions.py`: 展示项目的测试文件组织方式

  **Acceptance Criteria**:

  **QA Scenarios (MANDATORY):**

  ```
  Scenario: [Happy path — test file created and syntactically correct]
    Tool: Bash (python)
    Preconditions: test/unit/ directory exists
    Steps:
      1. ls -la test/unit/test_gradient_flow_mpc.py
      2. python -m py_compile test/unit/test_gradient_flow_mpc.py
    Expected Result: File exists and compiles without syntax errors
    Failure Indicators: "No such file" or "SyntaxError"
    Evidence: .sisyphus/evidence/task-2-file-check.txt

  Scenario: [Import check — test can import required modules]
    Tool: Bash (python)
    Preconditions: Test file created
    Steps:
      1. python -c "import test.unit.test_gradient_flow_mpc"
    Expected Result: Import succeeds (may fail on model load, but no import errors)
    Evidence: .sisyphus/evidence/task-2-import-check.txt
  ```

  **Commit**: YES (standalone)
  - Message: `test(mpc): add gradient flow verification test`
  - Files: `test/unit/test_gradient_flow_mpc.py`
  - Pre-commit: `python -m py_compile test/unit/test_gradient_flow_mpc.py`

---

- [ ] 3. Fix gradient flow blocker 2: __call__() numpy conversion

  **What to do**:
  - 修改 `mpc/gaussian_dynamics_model.py` 的 `__call__()` 方法 (lines 394-452)
  - 当 `grad_enabled=True` 时，返回 torch.Tensor 而非 numpy array
  - 当 `grad_enabled=False` 时，保持现有 numpy 行为 (后向兼容)
  - 传递 `grad_enabled` 参数到 `render_with_control()`

  **Must NOT do**:
  - 不破坏现有 CEM 的 numpy 接口
  - 不改变返回字典的 key 名称 ('rgb')

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: 单文件修改，约 15 行代码变更
  - **Skills**: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: NO (depends on Task 1)
  - **Parallel Group**: Wave 1 (sequential after Task 1)
  - **Blocks**: Tasks 4, 7
  - **Blocked By**: Task 1

  **References**:

  **Pattern References**:
  - `mpc/gaussian_dynamics_model.py:394-452` — 当前 `__call__()` 实现
  - `.sisyphus/docs/gradient-based-mpc-analysis.md:1318-1352` — 修复代码模板

  **WHY Each Reference Matters**:
  - `gaussian_dynamics_model.py:394-452`: 阻塞点 2 的确切位置，line 443 的 `.cpu().numpy()`
  - `gradient-based-mpc-analysis.md:1318-1352`: 提供条件分支逻辑，保持后向兼容

  **Acceptance Criteria**:

  **QA Scenarios (MANDATORY):**

  ```
  Scenario: [Happy path — grad_enabled=True returns torch.Tensor]
    Tool: Bash (python)
    Preconditions: Task 1 completed
    Steps:
      1. grep -A5 "if grad_enabled:" mpc/gaussian_dynamics_model.py | grep -v numpy
      2. Verify torch.Tensor is returned (not converted to numpy)
    Expected Result: Code path for grad_enabled=True keeps tensors on GPU
    Failure Indicators: "numpy" appears in grad_enabled=True branch
    Evidence: .sisyphus/evidence/task-3-tensor-check.txt

  Scenario: [Backward compat — grad_enabled=False returns numpy]
    Tool: Bash (python)
    Preconditions: Task 1 completed
    Steps:
      1. grep -A5 "else:" mpc/gaussian_dynamics_model.py | grep numpy
      2. Verify numpy conversion remains in grad_enabled=False branch
    Expected Result: Existing CEM code still receives numpy arrays
    Evidence: .sisyphus/evidence/task-3-numpy-check.txt
  ```

  **Commit**: YES (group with Task 1)
  - Message: `fix(mpc): enable gradient flow in render_with_control`
  - Files: `mpc/gaussian_dynamics_model.py`
  - Pre-commit: N/A

---

- [ ] 4. Update demo_flow_guided_mpc.py with CEM-GD support

  **What to do**:
  - 导入 `CEMGDOptimizer` from `mpc.cem_gd`
  - 修改 `create_optimizer()` 函数 (或等效位置) 根据参数选择优化器
  - 添加 CEM-GD 特有超参数 (`num_grad_opt_seqs`, `start_lr`, `max_iterations`)
  - 默认使用 CEM-GD (可通过 `--optimizer cem` 回退)

  **Must NOT do**:
  - 不删除现有 CEM 代码路径
  - 不修改 Flow Objectives 实现

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
    - Reason: 涉及多处修改，需要理解优化器接口
  - **Skills**: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: NO (depends on Tasks 1, 3)
  - **Parallel Group**: Wave 2
  - **Blocks**: Task 7
  - **Blocked By**: Tasks 1, 3, 5

  **References**:

  **Pattern References**:
  - `demo_flow_guided_mpc.py:41` — 当前 CEMOptimizer 导入
  - `demo_flow_guided_mpc.py:517-528` — 当前优化器创建代码
  - `mpc/cem_gd.py:206-257` — CEMGDOptimizer 构造函数签名

  **API/Type References**:
  - `mpc/cem_gd.py:CEMGDOptimizer.__init__()` — 新增参数: `num_grad_opt_seqs`, `start_lr`, `factor_shrink`, `max_tries`, `max_iterations`

  **WHY Each Reference Matters**:
  - `demo_flow_guided_mpc.py:41`: 需要添加 CEMGDOptimizer 导入
  - `demo_flow_guided_mpc.py:517-528`: 需要修改为条件分支，根据 optimizer 参数选择
  - `mpc/cem_gd.py:206-257`: CEMGDOptimizer 构造函数，了解额外参数

  **Acceptance Criteria**:

  **QA Scenarios (MANDATORY):**

  ```
  Scenario: [Happy path — CEM-GD optimizer loads]
    Tool: Bash (python)
    Preconditions: Tasks 1, 3 completed
    Steps:
      1. grep -n "CEMGDOptimizer" demo_flow_guided_mpc.py
      2. Assert import statement exists
      3. Assert optimizer selection logic exists
    Expected Result: CEMGDOptimizer is imported and used
    Failure Indicators: No matches for "CEMGDOptimizer"
    Evidence: .sisyphus/evidence/task-4-import-check.txt

  Scenario: [Integration — demo runs with --optimizer cem-gd]
    Tool: Bash (python)
    Preconditions: All Wave 1 tasks completed
    Steps:
      1. python demo_flow_guided_mpc.py --help | grep -i optimizer
      2. Assert "--optimizer" option exists
    Expected Result: CLI shows optimizer option
    Evidence: .sisyphus/evidence/task-4-cli-check.txt
  ```

  **Commit**: YES (standalone)
  - Message: `feat(mpc): add CEM-GD optimizer support to demo`
  - Files: `demo_flow_guided_mpc.py`
  - Pre-commit: `python -m py_compile demo_flow_guided_mpc.py`

---

- [ ] 5. Add --optimizer argument to demo script

  **What to do**:
  - 在 `demo_flow_guided_mpc.py` 的 argparse 中添加 `--optimizer` 参数
  - 选项: `cem`, `cem-gd` (default: `cem-gd`)
  - 添加 CEM-GD 特有超参数: `--grad_steps`, `--grad_lr`, `--num_grad_seqs`

  **Must NOT do**:
  - 不改变现有参数的默认值

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: argparse 参数添加，简单修改
  - **Skills**: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: YES (only depends on Task 1 for context)
  - **Parallel Group**: Wave 2 (with Task 4)
  - **Blocks**: Task 4
  - **Blocked By**: Task 1

  **References**:

  **Pattern References**:
  - `demo_flow_guided_mpc.py` — 现有 argparse 参数定义 (搜索 `argparse` 或 `add_argument`)

  **External References**:
  - `.sisyphus/docs/cem-gd-flow-competitive-analysis.md:82-96` — CEM-GD 超参数推荐值

  **WHY Each Reference Matters**:
  - `demo_flow_guided_mpc.py`: 找到现有参数定义位置，保持一致风格
  - `cem-gd-flow-competitive-analysis.md`: 推荐的超参数值 (lr=0.01, max_iterations=15)

  **Acceptance Criteria**:

  **QA Scenarios (MANDATORY):**

  ```
  Scenario: [Happy path — --optimizer argument works]
    Tool: Bash (python)
    Preconditions: None
    Steps:
      1. python demo_flow_guided_mpc.py --help 2>&1 | grep -E "(--optimizer|cem-gd|cem)"
    Expected Result: Help text shows --optimizer with cem/cem-gd options
    Failure Indicators: No --optimizer in help output
    Evidence: .sisyphus/evidence/task-5-help-check.txt

  Scenario: [Happy path — CEM-GD specific params exist]
    Tool: Bash
    Preconditions: None
    Steps:
      1. python demo_flow_guided_mpc.py --help 2>&1 | grep -E "(grad_steps|grad_lr|num_grad_seqs)"
    Expected Result: CEM-GD specific params shown
    Evidence: .sisyphus/evidence/task-5-params-check.txt
  ```

  **Commit**: YES (group with Task 4)
  - Message: `feat(mpc): add CEM-GD optimizer support to demo`
  - Files: `demo_flow_guided_mpc.py`

---

- [ ] 6. Create benchmark script for CEM vs CEM-GD

  **What to do**:
  - 创建 `test/scripts/benchmark_cem_vs_cemgd.py`
  - 对比指标: 样本数、迭代次数、墙钟时间、最终 reward
  - 使用相同的 goal 和初始条件
  - 输出 markdown 表格格式结果

  **Must NOT do**:
  - 不修改优化器实现
  - 不创建复杂的测试数据

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
    - Reason: 需要理解两种优化器接口，编写公平对比逻辑
  - **Skills**: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2 (independent)
  - **Blocks**: Task 7
  - **Blocked By**: None (can use existing code)

  **References**:

  **Pattern References**:
  - `mpc/cem.py:CEMOptimizer` — CEM 接口
  - `mpc/cem_gd.py:CEMGDOptimizer` — CEM-GD 接口

  **External References**:
  - `.sisyphus/docs/cem-gd-flow-competitive-analysis.md:201-250` — 预期性能对比数据

  **WHY Each Reference Matters**:
  - `cem.py`, `cem_gd.py`: 了解两种优化器的 API，确保公平对比
  - `cem-gd-flow-competitive-analysis.md`: 预期结果，验证基准测试合理性

  **Acceptance Criteria**:

  **QA Scenarios (MANDATORY):**

  ```
  Scenario: [Happy path — benchmark script runs]
    Tool: Bash (python)
    Preconditions: Script created
    Steps:
      1. python -m py_compile test/scripts/benchmark_cem_vs_cemgd.py
      2. Assert no syntax errors
    Expected Result: Script compiles successfully
    Failure Indicators: SyntaxError
    Evidence: .sisyphus/evidence/task-6-compile-check.txt

  Scenario: [Output format — markdown table generated]
    Tool: Bash
    Preconditions: Script created
    Steps:
      1. grep -E "^\|.*\|$" test/scripts/benchmark_cem_vs_cemgd.py | head -3
      2. Assert markdown table format in output logic
    Expected Result: Script generates markdown table
    Evidence: .sisyphus/evidence/task-6-format-check.txt
  ```

  **Commit**: YES (standalone)
  - Message: `test(mpc): add CEM vs CEM-GD benchmark script`
  - Files: `test/scripts/benchmark_cem_vs_cemgd.py`
  - Pre-commit: `python -m py_compile test/scripts/benchmark_cem_vs_cemgd.py`

---

- [ ] 7. Run full integration test

  **What to do**:
  - 运行梯度流测试: `python test/unit/test_gradient_flow_mpc.py`
  - 运行 demo 脚本 with CEM-GD: 验证端到端工作
  - 运行 demo 脚本 with CEM: 验证后向兼容
  - 收集所有测试 evidence

  **Must NOT do**:
  - 不跳过失败的测试
  - 不修改代码 (只运行测试)

  **Recommended Agent Profile**:
  - **Category**: `deep`
    - Reason: 需要运行多个测试，分析结果，可能需要调试
  - **Skills**: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: NO (验证阶段)
  - **Parallel Group**: Wave 3
  - **Blocks**: F1-F4
  - **Blocked By**: Tasks 1-6

  **References**:

  **Pattern References**:
  - `test/unit/test_gradient_flow_mpc.py` — 梯度流测试 (Task 2 创建)
  - `demo_flow_guided_mpc.py` — 演示脚本 (Task 4 更新)

  **WHY Each Reference Matters**:
  - 这是验证任务，需要运行前面创建的所有测试和修改

  **Acceptance Criteria**:

  **QA Scenarios (MANDATORY):**

  ```
  Scenario: [Happy path — gradient flow test passes]
    Tool: Bash (python)
    Preconditions: Tasks 1, 2, 3 completed
    Steps:
      1. cd /home/ubuntu/yyf/4DGaussians
      2. python test/unit/test_gradient_flow_mpc.py 2>&1
      3. Assert output contains "PASSED" or exit code 0
    Expected Result: Gradient flow test passes
    Failure Indicators: "FAILED", "Error", non-zero exit code
    Evidence: .sisyphus/evidence/task-7-gradient-test.txt

  Scenario: [Integration — CEM-GD demo runs without error]
    Tool: Bash (python)
    Preconditions: Tasks 1-5 completed
    Steps:
      1. python demo_flow_guided_mpc.py --optimizer cem-gd --num_steps 2 --horizon 2 --num_samples 8 2>&1 | head -50
      2. Assert no Python exceptions
    Expected Result: Demo starts without immediate crash (may fail on missing model, but that's OK)
    Evidence: .sisyphus/evidence/task-7-cemgd-demo.txt

  Scenario: [Backward compat — CEM demo still works]
    Tool: Bash (python)
    Preconditions: Tasks 1-5 completed
    Steps:
      1. python demo_flow_guided_mpc.py --optimizer cem --num_steps 2 --horizon 2 --num_samples 8 2>&1 | head -50
      2. Assert no Python exceptions
    Expected Result: CEM mode still functional
    Evidence: .sisyphus/evidence/task-7-cem-demo.txt
  ```

  **Commit**: NO (validation only)

---

- [ ] 8. Update mpc/AGENTS.md documentation

  **What to do**:
  - 更新 `mpc/AGENTS.md` 的 Optimizers 部分
  - 添加 CEM-GD 使用说明和超参数
  - 添加何时使用 CEM vs CEM-GD 的指南

  **Must NOT do**:
  - 不删除现有 CEM 文档

  **Recommended Agent Profile**:
  - **Category**: `writing`
    - Reason: 文档更新任务
  - **Skills**: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: YES (only reads Task 4 changes)
  - **Parallel Group**: Wave 3 (with Task 7)
  - **Blocks**: F1
  - **Blocked By**: Task 4

  **References**:

  **Pattern References**:
  - `mpc/AGENTS.md` — 现有 MPC 文档

  **External References**:
  - `.sisyphus/docs/cem-gd-flow-competitive-analysis.md` — 何时使用的决策矩阵

  **Acceptance Criteria**:

  **QA Scenarios (MANDATORY):**

  ```
  Scenario: [Happy path — CEM-GD section added]
    Tool: Bash (grep)
    Preconditions: Task completed
    Steps:
      1. grep -i "cem-gd\|cemgd" mpc/AGENTS.md
    Expected Result: CEM-GD documentation exists
    Failure Indicators: No matches
    Evidence: .sisyphus/evidence/task-8-doc-check.txt
  ```

  **Commit**: YES (standalone)
  - Message: `docs(mpc): add CEM-GD optimizer documentation`
  - Files: `mpc/AGENTS.md`

---

## Final Verification Wave

> 4 review agents run in PARALLEL. ALL must APPROVE. Present consolidated results to user and get explicit "okay" before completing.
>
> **Do NOT auto-proceed after verification.** Rejection or user feedback -> fix -> re-run -> present again -> wait for okay.

- [ ] F1. **Plan Compliance Audit** — `oracle`
  Read the plan end-to-end. For each "Must Have": verify implementation exists. For each "Must NOT Have": search codebase for forbidden patterns. Compare deliverables against plan.
  Output: `Must Have [N/N] | Must NOT Have [N/N] | Tasks [N/N] | VERDICT: APPROVE/REJECT`

- [ ] F2. **Code Quality Review** — `unspecified-high`
  Run `python -m py_compile` on all modified files. Check for: unused imports, missing error handling, inconsistent naming. Review gradient flow implementation correctness.
  Output: `Compile [PASS/FAIL] | Style [N issues] | VERDICT`

- [ ] F3. **Real Manual QA** — `unspecified-high`
  Run gradient flow test. Run demo with CEM-GD. Run demo with CEM (backward compat). Verify no regressions.
  Output: `Tests [N/N pass] | Demo [CEM-GD OK/FAIL] | Demo [CEM OK/FAIL] | VERDICT`

- [ ] F4. **Scope Fidelity Check** — `deep`
  For each task: verify changes match spec. Check no unrelated files modified. Verify backward compatibility preserved.
  Output: `Tasks [N/N compliant] | Unaccounted [CLEAN/N files] | VERDICT`

---

## Commit Strategy

| Task | Commit | Message | Files |
|------|--------|---------|-------|
| 1, 3 | Combined | `fix(mpc): enable gradient flow in gaussian_dynamics_model` | `mpc/gaussian_dynamics_model.py` |
| 2 | Standalone | `test(mpc): add gradient flow verification test` | `test/unit/test_gradient_flow_mpc.py` |
| 4, 5 | Combined | `feat(mpc): add CEM-GD optimizer support to demo` | `demo_flow_guided_mpc.py` |
| 6 | Standalone | `test(mpc): add CEM vs CEM-GD benchmark script` | `test/scripts/benchmark_cem_vs_cemgd.py` |
| 8 | Standalone | `docs(mpc): add CEM-GD optimizer documentation` | `mpc/AGENTS.md` |

---

## Success Criteria

### Verification Commands
```bash
# 梯度流测试
python test/unit/test_gradient_flow_mpc.py
# Expected: "✅ Gradient flow test PASSED"

# CEM-GD demo (dry run with minimal params)
python demo_flow_guided_mpc.py --optimizer cem-gd --help
# Expected: Shows CEM-GD options

# 后向兼容
python demo_flow_guided_mpc.py --optimizer cem --help
# Expected: Shows CEM options (no errors)
```

### Final Checklist
- [ ] 梯度从 loss 流回 control_vec (Task 1, 3)
- [ ] 梯度流测试通过 (Task 2, 7)
- [ ] CEM-GD 优化器可用 (Task 4)
- [ ] `--optimizer` 参数工作 (Task 5)
- [ ] 后向兼容 (纯 CEM 仍可用)
- [ ] 文档更新 (Task 8)

---

## Risk Assessment

### Potential Issues

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| **渲染器不可微分** | 低 (已验证可微) | 高 | 回退到纯 CEM，记录问题 |
| **CEM-GD 性能不达预期** | 中 | 中 | 调整超参数，对比基准 |
| **后向兼容性破坏** | 低 | 高 | 完整测试 CEM 路径 |
| **内存溢出 (梯度累积)** | 中 | 中 | 使用 gradient checkpointing |

### Rollback Strategy
如果 CEM-GD 出现严重问题:
1. 恢复 `--optimizer` 默认值为 `cem`
2. 保留 CEM-GD 代码但标记为 experimental
3. 记录问题到 issue tracker

---

## Appendix: Code Snippets

### A.1 render_with_control() 修复 (Task 1)

```python
# mpc/gaussian_dynamics_model.py, line ~347
def render_with_control(self, control_vec, time=None, grad_enabled=False):
    """Render with control override.
    
    Args:
        control_vec: Control input
        time: Time value
        grad_enabled: If True, preserve gradients for backprop
    """
    # ... existing preprocessing ...
    
    if grad_enabled:
        # Preserve gradients for gradient-based planning
        render_pkg = render(
            self.camera,
            self.gaussians,
            self.pipe_params,
            self.background,
            override_control_vec=control_vec,
            stage="fine"
        )
    else:
        # Disable gradients for sampling-based planning (faster)
        with torch.no_grad():
            render_pkg = render(
                self.camera,
                self.gaussians,
                self.pipe_params,
                self.background,
                override_control_vec=control_vec,
                stage="fine"
            )
    
    return render_pkg["render"]
```

### A.2 __call__() 修复 (Task 3)

```python
# mpc/gaussian_dynamics_model.py, line ~435
if grad_enabled:
    # Gradient-enabled: keep tensors on device
    rendered_image = self.render_with_control(control_vec, time_val, grad_enabled=True)
    batch_predictions.append(rendered_image)  # Keep as tensor [C, H, W]
else:
    # Sampling-based: convert to numpy
    rendered_image = self.render_with_control(control_vec, time_val, grad_enabled=False)
    image_np = rendered_image.permute(1, 2, 0).cpu().numpy()
    batch_predictions.append(image_np)

# After loop, return appropriate format
if grad_enabled:
    # Return torch.Tensor [B, T_horizon, C, H, W]
    predictions_tensor = torch.stack([
        torch.stack(bp) for bp in all_predictions
    ])
    return {'rgb': predictions_tensor}
else:
    # Return numpy [B, T_horizon, H, W, C]
    return {'rgb': np.stack([np.stack(bp) for bp in all_predictions])}
```

### A.3 Demo 优化器选择 (Task 4)

```python
# demo_flow_guided_mpc.py
from mpc.cem import CEMOptimizer
from mpc.cem_gd import CEMGDOptimizer

def create_optimizer(optimizer_type, model, objective, sampler, **kwargs):
    if optimizer_type == 'cem-gd':
        return CEMGDOptimizer(
            model=model,
            objective=objective,
            sampler=sampler,
            num_grad_opt_seqs=kwargs.get('num_grad_seqs', 5),
            start_lr=kwargs.get('grad_lr', 0.01),
            max_iterations=kwargs.get('grad_steps', 15),
            # ... other CEM params ...
        )
    else:  # 'cem'
        return CEMOptimizer(
            model=model,
            objective=objective,
            sampler=sampler,
            # ... existing CEM params ...
        )
```

---

**Plan Status**: 🟢 Ready for Execution

**To begin execution**: Run `/start-work`
