# ControlProcessor → ActionProcessor 重命名计划

## TL;DR

> **Quick Summary**: 将 TriPlane deformation 网络中的 `control_processor` 重命名为 `action_processor`，以匹配已训练 checkpoint 的命名约定，确保向后兼容性。
> 
> **Deliverables**:
> - 修改 3 个核心文件（triplane.py, deformation_triplane.py, triplane_film_analyzer.py）
> - 更新配置参数名称（control_* → action_*）
> - 更新 scene/AGENTS.md 文档说明命名约定
> 
> **Estimated Effort**: Quick
> **Parallel Execution**: NO - 顺序执行（变量重命名需要确保一致性）
> **Critical Path**: Task 1 → Task 2 → Task 3 → Task 4 → Task 5

---

## Context

### Original Request
用户报告 checkpoint 加载错误：
```
RuntimeError: Missing key(s): "deformation_net.control_processor.*"
Unexpected key(s): "deformation_net.action_processor.*"
```

### Root Cause
- **现有代码**: 使用 `control_processor`（较新的命名）
- **训练的 checkpoint**: 使用 `action_processor`（较旧的命名）
- **用户选择**: 保持 checkpoint 命名（`action_processor`），修改代码以匹配

### Design Decision
**命名语义等价性**: 在本代码库中，`action` 和 `control` 是语义等价的术语，都指代控制/动作信号编码器。这不是功能差异，仅是历史演变导致的命名不一致。

---

## Work Objectives

### Core Objective
将代码中所有 `control_processor` 引用重命名为 `action_processor`，确保与已训练 checkpoint 的 state_dict keys 完全匹配。

### Concrete Deliverables
- `scene/triplane.py`: 类名 `ControlProcessor` → `ActionProcessor`
- `scene/deformation_triplane.py`: 导入、实例化、属性访问全部更新
- `utils/triplane_film_analyzer.py`: 属性访问更新
- `arguments/toyarm/triplane.py`: 配置参数名更新
- `scene/AGENTS.md`: 添加命名约定说明

### Definition of Done
- [x] `grep -r "control_processor"` 返回 0 结果（除了注释和文档）
- [x] `grep -r "ControlProcessor"` 返回 0 结果（除了注释和文档）
- [x] LSP diagnostics 无错误
- [x] 配置参数名全部更新（control_input_dim → action_input_dim 等）
- [x] scene/AGENTS.md 包含命名约定说明

### Must Have
- 完整重命名所有引用（class, attributes, variables, config params）
- 保持代码功能完全不变
- 文档记录命名历史

### Must NOT Have (Guardrails)
- 不修改 checkpoint 文件本身
- 不添加 key remapping 逻辑（直接改名即可）
- 不改变任何模型架构或功能
- 不引入新的依赖或抽象

---

## Verification Strategy (MANDATORY)

### Test Decision
- **Infrastructure exists**: YES (无需单元测试，使用 LSP + grep 验证)
- **Automated tests**: None（重命名操作通过静态检查验证）
- **Framework**: N/A

### QA Policy
每个任务使用 LSP diagnostics 和 grep 验证完整性。最终使用 checkpoint 加载测试验证。

---

## Execution Strategy

### Sequential Execution (必须按顺序)

```
Task 1: 重命名 scene/triplane.py 类定义
  ├── ControlProcessor → ActionProcessor
  └── 更新 print 语句

Task 2: 更新 scene/deformation_triplane.py
  ├── 导入语句更新
  ├── 实例化更新 (self.control_processor → self.action_processor)
  ├── 属性引用更新 (self.control_dim → self.action_dim)
  └── forward 方法中的变量名 (control_feat → action_feat)

Task 3: 更新 utils/triplane_film_analyzer.py
  ├── 属性提取更新 (self.control_processor → self.action_processor)
  └── 方法中的引用更新

Task 4: 更新配置参数
  ├── arguments/toyarm/triplane.py
  │   ├── control_input_dim → action_input_dim
  │   ├── control_use_pe → action_use_pe
  │   ├── control_num_frequencies → action_num_frequencies
  │   ├── control_hidden_dim → action_hidden_dim
  │   └── control_output_dim → action_output_dim
  └── scene/deformation_triplane.py 中的 getattr 调用

Task 5: 更新文档
  └── scene/AGENTS.md 添加命名约定说明

Task 6: 验证
  ├── LSP diagnostics
  ├── Grep 验证
  └── （可选）Checkpoint 加载测试
```

### Critical Path
Task 1 → Task 2 → Task 3 → Task 4 → Task 5 → Task 6

---

## TODOs

- [x] 1. 重命名 `scene/triplane.py` 中的 `ControlProcessor` 类

  **What to do**:
  - 将 class 名从 `ControlProcessor` 改为 `ActionProcessor`
  - 更新 docstring 中的描述（control → action）
  - 更新 print 语句中的类名引用（line 311）

  **Must NOT do**:
  - 不修改任何方法逻辑或实现
  - 不改变类的功能或接口

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: 简单的类名重命名操作
  - **Skills**: []
    - Reason: 不需要特殊技能，基础 edit 操作即可

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Sequential
  - **Blocks**: Task 2, Task 3（依赖此类定义）
  - **Blocked By**: None（可立即开始）

  **References**:
  - `scene/triplane.py:252-311` - `ControlProcessor` 类定义及其方法
  - `scene/triplane.py:311` - Print 语句需要更新类名

  **Acceptance Criteria**:
  - [ ] `grep "class ControlProcessor" scene/triplane.py` 返回 0 结果
  - [ ] `grep "class ActionProcessor" scene/triplane.py` 返回 1 结果
  - [ ] `grep "ControlProcessor" scene/triplane.py` 返回 0 结果（除注释外）

  **QA Scenarios**:

  ```
  Scenario: 类名重命名完成
    Tool: Bash (grep)
    Preconditions: scene/triplane.py 已修改
    Steps:
      1. grep "class ActionProcessor" scene/triplane.py
      2. 确认 line 252 包含新类名
      3. grep "ControlProcessor" scene/triplane.py
      4. 确认仅在 docstring 或注释中出现（如果有）
    Expected Result: 类名完全更新，无遗留引用
    Failure Indicators: 仍存在 ControlProcessor 非注释引用
    Evidence: .sisyphus/evidence/task-1-class-rename.txt
  ```

  **Evidence to Capture**:
  - [x] task-1-class-rename.txt: grep 验证输出

  **Commit**: NO（与 Task 2 一起提交）

---

- [x] 2. 更新 `scene/deformation_triplane.py` 中的所有引用

  **What to do**:
  - Line 7: 导入语句 `from scene.triplane import TriPlaneField, ControlProcessor` → `ActionProcessor`
  - Line 194: 实例化 `self.control_processor = ControlProcessor(...)` → `self.action_processor = ActionProcessor(...)`
  - Line 206: 属性引用 `self.control_dim = self.control_processor.output_dim` → `self.action_dim = self.action_processor.output_dim`
  - Line 187-192: getattr 调用中的参数名
    - `control_use_pe` → `action_use_pe`
    - `control_num_freq` → `action_num_freq`（变量名）
    - `control_input_dim` → `action_input_dim`
    - `control_hidden` → `action_hidden`（变量名）
    - `control_output_dim` → `action_output_dim`
  - Line 214, 242-243: docstring 和 print 语句中的 `control_dim` → `action_dim`
  - Line 330, 346: 方法中的变量名
    - `control_feat = self.control_processor(control_vec)` → `action_feat = self.action_processor(action_vec)`
    - 参数名 `control_vec` → `action_vec`

  **Must NOT do**:
  - 不改变方法逻辑或功能
  - 不修改 FiLM fusion 的实现
  - 不改变 forward 方法的输入输出接口

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: 系统性的变量重命名，无复杂逻辑
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Sequential
  - **Blocks**: Task 3, Task 4
  - **Blocked By**: Task 1（依赖 ActionProcessor 类定义）

  **References**:
  - `scene/deformation_triplane.py:7` - 导入语句
  - `scene/deformation_triplane.py:186-206` - 初始化代码
  - `scene/deformation_triplane.py:320-351` - `_fuse_spatial_control` 方法
  - `scene/triplane.py:252` - ActionProcessor 类定义（Task 1 完成后）

  **Acceptance Criteria**:
  - [ ] `grep "ControlProcessor" scene/deformation_triplane.py` 返回 0 结果
  - [ ] `grep "control_processor" scene/deformation_triplane.py` 返回 0 结果
  - [ ] `grep "control_feat" scene/deformation_triplane.py` 返回 0 结果
  - [ ] `grep "action_processor" scene/deformation_triplane.py` 返回至少 2 处（实例化 + 调用）
  - [ ] LSP diagnostics 无错误

  **QA Scenarios**:

  ```
  Scenario: 导入和实例化更新
    Tool: Bash (grep + python syntax check)
    Preconditions: Task 1 完成
    Steps:
      1. grep "from scene.triplane import.*ActionProcessor" scene/deformation_triplane.py
      2. grep "self.action_processor = ActionProcessor" scene/deformation_triplane.py
      3. python -m py_compile scene/deformation_triplane.py
    Expected Result: 导入正确，实例化正确，语法无误
    Failure Indicators: 导入错误或语法错误
    Evidence: .sisyphus/evidence/task-2-import-instantiate.txt

  Scenario: 属性引用完整更新
    Tool: Bash (grep)
    Preconditions: 所有 edit 完成
    Steps:
      1. grep -n "action_processor" scene/deformation_triplane.py
      2. 确认至少 2 处引用（实例化 + forward 调用）
      3. grep "control_processor" scene/deformation_triplane.py
      4. 确认返回 0 结果
    Expected Result: 所有引用已更新，无遗留
    Failure Indicators: 仍存在 control_processor 引用
    Evidence: .sisyphus/evidence/task-2-references.txt
  ```

  **Evidence to Capture**:
  - [x] task-2-import-instantiate.txt: 导入和实例化验证
  - [x] task-2-references.txt: 引用完整性验证

  **Commit**: YES（与 Task 1 一起）
  - Message: `refactor(scene): rename ControlProcessor to ActionProcessor for checkpoint compatibility`
  - Files: `scene/triplane.py`, `scene/deformation_triplane.py`
  - Pre-commit: `python -m py_compile scene/triplane.py scene/deformation_triplane.py`

---

- [x] 3. 更新 `utils/triplane_film_analyzer.py` 中的引用

  **What to do**:
  - Line 79: `self.control_processor = inner_net.control_processor` → `self.action_processor = inner_net.action_processor`
  - 所有方法中使用 `self.control_processor` 的地方（lines 177, 255, 338, 392, 418）更新为 `self.action_processor`
  - 变量名更新（如有）：`control_feat` → `action_feat`
  - Docstring 中的描述更新（lines 4-14 中的"Action编码"等描述保持，因为本就使用 action 术语）

  **Must NOT do**:
  - 不改变分析逻辑或可视化功能
  - 不修改方法签名或接口

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: 简单的属性引用更新
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Sequential
  - **Blocks**: Task 6（验证任务）
  - **Blocked By**: Task 2（依赖 deformation_triplane.py 更新）

  **References**:
  - `utils/triplane_film_analyzer.py:79` - 属性提取
  - `utils/triplane_film_analyzer.py:177, 255, 338, 392, 418` - 属性使用位置
  - `scene/deformation_triplane.py:194` - action_processor 实例（Task 2 完成后）

  **Acceptance Criteria**:
  - [ ] `grep "control_processor" utils/triplane_film_analyzer.py` 返回 0 结果
  - [ ] `grep "action_processor" utils/triplane_film_analyzer.py` 返回至少 6 处
  - [ ] Python 语法检查通过

  **QA Scenarios**:

  ```
  Scenario: 属性提取和使用更新
    Tool: Bash (grep)
    Preconditions: Task 2 完成
    Steps:
      1. grep -n "action_processor" utils/triplane_film_analyzer.py
      2. 确认 line 79 和其他使用位置全部更新
      3. grep "control_processor" utils/triplane_film_analyzer.py
      4. 确认返回 0 结果
    Expected Result: 所有引用更新完成
    Failure Indicators: 仍存在 control_processor 引用
    Evidence: .sisyphus/evidence/task-3-analyzer.txt
  ```

  **Evidence to Capture**:
  - [x] task-3-analyzer.txt: grep 验证输出

  **Commit**: YES
  - Message: `refactor(utils): update triplane_film_analyzer to use action_processor`
  - Files: `utils/triplane_film_analyzer.py`
  - Pre-commit: `python -m py_compile utils/triplane_film_analyzer.py`

---

- [x] 4. 更新配置参数名称

  **What to do**:
  
  **4.1 更新 `arguments/toyarm/triplane.py`**:
  - Line 32: `control_input_dim = 15` → `action_input_dim = 15`
  - Line 33: `control_use_pe = False` → `action_use_pe = False`
  - Line 34: `control_num_frequencies = 4` → `action_num_frequencies = 4`
  - Line 35: `control_hidden_dim = 128` → `action_hidden_dim = 128`
  - Line 36: `control_output_dim = 32` → `action_output_dim = 32`
  - 更新注释（Line 31: "Control Signal Configuration" → "Action Signal Configuration"）
  
  **4.2 更新 `scene/deformation_triplane.py` 中的 getattr 调用**:
  - Line 187: `getattr(args, 'control_use_pe', True)` → `getattr(args, 'action_use_pe', True)`
  - Line 188: `getattr(args, 'control_num_frequencies', 4)` → `getattr(args, 'action_num_frequencies', 4)`
  - Line 189: `getattr(args, 'control_input_dim', 6)` → `getattr(args, 'action_input_dim', 6)`
  - Line 190: `getattr(args, 'control_hidden_dim', 128)` → `getattr(args, 'action_hidden_dim', 128)`
  - Line 192: `getattr(args, 'control_output_dim', 64)` → `getattr(args, 'action_output_dim', 64)`
  - 更新变量名：`control_use_pe` → `action_use_pe`, `control_num_freq` → `action_num_freq`, 等

  **Must NOT do**:
  - 不改变配置值（只改名称）
  - 不修改其他无关配置参数
  - 不改变 getattr 的默认值

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: 配置参数重命名
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Sequential
  - **Blocks**: Task 6（验证任务）
  - **Blocked By**: Task 2（确保代码逻辑先更新）

  **References**:
  - `arguments/toyarm/triplane.py:31-37` - 配置参数定义
  - `scene/deformation_triplane.py:187-192` - 配置参数读取

  **Acceptance Criteria**:
  - [ ] `grep "control_input_dim" arguments/toyarm/triplane.py` 返回 0 结果
  - [ ] `grep "action_input_dim" arguments/toyarm/triplane.py` 返回 1 结果
  - [ ] `grep "control_use_pe" scene/deformation_triplane.py` 返回 0 结果
  - [ ] `grep "action_use_pe" scene/deformation_triplane.py` 返回至少 1 结果

  **QA Scenarios**:

  ```
  Scenario: 配置参数完整更新
    Tool: Bash (grep)
    Preconditions: 所有 edit 完成
    Steps:
      1. grep "control_.*_dim\|control_use_pe\|control_num_freq" arguments/toyarm/triplane.py
      2. 确认返回 0 结果
      3. grep "action_.*_dim\|action_use_pe\|action_num_freq" arguments/toyarm/triplane.py
      4. 确认返回 5 个参数定义
    Expected Result: 所有 control_* 参数已更新为 action_*
    Failure Indicators: 仍存在 control_* 参数
    Evidence: .sisyphus/evidence/task-4-config.txt
  ```

  **Evidence to Capture**:
  - [x] task-4-config.txt: 配置参数验证

  **Commit**: YES
  - Message: `refactor(config): rename control_* params to action_* for consistency`
  - Files: `arguments/toyarm/triplane.py`, `scene/deformation_triplane.py`
  - Pre-commit: `python -c "from arguments.toyarm.triplane import ModelHiddenParams; print(ModelHiddenParams)"`

---

- [x] 5. 更新 `scene/AGENTS.md` 文档

  **What to do**:
  - 在 "KEY ABSTRACTIONS" 或 "CONVENTIONS" 部分添加新的子节
  - 标题: "Naming Convention: action_processor vs control_processor"
  - 内容说明:
    - `action_processor` 和 `control_processor` 在本代码库中是语义等价的
    - 都指代控制/动作信号编码器（TriPlane 架构中）
    - 当前代码使用 `action_processor` 以匹配训练 checkpoint 命名
    - 历史原因：早期版本使用 action，后续曾短暂改为 control，现已统一回 action
    - 与 HexPlane 的 `control_encoder` 不同（那是压缩到 1D 的编码器）

  **Must NOT do**:
  - 不修改文档的其他部分
  - 不添加与此重命名无关的内容

  **Recommended Agent Profile**:
  - **Category**: `writing`
    - Reason: 文档编写任务
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES（可与 Task 4 并行）
  - **Parallel Group**: Wave 2
  - **Blocks**: Task 6（验证任务）
  - **Blocked By**: Task 1-3 完成（需要了解完整上下文）

  **References**:
  - `scene/AGENTS.md` - 现有文档结构
  - `scene/triplane.py:252-265` - ActionProcessor docstring
  - `scene/deformation.py` - HexPlane 的 ControlEncoder（对比参考）

  **Acceptance Criteria**:
  - [ ] `grep "action_processor.*control_processor" scene/AGENTS.md` 返回至少 1 处
  - [ ] 文档中明确说明两者语义等价
  - [ ] 说明了历史演变原因

  **QA Scenarios**:

  ```
  Scenario: 文档添加完成
    Tool: Read
    Preconditions: scene/AGENTS.md 已修改
    Steps:
      1. 读取 scene/AGENTS.md
      2. 查找新增的命名约定说明部分
      3. 确认包含关键信息：语义等价、历史原因、区别于 HexPlane
    Expected Result: 文档清晰说明命名约定
    Failure Indicators: 缺少关键信息或说明不清
    Evidence: .sisyphus/evidence/task-5-doc.txt
  ```

  **Evidence to Capture**:
  - [x] task-5-doc.txt: 新增文档内容摘录

  **Commit**: YES
  - Message: `docs(scene): document action_processor vs control_processor naming convention`
  - Files: `scene/AGENTS.md`
  - Pre-commit: None

---

- [x] 6. 全面验证

  **What to do**:
  
  **6.1 LSP Diagnostics**:
  - 运行 `lsp_diagnostics` 检查所有修改的文件
  - 确保无类型错误、引用错误
  
  **6.2 Grep 全量验证**:
  - `grep -r "ControlProcessor" scene/ utils/ arguments/`（应返回 0 结果，除注释）
  - `grep -r "control_processor" scene/ utils/ arguments/`（应返回 0 结果，除注释）
  - `grep -r "control_feat" scene/`（应返回 0 结果）
  - `grep -r "control_vec" scene/`（应返回 0 结果）
  - `grep -r "control_.*_dim\|control_use_pe\|control_num_freq" arguments/`（应返回 0 结果）
  
  **6.3 语法检查**:
  - `python -m py_compile scene/triplane.py`
  - `python -m py_compile scene/deformation_triplane.py`
  - `python -m py_compile utils/triplane_film_analyzer.py`
  
  **6.4 （可选）Checkpoint 加载测试**:
  - 如果用户提供 checkpoint 路径，尝试加载验证

  **Must NOT do**:
  - 不自动修复验证中发现的问题（报告给用户）
  - 不运行完整训练（太耗时）

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
    - Reason: 需要仔细验证多个维度
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Sequential（最后一步）
  - **Blocks**: 无（最终任务）
  - **Blocked By**: Task 1-5（需要所有修改完成）

  **References**:
  - 所有修改过的文件

  **Acceptance Criteria**:
  - [ ] LSP diagnostics 全部通过（0 errors）
  - [ ] Grep 验证：无遗留的 control_processor 引用
  - [ ] Python 语法检查全部通过
  - [ ] （可选）Checkpoint 加载成功

  **QA Scenarios**:

  ```
  Scenario: LSP 诊断通过
    Tool: lsp_diagnostics
    Preconditions: 所有代码修改完成
    Steps:
      1. lsp_diagnostics(filePath="scene/triplane.py")
      2. lsp_diagnostics(filePath="scene/deformation_triplane.py")
      3. lsp_diagnostics(filePath="utils/triplane_film_analyzer.py")
      4. 确认所有文件 0 errors
    Expected Result: 无类型错误、引用错误
    Failure Indicators: 存在 errors
    Evidence: .sisyphus/evidence/task-6-lsp.txt

  Scenario: Grep 全量验证无遗漏
    Tool: Bash (grep)
    Preconditions: 所有修改完成
    Steps:
      1. grep -r "ControlProcessor" scene/ utils/ arguments/
      2. grep -r "control_processor" scene/ utils/
      3. grep -r "control_feat\|control_vec" scene/
      4. grep -r "control_.*_dim\|control_use_pe" arguments/
      5. 确认所有命令返回 0 结果（或仅注释）
    Expected Result: 无任何 control_* 遗留引用
    Failure Indicators: 发现未更新的引用
    Evidence: .sisyphus/evidence/task-6-grep.txt

  Scenario: Python 语法检查通过
    Tool: Bash (python -m py_compile)
    Preconditions: 所有代码修改完成
    Steps:
      1. python -m py_compile scene/triplane.py scene/deformation_triplane.py utils/triplane_film_analyzer.py
      2. 确认无 SyntaxError
    Expected Result: 编译成功，无语法错误
    Failure Indicators: SyntaxError
    Evidence: .sisyphus/evidence/task-6-syntax.txt
  ```

  **Evidence to Capture**:
  - [x] task-6-lsp.txt: LSP 诊断结果
  - [x] task-6-grep.txt: Grep 验证输出
  - [x] task-6-syntax.txt: 语法检查输出

  **Commit**: NO（验证任务不产生代码变更）

---

## Final Verification Wave

（本计划无需 Final Verification Wave，因为每个任务都包含充分的 QA Scenarios）

---

## Commit Strategy

- **Commit 1**: Task 1-2 完成后
  - `refactor(scene): rename ControlProcessor to ActionProcessor for checkpoint compatibility`
  - Files: `scene/triplane.py`, `scene/deformation_triplane.py`

- **Commit 2**: Task 3 完成后
  - `refactor(utils): update triplane_film_analyzer to use action_processor`
  - Files: `utils/triplane_film_analyzer.py`

- **Commit 3**: Task 4 完成后
  - `refactor(config): rename control_* params to action_* for consistency`
  - Files: `arguments/toyarm/triplane.py`, `scene/deformation_triplane.py`

- **Commit 4**: Task 5 完成后
  - `docs(scene): document action_processor vs control_processor naming convention`
  - Files: `scene/AGENTS.md`

---

## Success Criteria

### Verification Commands
```bash
# 验证无遗留引用
grep -r "ControlProcessor" scene/ utils/ arguments/  # 应返回 0 结果
grep -r "control_processor" scene/ utils/            # 应返回 0 结果
grep -r "control_feat\|control_vec" scene/           # 应返回 0 结果

# 语法检查
python -m py_compile scene/triplane.py scene/deformation_triplane.py utils/triplane_film_analyzer.py

# （可选）测试 checkpoint 加载
python -c "
import torch
from scene.deformation_factory import create_deform_network
from arguments.toyarm.triplane import ModelHiddenParams

# 模拟加载
state_dict = torch.load('path/to/checkpoint.pth')
# 应该能够成功匹配 deformation_net.action_processor.* keys
"
```

### Final Checklist
- [x] 所有 `ControlProcessor` 引用已更新为 `ActionProcessor`
- [x] 所有 `control_processor` 属性已更新为 `action_processor`
- [x] 所有配置参数 `control_*` 已更新为 `action_*`
- [x] scene/AGENTS.md 包含命名约定说明
- [x] LSP diagnostics 通过
- [x] Python 语法检查通过
- [x] Grep 验证无遗留引用
