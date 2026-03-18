# Bidirectional Flow-Based Dynamic Tracking Point Update

## TL;DR

> **Quick Summary**: 修改MPC规划系统，使用双向光流（forward + backward）和一致性检查来动态更新追踪点。每步规划后通过反向光流传播和mask过滤更新追踪点，确保点始终位于运动区域。
> 
> **Deliverables**:
> - 扩展 `mpc/point_sampling.py`: 双向光流计算、一致性检查、光流传播功能
> - 修改 `test_cotracker_mpc.py`: 初始化使用双向光流mask、MPC循环中每步更新追踪点
> - 可视化功能: 保存每步光流、mask、追踪点演化过程
> 
> **Estimated Effort**: Medium (3-4 hours)
> **Parallel Execution**: NO - sequential (每个任务依赖前一个)
> **Critical Path**: Task 1 → Task 2 → Task 3 → Task 4 → Task 5

---

## Context

### Original Request

用户发现当前追踪点mask存在问题，可能导致目标帧的追踪点匹配不准确。原因是：
1. 只使用单向光流（初始帧→目标帧）计算mask
2. 每步规划后没有动态更新追踪点
3. 光流质量缺乏一致性验证

### User's Idea

1. **初始阶段**: 计算双向光流（forward + backward）用于mask生成和追踪点采样
2. **每步规划后**: 计算反向光流（当前帧→上一帧）更新mask和追踪点
3. **更新策略**: 光流传播现有点 + mask过滤无效点 + 补充新点

### Research Findings

当前代码结构：
- `mpc/point_sampling.py`: 已有 `compute_motion_mask()` 单向光流mask计算
- `test_cotracker_mpc.py` (lines 318-397): 初始化时采样追踪点，使用 `tracker.track()` 离线计算目标点
- MPC循环 (lines 591-651): 每步规划→渲染，但不更新追踪点

需要添加：
- 双向光流计算和forward-backward一致性检查
- 光流传播追踪点功能
- MPC循环中的动态追踪点更新逻辑

---

## Work Objectives

### Core Objective

实现基于双向光流一致性检查的动态追踪点更新系统，使MPC规划在每步执行后能够：
1. 通过反向光流传播现有追踪点
2. 使用mask过滤无效点
3. 补充新的高质量追踪点
4. 动态更新目标追踪点位置

### Concrete Deliverables

**新增功能 (mpc/point_sampling.py)**:
- `compute_bidirectional_flow_with_consistency()` - 双向光流+一致性检查
- `adaptive_motion_mask_with_consistency()` - 基于一致性的自适应motion mask
- `propagate_points_with_flow()` - 光流传播追踪点
- `update_tracking_points_dynamic()` - 完整的动态更新流程（传播+过滤+补充）

**修改功能 (test_cotracker_mpc.py)**:
- 初始化阶段: 使用双向光流mask采样追踪点
- MPC循环: 每步规划后调用动态更新函数
- 可视化: 保存每步的光流、mask、追踪点演化

### Definition of Done

- [ ] `bun test mpc/test_point_sampling.py` - 新功能单元测试通过
- [ ] 运行完整MPC测试，检查追踪点在每步都更新
- [ ] 可视化输出包含: 光流图、mask演化、追踪点变化
- [ ] 日志清晰显示每步追踪点数量变化（传播、过滤、补充）

### Must Have

- 双向光流一致性检查必须正确实现
- 光流传播必须保持追踪点连续性
- Mask过滤必须移除无效点
- 补充点必须来自motion mask区域
- 每步更新后追踪点数量保持在合理范围（200-400个）

### Must NOT Have (Guardrails)

- ❌ 不要在每步完全重新采样追踪点（会丢失长期追踪）
- ❌ 不要忽略一致性检查（会导致mask质量下降）
- ❌ 不要在追踪点数量不足时panic（平滑降级到grid采样）
- ❌ 不要保存过多中间可视化（每步最多3张图）
- ❌ 不要修改 `PointTracker` 类本身（只修改采样和更新逻辑）

---

## Verification Strategy

### Test Decision

- **Infrastructure exists**: YES (pytest, unittest)
- **Automated tests**: Tests-after (单元测试 + 集成测试)
- **Framework**: pytest
- **If TDD**: N/A

### QA Policy

每个任务包含Agent-Executed QA场景（见TODO部分），使用：
- **单元测试**: `pytest mpc/test_point_sampling.py` - 测试新函数正确性
- **集成测试**: 运行 `test_cotracker_mpc.py` 并检查输出
- **可视化验证**: 检查保存的图片是否显示追踪点正确更新

---

## Execution Strategy

### Parallel Execution Waves

```
Wave 1 (Foundation - 扩展point_sampling.py):
├── Task 1: 添加双向光流+一致性检查函数 [quick]

Wave 2 (Core Logic):
├── Task 2: 添加光流传播和mask过滤函数 [quick]
└── Task 3: 添加完整动态更新函数 [unspecified-high]

Wave 3 (Integration):
├── Task 4: 修改test_cotracker_mpc.py初始化逻辑 [unspecified-high]
└── Task 5: 添加MPC循环中的动态更新调用 [deep]

Wave 4 (Validation):
├── Task 6: 添加可视化保存功能 [visual-engineering]
└── Task 7: 运行完整测试并验证结果 [deep]
```

**Critical Path**: Task 1 → Task 2 → Task 3 → Task 4 → Task 5 → Task 6 → Task 7  
**Parallel Speedup**: None (sequential)  
**Max Concurrent**: 1

---

## TODOs

- [ ] 1. 扩展point_sampling.py: 添加双向光流+一致性检查函数

  **What to do**:
  - 在 `mpc/point_sampling.py` 末尾添加 `compute_bidirectional_flow_with_consistency()` 函数
  - 计算forward flow (img1→img2) 和 backward flow (img2→img1)
  - 实现forward-backward consistency check: `||flow_forward(x) + flow_backward(x+flow_forward(x))|| < threshold`
  - 返回 `flow_forward`, `flow_backward`, `consistency_mask`, `flow_magnitude`
  - 添加 `adaptive_motion_mask_with_consistency()` 包装函数，结合一致性mask和自适应阈值

  **Must NOT do**:
  - 不要修改现有 `compute_motion_mask()` 函数（保持向后兼容）
  - 不要在这个任务中添加光流传播逻辑（下一个任务）

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: []
  - **Reason**: 清晰定义的函数，参考现有GMFlow使用模式，代码量小

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Wave 1
  - **Blocks**: Task 2, Task 3, Task 4
  - **Blocked By**: None

  **References**:
  - `mpc/point_sampling.py:298-377` - 现有 `compute_motion_mask()` 实现，复用GMFlow初始化代码
  - `mpc/point_sampling.py:328-343` - GMFlow模型加载模式
  - `mpc/point_sampling.py:346-360` - 光流计算和magnitude提取
  - Forward-backward consistency theory: UnFlow (ICCV 2017)
  - Consistency threshold典型值: 1-5 pixels (default 3.0)

  **Acceptance Criteria**:
  - [ ] 函数 `compute_bidirectional_flow_with_consistency()` 添加成功
  - [ ] 函数 `adaptive_motion_mask_with_consistency()` 添加成功
  - [ ] 两个函数都有完整docstring（包含Args、Returns、References）
  - [ ] Python语法正确（可通过 `python -m py_compile mpc/point_sampling.py`）

  **QA Scenarios**:
  ```
  Scenario: 双向光流计算 - 正常运动场景
    Tool: Bash (python REPL)
    Preconditions: 
      - 有两张测试图片: assets/user_provided/initial_frame.jpg, target_frame.jpg
      - Images shape: (480, 480, 3), dtype: float32, range: [0, 1]
    Steps:
      1. 导入函数: from mpc.point_sampling import compute_bidirectional_flow_with_consistency
      2. 加载图片: img1, img2 = load_test_images()
      3. 调用: flow_f, flow_b, cons_mask, mag = compute_bidirectional_flow_with_consistency(img1, img2, 'cuda:0', 3.0)
      4. 检查返回值形状: flow_f.shape == (480, 480, 2), cons_mask.shape == (480, 480), mag.shape == (480, 480)
      5. 检查一致性比例: 50% < cons_mask.mean() < 95% (合理范围)
    Expected Result: 
      - 所有shape正确
      - 一致性比例在合理范围内
      - 打印日志包含 "[BiFlow] Consistency check: XX.X% pixels consistent"
    Evidence: .sisyphus/evidence/task-1-biflow-normal.txt (保存shape和统计信息)

  Scenario: 边界情况 - 静态场景（无运动）
    Tool: Bash (python REPL)
    Preconditions: img1 = img2 (identical images)
    Steps:
      1. 调用双向光流函数
      2. 检查 flow_magnitude.max() ≈ 0
      3. 检查 consistency_mask.mean() > 0.95 (几乎全部一致)
    Expected Result:
      - 静态场景也能正确处理
      - 一致性mask接近100%
    Evidence: .sisyphus/evidence/task-1-biflow-static.txt
  ```

  **Commit**: YES
  - Message: `feat(mpc): add bidirectional flow with consistency check`
  - Files: `mpc/point_sampling.py`
  - Pre-commit: `python -m py_compile mpc/point_sampling.py`

---

- [ ] 2. 扩展point_sampling.py: 添加光流传播和mask过滤函数

  **What to do**:
  - 添加 `propagate_points_with_flow(points, flow_field, mask=None)` 函数
  - 给定点坐标 `(N, 2)` 和光流场 `(H, W, 2)`, 计算传播后的新坐标
  - 使用双线性插值或最近邻从flow_field中采样每个点的flow向量
  - 如果提供mask, 过滤掉传播后不在mask内的点
  - 返回 `propagated_points (M, 2)` 和 `valid_mask (N,)` boolean数组

  **Must NOT do**:
  - 不要在这个函数中添加补充新点的逻辑（Task 3）
  - 不要修改input points（创建新数组）

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: []
  - **Reason**: 简单的numpy array操作和插值

  **Parallelization**:
  - **Can Run In Parallel**: NO (depends on Task 1)
  - **Parallel Group**: Wave 2
  - **Blocks**: Task 3
  - **Blocked By**: Task 1

  **References**:
  - Numpy indexing for sampling: `flow_at_points = flow_field[y_coords, x_coords]`
  - Similar logic in: `mpc/flow_objectives.py` - 光流采样模式
  - Mask filtering: `propagated_points = new_points[valid_mask]`

  **Acceptance Criteria**:
  - [ ] 函数 `propagate_points_with_flow()` 添加成功
  - [ ] 函数有完整docstring
  - [ ] 边界情况处理: 点超出图像范围时clip到边界
  - [ ] Python语法正确

  **QA Scenarios**:
  ```
  Scenario: 光流传播 - 向右平移10像素
    Tool: Bash (python REPL)
    Preconditions: 
      - 创建人工光流场: flow = np.zeros((480, 480, 2)); flow[:, :, 0] = 10  # 全部向右10px
      - 初始点: points = np.array([[100, 100], [200, 200], [300, 300]])
    Steps:
      1. 导入: from mpc.point_sampling import propagate_points_with_flow
      2. 传播: new_pts, valid = propagate_points_with_flow(points, flow, mask=None)
      3. 检查: new_pts[:, 0] ≈ points[:, 0] + 10 (x坐标增加10)
      4. 检查: new_pts[:, 1] ≈ points[:, 1] (y坐标不变)
      5. 检查: valid.all() == True (所有点有效)
    Expected Result: 点正确向右平移10像素
    Evidence: .sisyphus/evidence/task-2-propagate-shift.txt

  Scenario: Mask过滤 - 部分点移出mask
    Tool: Bash (python REPL)
    Preconditions:
      - 光流向右移动
      - Mask只覆盖左半边: mask = np.zeros((480, 480), bool); mask[:, :240] = True
      - 初始点在左半边，传播后会移到右半边（mask外）
    Steps:
      1. 传播: new_pts, valid = propagate_points_with_flow(points, flow, mask)
      2. 检查: len(new_pts) < len(points) (部分点被过滤)
      3. 检查: new_pts都在mask内
    Expected Result: 移出mask的点被正确过滤
    Evidence: .sisyphus/evidence/task-2-propagate-mask.txt
  ```

  **Commit**: YES
  - Message: `feat(mpc): add optical flow point propagation with mask filtering`
  - Files: `mpc/point_sampling.py`
  - Pre-commit: `python -m py_compile mpc/point_sampling.py`

---

- [ ] 3. 扩展point_sampling.py: 添加完整动态更新函数

  **What to do**:
  - 添加 `update_tracking_points_dynamic(current_points, current_image, target_image, prev_image, num_points_target, device)` 函数
  - **Step 1**: 计算 `current_image → prev_image` 的反向光流（用于传播现有点）
  - **Step 2**: 使用 `propagate_points_with_flow()` 传播 `current_points`
  - **Step 3**: 计算 `current_image → target_image` 的双向光流和mask（用于补充新点）
  - **Step 4**: 如果传播后点数 < `num_points_target * 0.7`, 从motion mask中补充新点
  - **Step 5**: 使用spatial NMS确保点分布均匀
  - **Step 6**: 计算新的target points（通过 `current_image → target_image` 的光流传播current points）
  - 返回: `updated_current_points, updated_target_points, motion_mask_debug`

  **Must NOT do**:
  - 不要在这个函数中处理可视化（Task 6）
  - 不要过度补充点（最多补充到 `num_points_target`）

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
  - **Skills**: []
  - **Reason**: 复杂的多步骤逻辑，需要仔细协调各个子函数

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Wave 2
  - **Blocks**: Task 5 (MPC集成需要这个函数)
  - **Blocked By**: Task 1, Task 2

  **References**:
  - `mpc/point_sampling.py:431-561` - `sample_motion_driven_points()` 采样模式（复用补充点逻辑）
  - `mpc/point_sampling.py:507-508` - Spatial NMS应用
  - Task 1的双向光流函数
  - Task 2的光流传播函数

  **Acceptance Criteria**:
  - [ ] 函数 `update_tracking_points_dynamic()` 添加成功
  - [ ] 函数正确处理点数不足的情况（补充新点）
  - [ ] 返回的points数量在合理范围内（`num_points_target * 0.8` ~ `num_points_target * 1.0`）
  - [ ] 打印日志清晰显示: 传播后点数、补充点数、最终点数
  - [ ] Python语法正确

  **QA Scenarios**:
  ```
  Scenario: 动态更新 - 正常情况（60%点保留）
    Tool: Bash (python REPL)
    Preconditions:
      - current_points: 300个点
      - 三张图片: prev, current, target
      - 60%的点经过光流传播后仍在mask内
    Steps:
      1. 导入: from mpc.point_sampling import update_tracking_points_dynamic
      2. 调用: new_curr, new_tgt, mask = update_tracking_points_dynamic(
                   current_points, current_img, target_img, prev_img, 300, 'cuda:0')
      3. 检查: 180 < len(new_curr) < 300 (传播保留60% + 补充)
      4. 检查: len(new_curr) == len(new_tgt) (current和target点数相等)
      5. 检查日志: 包含 "Propagated: XXX → YYY valid, Supplemented: ZZZ"
    Expected Result: 点数保持在目标数量，日志清晰
    Evidence: .sisyphus/evidence/task-3-update-normal.txt

  Scenario: 动态更新 - 点大量丢失（仅20%保留）
    Tool: Bash (python REPL)
    Preconditions:
      - 只有20%的点在传播后有效（运动剧烈或遮挡严重）
      - 需要补充大量新点
    Steps:
      1. 调用更新函数
      2. 检查: 最终点数接近num_points_target（通过补充达到）
      3. 检查: 新补充的点都在motion mask内
    Expected Result: 平滑降级，补充足够多的新点
    Evidence: .sisyphus/evidence/task-3-update-loss.txt
  ```

  **Commit**: YES
  - Message: `feat(mpc): add dynamic tracking point update with flow propagation`
  - Files: `mpc/point_sampling.py`
  - Pre-commit: `python -m py_compile mpc/point_sampling.py`

---

- [ ] 4. 修改test_cotracker_mpc.py: 初始化使用双向光流mask

  **What to do**:
  - 修改 `test_cotracker_mpc.py` lines 318-397 的追踪点采样逻辑
  - 当 `args.sampling_method == "motion_mask"` 时，调用新的 `adaptive_motion_mask_with_consistency()` 替代原来的 `compute_motion_mask()`
  - 保持其他采样方法不变（shi_tomasi, combined, etc.）
  - 在初始化阶段也保存一致性mask的可视化（用于debug）

  **Must NOT do**:
  - 不要删除原有的采样方法选项
  - 不要修改default采样方法（保持向后兼容）
  - 不要在这个任务中修改MPC循环逻辑（Task 5）

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
  - **Skills**: []
  - **Reason**: 需要理解现有代码结构并谨慎修改

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Wave 3
  - **Blocks**: Task 5
  - **Blocked By**: Task 1, Task 2, Task 3

  **References**:
  - `test_cotracker_mpc.py:355-365` - 现有motion_mask采样方法调用
  - `mpc/point_sampling.py:431-461` - `sample_motion_driven_points()` 内部调用 `compute_motion_mask()`
  - Task 1的 `adaptive_motion_mask_with_consistency()` 函数

  **Acceptance Criteria**:
  - [ ] `test_cotracker_mpc.py` 修改成功
  - [ ] 使用motion_mask方法时调用新的一致性检查函数
  - [ ] 初始化日志包含一致性统计信息
  - [ ] Python语法正确，可以正常运行

  **QA Scenarios**:
  ```
  Scenario: 初始化 - 使用双向光流mask采样
    Tool: Bash (直接运行脚本)
    Preconditions: 
      - 测试图片: assets/user_provided/initial_frame.jpg, target_frame.jpg
      - 模型checkpoint存在
    Steps:
      1. 运行: python test_cotracker_mpc.py --sampling_method motion_mask --num_steps 1 --device cuda:1 --output_dir outputs/test_biflow_init
      2. 检查日志包含: "[BiFlow] Consistency check: XX.X% pixels consistent"
      3. 检查日志包含: "[AdaptiveMask] Motion mask coverage: XX.X%"
      4. 检查输出目录: 01_initial_with_points.png 存在
      5. 检查点采样成功（日志显示点数）
    Expected Result: 
      - 脚本运行无错误
      - 一致性检查日志正确输出
      - 追踪点采样成功
    Failure Indicators:
      - Import error: adaptive_motion_mask_with_consistency not found
      - 函数调用错误: 参数不匹配
    Evidence: .sisyphus/evidence/task-4-init-biflow.log (完整运行日志)

  Scenario: 初始化 - 使用其他采样方法（向后兼容）
    Tool: Bash
    Steps:
      1. 运行: python test_cotracker_mpc.py --sampling_method combined --num_steps 1 --device cuda:1 --output_dir outputs/test_combined_compat
      2. 检查: 不应调用双向光流（日志中无"[BiFlow]"）
      3. 检查: 采样成功完成
    Expected Result: 其他采样方法不受影响
    Evidence: .sisyphus/evidence/task-4-init-compat.log
  ```

  **Commit**: YES
  - Message: `feat(mpc): use bidirectional flow mask for initial point sampling`
  - Files: `test_cotracker_mpc.py`
  - Pre-commit: `python -m py_compile test_cotracker_mpc.py`

---

- [ ] 5. 修改test_cotracker_mpc.py: MPC循环中添加动态追踪点更新

  **What to do**:
  - 修改 `test_cotracker_mpc.py` 的MPC循环部分（约lines 550-651）
  - 在每步规划完成、渲染当前帧后（`step_rendered_rgb`生成后），添加追踪点更新逻辑：
    ```python
    # After rendering current frame
    if t > 0:  # Skip first step (no previous frame)
        updated_current_points, updated_target_points, debug_mask = \
            update_tracking_points_dynamic(
                current_points=goal["current_points"].cpu().numpy(),
                current_image=step_rendered_rgb,
                target_image=target_image,
                prev_image=prev_rendered_rgb,
                num_points_target=args.num_tracking_points,
                device=args.device
            )
        # Update goal dictionary
        goal["current_points"] = torch.from_numpy(updated_current_points).to(args.device)
        goal["target_points"] = torch.from_numpy(updated_target_points).to(args.device)
        # Save prev frame for next iteration
        prev_rendered_rgb = step_rendered_rgb.copy()
    ```
  - 需要在循环开始前初始化 `prev_rendered_rgb = initial_image`
  - 打印每步的追踪点数量变化

  **Must NOT do**:
  - 不要修改第一步（t=0）的逻辑（没有prev frame）
  - 不要改变原有的action执行和渲染逻辑
  - 不要在每步都完全重新采样（使用动态更新函数）

  **Recommended Agent Profile**:
  - **Category**: `deep`
  - **Skills**: []
  - **Reason**: 需要深入理解MPC循环逻辑，仔细插入更新代码而不破坏原有流程

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Wave 3
  - **Blocks**: Task 6
  - **Blocked By**: Task 3, Task 4

  **References**:
  - `test_cotracker_mpc.py:591-651` - MPC主循环结构
  - `test_cotracker_mpc.py:608-620` - 渲染逻辑（`step_rendered_rgb`生成位置）
  - Task 3的 `update_tracking_points_dynamic()` 函数
  - `test_cotracker_mpc.py:479-488` - goal字典初始化（包含current_points, target_points）

  **Acceptance Criteria**:
  - [ ] MPC循环修改成功
  - [ ] 每步（t>0）都调用动态更新函数
  - [ ] goal字典正确更新（current_points和target_points都更新）
  - [ ] 日志清晰显示每步追踪点数量变化
  - [ ] 第一步（t=0）不调用更新（使用初始采样的点）
  - [ ] Python语法正确

  **QA Scenarios**:
  ```
  Scenario: MPC循环 - 追踪点动态更新（10步）
    Tool: Bash (完整MPC测试)
    Preconditions:
      - 完整的测试环境
      - 使用motion_mask采样方法
    Steps:
      1. 运行: python test_cotracker_mpc.py --sampling_method motion_mask --num_steps 10 --horizon 10 --device cuda:1 --output_dir outputs/test_dynamic_update_full
      2. 检查日志: 每步（Step 2-10）都包含追踪点更新信息
      3. 检查日志格式: "Step X: Points updated XXX → YYY (propagated: AAA, supplemented: BBB)"
      4. 检查: 追踪点数量在整个过程中保持在200-400范围内
      5. 检查: 规划正常完成，生成所有step_XXXX_rendered.png
    Expected Result:
      - 每步都成功更新追踪点
      - 点数量保持稳定
      - MPC规划正常完成
    Failure Indicators:
      - 更新函数抛出异常
      - 追踪点数量突然降到很低（<50）
      - goal字典更新失败导致tracking objective报错
    Evidence: .sisyphus/evidence/task-5-mpc-dynamic.log (完整日志 + metrics.json)

  Scenario: MPC循环 - 第一步不更新（边界情况）
    Tool: Bash (检查日志)
    Steps:
      1. 运行1步测试: python test_cotracker_mpc.py --num_steps 1 ...
      2. 检查日志: Step 1不应包含"Points updated"（因为t=0没有prev frame）
    Expected Result: 第一步跳过更新，使用初始采样点
    Evidence: .sisyphus/evidence/task-5-mpc-first-step.log
  ```

  **Commit**: YES
  - Message: `feat(mpc): add dynamic tracking point update in MPC loop`
  - Files: `test_cotracker_mpc.py`
  - Pre-commit: `python -m py_compile test_cotracker_mpc.py && python test_cotracker_mpc.py --help > /dev/null`

---

- [ ] 6. 添加可视化: 保存每步光流、mask、追踪点演化

  **What to do**:
  - 在 `test_cotracker_mpc.py` MPC循环中，每步保存以下可视化（如果 `--save_flow_debug` flag开启）：
    1. `step_XXXX_flow_magnitude.png` - 光流幅度图（彩色热力图）
    2. `step_XXXX_consistency_mask.png` - 一致性mask叠加在当前帧上
    3. `step_XXXX_points_evolution.png` - 追踪点变化（红色=传播保留，绿色=新补充）
  - 添加命令行参数: `--save_flow_debug` (default: False)
  - 添加命令行参数: `--flow_debug_interval` (default: 1, 每步保存；可设为2表示每2步保存)
  - 使用matplotlib或cv2生成可视化

  **Must NOT do**:
  - 不要默认保存所有可视化（会产生大量文件）
  - 不要在可视化失败时中断MPC循环（try-except包裹）
  - 不要保存原始flow field数组（太大，只保存图片）

  **Recommended Agent Profile**:
  - **Category**: `visual-engineering`
  - **Skills**: []
  - **Reason**: 可视化任务，需要生成美观的调试图片

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Wave 4
  - **Blocks**: None
  - **Blocked By**: Task 5

  **References**:
  - `test_cotracker_mpc.py:135-149` - 现有 `visualize_points()` 函数
  - `mpc/point_sampling.py:524-556` - motion mask可视化示例
  - Matplotlib heatmap: `plt.imshow(flow_mag, cmap='hot')`
  - 叠加mask: `ax.imshow(mask, alpha=0.3, cmap='Reds')`

  **Acceptance Criteria**:
  - [ ] 添加 `--save_flow_debug` 和 `--flow_debug_interval` 参数
  - [ ] 可视化函数实现完成（生成3种图片）
  - [ ] 可视化代码包裹在 `if args.save_flow_debug:` 和 `try-except` 中
  - [ ] 生成的图片清晰易读（包含标题、图例）
  - [ ] 不影响MPC主循环性能（可视化耗时<0.5秒/步）

  **QA Scenarios**:
  ```
  Scenario: 可视化 - 开启debug模式保存所有图片
    Tool: Bash
    Steps:
      1. 运行: python test_cotracker_mpc.py --num_steps 5 --save_flow_debug --flow_debug_interval 1 --device cuda:1 --output_dir outputs/test_viz_debug
      2. 检查输出目录包含:
         - step_0002_flow_magnitude.png (步骤2开始有prev frame)
         - step_0002_consistency_mask.png
         - step_0002_points_evolution.png
         - ... (步骤3-5同样的3个文件)
      3. 打开图片验证: 光流热力图颜色映射正确，mask叠加清晰，追踪点颜色区分明显
    Expected Result: 所有可视化图片生成且质量良好
    Evidence: .sisyphus/evidence/task-6-viz-debug/ (保存3张示例图片)

  Scenario: 可视化 - 关闭debug模式（默认）
    Tool: Bash
    Steps:
      1. 运行: python test_cotracker_mpc.py --num_steps 5 --device cuda:1 --output_dir outputs/test_viz_off
      2. 检查: 输出目录不包含 *_flow_magnitude.png 等debug图片
      3. 检查: 只有step_XXXX_rendered.png 和 step_XXXX_with_points.png（原有文件）
    Expected Result: 默认不生成额外可视化，保持向后兼容
    Evidence: .sisyphus/evidence/task-6-viz-off.log (ls输出)
  ```

  **Commit**: YES
  - Message: `feat(mpc): add optical flow and mask visualization for debugging`
  - Files: `test_cotracker_mpc.py`
  - Pre-commit: `python -m py_compile test_cotracker_mpc.py`

---

- [ ] 7. 测试验证: 运行完整MPC测试并验证追踪点质量

  **What to do**:
  - 运行完整10步MPC测试，使用以下配置:
    ```bash
    python test_cotracker_mpc.py \
      --camera_name cam06 \
      --initial_frame_name frame_00001 \
      --sampling_method motion_mask \
      --num_steps 10 \
      --horizon 10 \
      --device cuda:1 \
      --save_flow_debug \
      --output_dir outputs/test_biflow_validation_full
    ```
  - 验证以下指标:
    1. **追踪点数量稳定性**: 每步点数保持在200-400范围内
    2. **追踪点质量**: 目标帧匹配距离 < 原baseline（对比outputs/test_cam06_with_initial_u_correct/metrics.json）
    3. **可视化完整性**: 所有debug图片生成且清晰
    4. **性能开销**: 每步增加的时间 < 5秒（光流计算+更新）
  - 生成对比报告: `.sisyphus/evidence/task-7-validation-report.md`

  **Must NOT do**:
  - 不要修改任何代码（纯验证任务）
  - 不要跳过完整10步测试（确保长期追踪稳定性）

  **Recommended Agent Profile**:
  - **Category**: `deep`
  - **Skills**: []
  - **Reason**: 需要深入分析测试结果，对比多个指标

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Wave 4
  - **Blocks**: None
  - **Blocked By**: Task 1-6 (所有前置任务)

  **References**:
  - `outputs/test_cam06_with_initial_u_correct/metrics.json` - Baseline指标
  - `outputs/test_cam06_with_initial_u_correct/loss_history.csv` - Baseline loss曲线

  **Acceptance Criteria**:
  - [ ] 完整10步测试成功运行
  - [ ] 追踪点数量在每步保持稳定（标准差 < 50个点）
  - [ ] 生成验证报告，包含指标对比表格
  - [ ] 所有QA场景通过

  **QA Scenarios**:
  ```
  Scenario: 完整MPC测试 - 10步规划验证
    Tool: Bash (完整测试)
    Preconditions: Task 1-6全部完成
    Steps:
      1. 运行完整测试命令（见上方）
      2. 检查日志: 每步都包含追踪点更新信息
      3. 提取每步点数: grep "Points updated" outputs/test_biflow_validation_full/*.log
      4. 检查: 点数标准差 < 50
      5. 对比metrics.json: mean_dist是否低于baseline（224.35像素）
      6. 检查性能: 每步时间增加 < 5秒
    Expected Result:
      - 测试成功完成
      - 追踪点数量稳定
      - 追踪质量提升（mean_dist降低）
      - 性能开销可接受
    Failure Indicators:
      - 某步点数突然降到<100（更新逻辑失败）
      - mean_dist反而升高（追踪质量下降）
      - 某步耗时>10秒（光流计算过慢）
    Evidence: 
      - .sisyphus/evidence/task-7-validation-full.log (完整日志)
      - .sisyphus/evidence/task-7-metrics-comparison.txt (指标对比)
      - .sisyphus/evidence/task-7-validation-report.md (完整报告)

  Scenario: 可视化验证 - 检查追踪点演化图
    Tool: Manual (人工查看图片)
    Steps:
      1. 打开: outputs/test_biflow_validation_full/step_0005_points_evolution.png
      2. 检查: 红色点（传播保留）集中在运动区域
      3. 检查: 绿色点（新补充）也在运动区域
      4. 检查: 点分布均匀，无明显聚集
    Expected Result: 可视化清晰显示追踪点动态更新过程
    Evidence: .sisyphus/evidence/task-7-viz-sample/ (保存3张代表性图片)
  ```

  **Commit**: YES
  - Message: `test(mpc): validate bidirectional flow tracking quality`
  - Files: `.sisyphus/evidence/task-7-validation-report.md` (验证报告)
  - Pre-commit: N/A (不修改代码)

---

## Final Verification Wave

> 所有实现任务完成后，运行以下4个独立验证任务（并行执行）

- [ ] F1. **单元测试验证** — `unspecified-high`
  
  创建 `mpc/test_point_sampling.py` 单元测试文件，测试新添加的函数：
  - `test_bidirectional_flow_basic()` - 双向光流计算基本功能
  - `test_consistency_check()` - 一致性检查正确性
  - `test_propagate_points()` - 光流传播准确性
  - `test_dynamic_update()` - 完整更新流程
  
  运行: `pytest mpc/test_point_sampling.py -v`
  
  输出: `Tests [N/N pass] | Coverage [N functions] | VERDICT: PASS/FAIL`

- [ ] F2. **性能基准测试** — `unspecified-high`
  
  测量新功能的性能开销：
  - 双向光流计算时间（ms）
  - 追踪点更新时间（ms）  
  - 每步总额外时间（ms）
  - 内存占用增加（MB）
  
  对比baseline（无动态更新）vs 新版本（有动态更新），确保：
  - 每步额外时间 < 5秒
  - 内存增加 < 500MB
  
  输出: `Performance [Baseline vs New] | Extra Time [X.XXs/step] | Memory [+XXX MB] | VERDICT: ACCEPT/REJECT`

- [ ] F3. **追踪质量对比测试** — `deep`
  
  运行3个测试场景，对比追踪质量:
  1. **Baseline**: 原始固定追踪点（不更新）
  2. **New**: 双向光流动态更新
  3. **每个场景**: 计算最终mean_dist, 追踪点存活率, loss收敛速度
  
  场景:
  - `cam06/frame_00001 → frame_00018` (当前测试)
  - `cam01/frame_00001 → frame_00020` (不同相机)
  - `cam06/frame_00050 → frame_00070` (不同时间段)
  
  输出: `Quality [3/3 scenarios] | mean_dist improvement [X%] | VERDICT: BETTER/WORSE/SAME`

- [ ] F4. **边界情况鲁棒性测试** — `deep`
  
  测试极端情况下的系统鲁棒性:
  1. **静态场景**: initial = target（无运动）
  2. **剧烈运动**: 相隔50帧的图片（大位移）
  3. **遮挡严重**: 80%点移出视野
  4. **纹理缺失**: 白墙场景（光流失效）
  
  每个场景验证:
  - 不crash（异常处理正确）
  - 降级策略生效（fallback到grid采样）
  - 日志包含WARNING信息
  
  输出: `Robustness [4/4 scenarios handled] | Crashes [0] | VERDICT: ROBUST/FRAGILE`

---

## Commit Strategy

每个任务完成后立即commit（见各任务的Commit部分）。

**最终提交顺序**:
1. `feat(mpc): add bidirectional flow with consistency check`
2. `feat(mpc): add optical flow point propagation with mask filtering`
3. `feat(mpc): add dynamic tracking point update with flow propagation`
4. `feat(mpc): use bidirectional flow mask for initial point sampling`
5. `feat(mpc): add dynamic tracking point update in MPC loop`
6. `feat(mpc): add optical flow and mask visualization for debugging`
7. `test(mpc): validate bidirectional flow tracking quality`

**Squash策略**: 如果需要，可以squash commits 1-3（基础函数）为单个commit，保持commits 4-6（集成）独立。

---

## Success Criteria

### 功能验证

```bash
# 所有新函数可导入
python -c "from mpc.point_sampling import compute_bidirectional_flow_with_consistency, propagate_points_with_flow, update_tracking_points_dynamic"

# MPC测试成功运行
python test_cotracker_mpc.py --sampling_method motion_mask --num_steps 10 --device cuda:1 --output_dir outputs/final_test
```

### 质量指标

- [ ] 追踪点数量稳定性: 标准差 < 50个点（10步测试）
- [ ] 追踪质量提升: mean_dist降低 > 10%（对比baseline）
- [ ] 性能开销: 每步额外时间 < 5秒
- [ ] 可视化完整: 所有debug图片生成且清晰
- [ ] 代码质量: 所有新函数有docstring，无syntax error

### 最终检查清单

- [ ] 所有7个TODO任务标记为completed
- [ ] 所有4个Final Verification任务PASS
- [ ] Git commits按照Commit Strategy提交
- [ ] 验证报告完整（.sisyphus/evidence/task-7-validation-report.md）
- [ ] 用户测试: 运行成功，追踪点质量改善明显

---

## 附录: 用户原始需求

**问题描述**:
> 1. 首先现在的mask看着是有点问题的，可能是光流导致的？因为计算光流是通过初始帧和结束帧之间的差别在初始帧基础上计算的，这可能导致了结束帧的追踪点匹配不到

**解决方案**:
> 2. 所以现在想要完成的idea是，首先初始状态通过初始帧和目标帧计算两次光流（正向和反向），在光流mask的基础上进行追踪点的获取
> 3. 在每一步规划完成之后计算当前帧和上一帧之间的光流（注意是反向计算），再通过这个光流mask来进行追踪点的捕获
> 4. 在此基础上再继续进行规划的尝试

**用户选择的策略**:
- 追踪点更新策略: **光流传播+mask过滤**（保留现有点，mask过滤，补充新点）
- 双向光流一致性检查: **需要**（提高mask质量）
- 光流mask阈值策略: **自适应阈值**（根据场景自动调整）
