# Learnings: Motion-Driven Mask Sampling

This file accumulates knowledge from task execution.


## [2026-03-10T03:30] Morphological Operations Research

**Source**: Background agent exploration (bg_d88bc87f)

### Kernel Convention
- **Standard**: `np.ones((5, 5), np.uint8)` across codebase
- **No variations found**: All morphology uses 5x5 square kernel

### Operation Sequence
- **Pattern**: MORPH_CLOSE → MORPH_OPEN
  - Close: Fills small holes, connects close regions
  - Open: Removes small isolated noise
- **Reference**: `test_cotracker_mpc.py:78-80`

### Data Type Pipeline
```python
# Boolean → uint8 → morphology → boolean
mask_bool = flow_magnitude > threshold
mask_uint8 = mask_bool.astype(np.uint8)
mask_uint8 = cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, kernel)
mask_uint8 = cv2.morphologyEx(mask_uint8, cv2.MORPH_OPEN, kernel)
mask_clean = mask_uint8.astype(bool)
```

### Image Conversion Convention
- **Internal**: float [0, 1]
- **Visualization/Save**: `(image * 255).astype(np.uint8)`
- **Helper pattern**: `to8b = lambda x: (255*np.clip(x,0,1)).astype(np.uint8)`

**Action for Task 1**: Use exact 5x5 kernel, MORPH_CLOSE→MORPH_OPEN sequence


## [2026-03-10T03:32] GMFlow Initialization & Memory Management

**Source**: Background agent exploration (bg_235503e4)

### Checkpoint Path
- **Canonical**: `gmflow/checkpoints/gmflow_sintel-0c07dcb3.pth`
- **Location verified**: File exists (18MB)
- **Config source**: `gmflow/config.py` uses `'./gmflow/checkpoints/...'`

### Initialization Pattern (from demo_flow_guided_mpc.py)
```python
from gmflow.config import get_cfg as get_gmflow_cfg
from gmflow.gmflow import GMFlow

gmflow_cfg = get_gmflow_cfg()
flownet = GMFlow(
    feature_channels=gmflow_cfg.feature_channels,
    num_scales=gmflow_cfg.num_scales,
    upsample_factor=gmflow_cfg.upsample_factor,
    num_head=gmflow_cfg.num_head,
    attention_type=gmflow_cfg.attention_type,
    ffn_dim_expansion=gmflow_cfg.ffn_dim_expansion,
    num_transformer_layers=gmflow_cfg.num_transformer_layers,
).to(device)

checkpoint = torch.load(gmflow_cfg.model, map_location="cpu")
weights = checkpoint["model"] if "model" in checkpoint else checkpoint
flownet.load_state_dict(weights, strict=True)
flownet.eval()
```

### Memory Cleanup
- **CRITICAL**: No production code deletes flownet after use
- **Recommendation**: Explicit `del flownet` + `torch.cuda.empty_cache()`
- **Pattern**: Load to CPU → move to GPU → compute → cleanup

### Safe Loading Pattern
```python
# Load to CPU first (reduces peak GPU memory)
checkpoint = torch.load(checkpoint_path, map_location='cpu')
weights = checkpoint['model'] if isinstance(checkpoint, dict) and 'model' in checkpoint else checkpoint
flownet.load_state_dict(weights, strict=False)
flownet.to(device)
flownet.eval()

# ... use flownet ...

# Cleanup
del flownet
torch.cuda.empty_cache()
```

**Action for Task 1**: Use safe loading pattern + explicit cleanup


## [2026-03-10T03:32] Adaptive Flow Thresholding Research

**Source**: Background agent librarian (bg_0514f0b9)

### Recommended Strategy: Hybrid Adaptive Thresholding

#### 1. Percentile Value
- **Recommended**: 70th percentile (top 30% of motion)
- **Current code**: 75th percentile (demo_flow_guided_mpc.py:388)
- **Rationale**: Balance between sensitivity and noise rejection

#### 2. Minimum Magnitude
- **Recommended**: 1.0 pixels
- **Source**: UnFlow β=0.5 parameter
- **Purpose**: Filter sensor noise and camera jitter

#### 3. Coverage Guardrails
- **Minimum**: 5% (detect static scenes)
- **Maximum**: 95% (detect camera shake)
- **Fallbacks**:
  - coverage < 5% → use 95th percentile
  - coverage > 95% → use 90th percentile

### Academic Foundations

**UnFlow (ICCV 2017)**:
- Formula: `threshold = 0.01 * flow_magnitude + 0.5`
- α=0.01 (magnitude scaling), β=0.5 (baseline)
- Used for occlusion detection

**GMFlow (CVPR 2022)**:
- Adopted UnFlow formula in `gmflow/geometry.py:75-96`
- Explicit citation: "alpha and beta values are following UnFlow"

**Rerender_A_Video (SIGGRAPH Asia 2023)**:
- UnFlow + pixel consistency (25% intensity threshold)

### Implementation Pattern
```python
flow_magnitude = np.linalg.norm(flow_field, axis=-1)

# Step 1: Filter noise
above_min = flow_magnitude > 1.0
threshold = np.percentile(flow_magnitude[above_min], 70.0)

# Step 2: Create mask
motion_mask = flow_magnitude > threshold
coverage = motion_mask.sum() / motion_mask.size

# Step 3: Guardrails
if coverage < 0.05:
    threshold = np.percentile(flow_magnitude.flatten(), 95)
elif coverage > 0.95:
    threshold = np.percentile(flow_magnitude.flatten(), 90)
```

**Action for Task 1**: Use 70th percentile + 1.0px minimum + 5-95% guardrails

