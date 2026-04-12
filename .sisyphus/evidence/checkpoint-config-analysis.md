# Checkpoint Configuration Analysis

**Generated:** 2026-03-31  
**Checkpoint Path:** `/home/ubuntu/yyf/4DGaussians/outputs/dm_control_push8/point_cloud/iteration_20000/deformation.pth`

---

## Executive Summary

✅ **Checkpoint successfully loaded and analyzed**  
✅ **Configuration reverse-engineered from tensor shapes**  
❌ **Critical mismatch identified: `getattr()` defaults in `deformation_triplane.py` don't match checkpoint**

---

## Checkpoint Structure

### Top-Level Keys
```
pos_poc                       → [10]
rotation_scaling_poc          → [2]
opacity_poc                   → [2]
deformation_net.*             → 66 parameters (all network weights)
```

### ActionProcessor Parameters (11 total)

| Parameter Name | Shape | Interpretation |
|----------------|-------|----------------|
| `action_processor.mlp.0.weight` | `[128, 15]` | **input_dim = 15, hidden_dim = 128** |
| `action_processor.mlp.0.bias` | `[128]` | Hidden layer bias |
| `action_processor.mlp.1.weight` | `[128]` | LayerNorm weight |
| `action_processor.mlp.1.bias` | `[128]` | LayerNorm bias |
| `action_processor.mlp.3.weight` | `[128, 128]` | Hidden → Hidden |
| `action_processor.mlp.3.bias` | `[128]` | Hidden layer bias |
| `action_processor.mlp.4.weight` | `[128]` | LayerNorm weight |
| `action_processor.mlp.4.bias` | `[128]` | LayerNorm bias |
| `action_processor.mlp.6.weight` | `[32, 128]` | **Hidden → output_dim = 32** |
| `action_processor.mlp.6.bias` | `[32]` | Output bias |
| **MISSING KEY** | `freq_bands` | ❌ **PE NOT used** |

---

## Configuration Reverse Engineering

### Checkpoint Training Config (Inferred)

```python
# Arguments used during training (reverse-engineered from shapes)
ModelHiddenParams = {
    'action_input_dim': 15,         # ✅ From mlp.0.weight: [128, 15] → 15
    'action_use_pe': False,         # ✅ No freq_bands key in checkpoint
    'action_num_frequencies': 4,    # ⚠️  Unused (PE disabled)
    'action_hidden_dim': 128,       # ✅ From mlp.0.weight: [128, 15] → 128
    'action_output_dim': 32,        # ✅ From mlp.6.weight: [32, 128] → 32
}
```

### FiLM Generator Dependency Verification

All 3 FiLM blocks confirm `action_output_dim = 32`:

| FiLM Block | film_generator.0.weight Shape | Expected Input Dim | Status |
|------------|------------------------------|-------------------|--------|
| Block 0 | `[128, 32]` | 32 | ✅ Match |
| Block 1 | `[128, 32]` | 32 | ✅ Match |
| Block 2 | `[128, 32]` | 32 | ✅ Match |

---

## Current Code Configuration

### Config File: `arguments/toyarm/triplane.py`

```python
# Lines 32-36
action_input_dim = 15,         # ✅ MATCHES checkpoint
action_use_pe = False,         # ✅ MATCHES checkpoint
action_num_frequencies = 4,    # ✅ Unused but safe
action_hidden_dim = 128,       # ✅ MATCHES checkpoint
action_output_dim = 32,        # ✅ MATCHES checkpoint
```

**Status:** ✅ Config file is **CORRECT** and matches checkpoint!

### Code: `scene/deformation_triplane.py` (Lines 187-192)

```python
# ❌ PROBLEM: getattr() defaults DON'T match checkpoint
action_use_pe = getattr(args, 'action_use_pe', True)         # Default: True  (should be False)
action_num_freq = getattr(args, 'action_num_frequencies', 4) # Default: 4     (OK, unused)
action_input_dim = getattr(args, 'action_input_dim', 6)      # Default: 6     (should be 15)
action_hidden = getattr(args, 'action_hidden_dim', 128)      # Default: 128   (OK)
action_output_dim = getattr(args, 'action_output_dim', 64)   # Default: 64    (should be 32)
```

**Status:** ❌ Code has **WRONG DEFAULTS** that override the config file!

---

## Root Cause Analysis

### Why Shape Mismatch Occurs

**Scenario 1: Config not loaded**
If `arguments/toyarm/triplane.py` is not passed via `--configs`, the model uses `getattr()` defaults:
- `action_input_dim = 6` (default) instead of `15` (checkpoint)
- `action_output_dim = 64` (default) instead of `32` (checkpoint)
- `action_use_pe = True` (default) instead of `False` (checkpoint)

**Scenario 2: Config loaded but defaults still applied**
If `args` object doesn't have the attribute (due to inheritance issues or missing fields), `getattr()` falls back to the wrong default.

### Expected vs Actual Shapes

| Parameter | Checkpoint Shape | Current Model Shape (with wrong defaults) | Status |
|-----------|------------------|------------------------------------------|--------|
| `mlp.0.weight` | `[128, 15]` | `[128, 6]` or `[128, 42]` (if PE enabled) | ❌ MISMATCH |
| `mlp.6.weight` | `[32, 128]` | `[64, 128]` | ❌ MISMATCH |
| `film_generator.0.weight` | `[128, 32]` | `[128, 64]` | ❌ MISMATCH |

**Note:** When `action_use_pe=True` with `num_frequencies=4`, input dim becomes `6 * (1 + 2*4) = 54`, not 42 (typo above).

---

## Impact Analysis

### Files Affected by Wrong Defaults

1. **scene/deformation_triplane.py:187-192** - ActionProcessor initialization
2. **scene/deformation_triplane.py:330** - FiLM block expects `action_output_dim`
3. **scene/deformation_triplane.py:346** - FiLM fusion logic

### Downstream Effects

- ❌ `load_state_dict()` fails with size mismatch
- ❌ Cannot resume training from checkpoint
- ❌ Cannot use trained model for inference
- ❌ Cannot transfer weights to new experiments

---

## Verification Tests

### Test 1: Checkpoint Keys
```bash
✅ PASS - All keys follow `deformation_net.*` naming convention
✅ PASS - No `control_processor` keys (rename was successful)
✅ PASS - `action_processor` keys present
```

### Test 2: Shape Consistency
```bash
✅ PASS - ActionProcessor mlp layer shapes are consistent
✅ PASS - FiLM generators all expect 32-dim input
✅ PASS - TriPlane grids have expected multi-resolution structure
```

### Test 3: PE Detection
```bash
✅ PASS - No `freq_bands` key in checkpoint
✅ CONFIRMED - Positional encoding was NOT used during training
```

---

## Recommended Solution

### Option A: Fix getattr() Defaults (RECOMMENDED)

**Strategy:** Update `scene/deformation_triplane.py` to use checkpoint-compatible defaults.

**Changes:**
```python
# Line 187-192 (BEFORE)
action_use_pe = getattr(args, 'action_use_pe', True)         # ❌ Wrong
action_input_dim = getattr(args, 'action_input_dim', 6)      # ❌ Wrong
action_output_dim = getattr(args, 'action_output_dim', 64)   # ❌ Wrong

# Line 187-192 (AFTER)
action_use_pe = getattr(args, 'action_use_pe', False)        # ✅ Match checkpoint
action_input_dim = getattr(args, 'action_input_dim', 15)     # ✅ Match checkpoint
action_output_dim = getattr(args, 'action_output_dim', 32)   # ✅ Match checkpoint
```

**Pros:**
- ✅ Minimal code change (3 lines)
- ✅ No config file changes needed
- ✅ Works even if config not explicitly loaded
- ✅ Backward compatible with trained checkpoints

**Cons:**
- ⚠️ Changes default behavior for new projects (but they should use config files anyway)

---

### Option B: Add Config Validation (DEFENSIVE)

**Strategy:** Add explicit validation to ensure config is loaded.

**Changes:**
```python
# After line 186, add:
required_attrs = ['action_input_dim', 'action_use_pe', 'action_output_dim']
missing = [attr for attr in required_attrs if not hasattr(args, attr)]
if missing:
    raise ValueError(f"Missing required config attributes: {missing}. "
                     f"Did you forget to pass --configs arguments/toyarm/triplane.py?")
```

**Pros:**
- ✅ Forces explicit configuration
- ✅ Clear error message for users

**Cons:**
- ❌ Breaks if config not loaded (more strict)
- ❌ Doesn't help if user forgot to pass config

---

### Option C: Update Base Config Defaults (NOT RECOMMENDED)

**Strategy:** Change defaults in `arguments/__init__.py` ModelHiddenParams base class.

**Cons:**
- ❌ Affects ALL datasets, not just ToyArm
- ❌ May break other experiments
- ❌ Violates principle of dataset-specific configs

---

## Recommended Action Plan

1. ✅ **Fix getattr() defaults** (Option A) - Primary solution
2. ✅ **Add inline comment** explaining why these are the defaults
3. ⚠️  **Optional:** Add config validation (Option B) for defense-in-depth
4. ✅ **Test:** Load checkpoint after fix to verify `load_state_dict()` succeeds

---

## TriPlane Grid Configuration (For Reference)

Multi-resolution spatial grids (3 levels × 3 planes):

| Level | Resolution | Shape per Plane | Total Params |
|-------|-----------|----------------|--------------|
| 0 | 128×128 | `[1, 32, 128, 128]` | 3 × 512K = 1.5M |
| 1 | 256×256 | `[1, 32, 256, 256]` | 3 × 2M = 6M |
| 2 | 512×512 | `[1, 32, 512, 512]` | 3 × 8M = 24M |

**Total Grid Params:** ~31.5M parameters

---

## Summary

| Item | Checkpoint | Config File | Code Defaults | Status |
|------|-----------|-------------|---------------|--------|
| `action_input_dim` | 15 | ✅ 15 | ❌ 6 | **FIX NEEDED** |
| `action_use_pe` | False | ✅ False | ❌ True | **FIX NEEDED** |
| `action_hidden_dim` | 128 | ✅ 128 | ✅ 128 | OK |
| `action_output_dim` | 32 | ✅ 32 | ❌ 64 | **FIX NEEDED** |

**Conclusion:** Config file is correct, but code defaults are wrong. Fix 3 lines in `deformation_triplane.py`.
