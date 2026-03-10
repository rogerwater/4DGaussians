# Arguments Module - Configuration System

**Generated:** 2026-03-04  
**Scope:** Python class-based configuration for 4DGaussians

## OVERVIEW

Configuration via Python classes (NOT YAML/JSON). Uses inheritance with `_base_` pointers and nested parameter dicts for multi-dimensional (4D space-time) grid configs.

## STRUCTURE

```
arguments/
├── __init__.py                # ParamGroup base + ModelParams/OptimizationParams
├── dnerf/                     # D-NeRF synthetic scenes
│   ├── dnerf_default.py       # Base defaults
│   └── bouncingballs.py       # Scene-specific overrides
├── dynerf/                    # DyNeRF real video datasets
├── hypernerf/                 # HyperNeRF datasets
├── multipleview/              # Multi-view datasets
└── toyarm/                    # ToyArm control datasets
```

## HOW CONFIGS WORK

### Class-Based Inheritance
```python
# arguments/dnerf/bouncingballs.py
ModelParams._base_ = "arguments/dnerf/dnerf_default.py::ModelParams"
ModelParams.model_path = "output/dnerf/bouncingballs"
ModelParams.resolution = 2  # Override parent
```

### Nested Parameters for 4D Grids
```python
ModelHiddenParams.kplanes_config = {
    'resolution': [64, 64, 64, 75],  # [X, Y, Z, Time]
    'grid_dimensions': 2,
    'input_coordinate_dim': 4,        # 4D = space + time
    ...
}
```

### CLI Flag Generation (ParamGroup Magic)
- **Leading underscore** → shorthand flag
  - `self._model_path` → `--model_path` + `-m` short flag
- **Booleans** → `action="store_true"`
- Auto-generates argparse from class attributes

## WHERE TO ADD NEW CONFIGS

1. **New dataset variant:** Create `arguments/<dataset_type>/<scene>.py`
2. **Set base:** `ModelParams._base_ = "arguments/<type>/default.py::ModelParams"`
3. **Override values:** Set class attributes
4. **Pass to train:** `--configs arguments/<type>/<scene>.py`

## CONVENTIONS

### Config Persistence (IMPORTANT)
- `train.py` writes `model_path/cfg_args` as `str(Namespace(...))`
- **Loaded via `eval()`** (security risk, but project convention)
- Resume runs merge saved cfg_args + CLI args
- **DO NOT** create YAML/JSON configs - breaks get_combined_args()

### Flag Behavior Gotchas
- Leading underscore creates shorthand: `_foo` → `--foo` + `-f`
- Booleans use store_true regardless of default value
- Test flag behavior before relying on it

## DEPRECATED PARAMETERS

**DO NOT USE:**
- `timebase_pe` - Legacy time encoder (replaced by control_encoder)
- `timenet_width`, `timenet_output` - Deprecated time encoder params
- Many flags marked `# useless` - kept for compatibility only

## QUICK REFERENCE

| Task | Action |
|------|--------|
| Add scene config | Create `arguments/<type>/<scene>.py` with `_base_` |
| Change hyperparams | Edit class attributes in config file |
| Test config | `python train.py --configs <path> --dry_run` |
| Inspect saved config | `cat output/<exp>/cfg_args` |

**When in doubt:** Check `arguments/__init__.py` for ParamGroup logic and base parameter definitions.
