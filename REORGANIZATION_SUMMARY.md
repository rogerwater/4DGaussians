# Test Folder Reorganization Summary

**Date**: 2026-03-18  
**Task**: Reorganize 4DGaussians repository to separate test/demo/experimental scripts from production code

## What Was Done

### 1. Created Test Folder Structure

```
test/
├── unit/                    # Unit tests (4 files)
├── integration/             # Integration tests (4 files)
├── demos/                   # Demo scripts (4 files)
├── scripts/                 # Experimental utilities (10 files)
├── notebooks/               # Jupyter notebooks (2 files)
└── README.md                # Testing guidelines
```

### 2. Files Moved (24 total)

#### From Root Directory → test/
- **Unit tests** (4 files):
  - `test_joint_angle_projection.py` → `test/unit/`
  - `test_mpc_integration.py` → `test/unit/`
  - `test_unit_circle_penalty.py` → `test/unit/`
  - `test_biflow_functions.py` → `test/unit/`

- **Integration tests** (4 files):
  - `test_cotracker_mpc.py` → `test/integration/`
  - `test_confidence_integration.py` → `test/integration/`
  - `test_confidence_tracking.py` → `test/integration/`
  - `test_point_tracker.py` → `test/integration/`

- **Demo scripts** (4 files):
  - `demo_cotracker_mpc.py` → `test/demos/`
  - `demo_flow_guided_mpc.py` → `test/demos/`
  - `visualize_flow_stages.py` → `test/demos/`
  - `run_render_based_test.sh` → `test/demos/`

- **Notebooks** (2 files):
  - `4DGaussians.ipynb` → `test/notebooks/`
  - `weight_visualization.ipynb` → `test/notebooks/`

#### From scripts/ → test/scripts/
- **Test utilities** (3 files):
  - `test_deformation_sensitivity.py`
  - `test_lpips_gradient.py`
  - `test_gradient_flow.py`

- **Experimental/analysis scripts** (7 files):
  - `inverse_control_recovery.py`
  - `run_control.py`
  - `summarize_inverse_results.py`
  - `synthesize_mpc_videos.py`
  - `README_synthesize_mpc_videos.md`
  - `select_image.py`
  - `grow_point.py`

### 3. Files Kept in Production Locations

#### Root Directory (Production Pipeline)
- `train.py` - Main training entry point
- `render.py` - Rendering entry point
- `metrics.py` - Evaluation metrics
- `full_eval.py` - Full evaluation pipeline
- `export_perframe_3DGS.py` - Utility for exporting Gaussians
- `merge_many_4dgs.py` - Utility for merging models
- `convert.py` - Data conversion utility
- `database.py` - Database utility

#### scripts/ (Data Preprocessing & Batch Training)
**Data preprocessing utilities** (kept as production tools):
- `preprocess_dynerf.py` - DyNeRF preprocessing (referenced in README)
- `downsample_point.py` - Point cloud downsampling (referenced in README)
- `colmap_converter.py` - COLMAP format conversion
- `extractimages.py` - Frame extraction
- `blender2colmap.py` - Blender to COLMAP conversion
- `llff2colmap.py` - LLFF to COLMAP conversion
- `hypernerf2colmap.py` - HyperNeRF to COLMAP conversion
- `merge_point.py` - Point cloud merging
- `train_test_split.py` - Dataset splitting

**Batch training scripts** (kept as production automation):
- `train_dnerf.sh` - Batch training for D-NeRF
- `train_dynerf.sh` - Batch training for DyNeRF
- `train_hyper_virg.sh` - Batch training for HyperNeRF
- `train_hyper_interp.sh` - Batch training for HyperNeRF interp
- `train_dynamic3dgs.sh` - Batch training for Dynamic 3DGS
- `train_dycheck.sh` - Batch training for DyCheck
- `train_with_flow.sh` - Training with optical flow
- `process_dnerf.sh` - Full D-NeRF pipeline

**Evaluation utilities** (kept as production tools):
- `read_all_metrics.py` - Aggregate metrics across scenes
- `cal_modelsize.py` - Model size calculation (referenced in README)

### 4. Documentation Updates

#### Created: `test/README.md`
Comprehensive testing guidelines including:
- Directory structure explanation
- How to run unit/integration/demo tests
- Test philosophy for research code
- Contributing guidelines for new tests
- Quick reference table

#### Updated: `AGENTS.md`
Added two new sections:

1. **Enhanced Testing section** (line 84-97):
   - Lists all test subdirectories
   - Explains purpose of each subdirectory
   - References `test/README.md` for details

2. **New rule #8 in "Important Notes for Agents"** (line 312-319):
   - Explicit test file placement rules
   - Lists target directories for each type of test
   - **Strict rule**: Never create test files in root or scripts/

## Benefits

### 1. Clean Repository Structure
- ✅ Root directory now contains only production scripts (8 Python files)
- ✅ `scripts/` folder contains only data-prep and batch-training tools
- ✅ All test/demo/experimental code isolated in `test/`

### 2. Clear Guidelines for Future Development
- ✅ AGENTS.md explicitly documents where to place test files
- ✅ `test/README.md` provides detailed testing conventions
- ✅ Directory structure mirrors best practices from top research repos (Nerfstudio, PyTorch)

### 3. Improved Maintainability
- ✅ Easy to identify test vs production code
- ✅ Test files organized by purpose (unit/integration/demos/experimental)
- ✅ Notebooks and shell scripts properly categorized

### 4. No Breaking Changes
- ✅ All production scripts remain in original locations
- ✅ README references to scripts/ utilities still valid
- ✅ Training workflows unchanged
- ✅ Only test/demo/experimental code was moved

## Verification Checklist

- [x] All test files moved to `test/` directory
- [x] Root directory clean of test files (only 8 production .py files)
- [x] `scripts/` directory clean of test files
- [x] `test/README.md` created with comprehensive guidelines
- [x] `AGENTS.md` updated with test placement rules
- [x] No production scripts were moved
- [x] All data preprocessing utilities remain in `scripts/`
- [x] All batch training scripts remain in `scripts/`

## Impact on Existing Workflows

### No Impact:
- Training workflows (`python train.py ...`)
- Rendering workflows (`python render.py ...`)
- Metrics evaluation (`python metrics.py ...`)
- Data preprocessing (`python scripts/preprocess_dynerf.py ...`)
- Batch training (`bash scripts/train_dnerf.sh ...`)

### Updated Paths:
- Running tests: `python test/unit/test_biflow_functions.py` (was `python test_biflow_functions.py`)
- Running demos: `python test/demos/demo_flow_guided_mpc.py` (was `python demo_flow_guided_mpc.py`)
- Opening notebooks: `jupyter notebook test/notebooks/4DGaussians.ipynb` (was `jupyter notebook 4DGaussians.ipynb`)

## Recommendations for Future Agents

When creating new test files, always:
1. Place in appropriate `test/` subdirectory
2. Follow naming conventions (`test_*.py` for tests, `demo_*.py` for demos)
3. Update `test/README.md` if adding new test categories
4. Never create test files in root directory or `scripts/`

## Research Repo Best Practices Applied

Based on analysis of successful research repositories (Nerfstudio, PyTorch Vision, SpacetimeGaussians):

✅ **Separate test directory** - Matches Nerfstudio pattern  
✅ **Organized by test type** - unit/integration/demos mirrors production repos  
✅ **Documented testing philosophy** - Research code validation approach  
✅ **Keep production simple** - Root directory has only core scripts  
✅ **Clear contributing guidelines** - AGENTS.md + test/README.md  

## Files Summary

| Category | Count | Location |
|----------|-------|----------|
| Unit tests | 4 | `test/unit/` |
| Integration tests | 4 | `test/integration/` |
| Demo scripts | 4 | `test/demos/` |
| Experimental utilities | 10 | `test/scripts/` |
| Jupyter notebooks | 2 | `test/notebooks/` |
| **Total moved** | **24** | `test/` |
| Production scripts (root) | 8 | Root directory |
| Production scripts (data prep) | 9 | `scripts/` |
| Production scripts (batch training) | 8 | `scripts/` |
| Production scripts (evaluation) | 2 | `scripts/` |

---

**Status**: ✅ Complete - Repository successfully reorganized with comprehensive documentation
