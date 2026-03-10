# 4DGaussians Agent Instructions

This document provides guidelines for AI coding agents working in the 4DGaussians codebase - a CVPR 2024 research project for 4D Gaussian Splatting for real-time dynamic scene rendering.

## Project Overview

4DGaussians is a **research-grade computer vision project** that extends 3D Gaussian Splatting to dynamic scenes using space-time feature grids (HexPlane/TriPlane architectures). The codebase includes:
- PyTorch-based neural rendering pipeline
- Custom CUDA kernels for Gaussian rasterization (in submodules)
- Model Predictive Control (MPC) for robotic planning
- Support for D-NeRF, HyperNeRF, DyNeRF, and custom multi-view datasets

**Key Technologies**: PyTorch 1.13.1, CUDA 11.6, mmcv, open3d, optical flow (GMFlow), LPIPS perceptual loss

---

## Build & Setup Commands

### Environment Setup
```bash
# Initial setup
git submodule update --init --recursive
conda create -n Gaussians4D python=3.7
conda activate Gaussians4D
pip install -r requirements.txt

# Build CUDA extensions (required before running)
pip install -e submodules/depth-diff-gaussian-rasterization
pip install -e submodules/simple-knn
```

### Training
```bash
# Train on D-NeRF synthetic scenes
python train.py -s data/dnerf/bouncingballs \
  --port 6017 \
  --expname "dnerf/bouncingballs" \
  --configs arguments/dnerf/bouncingballs.py

# Resume from checkpoint
python train.py -s data/dnerf/bouncingballs \
  --configs arguments/dnerf/bouncingballs.py \
  --checkpoint_iterations 200 \
  --start_checkpoint "output/dnerf/bouncingballs/chkpnt_coarse_200.pth"

# Batch training scripts
bash scripts/train_dnerf.sh dnerf        # Train all D-NeRF scenes
bash scripts/train_dynerf.sh dynerf      # Train all DyNeRF scenes
bash scripts/train_hyper_virg.sh hypernerf/virg  # Train HyperNeRF
```

### Rendering & Evaluation
```bash
# Render test views
python render.py --model_path "output/dnerf/bouncingballs/" \
  --skip_train \
  --configs arguments/dnerf/bouncingballs.py

# Compute metrics (PSNR, SSIM, LPIPS)
python metrics.py --model_path "output/dnerf/bouncingballs/"

# Full evaluation pipeline
python full_eval.py --model_path "output/dnerf/bouncingballs/"
```

### Utility Scripts
```bash
# Preprocess DyNeRF videos (extract frames)
python scripts/preprocess_dynerf.py --datadir data/dynerf/cut_roasted_beef

# Generate COLMAP point clouds
bash colmap.sh

# Downsample point clouds
python scripts/downsample_point.py --input_path data/points3D.ply --ratio 0.5

# Export per-frame Gaussians
python export_perframe_3DGS.py --model_path "output/dnerf/lego/"

# Run MPC control demo
python demo_flow_guided_mpc.py --config configs/4dgs_control.yaml
```

### Testing
**Note**: This is a research codebase without formal unit tests. Validation is performed via:
- `metrics.py` - Quantitative evaluation (PSNR, SSIM, LPIPS)
- `render.py` - Visual inspection of rendered images
- Manual verification against paper benchmarks

---

## Code Style Guidelines

### General Principles
- **Research-oriented code**: Prioritize clarity and reproducibility over production-grade abstraction
- **4-space indentation** (no tabs)
- **Copyright headers**: All files include INRIA GRAPHDECO research group headers
- **Extensive inline comments**: Explain architectural choices and paper references

### Import Organization
```python
# Standard library imports (sorted alphabetically)
import os
import sys
import random
from collections import defaultdict

# Third-party imports (sorted)
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

# Local imports (absolute from project root)
from scene import Scene, GaussianModel
from gaussian_renderer import render
from arguments import ModelParams, PipelineParams, OptimizationParams
from utils.loss_utils import l1_loss, ssim, lpips_loss
from utils.general_utils import safe_state
```

**Import Rules**:
- Use **absolute imports** from project root (e.g., `from scene import ...`)
- Group imports: standard library → third-party → local modules
- Avoid wildcard imports (`from module import *`)
- Import only what's needed, but don't over-split (group related imports)

### Naming Conventions
```python
# Classes: PascalCase
class GaussianModel:
class HexPlaneAnalyzer:

# Functions/methods: snake_case
def scene_reconstruction(dataset, opt, hyper):
def build_frame_pair_index(cameras):

# Variables: snake_case
model_params = ...
deformation_table = ...

# Constants: SCREAMING_SNAKE_CASE
TENSORBOARD_FOUND = True
MAX_ITERATIONS = 30000

# Private attributes: leading underscore
self._xyz = torch.empty(0)
self._features_dc = torch.empty(0)

# Lambda functions: lowercase with underscores
to8b = lambda x : (255*np.clip(x.cpu().numpy(),0,1)).astype(np.uint8)
```

### Type Hints
**Not strictly enforced** (research codebase), but use when it improves clarity:
```python
def __init__(self, sh_degree : int, args):
    # Basic type hints on critical parameters

def extract(self, args) -> GroupParams:
    # Return type hints on public APIs
```

### Configuration Management
Use **class-based parameter groups** (not YAML/JSON):
```python
# arguments/__init__.py
class ModelParams(ParamGroup):
    def __init__(self, parser, sentinel=False):
        self.sh_degree = 3
        self._source_path = ""
        self.data_device = "cuda"
        super().__init__(parser, "Loading Parameters", sentinel)

# Dataset-specific configs inherit from base configs
# arguments/dnerf/bouncingballs.py extends arguments/dnerf/dnerf_default.py
```

### Error Handling
```python
# Graceful degradation for optional features
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

# Guard against invalid states
if valid_mask.sum() == 0:
    return torch.tensor(0.0, device=pred_depth.device)

# Assertions for developer errors (not user input validation)
assert len(cameras) > 0, "No cameras found in scene"
```

### Tensor Operations
```python
# Always specify device explicitly
device = torch.device("cuda")
self._xyz = self._xyz.to(device)

# Use in-place operations when safe
self.optimizer.zero_grad(set_to_none=True)  # More efficient than zero_grad()

# Detach gradients when not needed
with torch.no_grad():
    rendered_image = render(viewpoint_camera, gaussians)
```

### Factory Pattern for Architecture Selection
```python
# scene/deformation_factory.py
def create_deform_network(args):
    """Factory pattern for selecting deformation architecture"""
    if args.use_triplane:
        return deform_network_triplane(args)
    else:
        return deform_network(args)  # HexPlane (default)
```

### Logging & Progress
```python
# Use print() for important milestones
print(f"[GMFlow] loading checkpoint from {gmflow_cfg.model}")

# Use tqdm for training loops
for iteration in tqdm(range(first_iter, opt.iterations), desc="Training"):
    # training code

# TensorBoard for metrics (if available)
if TENSORBOARD_FOUND and tb_writer:
    tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
```

---

## File Organization Patterns

```
scene/              # Core data structures
├── gaussian_model.py         # 3D Gaussian primitives
├── deformation.py            # HexPlane deformation network
├── deformation_triplane.py   # TriPlane deformation network
├── deformation_factory.py    # Architecture selection factory

utils/              # Shared utilities (19 files)
├── loss_utils.py            # L1, SSIM, LPIPS losses
├── flow_utils.py            # Optical flow processing
├── general_utils.py         # Math utilities (inverse_sigmoid, etc.)

arguments/          # Python-based config files
├── __init__.py              # Base parameter classes
├── dnerf/                   # D-NeRF dataset configs (with inheritance)
├── hypernerf/               # HyperNeRF configs

mpc/                # Model Predictive Control module
scripts/            # Helper scripts (preprocessing, batch training)
```

---

## Common Patterns

### State Capture & Restoration (Checkpointing)
```python
def capture(self):
    """Capture model state for checkpointing"""
    return (self._xyz, self._features_dc, self.optimizer.state_dict(), ...)

def restore(self, model_args, training_args):
    """Restore from checkpoint"""
    (self._xyz, self._features_dc, opt_dict, ...) = model_args
    self.optimizer.load_state_dict(opt_dict)
```

### Configuration Inheritance
```python
# arguments/dnerf/bouncingballs.py
ModelParams._base_ = arguments/dnerf/dnerf_default.py::ModelParams
ModelParams.model_path = "output/dnerf/bouncingballs"
ModelParams.resolution = 2  # Override parent value
```

---

## Important Notes for Agents

1. **No formal linting/formatting**: Code style is consistent but not enforced by tools. Follow existing patterns.

2. **Submodules are CUDA extensions**: Do not modify `submodules/` (diff-gaussian-rasterization, simple-knn) - they are external dependencies.

3. **Configuration via Python**: Do not create YAML/JSON configs - use Python class-based configs in `arguments/`.

4. **Outputs are gitignored**: Training outputs go to `outputs/` (not tracked). Use `--model_path` to specify output location.

5. **Two-stage training**: Training has "coarse" and "fine" stages with separate checkpoints (`chkpnt_coarse_*.pth`, `chkpnt_fine_*.pth`).

6. **Architecture variants**: The codebase supports both HexPlane (6-plane space-time) and TriPlane (3-plane spatial) architectures. Check `args.use_triplane` to determine which is active.

7. **Research focus**: Prioritize correctness and reproducibility over optimization. Avoid refactoring unless explicitly requested.

---

## Quick Reference

| Task | Command |
|------|---------|
| Train single scene | `python train.py -s data/dnerf/lego --configs arguments/dnerf/lego.py` |
| Render test views | `python render.py --model_path output/dnerf/lego/ --skip_train` |
| Compute metrics | `python metrics.py --model_path output/dnerf/lego/` |
| Export Gaussians | `python export_perframe_3DGS.py --model_path output/dnerf/lego/` |
| Check model size | `python scripts/cal_modelsize.py --path output/dnerf/lego/` |

**When in doubt**: Check existing code in `train.py`, `scene/gaussian_model.py`, or reference configs in `arguments/dnerf/`.
