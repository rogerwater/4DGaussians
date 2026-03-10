# Scene Module - Core Data Structures

**Generated:** 2026-03-04  
**Scope:** Gaussian models, deformation networks, dataset loaders

## OVERVIEW

Core rendering primitives: 3D Gaussians, space-time deformation, camera/dataset wrappers. Two architecture variants (HexPlane/TriPlane) selected via factory.

## STRUCTURE

```
scene/
├── gaussian_model.py         # GaussianModel - 3D Gaussian primitives
├── deformation.py            # HexPlane deformation (6-plane space-time)
├── deformation_triplane.py   # TriPlane deformation (3-plane spatial)
├── deformation_factory.py    # Architecture selection factory
├── dataset_readers.py        # Auto-detect dataset type (Colmap/Blender/etc)
├── neural_3D_dataset_NDC.py  # DyNeRF/NeRF video dataset loader
├── toyarm_dataset.py         # ToyArm control dataset
├── multipleview_dataset.py   # Multi-view dataset
├── hyper_loader.py           # HyperNeRF dataset loader
└── camera.py                 # Camera primitives & transforms
```

## KEY ABSTRACTIONS

### GaussianModel (gaussian_model.py)
**Core model state:** xyz, features, scaling, rotation, opacity  
**Key methods:**
- `training_setup()` - Initialize optimizers
- `capture()` / `restore()` - Checkpoint save/load
- `save_ply()` - Export point cloud
- `save_deformation()` - Save deformation network
- Densification/pruning logic

**Device:** Always explicitly set to CUDA

### Deformation Networks
**Two variants:**
- **HexPlane** (deformation.py) - 6-plane space-time (default)
- **TriPlane** (deformation_triplane.py) - 3-plane spatial + FiLM

**Selection via factory:**
```python
# deformation_factory.py
if args.use_triplane:
    return deform_network_triplane(args)
else:
    return deform_network(args)  # HexPlane
```

### Dataset Auto-Detection (dataset_readers.py)
Detects type by files present:
- `sparse/` → Colmap
- `transforms_train.json` → Blender
- `poses_bounds.npy` → DyNeRF/LLFF
- `transforms.json` → ToyArm
- `points3D_multipleview.ply` → MultipleView

## CHECKPOINTING

### Save Locations
```
model_path/
├── chkpnt_coarse_<iter>.pth  # Coarse stage checkpoint
├── chkpnt_fine_<iter>.pth    # Fine stage checkpoint
└── point_cloud/iteration_<n>/
    ├── point_cloud.ply
    ├── deformation.pth
    ├── deformation_table.pth
    └── deformation_accum.pth
```

### Capture Format
```python
(
    self._xyz,
    self._features_dc,
    self._features_rest,
    self._scaling,
    self._rotation,
    self._opacity,
    self.max_radii2D,
    self.xyz_gradient_accum,
    self.denom,
    self.optimizer.state_dict(),
    self.spatial_lr_scale,
    (deformation model dict),
    (deformation_accum)
)
```

## CONVENTIONS

### Import Pattern
```python
from scene import Scene, GaussianModel
from scene.deformation_factory import create_deform_network
```

### Device Handling
```python
device = torch.device("cuda")
self._xyz = self._xyz.to(device)  # Always explicit
```

### Two-Stage Training
- **Coarse** → initialize structure
- **Fine** → refine details
- Checkpoints named by stage

## NUMERICAL WARNINGS

**FIXME items in camera.py:**
- `R_to_q()` - Quaternion conversion may be unstable at theta==pi
- `lnR` calculation - "wei-chiu finds it weird"
- **Action:** Validate numerically before using for critical code

## WHERE TO LOOK

| Task | File |
|------|------|
| Modify Gaussian params | gaussian_model.py |
| Add dataset type | dataset_readers.py + new loader |
| Change deformation | deformation*.py + factory |
| Fix checkpoint format | gaussian_model.py capture/restore |
| Debug camera math | camera.py (but note FIXMEs) |

## ADDING NEW ARCHITECTURE VARIANT

1. Create `scene/deformation_<variant>.py`
2. Implement interface matching deformation.py
3. Add selection logic to `deformation_factory.py`
4. Add flag to `ModelParams` for variant selection

**When in doubt:** Check existing deformation implementations and factory pattern.
