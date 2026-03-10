# Integration Rules: Point Tracking MPC System

**Purpose**: Guidelines for modifying and extending the point tracking-based MPC reward system.  
**Audience**: Future developers working on 4DGaussians MPC module.  
**Last Updated**: March 6, 2026

---

## Table of Contents
1. [Core Principles](#core-principles)
2. [Modifying PointTracker](#modifying-pointtracker)
3. [Creating New Objectives](#creating-new-objectives)
4. [MPC Integration Patterns](#mpc-integration-patterns)
5. [Performance Considerations](#performance-considerations)
6. [Common Pitfalls](#common-pitfalls)
7. [Debugging Guide](#debugging-guide)

---

## Core Principles

### 1. Maintain Compatibility with Python 3.7 + PyTorch 1.13

**Why**: 4DGaussians is locked to Python 3.7 due to CUDA 11.6 and PyTorch 1.13 dependencies.

**Critical Rules**:

```python
# ❌ WRONG - Modern Python 3.9+ syntax
from typing import Optional
def track(self, points) -> tuple[Tensor, Tensor]:
    ...

# ✅ CORRECT - Python 3.7 compatible
from typing import Optional, Tuple
import torch
def track(self, points):
    # type: (torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]
    ...
```

**Type Hints Compatibility Table**:

| Modern (3.9+) | Legacy (3.7) | Import Required |
|---------------|--------------|-----------------|
| `tuple[int, int]` | `Tuple[int, int]` | `from typing import Tuple` |
| `list[str]` | `List[str]` | `from typing import List` |
| `dict[str, int]` | `Dict[str, int]` | `from typing import Dict` |
| `int \| None` | `Optional[int]` | `from typing import Optional` |

**PyTorch 1.13 Restrictions**:

```python
# ❌ NOT AVAILABLE in PyTorch 1.13
torch.load(path, weights_only=True)  # Added in 1.15
nn.LayerNorm(dim, bias=False)  # bias parameter added in 1.15

# ✅ WORKAROUNDS
torch.load(path, map_location=device)  # Use map_location instead
nn.LayerNorm(dim)  # Omit bias parameter (defaults to True)
```

---

### 2. Respect TAPIR's Input Requirements

**Critical**: TAPIR expects uint8 [0, 255], NOT float [0, 1].

```python
# ❌ WRONG - Will degrade tracking accuracy by 100x
video_float = load_images()  # (T, C, H, W) float [0, 1]
tracks = tracker.track(video_float, points)  # ERROR: Poor results

# ✅ CORRECT - Convert to uint8 first
video_float = load_images()  # (T, C, H, W) float [0, 1]
video_uint8 = (video_float * 255).clamp(0, 255).to(torch.uint8)
tracks = tracker.track(video_uint8, points)  # GOOD: Accurate tracking
```

**Why**: TAPIR's normalization layer (in `tapir_model.py`) assumes [0, 255] input:
```python
# Inside TAPIR model
normalized = (input / 255.0 - mean) / std  # Expects uint8 input
```

**Verification**: Check test results in `test_point_tracker.py`:
- Correct (uint8): 0.16 pixel mean error
- Incorrect (float): 15.3 pixel mean error (100x worse!)

---

### 3. Maintain MPC Interface Contracts

All objectives MUST return `(B, 1, 1)` shaped rewards:

```python
class CustomObjective(Objective):
    def compute_reward(self, rendered_images):
        # type: (torch.Tensor) -> torch.Tensor
        """
        Args:
            rendered_images: (B, T, C, H, W) rendered trajectories
            
        Returns:
            reward: (B, 1, 1) scalar reward per sample
                   MUST be this shape for MPC optimizer
        """
        # Your reward computation
        raw_reward = ...  # Shape: (B,)
        
        # CRITICAL: Reshape to (B, 1, 1)
        return raw_reward.view(B, 1, 1)
```

**Why This Shape**:
- MPC optimizer (CEM) expects consistent shape for broadcasting
- Allows stacking multiple objectives without reshaping
- Compatible with torch operations like `torch.cat([obj1, obj2], dim=1)`

---

## Modifying PointTracker

### Architecture Overview

```python
# mpc/point_tracker.py
class PointTracker:
    def __init__(self, checkpoint_path, device='cuda'):
        # Loads TAPIR model from checkpoint
        
    def track(self, video_tensor, initial_points):
        # Main tracking interface
        # Returns: (tracks, visibles)
```

### Adding New Tracking Features

**Example: Add occlusion confidence scores**

```python
# Step 1: Extend track() return signature
def track(self, video_tensor, initial_points):
    # type: (torch.Tensor, torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    """
    Returns:
        tracks: (B, N, T, 2) or (N, T, 2)
        visibles: (B, N, T) or (N, T) 
        confidences: (B, N, T) or (N, T)  # NEW
    """
    # Existing tracking code...
    tracks, visibles = self._internal_track(video_uint8, initial_points)
    
    # NEW: Extract confidence from TAPIR output
    # (Requires modifying tapir_inference.py to return confidence)
    confidences = self._extract_confidence(tracks)
    
    return tracks, visibles, confidences

# Step 2: Update all callers
# In cotracker_objectives.py:
tracks, visibles, confidences = self.tracker.track(...)
# Use confidences in reward computation
```

**⚠️ Breaking Change Checklist**:
- [ ] Update `cotracker_objectives.py` to handle new return signature
- [ ] Update `demo_cotracker_mpc.py` visualization code
- [ ] Update `test_point_tracker.py` assertions
- [ ] Document the change in this file and DEVELOPMENT_LOG.md

---

### Swapping to Different Tracker

**Example: Replace TAPIR with CoTracker2**

```python
# Step 1: Create new tracker class (keep same interface)
class CoTracker2Wrapper:
    """
    Drop-in replacement for PointTracker using CoTracker2.
    MUST maintain same API: track(video, points) -> (tracks, visibles)
    """
    def __init__(self, checkpoint_path, device='cuda'):
        from cotracker.predictor import CoTrackerPredictor
        self.model = CoTrackerPredictor(checkpoint=checkpoint_path)
        self.model = self.model.to(device)
        self.device = device
    
    def track(self, video_tensor, initial_points):
        # type: (torch.Tensor, torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]
        """
        Interface MUST match PointTracker.track() exactly.
        """
        # Ensure correct input format for CoTracker
        if video_tensor.dim() == 4:
            video_tensor = video_tensor.unsqueeze(0)  # Add batch dim
            
        # CoTracker specific preprocessing
        video_normalized = normalize_cotracker_input(video_tensor)
        
        # Track points
        tracks = self.model(video_normalized, queries=initial_points)
        
        # CoTracker doesn't return visibility - estimate it
        visibles = estimate_visibility_from_tracks(tracks)
        
        # CRITICAL: Return same shape as PointTracker
        return tracks, visibles

# Step 2: Use in objectives (no code change needed!)
# In cotracker_objectives.py __init__:
# tracker = CoTracker2Wrapper(checkpoint, device)  # Works!
# All other code unchanged because interface matches
```

**Key Point**: Maintain exact API contract - input/output shapes and types MUST match.

---

## Creating New Objectives

### Template for Point Tracking Objectives

```python
from mpc.objectives import Objective
import torch
from typing import Tuple

class MyPointTrackingObjective(Objective):
    """
    Template for creating new point-tracking-based objectives.
    """
    def __init__(self, tracker, initial_points, weight=1.0):
        # type: (PointTracker, torch.Tensor, float) -> None
        """
        Args:
            tracker: PointTracker instance
            initial_points: (N, 2) starting positions
            weight: Objective weight (default 1.0)
        """
        super(MyPointTrackingObjective, self).__init__(weight=weight)
        self.tracker = tracker
        self.initial_points = initial_points
        
        # Store any additional parameters
        self.device = initial_points.device
    
    def compute_reward(self, rendered_images):
        # type: (torch.Tensor) -> torch.Tensor
        """
        Compute reward from rendered trajectory.
        
        Args:
            rendered_images: (B, T, C, H, W) rendered frames
            
        Returns:
            reward: (B, 1, 1) scalar reward
        """
        B, T, C, H, W = rendered_images.shape
        
        # Step 1: Track points across trajectory
        tracks, visibles = self.tracker.track(
            rendered_images,  # (B, T, C, H, W)
            self.initial_points  # (N, 2)
        )
        # tracks: (B, N, T, 2)
        # visibles: (B, N, T)
        
        # Step 2: Compute your custom reward
        # Example: Penalize total motion
        motion = torch.diff(tracks, dim=2)  # (B, N, T-1, 2)
        motion_magnitude = torch.norm(motion, dim=-1)  # (B, N, T-1)
        total_motion = motion_magnitude.sum(dim=(1, 2))  # (B,)
        reward = -total_motion  # Negative (less motion = higher reward)
        
        # Step 3: CRITICAL - Reshape to (B, 1, 1)
        return reward.view(B, 1, 1)
```

---

### Example: Velocity Matching Objective

```python
class VelocityMatchingObjective(Objective):
    """
    Reward trajectories where point velocities match desired velocities.
    Useful for tasks like "move gripper smoothly at 10 cm/s".
    """
    def __init__(self, tracker, initial_points, target_velocity, weight=1.0):
        # type: (PointTracker, torch.Tensor, torch.Tensor, float) -> None
        """
        Args:
            target_velocity: (N, 2) desired [vx, vy] in pixels/frame
        """
        super(VelocityMatchingObjective, self).__init__(weight=weight)
        self.tracker = tracker
        self.initial_points = initial_points
        self.target_velocity = target_velocity  # (N, 2)
    
    def compute_reward(self, rendered_images):
        # type: (torch.Tensor) -> torch.Tensor
        B, T, C, H, W = rendered_images.shape
        
        # Track points
        tracks, visibles = self.tracker.track(rendered_images, self.initial_points)
        # tracks: (B, N, T, 2)
        
        # Compute actual velocities
        velocities = torch.diff(tracks, dim=2)  # (B, N, T-1, 2)
        
        # Expand target to match batch/time dimensions
        target_vel = self.target_velocity.unsqueeze(0).unsqueeze(2)  # (1, N, 1, 2)
        target_vel = target_vel.expand(B, -1, T-1, -1)  # (B, N, T-1, 2)
        
        # Compute velocity error
        vel_error = torch.norm(velocities - target_vel, dim=-1)  # (B, N, T-1)
        
        # Average error across points and time
        mean_error = vel_error.mean(dim=(1, 2))  # (B,)
        
        # Reward = negative error
        reward = -mean_error  # (B,)
        
        return reward.view(B, 1, 1)
```

---

### Example: Formation Maintenance Objective

```python
class FormationObjective(Objective):
    """
    Maintain relative distances between tracked points.
    Useful for multi-object manipulation (e.g., keep two fingers at fixed width).
    """
    def __init__(self, tracker, initial_points, target_distances, weight=1.0):
        # type: (PointTracker, torch.Tensor, torch.Tensor, float) -> None
        """
        Args:
            target_distances: (M,) desired pairwise distances
                             where M = N*(N-1)/2 for N points
        """
        super(FormationObjective, self).__init__(weight=weight)
        self.tracker = tracker
        self.initial_points = initial_points  # (N, 2)
        self.target_distances = target_distances  # (M,)
        
        # Precompute pairwise indices
        N = initial_points.shape[0]
        self.pair_indices = []
        for i in range(N):
            for j in range(i+1, N):
                self.pair_indices.append((i, j))
    
    def compute_reward(self, rendered_images):
        # type: (torch.Tensor) -> torch.Tensor
        B, T, C, H, W = rendered_images.shape
        
        # Track points
        tracks, visibles = self.tracker.track(rendered_images, self.initial_points)
        # tracks: (B, N, T, 2)
        
        # Compute pairwise distances at each timestep
        pairwise_dists = []
        for i, j in self.pair_indices:
            diff = tracks[:, i, :, :] - tracks[:, j, :, :]  # (B, T, 2)
            dist = torch.norm(diff, dim=-1)  # (B, T)
            pairwise_dists.append(dist)
        pairwise_dists = torch.stack(pairwise_dists, dim=1)  # (B, M, T)
        
        # Compare to target distances
        target = self.target_distances.view(1, -1, 1)  # (1, M, 1)
        target = target.expand(B, -1, T)  # (B, M, T)
        
        # Formation error
        errors = torch.abs(pairwise_dists - target)  # (B, M, T)
        mean_error = errors.mean(dim=(1, 2))  # (B,)
        
        # Reward = negative error
        reward = -mean_error
        return reward.view(B, 1, 1)
```

---

## MPC Integration Patterns

### Pattern 1: Single Objective

```python
# Simplest case - only point tracking
tracker = PointTracker(checkpoint_path="submodules/tapir_pytorch/causal_bootstapir_checkpoint.pt")
objective = PointTrackingObjective(
    target_points=target_points,
    initial_points=initial_points,
    tracker=tracker,
    weight=1.0
)

controller = FlowGuidedGaussianModel(
    gaussians=gaussians,
    objectives=[objective],  # Single objective
    horizon=10,
    num_samples=32
)
```

---

### Pattern 2: Multi-Objective (Tracking + Flow)

```python
# Combine point tracking with flow objectives
from mpc.flow_objectives import FlowConsistencyObjective

tracker = PointTracker(checkpoint_path)

# Objective 1: Reach target points
tracking_obj = PointTrackingObjective(
    target_points=target_points,
    initial_points=initial_points,
    tracker=tracker,
    weight=0.7  # 70% weight
)

# Objective 2: Match optical flow
flow_obj = FlowConsistencyObjective(
    target_flow=target_flow,
    weight=0.3  # 30% weight
)

controller = FlowGuidedGaussianModel(
    gaussians=gaussians,
    objectives=[tracking_obj, flow_obj],  # Multi-objective
    horizon=10,
    num_samples=32
)

# MPC automatically combines: total_reward = 0.7*track_reward + 0.3*flow_reward
```

**Weight Tuning Guidelines**:
- Start with equal weights (1.0 each) and adjust based on performance
- If tracking dominates, reduce its weight (e.g., 0.5)
- If flow matters more, increase its weight (e.g., 1.5)
- Use validation tasks to tune weights systematically

---

### Pattern 3: Adaptive Objective Switching

```python
# Different objectives at different planning stages
class AdaptiveMPCController:
    def __init__(self, gaussians, tracker, initial_points, target_points):
        self.gaussians = gaussians
        self.tracker = tracker
        self.initial_points = initial_points
        self.target_points = target_points
        self.current_step = 0
        
    def plan(self, current_state):
        # Phase 1 (steps 0-10): Coarse reaching with few points
        if self.current_step < 10:
            objective = PointTrackingObjective(
                target_points=self.target_points[::4],  # Subsample (every 4th point)
                initial_points=self.initial_points[::4],
                tracker=self.tracker,
                weight=1.0
            )
            num_samples = 64  # More exploration
            
        # Phase 2 (steps 10-20): Fine alignment with all points
        else:
            objective = PointTrackingObjective(
                target_points=self.target_points,  # All points
                initial_points=self.initial_points,
                tracker=self.tracker,
                weight=1.0
            )
            num_samples = 32  # Less exploration
        
        controller = FlowGuidedGaussianModel(
            gaussians=self.gaussians,
            objectives=[objective],
            horizon=10,
            num_samples=num_samples
        )
        
        action = controller.step(current_state)
        self.current_step += 1
        return action
```

---

## Performance Considerations

### 1. Tracking Frequency

**Problem**: Tracking on every MPC sample is expensive.

**Measurement**:
```python
# Typical MPC iteration costs:
# - 32 samples × 10 horizon = 320 render calls
# - Each render: ~2ms (4DGaussians)
# - Each track call: ~5ms (TAPIR)
# Total without tracking: 320 × 2ms = 640ms/iter
# Total with tracking: 320 × (2+5)ms = 2240ms/iter
# → 3.5x slowdown!
```

**Optimization Strategies**:

```python
# Strategy 1: Reduce point count
initial_points = sample_grid_points(image, N=64)  # Instead of 256
# → 4x speedup

# Strategy 2: Track only elite samples (after first CEM iteration)
class SelectiveTrackingObjective(Objective):
    def __init__(self, tracker, initial_points, target_points, elite_threshold=0.1):
        super(SelectiveTrackingObjective, self).__init__()
        self.tracker = tracker
        self.initial_points = initial_points
        self.target_points = target_points
        self.elite_threshold = elite_threshold
        self.iteration = 0
    
    def compute_reward(self, rendered_images):
        B = rendered_images.shape[0]
        
        # First iteration: Use cheap heuristic (e.g., flow magnitude)
        if self.iteration == 0:
            reward = self._cheap_heuristic(rendered_images)
        # Later iterations: Track only top 10% samples
        else:
            reward = self._selective_tracking(rendered_images)
        
        self.iteration += 1
        return reward.view(B, 1, 1)

# Strategy 3: Reduce horizon
controller = FlowGuidedGaussianModel(
    objectives=[objective],
    horizon=5,  # Instead of 10 → 2x speedup
    num_samples=32
)
```

**Benchmarking Template**:
```python
import time

def benchmark_objective(objective, rendered_images, num_trials=10):
    times = []
    for _ in range(num_trials):
        start = time.time()
        reward = objective.compute_reward(rendered_images)
        torch.cuda.synchronize()  # Wait for GPU
        elapsed = time.time() - start
        times.append(elapsed)
    
    print(f"Mean: {np.mean(times)*1000:.2f}ms")
    print(f"Std: {np.std(times)*1000:.2f}ms")
    return times

# Usage
tracker = PointTracker(checkpoint_path)
objective = PointTrackingObjective(tracker, initial_points, target_points)
rendered = torch.randn(32, 10, 3, 256, 256).cuda()  # Fake data
benchmark_objective(objective, rendered)
```

---

### 2. Memory Management

**Problem**: Tracking large batches can cause OOM errors.

```python
# BAD: Processes entire batch at once
def compute_reward(self, rendered_images):
    # (32, 10, 3, 256, 256) × 256 points = potential OOM
    tracks, _ = self.tracker.track(rendered_images, self.initial_points)
    ...

# GOOD: Chunk processing
def compute_reward(self, rendered_images):
    B = rendered_images.shape[0]
    chunk_size = 8  # Process 8 samples at a time
    
    all_rewards = []
    for i in range(0, B, chunk_size):
        chunk = rendered_images[i:i+chunk_size]
        tracks, _ = self.tracker.track(chunk, self.initial_points)
        reward_chunk = self._compute_reward_from_tracks(tracks)
        all_rewards.append(reward_chunk)
    
    reward = torch.cat(all_rewards, dim=0)
    return reward.view(B, 1, 1)
```

**Memory Profiling**:
```python
import torch
torch.cuda.reset_peak_memory_stats()

# Your code here
reward = objective.compute_reward(rendered_images)

peak_mem = torch.cuda.max_memory_allocated() / 1024**3  # GB
print(f"Peak memory: {peak_mem:.2f} GB")
```

---

### 3. Caching Strategies

**Problem**: Repeated tracking of same initial state is wasteful.

```python
class CachedPointTrackingObjective(Objective):
    """
    Cache tracking results for repeated states.
    Useful when MPC reuses similar initial conditions.
    """
    def __init__(self, tracker, initial_points, target_points, cache_size=100):
        super(CachedPointTrackingObjective, self).__init__()
        self.tracker = tracker
        self.initial_points = initial_points
        self.target_points = target_points
        
        # Simple hash-based cache
        self.cache = {}  # {image_hash: (tracks, visibles)}
        self.cache_size = cache_size
    
    def _hash_image(self, image):
        # type: (torch.Tensor) -> int
        """Simple hash for image (use first frame only)"""
        first_frame = image[0, 0]  # (C, H, W)
        # Downsample and hash
        downsampled = F.avg_pool2d(first_frame.unsqueeze(0), 8)
        return hash(downsampled.cpu().numpy().tobytes())
    
    def compute_reward(self, rendered_images):
        B, T, C, H, W = rendered_images.shape
        
        rewards = []
        for b in range(B):
            vid = rendered_images[b]  # (T, C, H, W)
            vid_hash = self._hash_image(vid)
            
            # Check cache
            if vid_hash in self.cache:
                tracks, visibles = self.cache[vid_hash]
            else:
                # Cache miss - compute tracking
                tracks, visibles = self.tracker.track(vid, self.initial_points)
                
                # Add to cache (evict oldest if full)
                if len(self.cache) >= self.cache_size:
                    oldest_key = next(iter(self.cache))
                    del self.cache[oldest_key]
                self.cache[vid_hash] = (tracks, visibles)
            
            # Compute reward
            diff = tracks - self.target_points.unsqueeze(1)  # (N, T, 2)
            dist = torch.norm(diff, dim=-1).mean()  # Scalar
            rewards.append(-dist)
        
        reward = torch.stack(rewards)  # (B,)
        return reward.view(B, 1, 1)
```

---

## Common Pitfalls

### Pitfall 1: Forgetting Shape Constraints

```python
# ❌ WRONG - Returns wrong shape
def compute_reward(self, rendered_images):
    tracks, _ = self.tracker.track(rendered_images, self.initial_points)
    diff = tracks - self.target_points
    reward = -torch.norm(diff, dim=-1).mean()
    return reward  # Shape: () - WRONG!

# ✅ CORRECT
def compute_reward(self, rendered_images):
    B = rendered_images.shape[0]
    tracks, _ = self.tracker.track(rendered_images, self.initial_points)
    diff = tracks - self.target_points.unsqueeze(0).unsqueeze(2)
    distances = torch.norm(diff, dim=-1)  # (B, N, T)
    reward = -distances.mean(dim=(1, 2))  # (B,)
    return reward.view(B, 1, 1)  # (B, 1, 1) - CORRECT!
```

---

### Pitfall 2: Ignoring Batch vs Single Input

```python
# ❌ WRONG - Assumes always batched
def track(self, video_tensor, initial_points):
    B, T, C, H, W = video_tensor.shape  # Fails if input is (T, C, H, W)!
    ...

# ✅ CORRECT - Handle both cases
def track(self, video_tensor, initial_points):
    # Check if batched
    if video_tensor.dim() == 4:  # (T, C, H, W)
        video_tensor = video_tensor.unsqueeze(0)  # Add batch dim
        unbatch_output = True
    else:  # (B, T, C, H, W)
        unbatch_output = False
    
    B, T, C, H, W = video_tensor.shape
    
    # ... tracking code ...
    
    if unbatch_output:
        tracks = tracks.squeeze(0)  # Remove batch dim
        visibles = visibles.squeeze(0)
    
    return tracks, visibles
```

---

### Pitfall 3: Float vs Uint8 Confusion

```python
# ❌ WRONG - Passing float to TAPIR
rendered = gaussians.render(camera)  # Returns float [0, 1]
tracks = tracker.track(rendered, points)  # Poor accuracy!

# ✅ CORRECT - Convert to uint8 first
rendered = gaussians.render(camera)  # float [0, 1]
rendered_uint8 = (rendered * 255).clamp(0, 255).to(torch.uint8)
tracks = tracker.track(rendered_uint8, points)  # Good accuracy!

# ✅ EVEN BETTER - Let PointTracker handle conversion
# (Already implemented in mpc/point_tracker.py line 105-109)
tracks = tracker.track(rendered, points)  # Converts internally
```

---

### Pitfall 4: Not Resetting TAPIR State

```python
# ❌ WRONG - State leaks between sequences
tracker = PointTracker(checkpoint_path)

for episode in episodes:
    for step in episode:
        # TAPIR maintains causal state from previous episode!
        tracks = tracker.track(video, points)  # Contaminated!

# ✅ CORRECT - Reset state between sequences
tracker = PointTracker(checkpoint_path)

for episode in episodes:
    tracker.model.set_points(points)  # Reset state
    
    for step in episode:
        tracks = tracker.track(video, points)  # Clean state
```

---

## Debugging Guide

### Step 1: Verify Tracking Accuracy

```python
# Run standalone test
python test_point_tracker.py

# Expected output:
# Mean tracking error: <0.5 pixels
# All points visible: True

# If error > 1 pixel:
# - Check uint8 conversion
# - Check checkpoint loaded correctly
# - Check TAPIR patches applied
```

---

### Step 2: Visualize Tracked Points

```python
def visualize_tracks(image, tracks, save_path):
    """
    Overlay tracked points on image.
    
    Args:
        image: (H, W, 3) numpy uint8
        tracks: (N, 2) numpy float [x, y] positions
        save_path: Output path for visualization
    """
    import cv2
    
    vis = image.copy()
    for i, (x, y) in enumerate(tracks):
        # Draw point
        cv2.circle(vis, (int(x), int(y)), 3, (0, 255, 0), -1)
        # Draw index
        cv2.putText(vis, str(i), (int(x)+5, int(y)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
    
    cv2.imwrite(save_path, vis)
    print(f"Saved visualization to {save_path}")

# Usage in demo script
initial_img = load_image("initial.png")
tracks, _ = tracker.track(video, initial_points)
visualize_tracks(initial_img, tracks[:, 0, :].cpu().numpy(), "debug_frame0.png")
visualize_tracks(target_img, tracks[:, -1, :].cpu().numpy(), "debug_target.png")
```

---

### Step 3: Check Reward Gradients (for differentiable tracking)

```python
def check_reward_gradient(objective, rendered_images):
    """
    Verify reward responds to changes in rendered images.
    """
    rendered_images.requires_grad = True
    
    reward = objective.compute_reward(rendered_images)
    reward.sum().backward()
    
    grad_norm = rendered_images.grad.norm().item()
    print(f"Gradient norm: {grad_norm}")
    
    if grad_norm < 1e-6:
        print("⚠️ WARNING: Gradient is near zero - reward may not be differentiable")
    else:
        print("✓ Gradient non-zero - reward is responsive")

# Usage
rendered = torch.randn(4, 10, 3, 256, 256, requires_grad=True).cuda()
check_reward_gradient(objective, rendered)
```

---

### Step 4: Profile Performance Bottlenecks

```python
import torch.autograd.profiler as profiler

with profiler.profile(use_cuda=True, record_shapes=True) as prof:
    reward = objective.compute_reward(rendered_images)

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

# Look for:
# - TAPIR forward pass time
# - Data transfer (CPU↔GPU)
# - Unnecessary copies
```

---

### Step 5: Debug MPC Integration

```python
# Add logging to objective
class DebugPointTrackingObjective(PointTrackingObjective):
    def compute_reward(self, rendered_images):
        reward = super().compute_reward(rendered_images)
        
        # Log reward statistics
        print(f"Reward mean: {reward.mean().item():.3f}")
        print(f"Reward std: {reward.std().item():.3f}")
        print(f"Reward min: {reward.min().item():.3f}")
        print(f"Reward max: {reward.max().item():.3f}")
        
        # Check for NaN/Inf
        if torch.isnan(reward).any():
            print("⚠️ WARNING: NaN in reward!")
        if torch.isinf(reward).any():
            print("⚠️ WARNING: Inf in reward!")
        
        return reward
```

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2026-03-06 | Initial version - TAPIR integration rules |

---

## See Also

- `doc/DEVELOPMENT_LOG.md`: Chronological development history
- `doc/PROMPT_TO_CHANGES.md`: Mapping of user requests to code changes
- `mpc/AGENTS.md`: MPC module architecture documentation
- `test_point_tracker.py`: Reference test implementation
