"""
Point sampling strategies for robust tracking in MPC.

Based on research from:
- RoboTAP (DeepMind 2024): Motion clustering
- ReKep (Stanford/Columbia 2024): Semantic keypoints  
- SuperPoint (MagicLeap 2020): Corner detection
- FlowTrack (CVPR 2024): Hybrid tracking
"""

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import os

# Motion mask computation (Task 1)
from gmflow.config import get_cfg as get_gmflow_cfg
from gmflow.gmflow import GMFlow


def sample_shi_tomasi_points(image, num_points=256, quality_level=0.01, min_distance=8):
    """
    Sample Shi-Tomasi corner points (Good Features to Track).
    
    Better than Sobel for robot arms - focuses on corners/joints.
    
    Args:
        image: (H, W, 3) RGB image, [0, 1] float
        num_points: Target number of points
        quality_level: Quality threshold (0.001-0.1, lower=more points)
        min_distance: Minimum pixel distance between points
    
    Returns:
        points: (N, 2) array of [x, y] coordinates
    """
    # Convert to grayscale uint8
    gray = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    
    # Detect corners
    corners = cv2.goodFeaturesToTrack(
        gray,
        maxCorners=num_points,
        qualityLevel=quality_level,
        minDistance=min_distance,
        blockSize=7,
        useHarrisDetector=False,  # Shi-Tomasi
        k=0.04
    )
    
    if corners is None:
        # Fallback to grid if no corners found
        print("[PointSampling] WARNING: No Shi-Tomasi corners found, falling back to grid")
        return sample_uniform_grid(image, num_points)
    
    # Reshape from (N, 1, 2) to (N, 2)
    points = corners.squeeze(1)
    
    # Pad with grid points if insufficient
    if len(points) < num_points:
        grid_points = sample_uniform_grid(image, num_points - len(points))
        points = np.vstack([points, grid_points])
    
    return points[:num_points]


def sample_uniform_grid(image, num_points=256, border=0.05):
    """Simple uniform grid sampling (baseline)."""
    H, W = image.shape[:2]
    border_px = int(border * min(H, W))
    
    grid_size = int(np.ceil(np.sqrt(num_points)))
    y = np.linspace(border_px, H - border_px - 1, grid_size)
    x = np.linspace(border_px, W - border_px - 1, grid_size)
    xx, yy = np.meshgrid(x, y)
    
    points = np.stack([xx.flatten(), yy.flatten()], axis=-1)
    return points[:num_points]


def sample_texture_points(image, num_points=256, laplacian_threshold=0.1):
    """
    Sample points in high-texture regions using Laplacian variance.
    
    Useful for trackable features with local structure.
    """
    # Convert to grayscale
    gray = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    
    # Compute Laplacian (texture measure)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F, ksize=3)
    texture_map = np.abs(laplacian)
    
    # Normalize
    texture_map = (texture_map - texture_map.min()) / (texture_map.max() - texture_map.min() + 1e-8)
    
    # Threshold
    high_texture_mask = texture_map > laplacian_threshold
    
    # Sample from high-texture regions
    texture_coords = np.argwhere(high_texture_mask)  # (N, 2) [y, x]
    
    if len(texture_coords) > num_points:
        # Random sampling
        indices = np.random.choice(len(texture_coords), num_points, replace=False)
        points = texture_coords[indices][:, [1, 0]]  # Convert to [x, y]
    elif len(texture_coords) > 0:
        # Pad with grid
        points = texture_coords[:, [1, 0]]
        grid_points = sample_uniform_grid(image, num_points - len(points))
        points = np.vstack([points, grid_points])
    else:
        # Fallback
        points = sample_uniform_grid(image, num_points)
    
    return points[:num_points]


def sample_combined(
    image, 
    num_points=256,
    corner_weight=0.5,
    texture_weight=0.3,
    grid_weight=0.2,
    nms_radius=8
):
    """
    Combined sampling: weighted mix of corner + texture + grid.
    
    Recommended for robotics - balances structure, trackability, and coverage.
    
    Args:
        image: (H, W, 3) RGB image
        num_points: Total points to sample
        corner_weight: Fraction from Shi-Tomasi corners
        texture_weight: Fraction from high-texture regions
        grid_weight: Fraction from uniform grid
        nms_radius: Non-maximum suppression radius (spatial diversity)
    
    Returns:
        points: (N, 2) array [x, y]
    """
    num_corner = int(num_points * corner_weight)
    num_texture = int(num_points * texture_weight)
    num_grid = num_points - num_corner - num_texture
    
    # Sample from each strategy
    corner_points = sample_shi_tomasi_points(
        image, 
        num_corner, 
        quality_level=0.01,
        min_distance=nms_radius
    )
    
    texture_points = sample_texture_points(
        image,
        num_texture,
        laplacian_threshold=0.1
    )
    
    grid_points = sample_uniform_grid(image, num_grid, border=0.05)
    
    # Combine
    all_points = np.vstack([corner_points, texture_points, grid_points])
    
    # Apply spatial NMS for diversity
    all_points = apply_nms(all_points, radius=nms_radius)
    
    # Shuffle
    np.random.shuffle(all_points)
    
    return all_points[:num_points]


def apply_nms(points, radius=8):
    """Non-maximum suppression for spatial diversity."""
    if len(points) == 0:
        return points
    
    # Sort by score (use distance from center as proxy)
    H_center, W_center = points.mean(axis=0)
    distances = np.sqrt(
        (points[:, 0] - W_center)**2 + (points[:, 1] - H_center)**2
    )
    sorted_indices = np.argsort(distances)
    
    kept = []
    for idx in sorted_indices:
        point = points[idx]
        
        # Check distance to already kept points
        if len(kept) == 0:
            kept.append(point)
            continue
        
        kept_array = np.array(kept)
        dists = np.sqrt(
            (kept_array[:, 0] - point[0])**2 + 
            (kept_array[:, 1] - point[1])**2
        )
        
        if dists.min() > radius:
            kept.append(point)
    
    return np.array(kept)


def detect_tracking_failure(tracks, visibles, flow_forward=None, flow_backward=None):
    """
    Detect tracking failure using multiple heuristics.
    
    Based on fault-tolerant visual servoing research (2024).
    
    Returns:
        failed: bool, True if tracking should be re-initialized
        reason: str, description of failure mode
    """
    # Check 1: Visibility ratio
    valid_ratio = visibles.sum() / len(tracks)
    if valid_ratio < 0.5:
        return True, f"visibility_low ({valid_ratio:.2f})"
    
    # Check 2: Forward-backward consistency (if optical flow available)
    if flow_forward is not None and flow_backward is not None:
        # Warp forward flow by backward flow
        H, W = flow_forward.shape[1:3]
        grid_y, grid_x = torch.meshgrid(
            torch.arange(H, device=flow_forward.device),
            torch.arange(W, device=flow_forward.device),
            indexing='ij'
        )
        identity_flow = torch.stack([grid_x, grid_y], dim=0).float()
        
        # Warp
        flow_fwd_normalized = flow_forward / torch.tensor(
            [W, H], device=flow_forward.device
        ).view(1, 2, 1, 1)
        
        warped = F.grid_sample(
            flow_backward,
            (identity_flow + flow_forward).permute(1, 2, 0)[None] / torch.tensor(
                [W, H], device=flow_forward.device
            ) * 2 - 1,
            align_corners=False
        )
        
        consistency_error = torch.norm(flow_forward + warped, dim=1).mean()
        
        if consistency_error > 5.0:  # pixels
            return True, f"flow_inconsistency ({consistency_error:.1f}px)"
    
    # Check 3: Spatial clustering (points collapsed to single location)
    if tracks.ndim == 3:  # (N, T, 2)
        last_positions = tracks[:, -1, :]
    else:  # (N, 2)
        last_positions = tracks
    
    std_x = last_positions[:, 0].std()
    std_y = last_positions[:, 1].std()
    
    if std_x < 10 or std_y < 10:  # Less than 10px spread
        return True, f"spatial_collapse (std_x={std_x:.1f}, std_y={std_y:.1f})"
    
    return False, "ok"


# Diagnostic functions
def compute_point_quality_scores(image, points):
    """
    Compute per-point quality scores for filtering.
    
    Returns:
        scores: (N,) array, higher = better quality
    """
    gray = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    
    scores = []
    for pt in points:
        x, y = int(pt[0]), int(pt[1])
        
        # Extract patch (7x7)
        patch = gray[max(0, y-3):min(gray.shape[0], y+4), 
                     max(0, x-3):min(gray.shape[1], x+4)]
        
        if patch.size == 0:
            scores.append(0.0)
            continue
        
        # Compute Laplacian variance (texture)
        laplacian = cv2.Laplacian(patch, cv2.CV_64F)
        texture_score = np.var(laplacian)
        
        scores.append(texture_score)
    
    return np.array(scores)


def compute_motion_mask(img1, img2, device='cuda:0', percentile=70, min_magnitude=1.0,
                        save_diagnostics=False, output_dir='outputs/test_motion'):
    """
    Compute motion mask between two frames using GMFlow optical flow.
    
    Uses adaptive percentile-based thresholding to identify moving regions,
    with morphological post-processing for clean boundaries.
    
    Args:
        img1: (H, W, 3) RGB image [0, 1] float (initial frame)
        img2: (H, W, 3) RGB image [0, 1] float (target frame)
        device: CUDA device string (e.g., 'cuda:0')
        percentile: Percentile threshold for motion detection (70 = top 30% of motion)
        min_magnitude: Minimum flow magnitude in pixels (noise filter, default 1.0px)
        save_diagnostics: Whether to save flow magnitude visualization
        output_dir: Directory for diagnostic outputs
    
    Returns:
        motion_mask: (H, W) boolean array, True = motion detected
        flow_magnitude: (H, W) float array, flow magnitude in pixels
    
    References:
        - GMFlow initialization: demo_flow_guided_mpc.py:271-288
        - Adaptive threshold: UnFlow (ICCV 2017), demo_flow_guided_mpc.py:312
        - Morphological ops: test_cotracker_mpc.py:78-80
    """
    H, W = img1.shape[:2]
    device_obj = torch.device(device)
    
    # ============ GMFlow Initialization ============
    gmflow_cfg = get_gmflow_cfg()
    flownet = GMFlow(
        feature_channels=gmflow_cfg.feature_channels,
        num_scales=gmflow_cfg.num_scales,
        upsample_factor=gmflow_cfg.upsample_factor,
        num_head=gmflow_cfg.num_head,
        attention_type=gmflow_cfg.attention_type,
        ffn_dim_expansion=gmflow_cfg.ffn_dim_expansion,
        num_transformer_layers=gmflow_cfg.num_transformer_layers,
    ).to(device_obj)
    
    # Safe checkpoint loading (CPU → GPU)
    checkpoint = torch.load(gmflow_cfg.model, map_location="cpu")
    weights = checkpoint["model"] if "model" in checkpoint else checkpoint
    flownet.load_state_dict(weights, strict=True)
    flownet.eval()
    
    # ============ Flow Computation ============
    # Prepare images: (H, W, 3) float → (1, 3, H, W) tensor
    img1_t = torch.from_numpy(img1).permute(2, 0, 1).unsqueeze(0).float().to(device_obj)
    img2_t = torch.from_numpy(img2).permute(2, 0, 1).unsqueeze(0).float().to(device_obj)
    
    with torch.no_grad():
        flow_predictions = flownet(
            img1_t, img2_t,
            attn_splits_list=[2],
            corr_radius_list=[-1],
            prop_radius_list=[-1],
        )
        flow_field = flow_predictions[-1]  # (1, 2, H, W)
    
    flow_field = flow_field[0].permute(1, 2, 0).cpu().numpy()  # (H, W, 2)
    flow_magnitude = np.linalg.norm(flow_field, axis=-1)  # (H, W)
    
    # ============ Adaptive Percentile Thresholding ============
    # Step 1: Filter noise
    above_min = flow_magnitude > min_magnitude
    if above_min.sum() == 0:
        print(f"[MotionMask] WARNING: No motion above {min_magnitude}px, using all pixels")
        above_min = np.ones_like(flow_magnitude, dtype=bool)
    
    threshold = np.percentile(flow_magnitude[above_min], percentile)
    
    # Step 2: Create mask
    motion_mask = flow_magnitude > threshold
    coverage = motion_mask.sum() / motion_mask.size
    
    # Step 3: Guardrails (handle edge cases)
    if coverage < 0.05:
        print(f"[MotionMask] WARNING: Low coverage {coverage:.1%}, fallback to 95th percentile")
        threshold = np.percentile(flow_magnitude.flatten(), 95)
        motion_mask = flow_magnitude > threshold
        coverage = motion_mask.sum() / motion_mask.size
    elif coverage > 0.95:
        print(f"[MotionMask] WARNING: High coverage {coverage:.1%} (camera motion?), fallback to 90th percentile")
        threshold = np.percentile(flow_magnitude.flatten(), 90)
        motion_mask = flow_magnitude > threshold
        coverage = motion_mask.sum() / motion_mask.size
    
    # ============ Morphological Post-Processing ============
    kernel = np.ones((5, 5), np.uint8)
    mask_uint8 = motion_mask.astype(np.uint8)
    mask_uint8 = cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, kernel)  # Fill holes
    mask_uint8 = cv2.morphologyEx(mask_uint8, cv2.MORPH_OPEN, kernel)   # Remove noise
    motion_mask = mask_uint8.astype(bool)
    
    # Update coverage after morphological ops
    coverage = motion_mask.sum() / motion_mask.size
    
    # ============ Validation ============
    assert 0.01 < coverage < 0.8, \
        f"Motion mask coverage {coverage:.1%} out of valid range [1%, 80%]. " \
        f"Too low = static scene, too high = camera motion."
    
    print(f"[MotionMask] Coverage: {coverage:.1%}, Threshold: {threshold:.2f}px, " \
          f"Flow range: [{flow_magnitude.min():.2f}, {flow_magnitude.max():.2f}]px")
    
    # ============ Diagnostic Visualization ============
    if save_diagnostics:
        try:
            import matplotlib.pyplot as plt
            os.makedirs(output_dir, exist_ok=True)
            
            # Flow magnitude heatmap
            plt.figure(figsize=(10, 8))
            plt.imshow(flow_magnitude, cmap='jet')
            plt.colorbar(label='Flow Magnitude (pixels)')
            plt.title(f'Optical Flow Magnitude\n(threshold={threshold:.2f}px, coverage={coverage:.1%})')
            plt.tight_layout()
            plt.savefig(f'{output_dir}/flow_magnitude_heatmap.png', dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"[MotionMask] Saved diagnostic to {output_dir}/flow_magnitude_heatmap.png")
        except ImportError:
            print("[MotionMask] WARNING: matplotlib not available, skipping diagnostics")
    
    # ============ GPU Memory Cleanup ============
    del flownet
    torch.cuda.empty_cache()
    
    return motion_mask, flow_magnitude


def sample_motion_driven_points(img1, img2, num_points=384, device='cuda:0',
                                 motion_ratio=0.7, nms_radius=8,
                                 save_diagnostics=False, output_dir='outputs/test_motion'):
    """
    Sample tracking points focused on motion regions using GMFlow.
    
    Combines motion-based sampling (70%) with corner detection (30%) for robust tracking.
    Includes fallback strategies for edge cases (static scenes, camera motion).
    
    Args:
        img1: (H, W, 3) RGB image [0, 1] float (initial frame)
        img2: (H, W, 3) RGB image [0, 1] float (target frame)
        num_points: Target number of tracking points
        device: CUDA device string
        motion_ratio: Fraction of points from motion regions (default 0.7 = 70%)
        nms_radius: Minimum distance between points (pixels)
        save_diagnostics: Whether to save visualization
        output_dir: Directory for diagnostic outputs
    
    Returns:
        points: (N, 2) numpy array of [x, y] coordinates
    
    References:
        - Motion mask: compute_motion_mask() (Task 1)
        - Hybrid sampling: sample_combined() (lines 138-172)
        - Mask-constrained sampling: test_cotracker_mpc.py:84-129
    """
    H, W = img1.shape[:2]
    
    # ============ Compute Motion Mask ============
    motion_mask, flow_magnitude = compute_motion_mask(img1, img2, device=device,
                                                       save_diagnostics=False,
                                                       output_dir=output_dir)
    
    coverage = motion_mask.sum() / motion_mask.size
    
    # ============ Fallback Strategies ============
    # Case 1: Nearly static scene (coverage < 1%)
    if coverage < 0.01:
        print(f"[MotionSampling] WARNING: Static scene (coverage {coverage:.1%}), fallback to uniform grid")
        return sample_uniform_grid(img1, num_points, border=0.05)
    
    # Case 2: Camera motion (coverage > 80%)
    if coverage > 0.80:
        print(f"[MotionSampling] WARNING: High motion (coverage {coverage:.1%}), fallback to Shi-Tomasi")
        return sample_shi_tomasi_points(img1, num_points, quality_level=0.01, min_distance=nms_radius)
    
    # ============ Motion-Driven Hybrid Sampling ============
    num_motion_points = int(num_points * motion_ratio)
    num_corner_points = num_points - num_motion_points
    
    # Part 1: Sample from motion regions
    motion_coords = np.argwhere(motion_mask)  # (N, 2) [y, x]
    
    if len(motion_coords) > 0:
        # Weight sampling by flow magnitude for better feature selection
        motion_y, motion_x = motion_coords[:, 0], motion_coords[:, 1]
        weights = flow_magnitude[motion_y, motion_x]
        
        # Normalize probabilities with zero-flow fallback
        if weights.sum() > 0:
            probabilities = weights / weights.sum()
        else:
            probabilities = np.ones(len(weights)) / len(weights)
        
        # Sample indices with replacement if needed
        num_to_sample = min(num_motion_points, len(motion_coords))
        indices = np.random.choice(len(motion_coords), size=num_to_sample, 
                                   replace=False, p=probabilities)
        motion_points = motion_coords[indices][:, [1, 0]]  # Convert to [x, y]
    else:
        motion_points = np.empty((0, 2))
    
    # Part 2: Sample Shi-Tomasi corners for texture quality
    corner_points = sample_shi_tomasi_points(img1, num_corner_points, 
                                             quality_level=0.01,
                                             min_distance=nms_radius)
    
    # ============ Combine and Filter ============
    all_points = np.vstack([motion_points, corner_points])
    
    # Apply spatial NMS for diversity
    all_points = apply_nms(all_points, radius=nms_radius)
    
    # Shuffle
    np.random.shuffle(all_points)
    
    # Ensure we have enough points (pad with grid if needed)
    if len(all_points) < num_points * 0.8:  # Less than 80% target
        print(f"[MotionSampling] WARNING: Only {len(all_points)} points after NMS, " \
              f"padding with grid to reach {num_points}")
        grid_points = sample_uniform_grid(img1, num_points - len(all_points), border=0.05)
        all_points = np.vstack([all_points, grid_points])
    
    final_points = all_points[:num_points]
    
    # ============ Diagnostic Visualization ============
    if save_diagnostics:
        try:
            import matplotlib.pyplot as plt
            os.makedirs(output_dir, exist_ok=True)
            
            # Motion mask overlay with sampled points
            fig, axes = plt.subplots(1, 2, figsize=(16, 8))
            
            # Left: Motion mask overlay
            axes[0].imshow(img1)
            axes[0].imshow(motion_mask, cmap='Reds', alpha=0.3)
            axes[0].set_title(f'Motion Mask (coverage={coverage:.1%})')
            axes[0].axis('off')
            
            # Right: Sampled points
            axes[1].imshow(img1)
            axes[1].scatter(final_points[:, 0], final_points[:, 1], 
                           c='lime', s=20, marker='o', alpha=0.8, edgecolors='black', linewidths=0.5)
            
            # Check how many points fall in motion regions
            points_in_motion = motion_mask[final_points[:, 1].astype(int), final_points[:, 0].astype(int)]
            motion_point_ratio = points_in_motion.sum() / len(final_points)
            
            axes[1].set_title(f'Sampled Points (n={len(final_points)}, {motion_point_ratio:.1%} in motion)')
            axes[1].axis('off')
            
            plt.tight_layout()
            plt.savefig(f'{output_dir}/motion_mask_with_points.png', dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"[MotionSampling] Saved diagnostic to {output_dir}/motion_mask_with_points.png")
            print(f"[MotionSampling] Points in motion regions: {motion_point_ratio:.1%}")
        except ImportError:
            print("[MotionSampling] WARNING: matplotlib not available, skipping diagnostics")
    
    print(f"[MotionSampling] Sampled {len(final_points)} points " \
          f"(motion={len(motion_points)}, corners={len(corner_points)}, final after NMS={len(final_points)})")
    
    return final_points


# ============================================================================
# Task 1: Bidirectional Optical Flow with Forward-Backward Consistency Check
# ============================================================================

def compute_bidirectional_flow_with_consistency(
    img1: np.ndarray,
    img2: np.ndarray,
    device: str = 'cuda:0',
    consistency_threshold: float = 3.0
):
    """
    Compute bidirectional optical flow with forward-backward consistency check.
    
    This function computes flow in both directions (img1→img2 and img2→img1) and
    identifies consistent pixels using the forward-backward consistency criterion:
    ||flow_forward(x) + flow_backward(x + flow_forward(x))|| < threshold
    
    Consistent pixels indicate reliable motion estimates, filtering out:
    - Occlusions (pixels visible in one frame but not the other)
    - Textureless regions (ambiguous matches)
    - Motion boundaries (unstable flow estimates)
    
    Args:
        img1: (H, W, 3) RGB image [0, 1] float (source frame)
        img2: (H, W, 3) RGB image [0, 1] float (target frame)
        device: CUDA device string (e.g., 'cuda:0')
        consistency_threshold: Max pixel distance for consistency (typical: 1-5px)
                              Lower = stricter (fewer but more reliable points)
                              Higher = looser (more points but less reliable)
    
    Returns:
        flow_forward: (H, W, 2) numpy array, flow from img1 to img2
        flow_backward: (H, W, 2) numpy array, flow from img2 to img1
        consistency_mask: (H, W) boolean array, True = consistent pixel
        flow_magnitude: (H, W) float array, forward flow magnitude in pixels
    
    References:
        - Forward-backward consistency: UnFlow (ICCV 2017)
        - GMFlow initialization: compute_motion_mask() lines 328-343
        
    Example:
        >>> flow_fwd, flow_bwd, mask, mag = compute_bidirectional_flow_with_consistency(
        ...     initial_frame, target_frame, device='cuda:1', consistency_threshold=3.0
        ... )
        >>> print(f"Consistent pixels: {mask.sum()} / {mask.size} ({mask.mean():.1%})")
    """
    H, W = img1.shape[:2]
    device_obj = torch.device(device)
    
    # ============ GMFlow Initialization ============
    gmflow_cfg = get_gmflow_cfg()
    flownet = GMFlow(
        feature_channels=gmflow_cfg.feature_channels,
        num_scales=gmflow_cfg.num_scales,
        upsample_factor=gmflow_cfg.upsample_factor,
        num_head=gmflow_cfg.num_head,
        attention_type=gmflow_cfg.attention_type,
        ffn_dim_expansion=gmflow_cfg.ffn_dim_expansion,
        num_transformer_layers=gmflow_cfg.num_transformer_layers,
    ).to(device_obj)
    
    # Safe checkpoint loading (CPU → GPU)
    checkpoint = torch.load(gmflow_cfg.model, map_location="cpu")
    weights = checkpoint["model"] if "model" in checkpoint else checkpoint
    flownet.load_state_dict(weights, strict=True)
    flownet.eval()
    
    # ============ Forward Flow: img1 → img2 ============
    img1_t = torch.from_numpy(img1).permute(2, 0, 1).unsqueeze(0).float().to(device_obj)
    img2_t = torch.from_numpy(img2).permute(2, 0, 1).unsqueeze(0).float().to(device_obj)
    
    with torch.no_grad():
        flow_forward_predictions = flownet(
            img1_t, img2_t,
            attn_splits_list=[2],
            corr_radius_list=[-1],
            prop_radius_list=[-1],
        )
        flow_forward_t = flow_forward_predictions[-1]  # (1, 2, H, W)
    
    # ============ Backward Flow: img2 → img1 ============
    with torch.no_grad():
        flow_backward_predictions = flownet(
            img2_t, img1_t,
            attn_splits_list=[2],
            corr_radius_list=[-1],
            prop_radius_list=[-1],
        )
        flow_backward_t = flow_backward_predictions[-1]  # (1, 2, H, W)
    
    # Convert to numpy: (1, 2, H, W) → (H, W, 2)
    flow_forward = flow_forward_t[0].permute(1, 2, 0).cpu().numpy()  # (H, W, 2)
    flow_backward = flow_backward_t[0].permute(1, 2, 0).cpu().numpy()  # (H, W, 2)
    
    # ============ Forward-Backward Consistency Check ============
    # For each pixel x in img1, compute warped position x' = x + flow_forward(x)
    # Then check if flow_backward(x') + flow_forward(x) ≈ 0
    
    # Create coordinate grid for img1
    y_coords, x_coords = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
    coords = np.stack([x_coords, y_coords], axis=-1).astype(np.float32)  # (H, W, 2) [x, y]
    
    # Warp coordinates: x' = x + flow_forward(x)
    warped_coords = coords + flow_forward  # (H, W, 2)
    
    # Sample flow_backward at warped coordinates using bilinear interpolation
    # OpenCV's remap expects separate x and y maps
    flow_backward_warped = cv2.remap(
        flow_backward,
        warped_coords[..., 0].astype(np.float32),  # x coordinates
        warped_coords[..., 1].astype(np.float32),  # y coordinates
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0
    )  # (H, W, 2)
    
    # Consistency error: ||flow_forward(x) + flow_backward(x')||
    consistency_error = np.linalg.norm(flow_forward + flow_backward_warped, axis=-1)  # (H, W)
    
    # Consistency mask: error < threshold
    consistency_mask = consistency_error < consistency_threshold  # (H, W) boolean
    
    # ============ Flow Magnitude ============
    flow_magnitude = np.linalg.norm(flow_forward, axis=-1)  # (H, W)
    
    # Clean up GPU memory
    del flownet, img1_t, img2_t, flow_forward_t, flow_backward_t
    torch.cuda.empty_cache()
    
    return flow_forward, flow_backward, consistency_mask, flow_magnitude


def adaptive_motion_mask_with_consistency(
    img1: np.ndarray,
    img2: np.ndarray,
    device: str = 'cuda:0',
    percentile: float = 70,
    min_magnitude: float = 1.0,
    consistency_threshold: float = 3.0,
    morphology_kernel_size: int = 5
):
    """
    Compute adaptive motion mask using bidirectional flow with consistency check.
    
    This function combines:
    1. Bidirectional optical flow (forward + backward)
    2. Forward-backward consistency check (filters unreliable regions)
    3. Adaptive percentile-based thresholding (adapts to motion scale)
    4. Morphological post-processing (clean boundaries)
    
    The resulting mask identifies pixels with:
    - Significant motion (above adaptive threshold)
    - Consistent flow in both directions (reliable estimates)
    - Clean region boundaries (no isolated pixels)
    
    Args:
        img1: (H, W, 3) RGB image [0, 1] float (source frame)
        img2: (H, W, 3) RGB image [0, 1] float (target frame)
        device: CUDA device string (e.g., 'cuda:0')
        percentile: Percentile threshold for motion detection (70 = top 30% of motion)
                   Higher = more restrictive (only strong motion)
                   Lower = more permissive (includes weak motion)
        min_magnitude: Minimum flow magnitude in pixels (noise filter, default 1.0px)
        consistency_threshold: Max pixel distance for forward-backward consistency (3.0px)
        morphology_kernel_size: Kernel size for morphological ops (default 5)
    
    Returns:
        motion_mask: (H, W) boolean array, True = reliable motion detected
        flow_forward: (H, W, 2) numpy array, forward flow field
        flow_magnitude: (H, W) float array, flow magnitude in pixels
        consistency_mask: (H, W) boolean array, True = consistent pixel
    
    References:
        - Adaptive threshold: compute_motion_mask() lines 357-368
        - Morphology: compute_motion_mask() lines 370-375
        
    Example:
        >>> mask, flow, mag, cons = adaptive_motion_mask_with_consistency(
        ...     initial_frame, target_frame, device='cuda:1', percentile=70
        ... )
        >>> print(f"Motion pixels: {mask.sum()} (magnitude: {mag[mask].mean():.1f}px)")
    """
    # ============ Compute Bidirectional Flow ============
    flow_forward, flow_backward, consistency_mask, flow_magnitude = \
        compute_bidirectional_flow_with_consistency(
            img1, img2, device=device, consistency_threshold=consistency_threshold
        )
    
    # ============ Adaptive Thresholding ============
    # Only consider consistent pixels for threshold computation
    consistent_magnitudes = flow_magnitude[consistency_mask]
    
    if len(consistent_magnitudes) == 0:
        # Fallback: no consistent pixels (rare, e.g., completely static scene)
        print("[AdaptiveMask] WARNING: No consistent pixels found, using global threshold")
        adaptive_threshold = np.percentile(flow_magnitude, percentile)
    else:
        # Compute percentile threshold from consistent pixels only
        adaptive_threshold = np.percentile(consistent_magnitudes, percentile)
    
    # Ensure minimum magnitude threshold
    adaptive_threshold = max(adaptive_threshold, min_magnitude)
    
    # ============ Motion Mask ============
    # Require BOTH high magnitude AND consistency
    magnitude_mask = flow_magnitude > adaptive_threshold  # (H, W) boolean
    motion_mask = magnitude_mask & consistency_mask  # (H, W) boolean
    
    # ============ Morphological Post-Processing ============
    # Remove small isolated regions and smooth boundaries
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morphology_kernel_size, morphology_kernel_size))
    
    # Opening: erosion followed by dilation (removes small noise)
    motion_mask_uint8 = motion_mask.astype(np.uint8) * 255
    motion_mask_cleaned = cv2.morphologyEx(motion_mask_uint8, cv2.MORPH_OPEN, kernel)
    
    # Closing: dilation followed by erosion (fills small holes)
    motion_mask_cleaned = cv2.morphologyEx(motion_mask_cleaned, cv2.MORPH_CLOSE, kernel)
    
    motion_mask_final = motion_mask_cleaned > 0  # Convert back to boolean
    
    return motion_mask_final, flow_forward, flow_magnitude, consistency_mask


# ============================================================================
# Task 2: Optical Flow-Based Point Propagation with Mask Filtering
# ============================================================================

def propagate_points_with_flow(
    points: np.ndarray,
    flow_field: np.ndarray,
    mask: np.ndarray = None,
    image_shape: tuple = None
):
    """
    Propagate tracking points using optical flow field with validity filtering.
    
    This function warps existing tracking points to their new positions using
    the optical flow field, then filters out invalid points based on:
    1. Mask validity (if provided) - points must land in valid regions
    2. Boundary checks - points must stay within image bounds
    
    This is the core operation for dynamic tracking point updates:
    - Preserves long-term tracking continuity (vs. resampling from scratch)
    - Filters out occluded/unreliable points via mask
    - Handles boundary cases gracefully
    
    Args:
        points: (N, 2) numpy array of [x, y] coordinates to propagate
        flow_field: (H, W, 2) numpy array of optical flow [dx, dy] at each pixel
        mask: (H, W) boolean array, optional validity mask (True = valid region)
              If provided, only points that land in valid regions are kept
              If None, only boundary checks are performed
        image_shape: (H, W) tuple, optional image dimensions for bounds checking
                     If None, inferred from flow_field shape
    
    Returns:
        propagated_points: (M, 2) numpy array, new [x, y] coordinates (M ≤ N)
        valid_indices: (M,) numpy array, indices of valid points in original array
                       Use this to update corresponding target points:
                       target_points_new = target_points[valid_indices]
    
    Algorithm:
        1. Sample flow at each point location (bilinear interpolation)
        2. Compute new positions: p' = p + flow(p)
        3. Filter by mask (if provided): keep only p' where mask(p') == True
        4. Filter by bounds: keep only p' where 0 ≤ x < W and 0 ≤ y < H
    
    Example:
        >>> # Propagate points from frame t to frame t+1
        >>> flow_t_to_t1 = compute_flow(frame_t, frame_t1)
        >>> consistency_mask = compute_consistency_mask(...)
        >>> new_points, valid_idx = propagate_points_with_flow(
        ...     current_points, flow_t_to_t1, mask=consistency_mask
        ... )
        >>> # Update target points accordingly
        >>> target_points_new = target_points[valid_idx]
    
    References:
        - Bilinear sampling: cv2.remap in compute_bidirectional_flow_with_consistency
        - Mask filtering strategy: User requirement "光流传播+mask过滤"
    """
    if len(points) == 0:
        return np.array([]).reshape(0, 2), np.array([], dtype=np.int64)
    
    H, W = flow_field.shape[:2]
    if image_shape is None:
        image_shape = (H, W)
    img_H, img_W = image_shape
    
    # ============ Sample Flow at Point Locations ============
    # cv2.remap expects float32 coordinates
    x_coords = points[:, 0].astype(np.float32)  # (N,)
    y_coords = points[:, 1].astype(np.float32)  # (N,)
    
    # Sample flow_x and flow_y separately
    flow_x = cv2.remap(
        flow_field[..., 0],  # (H, W) x-component
        x_coords.reshape(1, -1),  # (1, N) x coordinates
        y_coords.reshape(1, -1),  # (1, N) y coordinates
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0
    ).flatten()  # (N,)
    
    flow_y = cv2.remap(
        flow_field[..., 1],  # (H, W) y-component
        x_coords.reshape(1, -1),
        y_coords.reshape(1, -1),
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0
    ).flatten()  # (N,)
    
    # ============ Propagate Points ============
    propagated_x = points[:, 0] + flow_x  # (N,)
    propagated_y = points[:, 1] + flow_y  # (N,)
    propagated_points = np.stack([propagated_x, propagated_y], axis=-1)  # (N, 2)
    
    # ============ Validity Filtering ============
    valid_mask = np.ones(len(points), dtype=bool)  # Start with all valid
    
    # 1. Boundary check: must be within image bounds
    valid_mask &= (propagated_x >= 0) & (propagated_x < img_W)
    valid_mask &= (propagated_y >= 0) & (propagated_y < img_H)
    
    # 2. Mask check: must land in valid region (if mask provided)
    if mask is not None:
        # Sample mask at propagated positions (nearest neighbor)
        propagated_x_int = np.clip(propagated_x, 0, img_W - 1).astype(np.int32)
        propagated_y_int = np.clip(propagated_y, 0, img_H - 1).astype(np.int32)
        mask_values = mask[propagated_y_int, propagated_x_int]  # (N,) boolean
        valid_mask &= mask_values
    
    # ============ Extract Valid Points ============
    valid_indices = np.where(valid_mask)[0]  # (M,) indices of valid points
    propagated_points_valid = propagated_points[valid_mask]  # (M, 2)
    
    return propagated_points_valid, valid_indices


# ============================================================================
# Task 3: Complete Dynamic Tracking Point Update Function
# ============================================================================

def update_tracking_points_dynamic(
    current_points: np.ndarray,
    target_points: np.ndarray,
    current_image: np.ndarray,
    target_image: np.ndarray,
    prev_image: np.ndarray,
    num_points_target: int,
    device: str = 'cuda:0',
    consistency_threshold: float = 3.0,
    percentile: float = 70,
    min_points_ratio: float = 0.5,
    max_points_ratio: float = 1.5
):
    """
    Dynamically update tracking points using bidirectional optical flow.
    
    This is the MAIN function for dynamic tracking point updates in MPC planning.
    It implements the complete pipeline:
    
    1. **Propagate existing points** via reverse flow (current → prev)
       - Preserves long-term tracking continuity
       - Filters invalid points using consistency mask
    
    2. **Supplement new points** if count drops below threshold
       - Sample from motion regions (current → target flow)
       - Ensures sufficient tracking coverage
    
    3. **Update target points** to match propagated current points
       - Recompute correspondences via forward flow (current → target)
       - Maintains tracking coherence
    
    This approach combines:
    - Flow propagation (preserves existing tracks)
    - Mask filtering (removes unreliable points)
    - Dynamic supplementing (maintains point count)
    
    Args:
        current_points: (N, 2) numpy array, current frame tracking points [x, y]
        target_points: (N, 2) numpy array, corresponding target points [x, y]
        current_image: (H, W, 3) RGB [0, 1] float, current rendered frame
        target_image: (H, W, 3) RGB [0, 1] float, final target frame (fixed)
        prev_image: (H, W, 3) RGB [0, 1] float, previous frame (t-1)
        num_points_target: int, target number of tracking points (e.g., 384)
        device: str, CUDA device (e.g., 'cuda:1')
        consistency_threshold: float, forward-backward consistency threshold (3.0px)
        percentile: float, motion mask percentile threshold (70 = top 30%)
        min_points_ratio: float, minimum points ratio before supplementing (0.5 = 50%)
        max_points_ratio: float, maximum points ratio cap (1.5 = 150%)
    
    Returns:
        updated_current_points: (M, 2) numpy array, updated current points
        updated_target_points: (M, 2) numpy array, updated target points
        debug_info: dict with keys:
            - 'motion_mask': (H, W) boolean, motion mask for visualization
            - 'consistency_mask': (H, W) boolean, consistency mask
            - 'num_propagated': int, number of points after propagation
            - 'num_supplemented': int, number of points added
            - 'flow_magnitude': (H, W) float, flow magnitude for visualization
    
    Algorithm Flow:
        Step 1: Compute reverse flow (current → prev) with consistency check
        Step 2: Propagate current_points using reverse flow + mask filtering
        Step 3: Update target_points to match propagated current_points (keep valid indices)
        Step 4: If points < threshold, supplement from motion mask (current → target)
        Step 5: Recompute target_points for all current_points via forward flow
    
    Example Usage in MPC Loop:
        >>> for step in range(num_steps):
        ...     # Execute action, render current frame
        ...     rendered_image = render(...)
        ...     
        ...     # Update tracking points dynamically
        ...     if step > 0:  # Skip first step (no previous frame)
        ...         current_pts, target_pts, debug = update_tracking_points_dynamic(
        ...             current_points=goal["current_points"].cpu().numpy(),
        ...             target_points=goal["target_points"].cpu().numpy(),
        ...             current_image=rendered_image,
        ...             target_image=target_image_fixed,
        ...             prev_image=prev_rendered_image,
        ...             num_points_target=384,
        ...             device=device
        ...         )
        ...         goal["current_points"] = torch.from_numpy(current_pts).to(device)
        ...         goal["target_points"] = torch.from_numpy(target_pts).to(device)
        ...     
        ...     prev_rendered_image = rendered_image.copy()
    
    References:
        - User requirement: "在每一步规划完成之后计算当前帧和上一帧之间的光流（注意是反向计算）"
        - Strategy: "光流传播+mask过滤" (flow propagation + mask filtering)
    """
    H, W = current_image.shape[:2]
    
    # ============ Step 1: Compute Reverse Flow (current → prev) ============
    print(f"[DynamicUpdate] Step 1: Computing reverse flow (current → prev)...")
    flow_reverse, _, consistency_mask_reverse, _ = \
        compute_bidirectional_flow_with_consistency(
            current_image, prev_image,
            device=device,
            consistency_threshold=consistency_threshold
        )
    
    # ============ Step 2: Propagate Current Points ============
    print(f"[DynamicUpdate] Step 2: Propagating {len(current_points)} points...")
    propagated_current_points, valid_indices = propagate_points_with_flow(
        points=current_points,
        flow_field=flow_reverse,
        mask=consistency_mask_reverse,
        image_shape=(H, W)
    )
    
    num_propagated = len(propagated_current_points)
    print(f"[DynamicUpdate]   → Kept {num_propagated}/{len(current_points)} points "
          f"({num_propagated/len(current_points)*100:.1f}%)")
    
    # ============ Step 3: Update Target Points (keep valid indices) ============
    propagated_target_points = target_points[valid_indices]
    
    # ============ Step 4: Supplement Points if Needed ============
    min_points_threshold = int(num_points_target * min_points_ratio)
    num_supplemented = 0
    
    if num_propagated < min_points_threshold:
        num_needed = num_points_target - num_propagated
        print(f"[DynamicUpdate] Step 4: Points below threshold ({num_propagated} < {min_points_threshold}), "
              f"supplementing {num_needed} points...")
        
        # Compute motion mask (current → target) for sampling new points
        motion_mask, flow_forward, flow_magnitude_vis, consistency_mask_forward = \
            adaptive_motion_mask_with_consistency(
                current_image, target_image,
                device=device,
                percentile=percentile,
                consistency_threshold=consistency_threshold
            )
        
        # Sample new points from motion regions
        motion_coords = np.column_stack(np.where(motion_mask))  # (K, 2) [y, x]
        if len(motion_coords) > 0:
            # Convert to [x, y] format
            motion_points_candidates = motion_coords[:, [1, 0]].astype(np.float32)  # (K, 2) [x, y]
            
            # Compute flow magnitude for weighted sampling
            flow_magnitude = np.linalg.norm(flow_forward, axis=-1)
            motion_y, motion_x = motion_coords[:, 0], motion_coords[:, 1]
            weights = flow_magnitude[motion_y, motion_x]
            
            # Normalize probabilities with zero-flow fallback
            if weights.sum() > 0:
                probabilities = weights / weights.sum()
            else:
                probabilities = np.ones(len(weights)) / len(weights)
            
            # Randomly sample needed points
            if len(motion_points_candidates) >= num_needed:
                sampled_indices = np.random.choice(
                    len(motion_points_candidates),
                    size=num_needed,
                    replace=False,
                    p=probabilities
                )
            else:
                # Not enough motion points, take all available
                sampled_indices = np.arange(len(motion_points_candidates))
            
            new_current_points = motion_points_candidates[sampled_indices]  # (num_needed, 2)
            num_supplemented = len(new_current_points)
            
            # Concatenate with propagated points
            propagated_current_points = np.vstack([propagated_current_points, new_current_points])
            
            print(f"[DynamicUpdate]   → Supplemented {num_supplemented} points from motion mask")
        else:
            print(f"[DynamicUpdate]   → WARNING: No motion regions found, using grid sampling fallback")
            # Fallback: sample from grid
            grid_y, grid_x = np.meshgrid(
                np.linspace(0, H-1, int(np.sqrt(num_needed))+1),
                np.linspace(0, W-1, int(np.sqrt(num_needed))+1),
                indexing='ij'
            )
            grid_points = np.stack([grid_x.flatten(), grid_y.flatten()], axis=-1)[:num_needed]
            propagated_current_points = np.vstack([propagated_current_points, grid_points])
            num_supplemented = len(grid_points)
    else:
        # No supplementing needed, but still compute motion mask for debug visualization
        motion_mask, flow_forward, flow_magnitude_vis, consistency_mask_forward = \
            adaptive_motion_mask_with_consistency(
                current_image, target_image,
                device=device,
                percentile=percentile,
                consistency_threshold=consistency_threshold
            )
    
    # ============ Step 5: Recompute Target Points for All Current Points ============
    print(f"[DynamicUpdate] Step 5: Recomputing target points via forward flow...")
    
    # Compute forward flow if not already computed (happens when no supplementing)
    if num_supplemented == 0:
        # Already computed in Step 4, reuse flow_forward
        pass
    
    # Warp current points to target frame using forward flow
    new_target_points, valid_forward_indices = propagate_points_with_flow(
        points=propagated_current_points,
        flow_field=flow_forward,
        mask=consistency_mask_forward,
        image_shape=(H, W)
    )
    
    # Filter current points to match valid target points
    updated_current_points = propagated_current_points[valid_forward_indices]
    updated_target_points = new_target_points
    
    # ============ Cap Maximum Points ============
    max_points_cap = int(num_points_target * max_points_ratio)
    if len(updated_current_points) > max_points_cap:
        # Randomly downsample to cap
        keep_indices = np.random.choice(
            len(updated_current_points),
            size=max_points_cap,
            replace=False
        )
        updated_current_points = updated_current_points[keep_indices]
        updated_target_points = updated_target_points[keep_indices]
        print(f"[DynamicUpdate]   → Capped to {max_points_cap} points (max_ratio={max_points_ratio})")
    
    # ============ Final Statistics ============
    print(f"[DynamicUpdate] Final: {len(updated_current_points)} points "
          f"(propagated={num_propagated}, supplemented={num_supplemented}, "
          f"filtered={(num_propagated+num_supplemented)-len(updated_current_points)})")
    
    # ============ Debug Info ============
    debug_info = {
        'motion_mask': motion_mask,
        'consistency_mask': consistency_mask_forward,
        'num_propagated': num_propagated,
        'num_supplemented': num_supplemented,
        'flow_magnitude': flow_magnitude_vis
    }
    
    return updated_current_points, updated_target_points, debug_info
