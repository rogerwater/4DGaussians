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
        weights = weights / weights.sum()  # Normalize
        
        # Sample indices with replacement if needed
        num_to_sample = min(num_motion_points, len(motion_coords))
        indices = np.random.choice(len(motion_coords), size=num_to_sample, 
                                   replace=False, p=weights)
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
