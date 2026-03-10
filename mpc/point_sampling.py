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
