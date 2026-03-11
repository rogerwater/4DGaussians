#!/usr/bin/env python3
"""
Test script for Point Tracking MPC with specific requirements:
1. Use CoTracker to mark tracking points on initial and target frames, save visualizations
2. After each planning step, render next frame using optimal action and save
3. After planning, check if final image reaches target and compute difference
4. Save action sequence as .npy file
5. Initial action should not be all zeros - use CEM exploration
6. Limit action magnitude to prevent overflow
"""

import os
import sys
import argparse
import numpy as np
import torch
from pathlib import Path
from PIL import Image
import cv2
import json
import time

# Parse device early
def parse_device_early():
    for i, arg in enumerate(sys.argv):
        if arg == '--device' and i + 1 < len(sys.argv):
            device = sys.argv[i + 1]
            if device.startswith('cuda'):
                device_id = device.split(':')[1] if ':' in device else '0'
                os.environ['CUDA_VISIBLE_DEVICES'] = device_id
                return 'cuda:0', device_id
    return 'cuda:0', '0'

actual_device, actual_device_id = parse_device_early()

sys.path.insert(0, str(Path(__file__).parent))

# Import modules
from mpc.point_tracker import PointTracker
from mpc.cotracker_objectives import PointTrackingObjective
from mpc.flow_guided_gaussian_model import FlowGuidedGaussianDynamicsModel
from mpc.cem import CEMOptimizer
from mpc.sampler import CorrelatedNoiseSampler
from mpc.objectives import CombinedObjective
from mpc.utils import ObservationList
from mpc.agent import SimplePlanningAgent
from mpc import point_sampling  # New sampling module

def load_image(image_path, target_size=(480, 480)):
    """Load and resize image to 480x480"""
    img = Image.open(image_path).convert('RGB')
    img = img.resize((target_size[1], target_size[0]), Image.BILINEAR)
    img_array = np.array(img).astype(np.float32) / 255.0
    return img_array

def detect_object_regions(image, threshold=0.1):
    """
    Detect regions with significant color variation (likely objects/robot arm).
    Returns a mask where True indicates interesting regions.
    """
    # Convert to grayscale and compute local variance
    gray = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    
    # Apply bilateral filter to reduce noise while keeping edges
    filtered = cv2.bilateralFilter(gray, 9, 75, 75)
    
    # Compute gradient magnitude (edges indicate objects)
    grad_x = cv2.Sobel(filtered, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(filtered, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag = np.sqrt(grad_x**2 + grad_y**2)
    
    # Normalize to [0, 1]
    grad_mag = (grad_mag - grad_mag.min()) / (grad_mag.max() - grad_mag.min() + 1e-8)
    
    # Threshold to get object mask
    object_mask = grad_mag > threshold
    
    # Morphological operations to clean up mask
    kernel = np.ones((5, 5), np.uint8)
    object_mask = cv2.morphologyEx(object_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
    object_mask = cv2.morphologyEx(object_mask, cv2.MORPH_OPEN, kernel)
    
    return object_mask.astype(bool)

def sample_object_focused_points(image, num_points=256, object_ratio=0.7):
    """
    Sample tracking points with focus on objects and robot arm.
    
    Args:
        image: (H, W, 3) RGB image
        num_points: Total number of points to sample
        object_ratio: Fraction of points to place on objects (vs uniform background)
    
    Returns:
        points: (N, 2) array of [x, y] coordinates
    """
    H, W = image.shape[:2]
    
    # Detect object regions
    object_mask = detect_object_regions(image, threshold=0.1)
    
    # Number of points for objects vs background
    num_object_points = int(num_points * object_ratio)
    num_bg_points = num_points - num_object_points
    
    # Sample points on objects (dense)
    object_coords = np.argwhere(object_mask)  # (N, 2) [y, x]
    if len(object_coords) > 0:
        # Randomly sample from object pixels
        indices = np.random.choice(len(object_coords), 
                                   size=min(num_object_points, len(object_coords)), 
                                   replace=False)
        object_points = object_coords[indices][:, [1, 0]]  # Convert to [x, y]
    else:
        object_points = np.empty((0, 2))
    
    # Sample background points (sparse grid)
    if num_bg_points > 0:
        grid_size = int(np.ceil(np.sqrt(num_bg_points)))
        border = int(0.05 * min(H, W))  # 5% border
        y = np.linspace(border, H-border-1, grid_size)
        x = np.linspace(border, W-border-1, grid_size)
        xx, yy = np.meshgrid(x, y)
        bg_points = np.stack([xx.flatten(), yy.flatten()], axis=-1)[:num_bg_points]
    else:
        bg_points = np.empty((0, 2))
    
    # Combine and shuffle
    all_points = np.vstack([object_points, bg_points])
    np.random.shuffle(all_points)
    
    return all_points[:num_points]

def visualize_points(image, points, color=(0, 255, 0), radius=3):
    """Visualize tracking points on image"""
    # image: (H, W, 3) float [0, 1]
    # points: (N, 2) [x, y]
    vis = (image * 255).astype(np.uint8).copy()
    if points is None:
        return vis
        
    for i, p in enumerate(points):
        x, y = int(p[0]), int(p[1])
        if 0 <= x < vis.shape[1] and 0 <= y < vis.shape[0]:
            cv2.circle(vis, (x, y), radius, color, -1)
            # Add point index for first few points
            if i < 10:
                cv2.putText(vis, str(i), (x+5, y), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
    return vis

def compute_image_difference(img1, img2):
    """Compute MSE and PSNR between two images"""
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        psnr = float('inf')
    else:
        psnr = 20 * np.log10(1.0 / np.sqrt(mse))
    return mse, psnr

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, 
                        default="/home/ubuntu/yyf/4DGaussians/outputs/dm_control_push_test_flow2/point_cloud/iteration_10000")
    parser.add_argument("--initial_image", type=str,
                        default="/home/ubuntu/yyf/4DGaussians/assets/start-end/cam6_sample1_frame_00001.jpg")
    parser.add_argument("--target_image", type=str,
                        default="/home/ubuntu/yyf/4DGaussians/assets/start-end/cam6_sample1_frame_00018.jpg")
    parser.add_argument("--transforms_json", type=str,
                        default="/home/ubuntu/yyf/4DGaussians/assets/example_transforms.json")
    parser.add_argument("--control_dim", type=int, default=15)
    parser.add_argument("--num_steps", type=int, default=10, help="Number of MPC execution steps (reduced for faster testing)")
    parser.add_argument("--horizon", type=int, default=5, help="MPC planning horizon (reduced for faster planning)")
    parser.add_argument("--num_samples", type=int, default=48, help="CEM samples per iteration (balanced for speed and exploration)")
    parser.add_argument("--opt_iters", type=int, default=5, help="CEM optimization iterations")
    parser.add_argument("--num_tracking_points", type=int, default=384, help="Number of points to track (balanced for speed and coverage)")
    parser.add_argument("--tracking_weight", type=float, default=1.0)
    parser.add_argument("--sampling_method", type=str, default="combined",
                        choices=["sobel_hybrid", "shi_tomasi", "combined", "texture", "grid", "motion_mask"],
                        help="Point sampling strategy: sobel_hybrid (original), shi_tomasi (corners), "
                             "combined (50%% corners + 30%% texture + 20%% grid, recommended), texture (high-texture), "
                             "grid (uniform), motion_mask (GMFlow-based motion regions, 70%% motion + 30%% corners)")
    parser.add_argument("--resample_motion_mask_per_step", action="store_true", default=None,
                        help="Re-sample motion mask at every MPC step (only effective with --sampling_method=motion_mask). "
                             "Provides dense, accurate loss signals by keeping tracking points on moving objects. "
                             "Default: True for motion_mask, False for other methods.")
    parser.add_argument("--image_height", type=int, default=512, 
                        help="Image height for rendering (512x512 recommended for BootsTAPIR)")
    parser.add_argument("--image_width", type=int, default=512,
                        help="Image width for rendering (512x512 recommended for BootsTAPIR)")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--output_dir", type=str, default="./outputs/cotracker_test")
    parser.add_argument("--action_limit", type=float, default=0.8, 
                        help="Maximum action magnitude per step (increased to allow larger movements)")
    
    args = parser.parse_args()
    
    # Override device with remapped device (after CUDA_VISIBLE_DEVICES is set)
    args.device = actual_device
    
    # Set default for resample_motion_mask_per_step based on sampling_method
    if args.resample_motion_mask_per_step is None:
        args.resample_motion_mask_per_step = (args.sampling_method == "motion_mask")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("="*70)
    print("Point Tracking MPC Test (TAPIR)")
    print("="*70)
    print(f"Model: {args.model_path}")
    print(f"Initial: {args.initial_image}")
    print(f"Target: {args.target_image}")
    print(f"Output: {args.output_dir}")
    print(f"Action Limit: ±{args.action_limit}")
    print("="*70)
    
    # 1. Initialize Tracker
    print("\n[1/7] Initializing PointTracker (TAPIR)...")
    tracker = PointTracker(
        device=args.device,
        input_resolution=(args.image_height, args.image_width)
    )
    
    # 2. Load Images
    print("\n[2/7] Loading images...")
    initial_image = load_image(args.initial_image, (args.image_height, args.image_width))
    target_image = load_image(args.target_image, (args.image_height, args.image_width))
    
    # Save original images
    Image.fromarray((initial_image * 255).astype(np.uint8)).save(
        os.path.join(args.output_dir, "00_initial_image.png"))
    Image.fromarray((target_image * 255).astype(np.uint8)).save(
        os.path.join(args.output_dir, "00_target_image.png"))
    
    # 3. Setup Tracking Points
    print("\n[3/7] Setting up tracking points...")
    print(f"  Using sampling method: {args.sampling_method}")
    
    if args.sampling_method == "sobel_hybrid":
        initial_points = sample_object_focused_points(
            initial_image, 
            num_points=args.num_tracking_points,
            object_ratio=0.7
        )
        sampling_desc = "70% object-focused (Sobel+hybrid)"
    elif args.sampling_method == "shi_tomasi":
        initial_points = point_sampling.sample_shi_tomasi_points(
            initial_image, 
            num_points=args.num_tracking_points
        )
        sampling_desc = "Shi-Tomasi corners"
    elif args.sampling_method == "combined":
        initial_points = point_sampling.sample_combined(
            initial_image, 
            num_points=args.num_tracking_points,
            corner_weight=0.5,
            texture_weight=0.3,
            grid_weight=0.2
        )
        sampling_desc = "Combined (50% corners + 30% texture + 20% grid)"
    elif args.sampling_method == "texture":
        initial_points = point_sampling.sample_texture_points(
            initial_image, 
            num_points=args.num_tracking_points
        )
        sampling_desc = "High-texture regions"
    elif args.sampling_method == "grid":
        initial_points = point_sampling.sample_uniform_grid(
            initial_image, 
            num_points=args.num_tracking_points
        )
        sampling_desc = "Uniform grid"
    elif args.sampling_method == "motion_mask":
        initial_points = point_sampling.sample_motion_driven_points(
            initial_image,
            target_image,
            num_points=args.num_tracking_points,
            device=args.device,
            motion_ratio=0.7,
            save_diagnostics=True,
            output_dir=args.output_dir
        )
        sampling_desc = "Motion-driven (70% motion regions + 30% corners)"
    else:
        raise ValueError(f"Unknown sampling method: {args.sampling_method}")
    
    print(f"  Sampled {len(initial_points)} tracking points ({sampling_desc})")
    print(f"  Point coordinate range: x=[{initial_points[:, 0].min():.1f}, {initial_points[:, 0].max():.1f}], y=[{initial_points[:, 1].min():.1f}, {initial_points[:, 1].max():.1f}]")
    print(f"  Image shape: {initial_image.shape}")
    
    # Visualize initial points (RED)
    vis_initial = visualize_points(initial_image, initial_points, color=(255, 0, 0), radius=3)
    Image.fromarray(vis_initial).save(os.path.join(args.output_dir, "01_initial_with_points.png"))
    print(f"  Saved: 01_initial_with_points.png")
    
    # Track from initial to target (offline)
    print("  Computing target points by tracking Initial -> Target...")
    video_stack = np.stack([initial_image, target_image])  # (2, H, W, 3)
    video_tensor = torch.from_numpy(video_stack).permute(0, 3, 1, 2).unsqueeze(0).to(args.device).float()
    
    tracks, visibles = tracker.track(video_tensor, initial_points)
    if tracks is None:
        print("ERROR: Tracker failed. Aborting.")
        return
    
    # Extract target points (at frame 1)
    target_points = tracks[0, :, 1, :].cpu().numpy()  # (N, 2)
    target_visibles = visibles[0, :, 1].cpu().numpy()  # (N,)
    
    print(f"  Target point coordinate range: x=[{target_points[:, 0].min():.1f}, {target_points[:, 0].max():.1f}], y=[{target_points[:, 1].min():.1f}, {target_points[:, 1].max():.1f}]")
    
    # Visualize target points (GREEN)
    vis_target = visualize_points(target_image, target_points, color=(0, 255, 0), radius=3)
    Image.fromarray(vis_target).save(os.path.join(args.output_dir, "01_target_with_points.png"))
    print(f"  Saved: 01_target_with_points.png")
    print(f"  Visible target points: {target_visibles.sum()}/{len(target_visibles)}")
    
    # 4. Load Camera Transforms and Initial Control
    print("\n[4/7] Loading camera transforms and initial control...")
    transform_matrix = None
    focal_x = None
    focal_y = None
    cx = None
    cy = None
    initial_control = None
    
    if os.path.exists(args.transforms_json):
        with open(args.transforms_json, 'r') as f:
            transforms_data = json.load(f)
        
        # Camera parameters
        cameras_meta = transforms_data.get('cameras', [])
        if cameras_meta:
            camera_meta = cameras_meta[0]
            transform_matrix = np.array(camera_meta['transform_matrix'], dtype=np.float32)
            focal_x = camera_meta.get('fl_x') or camera_meta.get('focal_length')
            focal_y = camera_meta.get('fl_y') or camera_meta.get('focal_length')
            cx = camera_meta.get('cx', args.image_width / 2.0)
            cy = camera_meta.get('cy', args.image_height / 2.0)
            print(f"  Camera loaded: fx={focal_x}, fy={focal_y}, cx={cx}, cy={cy}")
        
        # Initial control from JSON
        frames = transforms_data.get('frames', [])
        image_filename = os.path.basename(args.initial_image)
        for frame in frames:
            if image_filename in frame.get('file_path', ''):
                if 'joint_pos' in frame:
                    initial_control = np.array(frame['joint_pos'], dtype=np.float32)
                    print(f"  Initial control loaded from JSON: shape={initial_control.shape}")
                    break
    
    # If no initial control found, we'll let CEM explore (don't use zeros!)
    if initial_control is None:
        print("  WARNING: No initial control found in JSON. Will use CEM exploration.")
        initial_control = np.zeros(args.control_dim, dtype=np.float32)
    
    # 5. Initialize Model
    print("\n[5/7] Initializing FlowGuidedGaussianDynamicsModel...")
    model = FlowGuidedGaussianDynamicsModel(
        model_path=args.model_path,
        iteration=12000,  # Use iteration 12000 as specified
        control_dim=args.control_dim,
        image_height=args.image_height,
        image_width=args.image_width,
        num_context=2,
        use_sparse_rendering=False,
        enable_flow_prediction=False,
        device=args.device,
        transform_matrix=transform_matrix,
        focal_x=focal_x,
        focal_y=focal_y,
        cx=cx,
        cy=cy
    )
    print("  Model loaded successfully")
    
    # 6. Setup MPC Optimizer
    print("\n[6/7] Setting up MPC Optimizer...")
    
    # Objective with advanced weighting strategies
    point_tracking_obj = PointTrackingObjective(
        tracker=tracker,
        weight=args.tracking_weight,
        rgb_key='rgb',
        goal_key='target_points',
        current_points_key='current_tracked_points',
        visibility_weight=True,      # Downweight occluded points
        endpoint_weight=3.0,         # 3x weight on final frame (endpoint priority)
        temporal_decay=0.7,          # Exponential decay for earlier frames
        smoothness_weight=0.01       # Small smoothness penalty
    )
    
    objective = CombinedObjective({
        'point_tracking': point_tracking_obj
    })
    
    # Sampler with action limits
    sampler = CorrelatedNoiseSampler(
        a_dim=args.control_dim,
        beta=0.9,  # Temporal correlation coefficient
        horizon=args.horizon
    )
    
    # CEM Optimizer
    optimizer = CEMOptimizer(
        model=model,
        objective=objective,
        sampler=sampler,
        a_dim=args.control_dim,
        horizon=args.horizon,
        num_samples=args.num_samples,
        elites_frac=0.1,  # 10% elite samples
        opt_iters=args.opt_iters,
        verbose=True
    )
    
    # Agent
    agent = SimplePlanningAgent(
        a_dim=args.control_dim,
        optimizer=optimizer,
        replan_interval=1,
        use_initial_action=True  # Use initial control if available
    )
    
    # 7. MPC Loop
    print("\n[7/7] Starting MPC Loop...")
    print(f"  Planning for {args.num_steps} steps with horizon={args.horizon}")
    
    current_tracked_points = initial_points.copy()
    current_image = initial_image.copy()
    
    observations = []
    actions = []
    step_losses = []  # Track loss at each step
    
    # Initial state
    initial_action_tensor = torch.tensor(initial_control, dtype=torch.float32, device=args.device)
    initial_rendered = model.render_with_control(initial_action_tensor)  # (3, H, W)
    initial_rendered_np = initial_rendered.permute(1, 2, 0).cpu().numpy()
    
    # Save initial rendered image
    Image.fromarray((initial_rendered_np * 255).astype(np.uint8)).save(
        os.path.join(args.output_dir, "step_0000_rendered.png"))
    
    # Observation history
    obs_history = ObservationList(
        data_dict={'rgb': initial_rendered_np[None]},
        image_shape=(args.image_height, args.image_width)
    )
    
    # Fill context
    for _ in range(model.num_context - 1):
        obs_history.append(obs_history[0])
    
    observations.append(initial_rendered_np)
    actions.append(initial_control.copy())
    
    # Main loop
    for step in range(1, args.num_steps + 1):
        print(f"\n--- Step {step}/{args.num_steps} ---")
        
        # 🆕 Per-step motion mask resampling (Solution A)
        if args.resample_motion_mask_per_step and args.sampling_method == "motion_mask":
            step_dir = os.path.join(args.output_dir, f"step_{step:03d}")
            os.makedirs(step_dir, exist_ok=True)
            
            print(f"  🔄 Re-computing motion mask: current frame → target frame")
            start_time = time.time()
            
            current_tracked_points = point_sampling.sample_motion_driven_points(
                current_image,      # Current frame (not initial!)
                target_image,       # Target frame (fixed)
                num_points=args.num_tracking_points,
                device=args.device,
                motion_ratio=0.7,
                save_diagnostics=True,
                output_dir=step_dir
            )
            
            gmflow_time = time.time() - start_time
            print(f"    ✓ Sampled {len(current_tracked_points)} points on moving objects ({gmflow_time:.2f}s)")
            
            # Compute target points via TAPIR (current → target)
            print(f"  📍 Computing target point positions via TAPIR...")
            video_tensor_to_target = torch.stack([
                torch.from_numpy(current_image).permute(2, 0, 1).float(),
                torch.from_numpy(target_image).permute(2, 0, 1).float()
            ], dim=0).unsqueeze(0).to(args.device)
            
            tracks_to_target, visibles_to_target = tracker.track(
                video_tensor_to_target,
                current_tracked_points
            )
            target_points = tracks_to_target[0, :, 1, :].cpu().numpy()
            
            # Visualize resampled points on current frame
            vis_current_points = visualize_points(current_image, current_tracked_points, color=(255, 255, 0), radius=2)
            Image.fromarray((vis_current_points).astype(np.uint8)).save(
                os.path.join(step_dir, f"current_with_resampled_points.png"))
            print(f"    ✓ Target points computed, diagnostics saved to {step_dir}/")
        
        # Set goal (pass as plain dict, not ObservationList)
        agent.set_goal({
            'target_points': target_points,  # (N, 2)
            'current_tracked_points': current_tracked_points  # (N, 2)
        })
        
        # Plan optimal action
        action = agent.act(t=step-1, obs_history=obs_history, state_obs_history=obs_history)
        
        # Collect loss information from optimizer (if planning occurred)
        # Note: At t=0, agent uses initial action without planning, so optimizer attributes may be None
        if optimizer.last_best_reward is not None:
            loss_info = {
                'step': step,
                'best_reward': optimizer.last_best_reward,
                'mean_reward': optimizer.last_mean_reward,
                'iterations': optimizer.last_rewards_history.copy()
            }
            step_losses.append(loss_info)
            
            print(f"  Best Reward: {loss_info['best_reward']:.4f}")
            print(f"  Mean Reward: {loss_info['mean_reward']:.4f}")
        else:
            print(f"  Using initial action (no planning at step {step})")
        
        # Clip action to prevent overflow (safety check)
        action = np.clip(action, -args.action_limit, args.action_limit)
        
        actions.append(action)
        print(f"  Action (first 5 dims): [{', '.join([f'{a:.4f}' for a in action[:5]])}...]")
        print(f"  Action magnitude: {np.linalg.norm(action):.4f}")
        
        # Render next frame with optimal action
        action_tensor = torch.tensor(action, dtype=torch.float32, device=args.device)
        next_image_tensor = model.render_with_control(action_tensor)
        next_image_np = next_image_tensor.permute(1, 2, 0).cpu().numpy()
        
        # Save rendered image
        Image.fromarray((next_image_np * 255).astype(np.uint8)).save(
            os.path.join(args.output_dir, f"step_{step:04d}_rendered.png"))
        
        # Update tracked points (track from current to next)
        video_stack = np.stack([current_image, next_image_np])  # (2, H, W, 3)
        video_tensor = torch.from_numpy(video_stack).permute(0, 3, 1, 2).unsqueeze(0).to(args.device).float()
        
        tracks, visibles = tracker.track(video_tensor, current_tracked_points)
        new_points = tracks[0, :, 1, :].cpu().numpy()
        new_visibles = visibles[0, :, 1].cpu().numpy()
        
        failed, failure_reason = point_sampling.detect_tracking_failure(
            tracks[0].cpu().numpy(),
            visibles[0].cpu().numpy()
        )
        
        if failed:
            print(f"  ⚠️ Tracking failure detected: {failure_reason}")
            print(f"     Re-sampling points using {args.sampling_method} method...")
            
            if args.sampling_method == "motion_mask":
                # 🆕 NEW: Motion mask re-sampling on failure
                new_points = point_sampling.sample_motion_driven_points(
                    next_image_np,
                    target_image,  # Use original target from initialization
                    num_points=args.num_tracking_points,
                    device=args.device,
                    motion_ratio=0.7,
                    save_diagnostics=False,  # Don't save diagnostics on failure (save time)
                    output_dir=args.output_dir
                )
                print(f"     Re-sampled {len(new_points)} points using motion mask")
            elif args.sampling_method == "sobel_hybrid":
                new_points = sample_object_focused_points(next_image_np, num_points=args.num_tracking_points, object_ratio=0.7)
            elif args.sampling_method == "shi_tomasi":
                new_points = point_sampling.sample_shi_tomasi_points(next_image_np, num_points=args.num_tracking_points)
            elif args.sampling_method == "combined":
                new_points = point_sampling.sample_combined(next_image_np, num_points=args.num_tracking_points)
            elif args.sampling_method == "texture":
                new_points = point_sampling.sample_texture_points(next_image_np, num_points=args.num_tracking_points)
            elif args.sampling_method == "grid":
                new_points = point_sampling.sample_uniform_grid(next_image_np, num_points=args.num_tracking_points)
            
            print(f"     Re-sampled {len(new_points)} new tracking points")
        else:
            visible_ratio = new_visibles.sum() / len(new_visibles)
            print(f"  Tracking status: OK ({new_visibles.sum()}/{len(new_visibles)} visible = {visible_ratio*100:.1f}%)")
        
        # Visualize tracked points on current frame
        vis_step = visualize_points(next_image_np, new_points, color=(0, 255, 255), radius=2)
        Image.fromarray(vis_step).save(os.path.join(args.output_dir, f"step_{step:04d}_with_points.png"))
        
        # Compute distance to target
        dist = np.linalg.norm(new_points - target_points, axis=-1).mean()
        print(f"  Avg Distance to Target: {dist:.4f} pixels")
        
        # Update state
        current_tracked_points = new_points
        current_image = next_image_np
        
        # Append to observation history (wrap in ObservationList)
        new_obs = ObservationList(
            data_dict={'rgb': next_image_np[None]},
            image_shape=(args.image_height, args.image_width)
        )
        obs_history.append(new_obs)
        observations.append(next_image_np)
    
    # 8. Final Analysis
    print("\n" + "="*70)
    print("Final Analysis")
    print("="*70)
    
    final_image = observations[-1]
    
    # Save final image
    Image.fromarray((final_image * 255).astype(np.uint8)).save(
        os.path.join(args.output_dir, "final_rendered.png"))
    
    # Compute difference with target
    mse, psnr = compute_image_difference(final_image, target_image)
    print(f"\nImage Quality Metrics:")
    print(f"  MSE (Final vs Target):  {mse:.6f}")
    print(f"  PSNR (Final vs Target): {psnr:.2f} dB")
    
    # Final point tracking distance
    final_dist = np.linalg.norm(current_tracked_points - target_points, axis=-1)
    print(f"\nPoint Tracking Metrics:")
    print(f"  Mean distance to target: {final_dist.mean():.4f} pixels")
    print(f"  Max distance to target:  {final_dist.max():.4f} pixels")
    print(f"  Min distance to target:  {final_dist.min():.4f} pixels")
    
    # Check if reached target (threshold: 10 pixels on average)
    threshold = 10.0
    reached = final_dist.mean() < threshold
    print(f"\nTarget Reached: {'YES ✓' if reached else 'NO ✗'} (threshold: {threshold} pixels)")
    
    # 9. Save Action Sequence
    print("\n" + "="*70)
    print("Saving Results")
    print("="*70)
    
    actions_array = np.array(actions)  # (num_steps+1, control_dim)
    action_save_path = os.path.join(args.output_dir, "action_sequence.npy")
    np.save(action_save_path, actions_array)
    print(f"✓ Action sequence saved: {action_save_path}")
    print(f"  Shape: {actions_array.shape}")
    
    # Save metrics
    metrics = {
        'mse': float(mse),
        'psnr': float(psnr),
        'mean_dist': float(final_dist.mean()),
        'max_dist': float(final_dist.max()),
        'min_dist': float(final_dist.min()),
        'reached_target': bool(reached),
        'num_steps': args.num_steps,
        'control_dim': args.control_dim,
        'step_losses': step_losses  # Include per-step loss history
    }
    
    metrics_path = os.path.join(args.output_dir, "metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"✓ Metrics saved: {metrics_path}")
    
    # Save detailed loss CSV for easy plotting
    loss_csv_path = os.path.join(args.output_dir, "loss_history.csv")
    with open(loss_csv_path, 'w') as f:
        f.write("step,iteration,best_reward,mean_reward,std_reward\n")
        for loss_info in step_losses:
            step_num = loss_info['step']
            for iter_info in loss_info['iterations']:
                f.write(f"{step_num},{iter_info['iteration']},{iter_info['best']},{iter_info['mean']},{iter_info['std']}\n")
    print(f"✓ Loss history saved: {loss_csv_path}")
    
    print("\n" + "="*70)
    print("Test Complete!")
    print(f"All outputs saved to: {args.output_dir}")
    print("="*70)

if __name__ == "__main__":
    main()
