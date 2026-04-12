#!/usr/bin/env python3
"""
Point Tracking MPC Demo using TAPIR (from im2flow2act)
"""

import os
import sys
import argparse
import numpy as np
import subprocess
from pathlib import Path
from PIL import Image
import cv2

def _pick_least_used_gpu(default_device_id='0'):
    """Pick the GPU with the lowest memory usage ratio via nvidia-smi."""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return default_device_id

    candidates = []
    for gpu_id, line in enumerate(result.stdout.strip().splitlines()):
        parts = [p.strip() for p in line.split(',')]
        if len(parts) != 2:
            continue
        try:
            used_mem = float(parts[0])
            total_mem = max(float(parts[1]), 1.0)
        except ValueError:
            continue
        usage_ratio = used_mem / total_mem
        candidates.append((usage_ratio, used_mem, gpu_id))

    if not candidates:
        return default_device_id

    candidates.sort(key=lambda item: (item[0], item[1], item[2]))
    return str(candidates[0][2])


def parse_device_early():
    for i, arg in enumerate(sys.argv):
        if arg == '--device' and i + 1 < len(sys.argv):
            device = sys.argv[i + 1]
            if device.startswith('cuda'):
                device_id = device.split(':')[1] if ':' in device else '0'
                os.environ['CUDA_VISIBLE_DEVICES'] = device_id
                print(f"[Device Mapping] Requested device: {device}")
                print(f"[Device Mapping] Set CUDA_VISIBLE_DEVICES={device_id}")
                print(f"[Device Mapping] PyTorch will use: cuda:0 (mapped to physical GPU {device_id})")
                return 'cuda:0', device_id

    existing_visible = os.environ.get('CUDA_VISIBLE_DEVICES')
    if existing_visible:
        primary_device = existing_visible.split(',')[0].strip()
        print(f"[Device Mapping] Using existing CUDA_VISIBLE_DEVICES={existing_visible}")
        print(f"[Device Mapping] PyTorch will use: cuda:0 (mapped to physical GPU {primary_device})")
        return 'cuda:0', (primary_device if primary_device else '0')

    best_device_id = _pick_least_used_gpu(default_device_id='0')
    os.environ['CUDA_VISIBLE_DEVICES'] = best_device_id
    print(f"[Device Mapping] Auto-selected GPU: {best_device_id}")
    print(f"[Device Mapping] Set CUDA_VISIBLE_DEVICES={best_device_id}")
    print(f"[Device Mapping] PyTorch will use: cuda:0 (mapped to physical GPU {best_device_id})")
    return 'cuda:0', best_device_id

actual_device, actual_device_id = parse_device_early()

import torch

sys.path.insert(0, str(Path(__file__).parent))

# Import PointTracker first to setup JAX env
from mpc.point_tracker import PointTracker
from mpc.cotracker_objectives import PointTrackingObjective

from mpc.flow_guided_gaussian_model import FlowGuidedGaussianDynamicsModel
from mpc.cem import CEMOptimizer
from mpc.cem_gd import CEMGDOptimizer
from mpc.sampler import CorrelatedNoiseSampler
from mpc.objectives import CombinedObjective
from mpc.utils import ObservationList
from mpc.agent import SimplePlanningAgent
from mpc import point_sampling

def load_image(image_path, target_size=(256, 256)):
    img = Image.open(image_path).convert('RGB')
    img = img.resize((target_size[1], target_size[0]), Image.BILINEAR)
    img_array = np.array(img).astype(np.float32) / 255.0
    return img_array

def sample_grid_points(H, W, num_points=256, border=0.1):
    # Sample points in a grid, avoiding borders
    grid_size = int(np.ceil(np.sqrt(num_points)))
    h_border = int(H * border)
    w_border = int(W * border)
    
    y = np.linspace(h_border, H-h_border-1, grid_size)
    x = np.linspace(w_border, W-w_border-1, grid_size)
    xx, yy = np.meshgrid(x, y)
    points = np.stack([xx.flatten(), yy.flatten()], axis=-1)
    return points[:num_points]

def visualize_points(image, points, color=(0, 255, 0), radius=2):
    # image: (H, W, 3) float [0, 1]
    # points: (N, 2) [x, y]
    vis = (image * 255).astype(np.uint8).copy()
    if points is None:
        return vis
        
    for p in points:
        x, y = int(p[0]), int(p[1])
        if 0 <= x < vis.shape[1] and 0 <= y < vis.shape[0]:
            cv2.circle(vis, (x, y), radius, color, -1)
    return vis

def run_cotracker_mpc(
    model_path: str,
    iteration: int,
    initial_image_path: str,
    target_image_path: str,
    control_dim: int,
    num_steps: int = 20,
    horizon: int = 5,
    num_samples: int = 32,
    opt_iters: int = 10,
    num_tracking_points: int = 256,
    tracking_weight: float = 1.0,
    image_height: int = 480,
    image_width: int = 480,
    device: str = "cuda:3",
    log_dir: str = "./outputs/cotracker_test",
    transforms_json_path: str = None,
    optimizer_type: str = "cem-gd",
    num_samples_init: int = 32,
    num_samples_replan: int = 16,
    num_grad_seqs: int = 3,
    grad_lr: float = 0.01,
    grad_steps: int = 15,
    gradient_device: str = None,
    sampling_method: str = "motion_mask",
    action_limit: float = 1.0,
    resample_motion_mask_per_step: bool = False,
):
    device = actual_device
    
    os.makedirs(log_dir, exist_ok=True)
    
    print("="*70)
    print("Point Tracking MPC (TAPIR)")
    print("="*70)
    
    # 1. Initialize Tracker
    print("Initializing PointTracker (TAPIR)...")
    tracker = PointTracker(device=device)
    
    # 2. Load Images
    print("Loading images...")
    initial_image = load_image(initial_image_path, (image_height, image_width))
    target_image = load_image(target_image_path, (image_height, image_width))
    
    # Save inputs
    Image.fromarray((initial_image * 255).astype(np.uint8)).save(os.path.join(log_dir, "initial_image.png"))
    Image.fromarray((target_image * 255).astype(np.uint8)).save(os.path.join(log_dir, "target_image.png"))
    
    # 3. Setup Tracking Points
    print("Setting up tracking points...")
    print(f"  Using sampling method: {sampling_method}")
    
    if sampling_method == "sobel_hybrid":
        initial_points = sample_object_focused_points(
            initial_image, 
            num_points=num_tracking_points,
            object_ratio=0.7
        )
        sampling_desc = "70% object-focused (Sobel+hybrid)"
    elif sampling_method == "shi_tomasi":
        initial_points = point_sampling.sample_shi_tomasi_points(
            initial_image, 
            num_points=num_tracking_points
        )
        sampling_desc = "Shi-Tomasi corners"
    elif sampling_method == "combined":
        initial_points = point_sampling.sample_combined(
            initial_image, 
            num_points=num_tracking_points,
            corner_weight=0.5,
            texture_weight=0.3,
            grid_weight=0.2
        )
        sampling_desc = "Combined (50% corners + 30% texture + 20% grid)"
    elif sampling_method == "texture":
        initial_points = point_sampling.sample_texture_points(
            initial_image, 
            num_points=num_tracking_points
        )
        sampling_desc = "High-texture regions"
    elif sampling_method == "grid":
        initial_points = sample_grid_points(image_height, image_width, num_points=num_tracking_points)
        sampling_desc = "Uniform grid"
    elif sampling_method == "motion_mask":
        print("  Computing bidirectional flow with consistency check...")
        motion_mask, flow_forward, flow_magnitude, consistency_mask = \
            point_sampling.adaptive_motion_mask_with_consistency(
                initial_image,
                target_image,
                device=device,
                percentile=70,
                min_magnitude=1.0,
                consistency_threshold=3.0,
                morphology_kernel_size=5
            )
        
        print(f"  Consistency: {consistency_mask.sum()}/{consistency_mask.size} pixels ({consistency_mask.mean()*100:.1f}%)")
        print(f"  Motion mask: {motion_mask.sum()}/{motion_mask.size} pixels ({motion_mask.mean()*100:.1f}%)")
        print(f"  Flow magnitude: mean={flow_magnitude[motion_mask].mean():.1f}px, max={flow_magnitude.max():.1f}px")
        
        motion_coords = np.column_stack(np.where(motion_mask))
        motion_points_candidates = motion_coords[:, [1, 0]].astype(np.float32)
        
        num_motion = int(num_tracking_points * 0.7)
        num_corners = num_tracking_points - num_motion
        
        if len(motion_points_candidates) >= num_motion:
            motion_y, motion_x = motion_coords[:, 0], motion_coords[:, 1]
            weights = flow_magnitude[motion_y, motion_x]
            
            if weights.sum() > 0:
                probabilities = weights / weights.sum()
            else:
                probabilities = np.ones(len(weights)) / len(weights)
            
            motion_indices = np.random.choice(len(motion_points_candidates), size=num_motion, replace=False, p=probabilities)
            motion_points = motion_points_candidates[motion_indices]
        else:
            motion_points = motion_points_candidates
            num_corners = num_tracking_points - len(motion_points)
        
        corner_points = point_sampling.sample_shi_tomasi_points(
            initial_image,
            num_points=num_corners,
            quality_level=0.01,
            min_distance=8
        )
        
        initial_points = np.vstack([motion_points, corner_points])
        
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        axes[0].imshow(flow_magnitude, cmap='hot')
        axes[0].set_title(f'Flow Magnitude (max={flow_magnitude.max():.1f}px)')
        axes[0].axis('off')
        
        axes[1].imshow(consistency_mask, cmap='gray')
        axes[1].set_title(f'Consistency Mask ({consistency_mask.mean()*100:.1f}%)')
        axes[1].axis('off')
        
        axes[2].imshow(initial_image)
        axes[2].imshow(motion_mask, alpha=0.3, cmap='Reds')
        axes[2].scatter(motion_points[:, 0], motion_points[:, 1], c='red', s=10, label=f'Motion ({len(motion_points)})')
        axes[2].scatter(corner_points[:, 0], corner_points[:, 1], c='blue', s=10, label=f'Corners ({len(corner_points)})')
        axes[2].set_title('Sampled Points on Motion Mask')
        axes[2].legend()
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(log_dir, "01_biflow_initialization.png"), dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: 01_biflow_initialization.png")
        
        sampling_desc = f"Bidirectional flow (motion={len(motion_points)}, corners={len(corner_points)})"
    else:
        raise ValueError(f"Unknown sampling method: {sampling_method}")
    
    print(f"  Sampled {len(initial_points)} tracking points ({sampling_desc})")
    
    # Visualize initial points
    vis_initial = visualize_points(initial_image, initial_points, color=(255, 0, 0), radius=3)
    Image.fromarray(vis_initial).save(os.path.join(log_dir, "01_initial_with_points.png"))
    
    # Find Target Points (Offline Tracking)
    print("  Computing target points by tracking Initial -> Target...")
    video_stack = np.stack([initial_image, target_image])
    video_tensor = torch.from_numpy(video_stack).permute(0, 3, 1, 2).unsqueeze(0).to(device).float()
    
    tracks, visibles = tracker.track(video_tensor, initial_points)
    if tracks is None:
        print("ERROR: Tracker failed. Aborting.")
        return
    
    target_points = tracks[0, :, 1, :].cpu().numpy()
    target_visibles = visibles[0, :, 1].cpu().numpy()
    
    print(f"  Target point coordinate range: x=[{target_points[:, 0].min():.1f}, {target_points[:, 0].max():.1f}], y=[{target_points[:, 1].min():.1f}, {target_points[:, 1].max():.1f}]")
    
    vis_target = visualize_points(target_image, target_points, color=(0, 255, 0), radius=3)
    Image.fromarray(vis_target).save(os.path.join(log_dir, "01_target_with_points.png"))
    print(f"  Saved: 01_target_with_points.png")
    print(f"  Visible target points: {target_visibles.sum()}/{len(target_visibles)}")
    
    # 4. Initialize Model
    print("Initializing FlowGuidedGaussianDynamicsModel...")
    
    # Load camera/control from JSON if available
    transform_matrix = None
    focal_x = None
    focal_y = None
    cx = None
    cy = None
    initial_control = np.zeros(control_dim, dtype=np.float32)
    
    if transforms_json_path and os.path.exists(transforms_json_path):
        import json
        with open(transforms_json_path, 'r') as f:
            transforms_data = json.load(f)
        
        cameras_meta = transforms_data.get('cameras', [])
        frames = transforms_data.get('frames', [])
        
        # Extract camera name from initial_image path (e.g., /path/to/cam06/frame_00001.jpg)
        camera_name = None
        path_parts = Path(initial_image_path).parts
        for part in reversed(path_parts):
            if part.startswith('cam') and part[3:].isdigit():
                camera_name = part
                break
        
        # Parse camera index from camera name (1-based → 0-based)
        if camera_name and cameras_meta:
            camera_number = int(camera_name.replace('cam', ''))
            camera_idx = camera_number - 1
            
            print(f"[Camera Selection] Detected camera: {camera_name} (camera_number={camera_number}, camera_idx={camera_idx})")
            
            if 0 <= camera_idx < len(cameras_meta):
                camera_meta = cameras_meta[camera_idx]
                transform_matrix = np.array(camera_meta['transform_matrix'], dtype=np.float32)
                focal_x = camera_meta.get('fl_x') or camera_meta.get('focal_length')
                focal_y = camera_meta.get('fl_y') or camera_meta.get('focal_length')
                cx = camera_meta.get('cx', image_width / 2.0)
                cy = camera_meta.get('cy', image_height / 2.0)
                
                print(f"[Camera Selection] focal_x={focal_x}, focal_y={focal_y}, cx={cx}, cy={cy}")
                print(f"[Camera Selection] transform_matrix shape: {transform_matrix.shape}")
                
                # Match frame by initial_image filename to get joint_pos
                image_filename = os.path.basename(initial_image_path)
                for frame in frames:
                    if image_filename in frame.get('file_path', ''):
                        if 'joint_pos' in frame:
                            initial_control = np.array(frame['joint_pos'], dtype=np.float32)
                        break
    
    model = FlowGuidedGaussianDynamicsModel(
        model_path=model_path,
        iteration=iteration,
        control_dim=control_dim,
        image_height=image_height,
        image_width=image_width,
        num_context=2,
        use_sparse_rendering=False, # Disable sparse rendering for now as point tracking needs full image usually
        enable_flow_prediction=False, # We don't use GMFlow prediction
        device=device,
        transform_matrix=transform_matrix,
        focal_x=focal_x,
        focal_y=focal_y,
        cx=cx,
        cy=cy
    )
    
    # 5. Setup MPC Optimizer
    print("Setting up MPC Optimizer...")
    
    # Define Objective
    point_tracking_obj = PointTrackingObjective(
        tracker=tracker,
        weight=tracking_weight,
        rgb_key='rgb',
        goal_key='target_points',
        current_points_key='current_tracked_points'
    )
    
    objective = CombinedObjective({
        'point_tracking': point_tracking_obj
    })
    
    sampler = CorrelatedNoiseSampler(
        a_dim=control_dim,
        beta=0.9,
        horizon=horizon
    )
    
    # Optimizer Selection: CEM vs CEM-GD
    if optimizer_type == 'cem-gd':
        print(f"Using CEM-GD optimizer (hybrid sampling + gradient descent)")
        print(f"  Samples: init={num_samples_init}, replan={num_samples_replan}")
        print(f"  Gradient refinement: top-{num_grad_seqs} seqs, lr={grad_lr}, steps={grad_steps}")
        
        optimizer = CEMGDOptimizer(
            model=model,
            objective=objective,
            sampler=sampler,
            a_dim=control_dim,
            horizon=horizon,
            num_samples_init=num_samples_init,
            num_samples_replan=num_samples_replan,
            elites_frac=0.1,
            opt_iters=opt_iters,
            alpha=0.1,
            verbose=True,
            num_grad_opt_seqs=num_grad_seqs,
            start_lr=grad_lr,
            max_iterations=grad_steps,
            gradient_device=gradient_device,
        )
    else:  # 'cem'
        print(f"Using CEM optimizer (pure sampling)")
        print(f"  Samples: {num_samples}, iterations: {opt_iters}")
        
        optimizer = CEMOptimizer(
            model=model,
            objective=objective,
            sampler=sampler,
            a_dim=control_dim,
            horizon=horizon,
            num_samples=num_samples,
            elites_frac=0.1,
            opt_iters=opt_iters,
            verbose=True
        )
    
    agent = SimplePlanningAgent(
        a_dim=control_dim,
        optimizer=optimizer,
        replan_interval=1,
        use_initial_action=False
    )
    
    # 6. MPC Loop
    print("Starting MPC Loop...")
    
    current_tracked_points = initial_points.copy()
    current_image = initial_image.copy()
    
    observations = []
    actions = []
    
    # Initial observation
    # Render initial state to ensure consistency
    initial_action_tensor = torch.tensor(initial_control, dtype=torch.float32, device=device)
    initial_rendered = model.render_with_control(initial_action_tensor) # (3, H, W)
    initial_rendered_np = initial_rendered.permute(1, 2, 0).cpu().numpy()
    
    obs_history = ObservationList(
        data_dict={'rgb': initial_rendered_np[None]},
        image_shape=(image_height, image_width)
    )
    # Fill context
    for _ in range(model.num_context - 1):
        obs_history.append(obs_history[0])
        
    observations.append(initial_rendered_np)
    actions.append(initial_control.copy())
    
    for step in range(1, num_steps + 1):
        print(f"\nStep {step}/{num_steps}")
        
        # Set Goal
        agent.set_goal({
            'target_points': target_points,
            'current_tracked_points': current_tracked_points
        })
        
        # Plan
        action = agent.act(t=step-1, obs_history=obs_history, state_obs_history=obs_history)
        
        # Clip action to prevent overflow
        action = np.clip(action, -action_limit, action_limit)
        
        actions.append(action)
        print(f"  Action (first 5 dims): [{', '.join([f'{a:.4f}' for a in action[:5]])}...]")
        print(f"  Action magnitude: {np.linalg.norm(action):.4f}")
        
        # Render next frame
        action_tensor = torch.tensor(action, dtype=torch.float32, device=device)
        next_image_tensor = model.render_with_control(action_tensor)
        next_image_np = next_image_tensor.permute(1, 2, 0).cpu().numpy()
        
        # Save rendered image
        Image.fromarray((next_image_np * 255).astype(np.uint8)).save(
            os.path.join(log_dir, f"step_{step:04d}_rendered.png"))
        
        # Update tracked points
        video_stack = np.stack([current_image, next_image_np])
        video_tensor = torch.from_numpy(video_stack).permute(0, 3, 1, 2).unsqueeze(0).to(device).float()
        
        tracks, visibles = tracker.track(video_tensor, current_tracked_points)
        new_points = tracks[0, :, 1, :].cpu().numpy()
        new_visibles = visibles[0, :, 1].cpu().numpy()
        
        # Visualize
        vis_step = visualize_points(next_image_np, new_points, color=(0, 255, 255), radius=2)
        Image.fromarray(vis_step).save(os.path.join(log_dir, f"step_{step:04d}_with_points.png"))
        
        # Compute distance to target
        dist = np.linalg.norm(new_points - target_points, axis=-1).mean()
        print(f"  Avg Distance to Target: {dist:.4f} pixels")
        print(f"  Visible points: {new_visibles.sum()}/{len(new_visibles)}")
        
        # Update state
        current_tracked_points = new_points
        current_image = next_image_np
        
        new_obs = ObservationList(
            data_dict={'rgb': next_image_np[None]},
            image_shape=(image_height, image_width)
        )
        obs_history.append(new_obs)
        observations.append(next_image_np)

    print("Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--initial_image", type=str, required=True)
    parser.add_argument("--target_image", type=str, required=True)
    parser.add_argument("--control_dim", type=int, default=15)
    parser.add_argument("--transforms_json", type=str, default="/home/ubuntu/yyf/4DGaussians/assets/example_transforms.json")
    parser.add_argument("--num_steps", type=int, default=20)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--image_height", type=int, default=480)
    parser.add_argument("--image_width", type=int, default=480)
    parser.add_argument("--device", type=str, default="cuda:3")
    parser.add_argument("--output_dir", type=str, default="./outputs/cotracker_demo")
    
    # Optimizer selection
    parser.add_argument("--optimizer", type=str, default="cem-gd", choices=["cem", "cem-gd"],
                        help="Optimizer type: 'cem' (pure sampling) or 'cem-gd' (hybrid sampling + gradient)")
    
    # CEM parameters
    parser.add_argument("--num_samples", type=int, default=32,
                        help="CEM: Number of samples per iteration (only used with --optimizer=cem)")
    parser.add_argument("--opt_iters", type=int, default=10,
                        help="Number of CEM optimization iterations")
    
    # CEM-GD parameters
    parser.add_argument("--num_samples_init", type=int, default=32,
                        help="CEM-GD: Number of samples for initial planning")
    parser.add_argument("--num_samples_replan", type=int, default=16,
                        help="CEM-GD: Number of samples for replanning steps")
    parser.add_argument("--num_grad_seqs", type=int, default=3,
                        help="CEM-GD: Number of top sequences to refine with gradient descent")
    parser.add_argument("--grad_lr", type=float, default=0.01,
                        help="CEM-GD: Adam learning rate for gradient refinement")
    parser.add_argument("--grad_steps", type=int, default=15,
                        help="CEM-GD: Number of gradient descent iterations")
    parser.add_argument("--gradient_device", type=str, default=None,
                        help="CEM-GD: Device for gradient descent (e.g., 'cuda:2'). If None, uses same device as model.")
    
    # Point sampling and constraints
    parser.add_argument("--num_tracking_points", type=int, default=256)
    parser.add_argument("--sampling_method", type=str, default="motion_mask",
                        choices=["sobel_hybrid", "shi_tomasi", "combined", "texture", "grid", "motion_mask"])
    parser.add_argument("--action_limit", type=float, default=1.0,
                        help="Maximum action magnitude per step")
    parser.add_argument("--resample_motion_mask_per_step", action="store_true", default=False,
                        help="Re-sample motion mask at every MPC step (only with --sampling_method=motion_mask)")
    
    args = parser.parse_args()
    
    args.device = actual_device
    
    if args.device.startswith('cuda'):
        print(f"\n[PyTorch Verification]")
        print(f"  args.device = {args.device}")
        print(f"  torch.cuda.is_available() = {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  torch.cuda.current_device() = {torch.cuda.current_device()}")
            print(f"  torch.cuda.get_device_name(0) = {torch.cuda.get_device_name(0)}")
            print(f"  Physical GPU ID (from CUDA_VISIBLE_DEVICES) = {actual_device_id}")
        print()
    
    run_cotracker_mpc(
        model_path=args.model_path,
        iteration=30000,
        initial_image_path=args.initial_image,
        target_image_path=args.target_image,
        control_dim=args.control_dim,
        transforms_json_path=args.transforms_json,
        num_steps=args.num_steps,
        horizon=args.horizon,
        image_height=args.image_height,
        image_width=args.image_width,
        device=args.device,
        log_dir=args.output_dir,
        optimizer_type=args.optimizer,
        num_samples=args.num_samples,
        opt_iters=args.opt_iters,
        num_samples_init=args.num_samples_init,
        num_samples_replan=args.num_samples_replan,
        num_grad_seqs=args.num_grad_seqs,
        grad_lr=args.grad_lr,
        grad_steps=args.grad_steps,
        gradient_device=args.gradient_device,
        num_tracking_points=args.num_tracking_points,
        sampling_method=args.sampling_method,
        action_limit=args.action_limit,
        resample_motion_mask_per_step=args.resample_motion_mask_per_step,
    )
