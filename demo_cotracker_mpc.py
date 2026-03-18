#!/usr/bin/env python3
"""
Point Tracking MPC Demo using TAPIR (from im2flow2act)
"""

import os
import sys
import argparse
import numpy as np
import torch
from pathlib import Path
from PIL import Image
import cv2

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

# Import PointTracker first to setup JAX env
from mpc.point_tracker import PointTracker
from mpc.cotracker_objectives import PointTrackingObjective

from mpc.flow_guided_gaussian_model import FlowGuidedGaussianDynamicsModel
from mpc.cem import CEMOptimizer
from mpc.sampler import CorrelatedNoiseSampler
from mpc.objectives import CombinedObjective
from mpc.utils import ObservationList
from mpc.agent import SimplePlanningAgent

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
    num_steps: int = 25,
    horizon: int = 10,
    num_samples: int = 32,
    opt_iters: int = 10,
    num_tracking_points: int = 256,
    tracking_weight: float = 1.0,
    image_height: int = 256,
    image_width: int = 256,
    device: str = "cuda:0",
    log_dir: str = "./outputs/cotracker_test",
    transforms_json_path: str = None,
):
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
    # Sample points on initial image
    initial_points = sample_grid_points(image_height, image_width, num_points=num_tracking_points)
    
    # Visualize initial points
    vis_initial = visualize_points(initial_image, initial_points, color=(255, 0, 0))
    Image.fromarray(vis_initial).save(os.path.join(log_dir, "initial_points.png"))
    
    # Find Target Points (Offline Tracking)
    print("Computing target points by tracking from Initial -> Target...")
    # Prepare video tensor: (1, 2, 3, H, W)
    video_stack = np.stack([initial_image, target_image]) # (2, H, W, 3)
    video_tensor = torch.from_numpy(video_stack).permute(0, 3, 1, 2).unsqueeze(0).to(device).float()
    
    # Track
    tracks, visibles = tracker.track(video_tensor, initial_points)
    # tracks: (1, N, 2, 2) -> [batch, points, time, xy]
    if tracks is None:
        print("Error: Tracker failed. Aborting.")
        return

    # Target points are at t=1 (second frame)
    target_points = tracks[0, :, 1, :].cpu().numpy() # (N, 2)
    target_visibles = visibles[0, :, 1].cpu().numpy() # (N,)
    
    # Filter invisible points? Maybe keep them but they might be unreliable.
    # For now, keep all.
    
    # Visualize target points
    vis_target = visualize_points(target_image, target_points, color=(0, 255, 0))
    Image.fromarray(vis_target).save(os.path.join(log_dir, "target_points_tracked.png"))
    
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
        
        # Camera
        cameras_meta = transforms_data.get('cameras', [])
        if cameras_meta:
            camera_meta = cameras_meta[0]
            transform_matrix = np.array(camera_meta['transform_matrix'], dtype=np.float32)
            focal_x = camera_meta.get('fl_x') or camera_meta.get('focal_length')
            focal_y = camera_meta.get('fl_y') or camera_meta.get('focal_length')
            cx = camera_meta.get('cx', image_width / 2.0)
            cy = camera_meta.get('cy', image_height / 2.0)
            
        # Initial Control
        frames = transforms_data.get('frames', [])
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
    
    sampler = CorrelatedNoiseSampler(a_dim=control_dim, horizon=horizon)
    
    optimizer = CEMOptimizer(
        model=model,
        objective=objective,
        sampler=sampler,
        a_dim=control_dim,
        horizon=horizon,
        num_samples=num_samples,
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
        
        # Set Goal (Current tracked points + Final Target)
        # We need to update current_tracked_points in the goal
        agent.set_goal(ObservationList(
            data_dict={
                'target_points': target_points[None], # (1, N, 2)
                'current_tracked_points': current_tracked_points[None] # (1, N, 2)
            },
            image_shape=(image_height, image_width)
        ))
        
        # Plan
        action = agent.act(t=step-1, obs_history=obs_history)
        actions.append(action)
        print(f"  Action: {action[:4]}...")
        
        # Execute
        action_tensor = torch.tensor(action, dtype=torch.float32, device=device)
        next_image_tensor = model.render_with_control(action_tensor)
        next_image_np = next_image_tensor.permute(1, 2, 0).cpu().numpy()
        
        # Update Tracked Points (Real execution tracking)
        # Track from Current Image -> Next Image
        # video: [Current, Next]
        video_stack = np.stack([current_image, next_image_np]) # (2, H, W, 3)
        video_tensor = torch.from_numpy(video_stack).permute(0, 3, 1, 2).unsqueeze(0).to(device).float()
        
        tracks, _ = tracker.track(video_tensor, current_tracked_points)
        # tracks: (1, N, 2, 2) -> points at t=1
        new_points = tracks[0, :, 1, :].cpu().numpy()
        
        # Visualize Step
        vis_step = visualize_points(next_image_np, new_points, color=(0, 255, 255))
        Image.fromarray(vis_step).save(os.path.join(log_dir, f"step_{step:04d}.png"))
        
        # Compute distance to target
        dist = np.linalg.norm(new_points - target_points, axis=-1).mean()
        print(f"  Avg Distance to Target: {dist:.4f}")
        
        # Update state
        current_tracked_points = new_points
        current_image = next_image_np
        obs_history.append({'rgb': next_image_np})
        observations.append(next_image_np)

    print("Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--initial_image", type=str, required=True)
    parser.add_argument("--target_image", type=str, required=True)
    parser.add_argument("--control_dim", type=int, default=15)
    parser.add_argument("--transforms_json", type=str, default="/home/ubuntu/yyf/4DGaussians/assets/example_transforms.json")
    args = parser.parse_args()
    
    run_cotracker_mpc(
        model_path=args.model_path,
        iteration=30000, # default assumption
        initial_image_path=args.initial_image,
        target_image_path=args.target_image,
        control_dim=args.control_dim,
        transforms_json_path=args.transforms_json
    )
