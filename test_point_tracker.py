
import os
import sys
import numpy as np
import torch
import cv2
from pathlib import Path

# Add current directory to path so we can import mpc
sys.path.insert(0, os.getcwd())

from mpc.point_tracker import PointTracker

def create_moving_dot_video(frames=5, height=256, width=256, dot_radius=5):
    # Create a synthetic video of a moving white dot on black background
    video = []
    gt_tracks = []
    
    start_pos = np.array([50, 50], dtype=np.float32)
    velocity = np.array([20, 10], dtype=np.float32)
    
    for t in range(frames):
        img = np.zeros((height, width, 3), dtype=np.uint8)
        pos = start_pos + t * velocity
        cv2.circle(img, (int(pos[0]), int(pos[1])), dot_radius, (255, 255, 255), -1)
        video.append(img)
        gt_tracks.append(pos)
        
    return np.stack(video), np.stack(gt_tracks)

def test_tracker():
    print("Testing PointTracker integration with TAPIR...")
    
    # 1. Initialize Tracker
    try:
        tracker = PointTracker(device='cuda:0')
        print("Tracker initialized successfully.")
    except Exception as e:
        print(f"Failed to initialize tracker: {e}")
        import traceback
        traceback.print_exc()
        return

    # 2. Create Synthetic Data
    frames = 5
    video_np, gt_tracks = create_moving_dot_video(frames=frames)
    
    # Convert to tensor: (B, T, C, H, W)
    # create_moving_dot_video returns (T, H, W, C)
    video_tensor = torch.from_numpy(video_np).permute(0, 3, 1, 2).unsqueeze(0).float() / 255.0
    video_tensor = video_tensor.to('cuda:0')
    
    print(f"Video shape: {video_tensor.shape}")
    
    # 3. Define Query Points (Track the center of the dot at t=0)
    initial_points = gt_tracks[0:1] # (1, 2) [x, y]
    print(f"Tracking point starting at: {initial_points}")
    
    # 4. Run Tracking
    try:
        tracks, visibles = tracker.track(video_tensor, initial_points)
        print("Tracking complete.")
    except Exception as e:
        print(f"Tracking failed: {e}")
        import traceback
        traceback.print_exc()
        return

    if tracks is None:
        print("Tracker returned None (probably TAPNET_AVAILABLE=False).")
        return

    # 5. Analyze Results
    # tracks: (1, 1, T, 2) -> (T, 2)
    pred_tracks = tracks[0, 0, :, :].cpu().numpy()
    
    print("\nResults:")
    for t in range(frames):
        gt = gt_tracks[t]
        pred = pred_tracks[t]
        err = np.linalg.norm(gt - pred)
        print(f"Frame {t}: GT={gt}, Pred={pred}, Error={err:.2f}")
        
    # Check if error is low (TAPIR should be very accurate on this simple case)
    mean_err = np.linalg.norm(gt_tracks - pred_tracks, axis=1).mean()
    print(f"\nMean Tracking Error: {mean_err:.2f}")
    
    if mean_err < 5.0:
        print("SUCCESS: Tracking error is low.")
    else:
        print("WARNING: Tracking error is high.")

if __name__ == "__main__":
    test_tracker()
