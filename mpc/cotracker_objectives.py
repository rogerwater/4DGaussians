from mpc.objectives import Objective
import torch
import numpy as np

class PointTrackingObjective(Objective):
    """
    Reward function based on point tracking using TAPIR (or other trackers).
    Minimizes the distance between tracked points and target points.
    
    Improvements for MPC:
    - Visibility weighting: Downweight occluded/uncertain points
    - Temporal weighting: Prioritize endpoint (final frame) over intermediate frames
    - Smoothness: Optionally penalize large point movements between frames
    """
    def __init__(self, tracker, weight=1.0, rgb_key='rgb', goal_key='target_points', 
                 current_points_key='current_tracked_points',
                 visibility_weight=True, endpoint_weight=3.0, temporal_decay=0.7,
                 smoothness_weight=0.0):
        """
        Args:
            tracker: Point tracker instance (e.g., PointTracker with TAPIR)
            weight: Overall objective weight
            rgb_key: Key for RGB images in prediction dict
            goal_key: Key for target points in goal dict
            current_points_key: Key for current points in goal dict
            visibility_weight: Whether to downweight invisible points
            endpoint_weight: Multiplier for final frame distance (endpoint priority)
            temporal_decay: Decay factor for earlier frames (1.0 = no decay, <1.0 = prefer later frames)
            smoothness_weight: Weight for point motion smoothness penalty (0 = disabled)
        """
        super().__init__(weight)
        self.tracker = tracker
        self.rgb_key = rgb_key
        self.goal_key = goal_key
        self.current_points_key = current_points_key
        self.visibility_weight = visibility_weight
        self.endpoint_weight = endpoint_weight
        self.temporal_decay = temporal_decay
        self.smoothness_weight = smoothness_weight

    def compute_reward(self, prediction, goal):
        """
        Compute reward based on point tracking with advanced weighting strategies.
        
        Args:
            prediction: dict containing:
                - rgb_key: (B, T, 3, H, W) or (B, T, H, W, 3) image sequence
            goal: dict containing:
                - goal_key: (N, 2) target coordinates [x, y]
                - current_points_key: (N, 2) current tracked points [x, y] (start of horizon)
        
        Returns:
            reward: (B, 1, 1)
        """
        if self.tracker is None:
            # Return zero reward if tracker is not available
            B = prediction[self.rgb_key].shape[0]
            device = prediction[self.rgb_key].device
            return torch.zeros((B, 1, 1), device=device)

        rgb_seq = prediction[self.rgb_key]
        
        # Convert numpy to torch if needed
        if isinstance(rgb_seq, np.ndarray):
            # Assume shape (B, T, H, W, 3) from dynamics model
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            rgb_seq = torch.from_numpy(rgb_seq).to(device).float()
            # Permute to (B, T, C, H, W) for tracker
            if rgb_seq.shape[-1] == 3:
                rgb_seq = rgb_seq.permute(0, 1, 4, 2, 3)
        else:
            # Already torch tensor
            device = rgb_seq.device
            # Ensure (B, T, C, H, W) format
            if rgb_seq.shape[-1] == 3:
                rgb_seq = rgb_seq.permute(0, 1, 4, 2, 3)
        
        B, T = rgb_seq.shape[0], rgb_seq.shape[1]

        # Get initial points (current state points)
        if self.current_points_key in goal:
             current_points = goal[self.current_points_key]
        else:
             print(f"Warning: {self.current_points_key} not found in goal. Returning 0 reward.")
             return torch.zeros((B, 1, 1), device=device)

        # Run tracker
        # tracks: (B, N, T, 2)
        # visibles: (B, N, T)
        tracks, visibles = self.tracker.track(rgb_seq, current_points)
        
        if tracks is None:
             return torch.zeros((B, 1, 1), device=device)

        # Compute distance to target
        if self.goal_key in goal:
            target_points = goal[self.goal_key] # (N, 2)
        else:
             print(f"Warning: {self.goal_key} not found in goal. Returning 0 reward.")
             return torch.zeros((B, 1, 1), device=device)
        
        # Ensure target_points is tensor
        if isinstance(target_points, np.ndarray):
            target_points = torch.from_numpy(target_points).to(device)
        elif isinstance(target_points, torch.Tensor):
            target_points = target_points.to(device)
            
        # target_points: (N, 2) -> (1, N, 1, 2) broadcastable to (B, N, T, 2)
        target_points_expanded = target_points.unsqueeze(0).unsqueeze(2)
        
        # Distance (L2)
        # tracks: (B, N, T, 2)
        diff = tracks - target_points_expanded
        dist = torch.norm(diff, dim=-1) # (B, N, T)
        
        # === ADVANCED WEIGHTING STRATEGIES ===
        
        # 1. Visibility weighting: Downweight occluded/invisible points
        if self.visibility_weight:
            # visibles: (B, N, T) boolean
            visibility_mask = visibles.float()
            # Apply visibility mask (0.1 weight for invisible, 1.0 for visible)
            visibility_mask = torch.where(visibility_mask > 0.5, 
                                         torch.ones_like(visibility_mask), 
                                         torch.ones_like(visibility_mask) * 0.1)
            dist = dist * visibility_mask
        
        # 2. Temporal weighting: Prioritize later frames (endpoint priority)
        # Create exponentially increasing weights: early frames get less weight
        # temporal_weights[t] = temporal_decay^(T-1-t)
        # t=0 (early): decay^(T-1), t=T-1 (endpoint): decay^0 = 1.0
        temporal_indices = torch.arange(T, device=device).float()
        temporal_weights = self.temporal_decay ** (T - 1 - temporal_indices)
        temporal_weights = temporal_weights.view(1, 1, T) # (1, 1, T)
        
        # Apply endpoint boost to final frame
        temporal_weights[:, :, -1] *= self.endpoint_weight
        
        # Normalize so mean weight is 1.0 (keeps loss scale consistent)
        temporal_weights = temporal_weights / temporal_weights.mean()
        
        # Apply temporal weights
        weighted_dist = dist * temporal_weights # (B, N, T)
        
        # 3. Smoothness penalty (optional): Penalize large point movements between frames
        smoothness_penalty = 0.0
        if self.smoothness_weight > 0:
            # Compute point velocities: diff between consecutive frames
            point_velocities = tracks[:, :, 1:, :] - tracks[:, :, :-1, :] # (B, N, T-1, 2)
            velocity_magnitudes = torch.norm(point_velocities, dim=-1) # (B, N, T-1)
            # Penalize large velocities (encourages smooth tracking)
            smoothness_penalty = self.smoothness_weight * velocity_magnitudes.mean()
        
        # === FINAL REWARD ===
        
        # Average over points and time
        avg_dist = weighted_dist.mean(dim=(1, 2)) # (B,)
        
        # Reward = -Distance - Smoothness Penalty
        reward = -avg_dist
        if self.smoothness_weight > 0:
            reward = reward - smoothness_penalty
        
        return reward.view(B, 1, 1)
