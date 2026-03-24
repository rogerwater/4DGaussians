import torch
import torch.nn.functional as F
from utils.graphics_utils import fov2focal


def lift_flow_to_3d(flow_2d, depth_t, depth_t1, viewpoint_cam, next_cam):
    H, W = flow_2d.shape[1:]
    device = flow_2d.device
    
    focal_x = fov2focal(viewpoint_cam.FoVx, W)
    focal_y = fov2focal(viewpoint_cam.FoVy, H)
    cx = W / 2.0
    cy = H / 2.0
    
    K = torch.tensor([
        [focal_x, 0, cx],
        [0, focal_y, cy],
        [0, 0, 1]
    ], device=device, dtype=torch.float32)
    
    K_inv = torch.inverse(K)
    
    v_coords, u_coords = torch.meshgrid(
        torch.arange(H, device=device, dtype=torch.float32),
        torch.arange(W, device=device, dtype=torch.float32),
        indexing='ij'
    )
    
    uv1 = torch.stack([u_coords, v_coords, torch.ones_like(u_coords)], dim=-1)
    
    xyz_cam_t = (K_inv @ uv1.unsqueeze(-1)).squeeze(-1) * depth_t.unsqueeze(-1)
    
    c2w_t = viewpoint_cam.world_view_transform.inverse().cuda()
    
    xyz_cam_t_homo = torch.cat([
        xyz_cam_t,
        torch.ones(H, W, 1, device=device)
    ], dim=-1)
    
    xyz_world_t = (c2w_t @ xyz_cam_t_homo.unsqueeze(-1)).squeeze(-1)[..., :3]
    
    u_next = u_coords + flow_2d[0]  # [H, W]
    v_next = v_coords + flow_2d[1]
    
    valid_mask = (u_next >= 0) & (u_next < W - 1) & \
                 (v_next >= 0) & (v_next < H - 1) & \
                 (depth_t > 0) & (depth_t1 > 0)
    
    u_next_norm = 2.0 * u_next / (W - 1) - 1.0
    v_next_norm = 2.0 * v_next / (H - 1) - 1.0
    grid = torch.stack([u_next_norm, v_next_norm], dim=-1).unsqueeze(0)  # [1, H, W, 2]
    
    depth_t1_warped = F.grid_sample(
        depth_t1.unsqueeze(0).unsqueeze(0),  # [1, 1, H, W]
        grid,
        mode='bilinear',
        padding_mode='zeros',
        align_corners=True
    ).squeeze()  # [H, W]
    
    uv1_next = torch.stack([u_next, v_next, torch.ones_like(u_next)], dim=-1)
    xyz_cam_t1 = (K_inv @ uv1_next.unsqueeze(-1)).squeeze(-1) * depth_t1_warped.unsqueeze(-1)
    
    c2w_t1 = next_cam.world_view_transform.inverse().cuda()
    xyz_cam_t1_homo = torch.cat([
        xyz_cam_t1,
        torch.ones(H, W, 1, device=device)
    ], dim=-1)
    xyz_world_t1 = (c2w_t1 @ xyz_cam_t1_homo.unsqueeze(-1)).squeeze(-1)[..., :3]
    
    motion_3d = xyz_world_t1 - xyz_world_t  # [H, W, 3]
    
    depth_diff = torch.abs(depth_t1_warped - depth_t)
    depth_threshold = 0.1 * depth_t 
    valid_mask = valid_mask & (depth_diff < depth_threshold)
    
    motion_3d = motion_3d * valid_mask.unsqueeze(-1).float()
    
    return motion_3d, valid_mask


def compute_motion_gradients_from_3d(gaussians_xyz, motion_3d, valid_mask,
                                     gs_per_pixel, weight_per_gs_pixel):
    N = gaussians_xyz.shape[0]
    K, H, W = gs_per_pixel.shape
    device = gaussians_xyz.device
    
    motion_norm = torch.norm(motion_3d, dim=-1)  # [H, W]
    motion_norm = motion_norm * valid_mask.float()
    
    motion_magnitude = torch.zeros(N, device=device)
    motion_direction = torch.zeros(N, 3, device=device)
    motion_count = torch.zeros(N, device=device)
    
    gs_indices = gs_per_pixel.reshape(-1)  # [K*H*W]
    weights = weight_per_gs_pixel.reshape(-1)  # [K*H*W]
    
    motion_3d_expanded = motion_3d.unsqueeze(0).expand(K, -1, -1, -1).reshape(-1, 3)  # [K*H*W, 3]
    motion_norm_expanded = motion_norm.unsqueeze(0).expand(K, -1, -1).reshape(-1)  # [K*H*W]
    valid_expanded = valid_mask.unsqueeze(0).expand(K, -1, -1).reshape(-1).float()  # [K*H*W]
    
    valid_gs = gs_indices >= 0
    gs_indices = gs_indices[valid_gs].long()  # 转换为 int64
    weights = weights[valid_gs]
    motion_3d_expanded = motion_3d_expanded[valid_gs]
    motion_norm_expanded = motion_norm_expanded[valid_gs]
    valid_expanded = valid_expanded[valid_gs]
    
    weighted_motion = motion_norm_expanded * weights * valid_expanded
    motion_magnitude.scatter_add_(0, gs_indices, weighted_motion)
    
    weighted_direction = motion_3d_expanded * (weights * valid_expanded).unsqueeze(-1)
    motion_direction.scatter_add_(0, gs_indices.unsqueeze(-1).expand(-1, 3), weighted_direction)
    
    motion_count.scatter_add_(0, gs_indices, weights * valid_expanded)
    
    motion_magnitude = motion_magnitude / (motion_count + 1e-7)
    motion_direction = motion_direction / (motion_count.unsqueeze(-1) + 1e-7)
    
    direction_norm = torch.norm(motion_direction, dim=-1, keepdim=True)
    motion_direction = motion_direction / (direction_norm + 1e-7)
    
    return motion_magnitude, motion_direction

def visualize_motion_field(motion_3d, valid_mask, save_path=None):
    import numpy as np
    
    motion_norm = torch.norm(motion_3d, dim=-1)  # [H, W]
    motion_norm = motion_norm * valid_mask.float()
    
    motion_vis = motion_norm.cpu().numpy()
    
    if motion_vis.max() > 0:
        motion_vis = motion_vis / motion_vis.max()
    
    if save_path is not None:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 8))
        plt.imshow(motion_vis, cmap='jet')
        plt.colorbar(label='Motion Magnitude (normalized)')
        plt.title('3D Motion Field Visualization')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"[Motion] Saved visualization to {save_path}")
    
    return motion_vis