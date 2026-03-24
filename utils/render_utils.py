import torch
@torch.no_grad()
def get_state_at_time(pc,viewpoint_camera):    
    means3D = pc.get_xyz
    time = torch.tensor(viewpoint_camera.time).to(means3D.device).repeat(means3D.shape[0],1)
    opacity = pc._opacity
    shs = pc.get_features

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = pc._scaling
    rotations = pc._rotation
    cov3D_precomp = None
    
    # Handle action_vec - check if viewpoint_camera has it, otherwise create default
    if hasattr(viewpoint_camera, 'action_vec') and viewpoint_camera.action_vec is not None:
        action_vec = viewpoint_camera.action_vec.to(means3D.device)
        if action_vec.dim() == 1:
            action_vec = action_vec.unsqueeze(0)  # [6] -> [1, 6]
        action_vec = action_vec.repeat(means3D.shape[0], 1)  # [1, 6] -> [N, 6]
    else:
        # Get the actual action dimension from the deformation model
        action_dim = getattr(pc._deformation.action_encoder, 'input_dim', 6)
        action_vec = torch.zeros(means3D.shape[0], action_dim, device=means3D.device)
    
    means3D_final, scales_final, rotations_final, opacity_final, shs_final = pc._deformation(means3D, scales, 
                                                                 rotations, opacity, shs,
                                                                 action_vec)

    return means3D_final, scales_final, rotations_final, opacity_final, shs_final
