#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh
from time import time as get_time
def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None, stage="fine", 
           cam_type = None, is_training = False, iteration = 0, override_action_vec = None):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    
    means3D = pc.get_xyz
    if cam_type != "PanopticSports":
        tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
        tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
        raster_settings = GaussianRasterizationSettings(
            image_height=int(viewpoint_camera.image_height),
            image_width=int(viewpoint_camera.image_width),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=bg_color,
            scale_modifier=scaling_modifier,
            viewmatrix=viewpoint_camera.world_view_transform.cuda(),
            projmatrix=viewpoint_camera.full_proj_transform.cuda(),
            sh_degree=pc.active_sh_degree,
            campos=viewpoint_camera.camera_center.cuda(),
            prefiltered=False,
            debug=pipe.debug
        )
        time = torch.tensor(viewpoint_camera.time).to(means3D.device).repeat(means3D.shape[0],1)
        if override_action_vec is not None:
            action_vec = override_action_vec.to(means3D.device)
            if action_vec.dim() == 1:
                action_vec = action_vec.unsqueeze(0)
            action_vec = action_vec.repeat(means3D.shape[0], 1)
        elif hasattr(viewpoint_camera, 'action_vec') and viewpoint_camera.action_vec is not None:
            action_vec = viewpoint_camera.action_vec.to(means3D.device)
            if action_vec.dim() == 1:
                action_vec = action_vec.unsqueeze(0)  # [6] -> [1, 6]
            action_vec = action_vec.repeat(means3D.shape[0], 1)  # [1, 6] -> [N, 6]
        else:
            action_vec = torch.zeros(means3D.shape[0], 6, device=means3D.device)
    else:
        raster_settings = viewpoint_camera['camera']
        time = torch.tensor(viewpoint_camera['time']).to(means3D.device).repeat(means3D.shape[0],1)
        action_vec = viewpoint_camera.get('action_vec', None)
        if action_vec is None:
            action_vec = torch.zeros(means3D.shape[0], 6, device=means3D.device)
        else:
            action_vec = action_vec.to(means3D.device)
            if action_vec.dim() == 1:
                action_vec = action_vec.unsqueeze(0)
            if action_vec.shape[0] == 1:
                action_vec = action_vec.repeat(means3D.shape[0], 1)
        

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    # means3D = pc.get_xyz
    # add deformation to each points
    # deformation = pc.get_deformation

    
    means2D = screenspace_points
    opacity = pc._opacity
    shs = pc.get_features

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc._scaling
        rotations = pc._rotation
    deformation_point = pc._deformation_table
    if "coarse" in stage:
        means3D_final, scales_final, rotations_final, opacity_final, shs_final = means3D, scales, rotations, opacity, shs
    elif "fine" in stage:
        # time0 = get_time()
        # means3D_deform, scales_deform, rotations_deform, opacity_deform = pc._deformation(means3D[deformation_point], scales[deformation_point], 
        #                                                                  rotations[deformation_point], opacity[deformation_point],
        #                                                                  time[deformation_point])
        means3D_final, scales_final, rotations_final, opacity_final, shs_final = pc._deformation(means3D, scales, 
                                                                 rotations, opacity, shs,
                                                                 action_vec)
    else:
        raise NotImplementedError



    # time2 = get_time()
    # print("asset value:",time2-time1)
    scales_final = pc.scaling_activation(scales_final)
    rotations_final = pc.rotation_activation(rotations_final)
    opacity = pc.opacity_activation(opacity_final)
    
    """DropGaussian Implementation"""
    if is_training and "fine" in stage:
        max_drop_rate = 0.2
        max_iterations = 12000
        current_drop_rate = max_drop_rate * min(iteration / max_iterations, 1.0)
        
        num_gaussians = opacity.shape[0]
        compensation = torch.ones(num_gaussians, dtype=torch.float32, device="cuda")
    
        dropout_layer = torch.nn.Dropout(p=current_drop_rate)
        compensation = dropout_layer(compensation)
        
        opacity = opacity * compensation.unsqueeze(1)
    
    
    # print(opacity.max())
    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    # shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.cuda().repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            pass
            # shs = 
    else:
        colors_precomp = override_color

    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    # time3 = get_time()
    raster_output = rasterizer(
        means3D = means3D_final,
        means2D = means2D,
        shs = shs_final,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales_final,
        rotations = rotations_final,
        cov3D_precomp = cov3D_precomp)
    
    rendered_image = raster_output[0]
    radii = raster_output[1]
    depth = raster_output[2]
    alpha = raster_output[3]
    proj_2D = raster_output[4]
    conic_2D = raster_output[5]
    conic_2D_inv = raster_output[6]
    gs_per_pixel = raster_output[7]
    weight_per_gs_pixel = raster_output[8]
    x_mu = raster_output[9]
    # time4 = get_time()
    # print("rasterization:",time4-time3)
    # breakpoint()
    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii,
            "depth": depth,
            "alpha": alpha,
            "proj_2D": proj_2D,
            "conic_2D": conic_2D,
            "conic_2D_inv": conic_2D_inv,
            "gs_per_pixel": gs_per_pixel,
            "weight_per_gs_pixel": weight_per_gs_pixel,
            "x_mu": x_mu}
