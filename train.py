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
import numpy as np
import random
import os, sys
import torch
import cv2
from random import randint
from utils.loss_utils import l1_loss, ssim, l2_loss, lpips_loss, depth_l1_loss, flow_loss
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams, ModelHiddenParams
from torch.utils.data import DataLoader
from utils.timer import Timer
from utils.loader_utils import FineSampler, get_stamp_list
import lpips
from utils.scene_utils import render_training_image
from time import time
import copy
from gmflow.gmflow import build_gmflow
from gmflow.config import get_cfg as get_gmflow_cfg
from utils.flow_utils import calculate_gs_flow, flow_to_image
from utils.feature_analyzer import HexPlaneAnalyzer
from utils.triplane_film_analyzer import TriPlaneFiLMAnalyzer


to8b = lambda x : (255*np.clip(x.cpu().numpy(),0,1)).astype(np.uint8)

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False
    

def build_frame_pair_index(cameras):
    from collections import defaultdict
    
    trajectory_frames = defaultdict(list)
    for cam in cameras:
        cam_idx = getattr(cam, 'camera_idx', 0)
        sample_idx = getattr(cam, 'sample_idx', 0)
        trajectory_frames[(cam_idx, sample_idx)].append((cam.uid, cam.time))
    
    for key in trajectory_frames:
        trajectory_frames[key].sort(key=lambda x: x[1])
        
    uid_to_next = {}
    for (cam_idx, sample_idx), frames in trajectory_frames.items():
        for i in range(len(frames) - 1):
            current_uid = frames[i][0]
            next_uid = frames[i + 1][0]
            uid_to_next[current_uid] = next_uid
    
    num_trajectories = len(trajectory_frames)
    num_pairs = len(uid_to_next)
    print(f"[Frame Pairing] Found {num_trajectories} trajectories (camera_idx, sample_idx combinations)")
    
    return uid_to_next


def scene_reconstruction(dataset, opt, hyper, pipe, testing_iterations, saving_iterations, 
                         checkpoint_iterations, checkpoint, debug_from,
                         gaussians, scene, stage, tb_writer, train_iter,timer):
    first_iter = 0

    gaussians.training_setup(opt)
    if checkpoint:
        # breakpoint()
        if stage == "coarse" and stage not in checkpoint:
            print("start from fine stage, skip coarse stage.")
            # process is in the coarse stage, but start from fine stage
            return
        if stage in checkpoint: 
            (model_params, first_iter) = torch.load(checkpoint)
            gaussians.restore(model_params, opt)
            
    # add gmflow for dynamic scene flow estimation
    flownet = None
    if getattr(opt, 'use_gmflow', True) and stage == "fine":
        gmflow_cfg = get_gmflow_cfg()
        print(f"[GMFlow] loading checkpoint from {gmflow_cfg.model} (stage={stage})")
        flownet = torch.nn.DataParallel(build_gmflow(gmflow_cfg))
        flownet = flownet.module
        checkpoint_flow = torch.load(gmflow_cfg.model, map_location="cpu")
        weights = checkpoint_flow["model"] if "model" in checkpoint_flow else checkpoint_flow
        flownet.load_state_dict(weights)
        flownet = flownet.cuda()
        flownet.eval()
        print("[GMFlow] loaded successfully and set to eval mode")

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    ema_psnr_for_log = 0.0

    final_iter = train_iter
    
    # Initialize Feature Analyzer for fine stage
    """
    feature_analyzer = None
    if stage == "fine" and hasattr(gaussians, '_deformation'):
        try:
            feature_analyzer = HexPlaneAnalyzer(
                deformation_net=gaussians._deformation,
                config=hyper,
                num_sample_positions=2000,
                num_sample_controls=50,
                enable_tsne=False  # t-SNE很慢，默认关闭
            )
            print("[FeatureAnalyzer] Initialized successfully for fine stage")
        except Exception as e:
            print(f"[FeatureAnalyzer] Failed to initialize: {e}")
            feature_analyzer = None
    """
    feature_analyzer = None
    if stage == "fine" and hasattr(gaussians, '_deformation'):
        try:
            feature_analyzer = TriPlaneFiLMAnalyzer(
                deformation_net=gaussians._deformation,
                config=hyper,
                num_sample_positions=2000,
                num_sample_actions=100,
                enable_tsne=True
            )
            print("[FeatureAnalyzer] Initialized TriPlaneFiLMAnalyzer")
        except Exception as e:
            print(f"[FeatureAnalyzer] Failed: {e}")
    
    progress_bar = tqdm(range(first_iter, final_iter), desc="Training progress")
    first_iter += 1
    # lpips_model = lpips.LPIPS(net="alex").cuda()
    video_cams = scene.getVideoCameras()
    test_cams = scene.getTestCameras()
    train_cams = scene.getTrainCameras()
    
    uid_to_next_uid = build_frame_pair_index(train_cams)
    print(f"[Frame Pairing] Built index for {len(uid_to_next_uid)} frame pairs")
    uid_to_camera = {cam.uid: cam for cam in train_cams}

    if not viewpoint_stack and not opt.dataloader:
        # dnerf's branch
        viewpoint_stack = [i for i in train_cams]
        temp_list = copy.deepcopy(viewpoint_stack)
    # 
    batch_size = opt.batch_size
    print("data loading done")
    if opt.dataloader:
        viewpoint_stack = scene.getTrainCameras()
        if opt.custom_sampler is not None:
            sampler = FineSampler(viewpoint_stack)
            viewpoint_stack_loader = DataLoader(viewpoint_stack, batch_size=batch_size,sampler=sampler,num_workers=16,collate_fn=list)
            random_loader = False
        else:
            viewpoint_stack_loader = DataLoader(viewpoint_stack, batch_size=batch_size,shuffle=True,num_workers=16,collate_fn=list)
            random_loader = True
        loader = iter(viewpoint_stack_loader)
    
    
    # dynerf, zerostamp_init
    # breakpoint()
    if stage == "coarse" and opt.zerostamp_init:
        load_in_memory = True
        # batch_size = 4
        temp_list = get_stamp_list(viewpoint_stack,0)
        viewpoint_stack = temp_list.copy()
    else:
        load_in_memory = False 
                            # 
    count = 0
    for iteration in range(first_iter, final_iter+1):        
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    count +=1
                    viewpoint_index = (count ) % len(video_cams)
                    if (count //(len(video_cams))) % 2 == 0:
                        viewpoint_index = viewpoint_index
                    else:
                        viewpoint_index = len(video_cams) - viewpoint_index - 1
                    # print(viewpoint_index)
                    viewpoint = video_cams[viewpoint_index]
                    custom_cam.time = viewpoint.time
                    # print(custom_cam.time, viewpoint_index, count)
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer, stage=stage, cam_type=scene.dataset_type)["render"]

                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive) :
                    break
            except Exception as e:
                print(e)
                network_gui.conn = None

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera

        # dynerf's branch
        if opt.dataloader and not load_in_memory:
            try:
                viewpoint_cams = next(loader)
            except StopIteration:
                print("reset dataloader into random dataloader.")
                if not random_loader:
                    viewpoint_stack_loader = DataLoader(viewpoint_stack, batch_size=opt.batch_size,shuffle=True,num_workers=32,collate_fn=list)
                    random_loader = True
                loader = iter(viewpoint_stack_loader)

        else:
            idx = 0
            viewpoint_cams = []

            while idx < batch_size :    
                    
                viewpoint_cam = viewpoint_stack.pop(randint(0,len(viewpoint_stack)-1))
                if not viewpoint_stack :
                    viewpoint_stack =  temp_list.copy()
                viewpoint_cams.append(viewpoint_cam)
                idx +=1
            if len(viewpoint_cams) == 0:
                continue
        # print(len(viewpoint_cams))     
        # breakpoint()   
        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True
        images = []
        gt_images = []
        
        next_gt_images = []
        valid_flow_mask = []
        
        radii_list = []
        visibility_filter_list = []
        viewspace_point_tensor_list = []
        
        # extra keys for depth loss
        rendered_depth_list = []
        gt_depth_list = []
        
        # extra keys for computing flow
        alpha_list = []
        proj_2D_list = []
        conic_2D_list = []
        conic_2D_inv_list = []
        gs_per_pixel_list = []
        weight_per_gs_pixel_list = []
        x_mu_list = []
        
        for viewpoint_cam in viewpoint_cams:
            # render_pkg = render(viewpoint_cam, gaussians, pipe, background, stage=stage,cam_type=scene.dataset_type)
            render_pkg = render(viewpoint_cam, gaussians, pipe, background, stage=stage, cam_type=scene.dataset_type, is_training=True, iteration=iteration)
            image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
            
            images.append(image.unsqueeze(0))
            if scene.dataset_type!="PanopticSports":
                gt_image = viewpoint_cam.original_image.cuda()
            else:
                gt_image  = viewpoint_cam['image'].cuda()
            
            gt_images.append(gt_image.unsqueeze(0))
            radii_list.append(radii.unsqueeze(0))
            visibility_filter_list.append(visibility_filter.unsqueeze(0))
            viewspace_point_tensor_list.append(viewspace_point_tensor)
            
            rendered_depth_list.append(render_pkg["depth"])
            gt_depth_list.append(getattr(viewpoint_cam, 'depth', None))
            
            if viewpoint_cam.uid in uid_to_next_uid:
                next_uid = uid_to_next_uid[viewpoint_cam.uid]
                next_cam = uid_to_camera[next_uid]
                
                if next_cam is not None:
                    if scene.dataset_type!="PanopticSports":
                        next_gt_image = next_cam.original_image.cuda()
                    else:
                        next_gt_image = next_cam['image'].cuda()
                    next_gt_images.append(next_gt_image.unsqueeze(0))
                    valid_flow_mask.append(True)
                    
                    if flownet is not None and stage == "fine":
                        with torch.no_grad():
                            render_flow = render(viewpoint_cam, gaussians, pipe, background, stage=stage,cam_type=scene.dataset_type, is_training=False)
                        alpha_list.append(render_flow["alpha"].unsqueeze(0))
                        proj_2D_list.append(render_flow["proj_2D"])
                        conic_2D_list.append(render_flow["conic_2D"])
                        conic_2D_inv_list.append(render_flow["conic_2D_inv"])
                        gs_per_pixel_list.append(render_flow["gs_per_pixel"])
                        weight_per_gs_pixel_list.append(render_flow["weight_per_gs_pixel"])
                        x_mu_list.append(render_flow["x_mu"])
                else:
                    next_gt_images.append(gt_image.unsqueeze(0))
                    valid_flow_mask.append(False)
            else:
                next_gt_images.append(gt_image.unsqueeze(0))
                valid_flow_mask.append(False)
                    

        radii = torch.cat(radii_list,0).max(dim=0).values
        visibility_filter = torch.cat(visibility_filter_list).any(dim=0)
        image_tensor = torch.cat(images,0)
        gt_image_tensor = torch.cat(gt_images,0)
        
        next_gt_image_tensor = torch.cat(next_gt_images, 0)
        valid_flow_mask = torch.tensor(valid_flow_mask, device="cuda")
        
        # Loss
        # breakpoint()
        Ll1 = l1_loss(image_tensor, gt_image_tensor[:,:3,:,:])

        psnr_ = psnr(image_tensor, gt_image_tensor).mean().double()
        # norm
        
        loss = Ll1
        if stage == "fine" and hyper.time_smoothness_weight != 0:
            # tv_loss = 0
            tv_loss = gaussians.compute_regulation(hyper.time_smoothness_weight, hyper.l1_time_planes, hyper.plane_tv_weight)
            loss += tv_loss
        if opt.lambda_dssim != 0:
            ssim_loss = ssim(image_tensor,gt_image_tensor)
            loss += opt.lambda_dssim * (1.0-ssim_loss)
        # if opt.lambda_lpips !=0:
        #     lpipsloss = lpips_loss(image_tensor,gt_image_tensor,lpips_model)
        #     loss += opt.lambda_lpips * lpipsloss

        # depth loss
        L_depth = None
        
        if opt.use_depth_loss and stage == "fine":
            depth_losses = []
            
            for batch_idx, viewpoint_cam in enumerate(viewpoint_cams):
                gt_depth = getattr(viewpoint_cam, 'depth', None)
                if gt_depth is None:
                    continue
                
                gt_depth = gt_depth.cuda() / opt.depth_scale
                
                rendered_depth = rendered_depth_list[batch_idx]
                
                if rendered_depth.dim() == 2:
                    rendered_depth = rendered_depth.unsqueeze(0)
                if gt_depth.dim() == 2:
                    gt_depth = gt_depth.unsqueeze(0)
                    
                valid_mask = gt_depth > 0
                
                d_loss = depth_l1_loss(rendered_depth, gt_depth, valid_mask)
                
                if not torch.isnan(d_loss) and d_loss > 0:
                    depth_losses.append(d_loss)
            
            if len(depth_losses) > 0:
                L_depth = torch.stack(depth_losses).mean()
                loss = loss + opt.lambda_depth * L_depth
        
        # flow loss
        Lflow = None
        gs_flow_list = []
        flow_2d_gt_list = []
        flow_viewpoint_list = []
        
        if flownet is not None and stage == "fine":
            num_valid = valid_flow_mask.sum().item()
            if num_valid > 0:
                H, W = gt_image_tensor.shape[-2], gt_image_tensor.shape[-1]
                flow_losses = []
                
                flow_idx = 0
                for batch_idx, viewpoint_cam in enumerate(viewpoint_cams):
                    if not valid_flow_mask[batch_idx]:
                        continue
                    
                    gt_image_t = gt_images[batch_idx]  # [1, C, H, W]
                    next_gt_image_t = next_gt_images[batch_idx]  # [1, C, H, W] 
                    
                    # 1. calculate GT flow
                    with torch.no_grad():
                        flow_preds = flownet(gt_image_t * 255, next_gt_image_t * 255)
                        
                        flow_high = flow_preds[0]
                        H_flow, W_flow = flow_high.shape[-2:]
                        
                        if (H_flow == H) and (W_flow == W):
                            flow_2d_gt = flow_high.squeeze(0)  # [2, H, W]
                        else:
                            flow_2d_gt = torch.nn.functional.interpolate(flow_high, size=(H, W), mode="bilinear").squeeze(0)
                            flow_2d_gt[0] *= float(W) / float(W_flow)
                            flow_2d_gt[1] *= float(H) / float(H_flow)
                        flow_2d_gt = flow_2d_gt.cuda()
                        
                    # 2. get control_vec of next frame
                    next_uid = uid_to_next_uid[viewpoint_cam.uid]
                    next_cam = uid_to_camera[next_uid]
                    next_control_vec = next_cam.control_vec
                    
                    # 3. render with control_vec of next frame
                    with torch.no_grad():
                        render_next = render(viewpoint_cam, gaussians, pipe, background, 
                                            stage=stage, cam_type=scene.dataset_type,
                                            is_training=False,
                                            override_control_vec=next_control_vec)
                        next_proj_2D = render_next["proj_2D"]
                        next_conic_2D = render_next["conic_2D"]
                    
                    # 4. calculate gs flow
                    gs_flow = calculate_gs_flow(
                        gs_per_pixel_list[flow_idx],
                        weight_per_gs_pixel_list[flow_idx],
                        next_conic_2D,
                        conic_2D_inv_list[flow_idx],
                        proj_2D_list[flow_idx],
                        next_proj_2D,
                        x_mu_list[flow_idx]
                    )
                    
                    # 5. calculate gs loss
                    gs_flow_list.append(gs_flow.detach().cpu())
                    flow_2d_gt_list.append(flow_2d_gt.detach().cpu())
                    flow_viewpoint_list.append(viewpoint_cam)
                    
                    flow_loss_val = flow_loss(gs_flow, flow_2d_gt.detach(), H, W)
                    flow_losses.append(flow_loss_val)
                    
                    flow_idx += 1
                
                if len(flow_losses) > 0:
                    Lflow = torch.stack(flow_losses).mean()
                    loss = loss + opt.flow_loss_weight * Lflow
             
                    
        loss.backward()
        if torch.isnan(loss).any():
            print("loss is nan,end training, reexecv program now.")
            os.execv(sys.executable, [sys.executable] + sys.argv)
        viewspace_point_tensor_grad = torch.zeros_like(viewspace_point_tensor)
        for idx in range(0, len(viewspace_point_tensor_list)):
            viewspace_point_tensor_grad = viewspace_point_tensor_grad + viewspace_point_tensor_list[idx].grad
        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            ema_psnr_for_log = 0.4 * psnr_ + 0.6 * ema_psnr_for_log
            total_point = gaussians._xyz.shape[0]
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}",
                                          "psnr": f"{psnr_:.{2}f}",
                                          "point":f"{total_point}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            timer.pause()
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, [pipe, background], stage, scene.dataset_type, 
                            gs_flow_list, flow_2d_gt_list, flow_viewpoint_list, L_depth=L_depth, Lflow=Lflow)
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration, stage)
                
                # Analyze HexPlane features when saving checkpoint
                if feature_analyzer is not None and stage == "fine":
                    # print(f"\n[ITER {iteration}] Analyzing HexPlane features...")
                    # feature_analyzer.analyze_and_log(tb_writer, iteration, scene)
                    print(f"\n[ITER {iteration}] Analyzing TriPlane+FiLM features...")
                    feature_analyzer.analyze_and_log(tb_writer, iteration, scene)
            if dataset.render_process:
                if (iteration < 1000 and iteration % 10 == 9) \
                    or (iteration < 3000 and iteration % 50 == 49) \
                        or (iteration < 60000 and iteration %  100 == 99) :
                    # breakpoint()
                        render_training_image(scene, gaussians, [test_cams[iteration%len(test_cams)]], render, pipe, background, stage+"test", iteration,timer.get_elapsed_time(),scene.dataset_type)
                        render_training_image(scene, gaussians, [train_cams[iteration%len(train_cams)]], render, pipe, background, stage+"train", iteration,timer.get_elapsed_time(),scene.dataset_type)
                        # render_training_image(scene, gaussians, train_cams, render, pipe, background, stage+"train", iteration,timer.get_elapsed_time(),scene.dataset_type)

                    # total_images.append(to8b(temp_image).transpose(1,2,0))
            timer.start()
            # Densification
            if iteration < opt.densify_until_iter :
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor_grad, visibility_filter)

                if stage == "coarse":
                    opacity_threshold = opt.opacity_threshold_coarse
                    densify_threshold = opt.densify_grad_threshold_coarse
                else:    
                    opacity_threshold = opt.opacity_threshold_fine_init - iteration*(opt.opacity_threshold_fine_init - opt.opacity_threshold_fine_after)/(opt.densify_until_iter)  
                    densify_threshold = opt.densify_grad_threshold_fine_init - iteration*(opt.densify_grad_threshold_fine_init - opt.densify_grad_threshold_after)/(opt.densify_until_iter )  
                if  iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0 and gaussians.get_xyz.shape[0]<360000:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    
                    gaussians.densify(densify_threshold, opacity_threshold, scene.cameras_extent, size_threshold, 5, 5, scene.model_path, iteration, stage)
                if  iteration > opt.pruning_from_iter and iteration % opt.pruning_interval == 0 and gaussians.get_xyz.shape[0]>200000:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None

                    gaussians.prune(densify_threshold, opacity_threshold, scene.cameras_extent, size_threshold)
                    
                # if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0 :
                if iteration % opt.densification_interval == 0 and gaussians.get_xyz.shape[0]<360000 and opt.add_point:
                    gaussians.grow(5,5,scene.model_path,iteration,stage)
                    # torch.cuda.empty_cache()
                if iteration % opt.opacity_reset_interval == 0:
                    print("reset opacity")
                    gaussians.reset_opacity()
                    
            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" +f"_{stage}_" + str(iteration) + ".pth")


def training(dataset, hyper, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, expname):
    # first_iter = 0
    tb_writer = prepare_output_and_logger(expname)
    gaussians = GaussianModel(dataset.sh_degree, hyper)
    dataset.model_path = args.model_path
    timer = Timer()
    scene = Scene(dataset, gaussians, load_coarse=None)
    timer.start()
    scene_reconstruction(dataset, opt, hyper, pipe, testing_iterations, saving_iterations,
                             checkpoint_iterations, checkpoint, debug_from,
                             gaussians, scene, "coarse", tb_writer, opt.coarse_iterations,timer)
    scene_reconstruction(dataset, opt, hyper, pipe, testing_iterations, saving_iterations,
                         checkpoint_iterations, checkpoint, debug_from,
                         gaussians, scene, "fine", tb_writer, opt.iterations,timer)


def prepare_output_and_logger(expname):    
    if not args.model_path:
        # if os.getenv('OAR_JOB_ID'):
        #     unique_str=os.getenv('OAR_JOB_ID')
        # else:
        #     unique_str = str(uuid.uuid4())
        unique_str = expname

        args.model_path = os.path.join("./output/", unique_str)
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer


def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, 
                    renderFunc, renderArgs, stage, dataset_type, gs_flow_list=None, flow_2d_gt_list=None, flow_viewpoint_list=None,
                    L_depth=None, Lflow=None):
    if tb_writer:
        tb_writer.add_scalar(f'{stage}/train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar(f'{stage}/train_loss_patchestotal_loss', loss.item(), iteration)
        tb_writer.add_scalar(f'{stage}/iter_time', elapsed, iteration)
        
        if L_depth is not None:
            tb_writer.add_scalar(f'{stage}/train_loss_patches/depth_loss', L_depth.item(), iteration)
        
        if Lflow is not None:
            tb_writer.add_scalar(f'{stage}/train_loss_patches/flow_loss', Lflow.item(), iteration)
        
    
    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()

        # Save optical flow visualizations of training viewpoints
        if gs_flow_list is not None and len(gs_flow_list) > 0 and stage == "fine":
            flow_save_dir = os.path.join(scene.model_path, f"flow_vis_iter{iteration}")
            os.makedirs(flow_save_dir, exist_ok=True)
            print(f"\n[Flow Visualization] Saving {len(gs_flow_list)} training flow pairs to {flow_save_dir}")
            
            flow_errors = []
            for idx, (gs_flow, flow_2d_gt, viewpoint) in enumerate(zip(gs_flow_list, flow_2d_gt_list, flow_viewpoint_list)):
                H, W = gs_flow.shape[1], gs_flow.shape[2]
                gs_flow_normalized = gs_flow.clone()
                gs_flow_normalized[0] = gs_flow[0] / W
                gs_flow_normalized[1] = gs_flow[1] / H 
                
                flow_gt_normalized = flow_2d_gt.clone()
                flow_gt_normalized[0] = flow_2d_gt[0] / W
                flow_gt_normalized[1] = flow_2d_gt[1] / H
                
                flow_error = torch.abs(gs_flow_normalized - flow_gt_normalized).mean().item()
                flow_errors.append(flow_error)
                
                flow_gt_vis = flow_to_image(flow_gt_normalized.permute(1, 2, 0).cpu().numpy(), convert_to_bgr=True)
                gs_flow_vis = flow_to_image(gs_flow_normalized.permute(1, 2, 0).cpu().numpy(), convert_to_bgr=True)
                
                flow_diff = torch.abs(gs_flow_normalized - flow_gt_normalized) * 10
                flow_diff_vis = flow_to_image(flow_diff.permute(1, 2, 0).cpu().numpy(), convert_to_bgr=True)
                
                save_name = f"train_{viewpoint.image_name}"
                cv2.imwrite(os.path.join(flow_save_dir, f"{save_name}_flow_gt.png"), flow_gt_vis)
                cv2.imwrite(os.path.join(flow_save_dir, f"{save_name}_gs_flow.png"), gs_flow_vis)
                cv2.imwrite(os.path.join(flow_save_dir, f"{save_name}_flow_diff.png"), flow_diff_vis)
                
                np.save(
                    os.path.join(flow_save_dir, f"{save_name}_flow_gt.npy"),
                    flow_2d_gt.numpy()
                )
                np.save(
                    os.path.join(flow_save_dir, f"{save_name}_gs_flow.npy"),
                    gs_flow.numpy()
                )
                np.save(
                    os.path.join(flow_save_dir, f"{save_name}_gs_flow_normalized.npy"), 
                    gs_flow_normalized.numpy()
                )
                
            if len(flow_errors) > 0:
                avg_error = sum(flow_errors) / len(flow_errors)
                print(f"[Flow] Training batch flow error: {avg_error:.6f}")

                if tb_writer:
                    tb_writer.add_scalar(f'{stage}/flow/avg_flow_error', avg_error, iteration)
                    
                    for viz_idx in range(min(3, len(gs_flow_list))):
                        gs_flow = gs_flow_list[viz_idx]
                        flow_2d_gt = flow_2d_gt_list[viz_idx]
                        viewpoint = flow_viewpoint_list[viz_idx]
                        
                        H, W = gs_flow.shape[1], gs_flow.shape[2]
                        gs_flow_norm = gs_flow.clone()
                        gs_flow_norm[0] = gs_flow[0] / W
                        gs_flow_norm[1] = gs_flow[1] / H
                        
                        flow_gt_norm = flow_2d_gt.clone()
                        flow_gt_norm[0] = flow_2d_gt[0] / W
                        flow_gt_norm[1] = flow_2d_gt[1] / H
                        
                        flow_gt_vis = flow_to_image(flow_gt_norm.permute(1, 2, 0).cpu().numpy(), convert_to_bgr=False)
                        gs_flow_vis = flow_to_image(gs_flow_norm.permute(1, 2, 0).cpu().numpy(), convert_to_bgr=False)
                        
                        flow_diff = torch.abs(gs_flow_norm - flow_gt_norm) * 10
                        flow_diff_vis = flow_to_image(flow_diff.permute(1, 2, 0).cpu().numpy(), convert_to_bgr=False)
                        
                        flow_gt_tensor = torch.from_numpy(flow_gt_vis).permute(2, 0, 1).float() / 255.0
                        gs_flow_tensor = torch.from_numpy(gs_flow_vis).permute(2, 0, 1).float() / 255.0
                        flow_diff_tensor = torch.from_numpy(flow_diff_vis).permute(2, 0, 1).float() / 255.0
                        
                        tb_writer.add_image(f'{stage}/flow/view_{viewpoint.image_name}/gt_flow', flow_gt_tensor, iteration)
                        tb_writer.add_image(f'{stage}/flow/view_{viewpoint.image_name}/gs_flow', gs_flow_tensor, iteration)
                        tb_writer.add_image(f'{stage}/flow/view_{viewpoint.image_name}/flow_diff', flow_diff_tensor, iteration)
             
        validation_configs = ({'name': 'test', 'cameras' : [scene.getTestCameras()[idx % len(scene.getTestCameras())] for idx in range(10, 5000, 299)]},
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(10, 5000, 299)]})
                
        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians,stage=stage, cam_type=dataset_type, *renderArgs)["render"], 0.0, 1.0)
                    if dataset_type == "PanopticSports":
                        gt_image = torch.clamp(viewpoint["image"].to("cuda"), 0.0, 1.0)
                    else:
                        gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    try:
                        if tb_writer and (idx < 5):
                            tb_writer.add_images(stage + "/"+config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                            if iteration == testing_iterations[0]:
                                tb_writer.add_images(stage + "/"+config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    except:
                        pass
                    l1_test += l1_loss(image, gt_image).mean().double()
                    # mask=viewpoint.mask
                    
                    psnr_test += psnr(image, gt_image, mask=None).mean().double()                 
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                # print("sh feature",scene.gaussians.get_features.shape)
                if tb_writer:
                    tb_writer.add_scalar(stage + "/"+config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(stage+"/"+config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram(f"{stage}/scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            
            tb_writer.add_scalar(f'{stage}/total_points', scene.gaussians.get_xyz.shape[0], iteration)
            tb_writer.add_scalar(f'{stage}/deformation_rate', scene.gaussians._deformation_table.sum()/scene.gaussians.get_xyz.shape[0], iteration)
            tb_writer.add_histogram(f"{stage}/scene/motion_histogram", scene.gaussians._deformation_accum.mean(dim=-1)/100, iteration,max_bins=500)
            
            if L_depth is not None:
                tb_writer.add_scalar(f'{stage}/eval/depth_loss', L_depth.item(), iteration)
            if Lflow is not None:
                tb_writer.add_scalar(f'{stage}/eval/flow_loss', Lflow.item(), iteration)
        
        torch.cuda.empty_cache()
        
    
def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True
if __name__ == "__main__":
    # Set up command line argument parser
    # torch.set_default_tensor_type('torch.FloatTensor')
    torch.cuda.empty_cache()
    parser = ArgumentParser(description="Training script parameters")
    setup_seed(6666)
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    hp = ModelHiddenParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000, 11000, 12000, 13000, 14000, 15000, 16000, 17000, 18000, 19000, 20000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000, 11000, 12000, 13000, 14000, 15000, 16000, 17000, 18000, 19000, 20000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--expname", type=str, default = "")
    parser.add_argument("--configs", type=str, default = "")
    
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    if args.configs:
        import mmcv
        from utils.params_utils import merge_hparams
        config = mmcv.Config.fromfile(args.configs)
        args = merge_hparams(args, config)
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), hp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from, args.expname)

    # All done
    print("\nTraining complete.")
