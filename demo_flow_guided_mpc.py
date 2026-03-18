#!/usr/bin/env python3
"""
光流引导的4DGaussians MPC控制 - 完整示例

从初始状态到目标状态，通过MPC规划控制序列并渲染整个过程
从transforms.json读取初始帧的joint_pos作为初始状态，基于此进行光流引导的MPC规划
"""

import os
import sys
import argparse
import numpy as np
import torch
from pathlib import Path
from PIL import Image
from typing import Tuple

# 必须在导入torch之前处理CUDA设备选择
def parse_device_early():
    """提前解析device参数以设置CUDA_VISIBLE_DEVICES"""
    for i, arg in enumerate(sys.argv):
        if arg == '--device' and i + 1 < len(sys.argv):
            device = sys.argv[i + 1]
            if device.startswith('cuda'):
                device_id = device.split(':')[1] if ':' in device else '0'
                os.environ['CUDA_VISIBLE_DEVICES'] = device_id
                # 返回cuda:0（因为设置CUDA_VISIBLE_DEVICES后只有一个设备）和原始设备ID
                return 'cuda:0', device_id
    return 'cuda:0', '0'

actual_device, actual_device_id = parse_device_early()

sys.path.insert(0, str(Path(__file__).parent))

from mpc.flow_guided_gaussian_model import FlowGuidedGaussianDynamicsModel
from mpc.flow_objectives import (
    FlowConsistencyObjective,
    FlowDirectionLoss,
    ActionRegularizationObjective,
)
from mpc.cem import CEMOptimizer
from mpc.sampler import CorrelatedNoiseSampler
from mpc.objectives import CombinedObjective, SquaredError, VGGPerceptualObjective
from mpc.utils import ObservationList, write_moviepy_gif
from mpc.agent import SimplePlanningAgent
import cv2

# Utility functions for control signal constraints
def sincos_to_angle(sin_val, cos_val):
    """Convert sin/cos to angle in radians"""
    return np.arctan2(sin_val, cos_val)

def angle_to_sincos(angle):
    """Convert angle to sin/cos representation"""
    return np.sin(angle), np.cos(angle)

def constrain_control_delta(prev_control, new_control, max_delta_deg=30.0, log_clamp: bool = True):
    """
    约束控制信号变化幅度，避免关节超限
    
    Args:
        prev_control: (15,) - 上一步的控制向量 [sin(θ1), cos(θ1), ..., sin(θ6), cos(θ6), grip1, grip2, grip3]
        new_control: (15,) - 新的控制向量
        max_delta_deg: 每个关节允许的最大角度变化（度）
    
    Returns:
        constrained_control: (15,) - 约束后的控制向量
    """
    max_delta_rad = np.deg2rad(max_delta_deg)
    constrained = new_control.copy()
    
    # 处理前12维（6个关节）
    for i in range(6):
        sin_idx = 2 * i
        cos_idx = 2 * i + 1
        
        # 提取前后的角度
        prev_angle = sincos_to_angle(prev_control[sin_idx], prev_control[cos_idx])
        new_angle = sincos_to_angle(new_control[sin_idx], new_control[cos_idx])
        
        # 计算角度差（处理周期性）
        delta = new_angle - prev_angle
        # 归一化到[-π, π]
        delta = np.arctan2(np.sin(delta), np.cos(delta))
        
        # 约束delta
        if abs(delta) > max_delta_rad:
            raw_deg = np.degrees(delta)
            delta = np.sign(delta) * max_delta_rad
            constrained_angle = prev_angle + delta
            constrained[sin_idx], constrained[cos_idx] = angle_to_sincos(constrained_angle)
            if log_clamp:
                print(f"    Joint {i}: clamped {raw_deg:.1f}° → {np.degrees(delta):.1f}°")
    
    # 后3维（gripper）不约束或使用较小的约束
    # 这里保持不变，如需要可添加速度限制
    
    return constrained


def normalize_sincos_control(control_vec):
    """Normalize sin/cos pairs to unit circle for 6 joints."""
    normalized = control_vec.copy()
    for i in range(6):
        sin_idx = 2 * i
        cos_idx = 2 * i + 1
        angle = sincos_to_angle(normalized[sin_idx], normalized[cos_idx])
        normalized[sin_idx], normalized[cos_idx] = angle_to_sincos(angle)
    return normalized

# Import GMFlow for optical flow estimation
try:
    from gmflow.gmflow import GMFlow
    from gmflow.config import get_cfg as get_gmflow_cfg
    GMFLOW_AVAILABLE = True
except ImportError:
    print("⚠ GMFlow not available, will use deformation-based flow prediction")
    GMFLOW_AVAILABLE = False


def visualize_flow_hsv(flow_field: np.ndarray) -> np.ndarray:
    """Convert optical flow to HSV color representation.
    
    Args:
        flow_field: (H, W, 2) - optical flow [dx, dy]
    
    Returns:
        hsv_image: (H, W, 3) - RGB visualization [0, 255]
    """
    H, W = flow_field.shape[:2]
    
    # Compute flow magnitude and angle
    fx, fy = flow_field[..., 0], flow_field[..., 1]
    magnitude = np.sqrt(fx**2 + fy**2)
    angle = np.arctan2(fy, fx)
    
    # Create HSV image
    hsv = np.zeros((H, W, 3), dtype=np.uint8)
    
    # Hue represents direction (0-180 in OpenCV)
    hsv[..., 0] = (angle + np.pi) / (2 * np.pi) * 180
    
    # Saturation is maxed out
    hsv[..., 1] = 255
    
    # Value represents magnitude (normalized)
    max_magnitude = np.percentile(magnitude, 95)  # Use 95th percentile for better visualization
    if max_magnitude > 0:
        hsv[..., 2] = np.clip(magnitude / max_magnitude * 255, 0, 255).astype(np.uint8)
    else:
        hsv[..., 2] = 0
    
    # Convert HSV to RGB
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return rgb


def visualize_sampled_flow_points(
    initial_image: np.ndarray,
    target_image: np.ndarray,
    source_points: np.ndarray,
    target_points: np.ndarray,
    flow_vectors: np.ndarray,
    arrow_scale: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Visualize sampled flow points with arrows on initial and target images.
    
    Args:
        initial_image: (H, W, 3) - initial image [0, 1]
        target_image: (H, W, 3) - target image [0, 1]
        source_points: (N, 2) - normalized source points [0, 1]
        target_points: (N, 2) - normalized target points [0, 1]
        flow_vectors: (N, 2) - normalized flow vectors
        arrow_scale: scaling factor for arrow length
    
    Returns:
        initial_vis: (H, W, 3) - initial image with flow arrows [0, 255]
        target_vis: (H, W, 3) - target image with endpoints [0, 255]
    """
    H, W = initial_image.shape[:2]
    
    # Convert to uint8 for OpenCV drawing
    initial_vis = (initial_image * 255).astype(np.uint8).copy()
    target_vis = (target_image * 255).astype(np.uint8).copy()
    
    # Convert normalized coordinates to pixel coordinates
    source_px = (source_points * [W, H]).astype(np.int32)
    target_px = (target_points * [W, H]).astype(np.int32)
    
    # Compute flow magnitude for color mapping
    flow_magnitude = np.linalg.norm(flow_vectors, axis=-1)
    max_mag = np.max(flow_magnitude) if np.max(flow_magnitude) > 0 else 1.0
    
    # Draw arrows on initial image and circles on target image
    for i in range(len(source_px)):
        src = tuple(source_px[i])
        tgt = tuple(target_px[i])
        
        # Color based on flow magnitude (blue=small, red=large)
        norm_mag = flow_magnitude[i] / max_mag
        color = (
            int(255 * norm_mag),      # R
            int(100 * (1 - norm_mag)), # G
            int(255 * (1 - norm_mag))  # B
        )
        
        # Draw on initial image: circle at source + arrow
        cv2.circle(initial_vis, src, radius=2, color=color, thickness=-1)
        
        # Draw arrow if flow is significant
        if flow_magnitude[i] > 0.001:
            arrow_end = (
                int(src[0] + flow_vectors[i, 0] * W * arrow_scale),
                int(src[1] + flow_vectors[i, 1] * H * arrow_scale)
            )
            cv2.arrowedLine(initial_vis, src, arrow_end, color, thickness=1, tipLength=0.3)
        
        # Draw on target image: circle at target
        cv2.circle(target_vis, tgt, radius=3, color=color, thickness=-1)
        cv2.circle(target_vis, tgt, radius=4, color=(255, 255, 255), thickness=1)
    
    return initial_vis, target_vis


def load_image(image_path, target_size=(256, 256)):
    """加载并调整图像大小"""
    img = Image.open(image_path).convert('RGB')
    img = img.resize((target_size[1], target_size[0]), Image.BILINEAR)
    img_array = np.array(img).astype(np.float32) / 255.0
    return img_array


def create_flow_goal_from_images(
    initial_image: np.ndarray,
    target_image: np.ndarray,
    num_flow_points: int = 512,
    device: str = "cuda",
    sampling_strategy: str = "adaptive",  # "uniform", "adaptive", "motion_only"
    motion_focus_ratio: float = 0.7,  # 聚焦运动区域的比例
) -> dict:
    """
    使用GMFlow从初始帧和目标帧生成目标光流
    
    Args:
        initial_image: (H, W, 3) - 初始图像 [0, 1]
        target_image: (H, W, 3) - 目标图像 [0, 1]
        num_flow_points: 采样的光流点数量
        device: 计算设备
        sampling_strategy: 采样策略
            - "uniform": 均匀网格采样（原始方法）
            - "adaptive": 70%运动区域 + 30%均匀覆盖（推荐）
            - "motion_only": 100%聚焦运动区域（最稠密）
        motion_focus_ratio: adaptive模式下聚焦运动区域的比例
    
    Returns:
        goal_dict: 包含目标光流的字典
        num_flow_points: 采样的光流点数量
        device: 计算设备
    
    Returns:
        goal_dict: 包含目标光流的字典
            - 'source_points': (N, 2) - 初始点位置 [x, y]
            - 'target_points': (N, 2) - 目标点位置 [x, y]
            - 'flow_vectors': (N, 2) - 光流向量
    """
    H, W, _ = initial_image.shape
    
    if GMFLOW_AVAILABLE:
        print("  使用GMFlow生成目标光流...")
        
        # 加载GMFlow模型
        gmflow_cfg = get_gmflow_cfg()
        flownet = GMFlow(
            feature_channels=gmflow_cfg.feature_channels,
            num_scales=gmflow_cfg.num_scales,
            upsample_factor=gmflow_cfg.upsample_factor,
            num_head=gmflow_cfg.num_head,
            attention_type=gmflow_cfg.attention_type,
            ffn_dim_expansion=gmflow_cfg.ffn_dim_expansion,
            num_transformer_layers=gmflow_cfg.num_transformer_layers,
        ).to(device)
        
        # 加载预训练权重
        checkpoint = torch.load(gmflow_cfg.model, map_location="cpu")
        weights = checkpoint["model"] if "model" in checkpoint else checkpoint
        flownet.load_state_dict(weights, strict=True)
        flownet.eval()
        print(f"  ✓ GMFlow加载成功")
        
        # 准备输入：(1, 3, H, W)
        img1 = torch.from_numpy(initial_image).permute(2, 0, 1).unsqueeze(0).float().to(device)
        img2 = torch.from_numpy(target_image).permute(2, 0, 1).unsqueeze(0).float().to(device)
        
        # GMFlow预测
        with torch.no_grad():
            flow_predictions = flownet(
                img1, img2,
                attn_splits_list=[2],
                corr_radius_list=[-1],
                prop_radius_list=[-1],
            )
            flow_field = flow_predictions[-1]  # (1, 2, H, W)
        
        flow_field = flow_field[0].permute(1, 2, 0).cpu().numpy()  # (H, W, 2)
        print(f"  ✓ Flow field generated: {flow_field.shape}")
        
        # Visualize flow field using HSV color space
        flow_hsv = visualize_flow_hsv(flow_field)
        
        # ============ 基于运动显著性的自适应采样 ============
        # 计算光流幅度（运动显著性）
        flow_magnitude = np.linalg.norm(flow_field, axis=-1)  # (H, W)
        
        if sampling_strategy == "uniform":
            # 原始均匀网格采样
            grid_size = int(np.ceil(np.sqrt(num_flow_points)))
            y_coords = np.linspace(0, H-1, grid_size, dtype=int)
            x_coords = np.linspace(0, W-1, grid_size, dtype=int)
            yy, xx = np.meshgrid(y_coords, x_coords)
            sample_coords = np.stack([xx.flatten(), yy.flatten()], axis=-1)[:num_flow_points]
            print(f"  Strategy: Uniform grid sampling")
            
        elif sampling_strategy == "motion_only":
            # 100%聚焦运动区域
            sampling_weights = flow_magnitude ** 2
            sampling_weights = sampling_weights / (sampling_weights.sum() + 1e-8)
            
            flat_weights = sampling_weights.flatten()
            flat_indices = np.arange(H * W)
            sampled_indices = np.random.choice(
                flat_indices, size=num_flow_points, replace=False, p=flat_weights
            )
            sample_coords = np.stack([
                sampled_indices % W,  # x
                sampled_indices // W  # y
            ], axis=-1)
            print(f"  Strategy: Motion-focused sampling (100%)")
            
        else:  # "adaptive" (default)
            # 混合采样：聚焦运动 + 全局覆盖
            num_motion_samples = int(num_flow_points * motion_focus_ratio)
            num_uniform_samples = num_flow_points - num_motion_samples
            
            # 计算采样权重
            sampling_weights = flow_magnitude ** 2
            sampling_weights = sampling_weights / (sampling_weights.sum() + 1e-8)
            
            # 方法1: 基于运动显著性的重要性采样
            flat_weights = sampling_weights.flatten()
            flat_indices = np.arange(H * W)
            motion_sampled_indices = np.random.choice(
                flat_indices, size=num_motion_samples, replace=False, p=flat_weights
            )
            motion_coords = np.stack([
                motion_sampled_indices % W,  # x
                motion_sampled_indices // W  # y
            ], axis=-1)
            
            # 方法2: 均匀网格采样（保证全局覆盖）
            grid_size = int(np.ceil(np.sqrt(num_uniform_samples)))
            y_coords = np.linspace(0, H-1, grid_size, dtype=int)
            x_coords = np.linspace(0, W-1, grid_size, dtype=int)
            yy, xx = np.meshgrid(y_coords, x_coords)
            uniform_coords = np.stack([xx.flatten(), yy.flatten()], axis=-1)[:num_uniform_samples]
            
            # 合并采样点
            sample_coords = np.vstack([motion_coords, uniform_coords])
            print(f"  Strategy: Adaptive sampling ({num_motion_samples} motion + {num_uniform_samples} uniform)")
        
        # 提取采样点的光流
        sampled_flow = flow_field[sample_coords[:, 1], sample_coords[:, 0]]  # (N, 2)
        sampled_magnitude = np.linalg.norm(sampled_flow, axis=-1)
        
        # 计算源点和目标点（归一化坐标）
        source_points = sample_coords.astype(np.float32) / [W, H]  # (N, 2)
        target_points = source_points + sampled_flow / [W, H]  # (N, 2)
        
        # 统计信息
        print(f"  ✓ Sampled {len(source_points)} flow points")
        print(f"  Overall mean magnitude: {np.mean(sampled_magnitude):.2f} pixels")
        if sampling_strategy == "adaptive":
            motion_mag = np.mean(sampled_magnitude[:int(num_flow_points * motion_focus_ratio)])
            uniform_mag = np.mean(sampled_magnitude[int(num_flow_points * motion_focus_ratio):])
            print(f"  Motion-focused mean: {motion_mag:.2f} px | Uniform mean: {uniform_mag:.2f} px")
            print(f"  ✓ Density improvement: {motion_mag / (np.mean(sampled_magnitude) + 1e-8):.2f}x in motion regions")
        
        # 计算采样密度指标
        motion_threshold = np.percentile(flow_magnitude.flatten(), 75)  # 运动阈值
        high_motion_pixels = np.sum(flow_magnitude > motion_threshold)
        sampled_high_motion = np.sum(sampled_magnitude > motion_threshold)
        coverage_ratio = sampled_high_motion / len(sampled_magnitude)
        print(f"  High-motion coverage: {sampled_high_motion}/{len(sampled_magnitude)} ({coverage_ratio*100:.1f}%)")
        
        return {
            'source_points': source_points,
            'target_points': target_points,
            'flow_vectors': sampled_flow / [W, H],  # 归一化
            'flow_hsv': flow_hsv,  # HSV visualization
            'sampling_strategy': sampling_strategy,
            'sampled_magnitude': sampled_magnitude,  # 采样点幅度
        }
    
    else:
        # 如果没有GMFlow，使用简单的网格采样（假设静止）
        print("  ⚠ GMFlow不可用，使用零光流")
        grid_size = int(np.ceil(np.sqrt(num_flow_points)))
        y_coords = np.linspace(0.1, 0.9, grid_size)
        x_coords = np.linspace(0.1, 0.9, grid_size)
        yy, xx = np.meshgrid(y_coords, x_coords)
        
        source_points = np.stack([xx.flatten(), yy.flatten()], axis=-1)[:num_flow_points]
        target_points = source_points.copy()  # 零光流
        
        return {
            'source_points': source_points,
            'target_points': target_points,
            'flow_vectors': np.zeros_like(source_points),
        }


def setup_flow_guided_cem(
    model,
    control_dim: int,
    horizon: int = 10,
    num_samples: int = 64,
    opt_iters: int = 10,
    flow_weight: float = 0.7,
    image_weight: float = 0.3,
    vgg_weight: float = 0.0,
    vgg_layer: str = "relu3_3",
    verbose: bool = True,
    direction_weight: float = 0.01,
    direction_loss_type: str = "cosine",
    action_regularization_weight: float = 0.0,
    max_action_delta: float = 0.5,
):
    """
    配置光流引导的CEM优化器
    
    Args:
        model: FlowGuidedGaussianDynamicsModel
        control_dim: 控制维度
        horizon: 规划视野
        num_samples: CEM采样数
        opt_iters: CEM优化迭代数
        vgg_weight: VGG感知损失权重
        direction_weight: 光流方向损失权重
        verbose: 是否打印详细信息
    
    Returns:
        optimizer: 配置好的CEMOptimizer
        objectives_dict: 目标函数字典（用于打印各项reward）
    """
    # 创建采样器
    sampler = CorrelatedNoiseSampler(
        a_dim=control_dim,
        horizon=horizon,
        beta=0.9
    )
    
    # flow_weight参数被保留以保持向后兼容，但不再使用FlowAlignmentObjective
    if flow_weight > 0:
        print(f"  ⚠ flow_weight={flow_weight} is ignored (FlowAlignmentObjective removed)")
    
    # 创建光流平滑目标
    flow_smoothness = FlowConsistencyObjective(
        weight=0.1,
        order=2,
    )
    
    # 初始化目标函数字典
    objectives_dict = {
        'flow_smoothness': flow_smoothness,
    }
    
    # 图像级别的loss使用VGGPerceptualObjective（它使用prediction和prev_rgb）
    if image_weight > 0:
        print(f"  ⚠ image_weight={image_weight} is ignored (RGB goal removed, use vgg_weight instead)")

    if vgg_weight > 0:
        vgg_perceptual = VGGPerceptualObjective(
            weight=vgg_weight,
            layer=vgg_layer,
            image_key='rgb',
            prev_key='prev_rgb',
            device=model.device,
        )
        objectives_dict['vgg_perceptual'] = vgg_perceptual

    # 创建光流方向指引损失（如果权重>0）
    if direction_weight > 0:
        flow_direction = FlowDirectionLoss(
            weight=direction_weight,
            loss_type=direction_loss_type,
            use_visibility_mask=True,
            temporal_weight_decay=0.95,
        )
        objectives_dict['flow_direction'] = flow_direction
        print(f"  ✓ Added flow direction loss: weight={direction_weight}, type={direction_loss_type}")

    # 创建动作正则化损失（如果权重>0）
    if action_regularization_weight > 0:
        action_regularization = ActionRegularizationObjective(
            weight=action_regularization_weight,
            penalty_type='delta',
            max_delta=max_action_delta,
            penalty_scale='quadratic',
            apply_to_joints_only=True,
            num_joints=6,
        )
        objectives_dict['action_regularization'] = action_regularization
        print(f"  ✓ Added action regularization: weight={action_regularization_weight}, max_delta={max_action_delta}")

    objective = CombinedObjective(objectives_dict)
    
    # 创建CEM优化器
    optimizer = CEMOptimizer(
        model=model,
        objective=objective,
        sampler=sampler,
        a_dim=control_dim,
        horizon=horizon,
        num_samples=num_samples,
        elites_frac=0.1,
        opt_iters=opt_iters,
        alpha=0.1,
        verbose=verbose,
    )
    
    return optimizer, objectives_dict


def run_mpc_from_images(
    model_path: str,
    iteration: int,
    initial_image_path: str,
    target_image_path: str,
    control_dim: int,
    num_steps: int = 25,
    horizon: int = 5,
    num_samples: int = 32,
    opt_iters: int = 10,
    flow_weight: float = 0.0,
    image_weight: float = 0.0,
    vgg_weight: float = 0.2,
    vgg_layer: str = "relu3_3",
    image_weight_schedule: str = "none",
    image_weight_start: float = 5.0,
    use_sparse_render: bool = True,
    sparse_ratio: float = 0.15,
    image_height: int = 480,
    image_width: int = 480,
    device: str = "cuda:0",
    log_dir: str = "./outputs/render_based_test",
    verbose: bool = True,
    sampling_strategy: str = "adaptive",
    motion_focus_ratio: float = 0.7,
    camera_distance: float = 2.0,
    camera_elevation: float = 0.0,
    camera_azimuth: float = 0.0,
    fov_degrees: float = 45.0,
    transforms_json_path: str = "/home/ubuntu/yyf/4DGaussians/assets/example_transforms.json",
    save_video: bool = False,
    video_fps: int = 5,
    direction_weight: float = 1,
    direction_loss_type: str = "cosine",
    action_regularization_weight: float = 0.0,
    max_action_delta: float = 0.5,
    use_motion_mask: bool = True,
    motion_threshold_percentile: float = 50.0,
    ):
    """
    从初始图像和目标图像运行MPC控制
    
    主要流程：
    1. 加载初始和目标图像
    2. 从transforms.json读取初始帧的joint_pos作为初始控制
    3. 使用GMFlow生成目标光流
    4. 初始化模型和优化器
    5. 运行MPC规划控制序列
    6. 执行并渲染整个轨迹

    Args:
        transforms_json_path: 从frames中读取joint_pos，并从cameras加载相机参数
    """
    os.makedirs(log_dir, exist_ok=True)
    
    print("="*70)
    print("Flow-Guided 4DGaussians MPC Control")
    print("="*70)
    print(f"Model path: {model_path}")
    print(f"Initial image: {initial_image_path}")
    print(f"Target image: {target_image_path}")
    print(f"Sparse rendering: {use_sparse_render}")
    print("="*70)
    
    # 1. 加载图像
    print("\n[1/9] Loading images...")
    initial_image = load_image(initial_image_path, (image_height, image_width))
    target_image = load_image(target_image_path, (image_height, image_width))
    print(f"  Initial image: {initial_image.shape}")
    print(f"  Target image: {target_image.shape}")
    
    # 保存输入图像到日志目录
    Image.fromarray((initial_image * 255).astype(np.uint8)).save(
        os.path.join(log_dir, "initial_image.png")
    )
    Image.fromarray((target_image * 255).astype(np.uint8)).save(
        os.path.join(log_dir, "target_image.png")
    )
    
    # 2. 生成目标光流
    print("\n[2/6] Generating target optical flow...")
    goal_flow_dict = create_flow_goal_from_images(
        initial_image,
        target_image,
        num_flow_points=512,
        device=device,
        sampling_strategy=sampling_strategy,
        motion_focus_ratio=motion_focus_ratio,
    )
    
    # Save HSV flow visualization
    if 'flow_hsv' in goal_flow_dict:
        Image.fromarray(goal_flow_dict['flow_hsv']).save(
            os.path.join(log_dir, "target_flow_hsv.png")
        )
        print("  ✓ Flow HSV visualization saved")
    
    # Visualize sampled flow points
    print("  Visualizing sampled flow points...")
    initial_with_arrows, target_with_points = visualize_sampled_flow_points(
        initial_image,
        target_image,
        goal_flow_dict['source_points'],
        goal_flow_dict['target_points'],
        goal_flow_dict['flow_vectors'],
        arrow_scale=2.0,  # Scale arrows for better visibility
    )
    
    Image.fromarray(initial_with_arrows).save(
        os.path.join(log_dir, "initial_with_flow_arrows.png")
    )
    Image.fromarray(target_with_points).save(
        os.path.join(log_dir, "target_with_flow_points.png")
    )
    
    # Create a side-by-side comparison
    comparison = np.hstack([initial_with_arrows, target_with_points])
    Image.fromarray(comparison).save(
        os.path.join(log_dir, "flow_points_comparison.png")
    )
    
    print(f"  ✓ Sampled flow visualization saved:")
    print(f"     Source image with {len(goal_flow_dict['source_points'])} flow arrows")
    
    # 3. 加载相机参数（如果提供了transforms.json）
    print("\n[3/9] Loading camera parameters...")
    transform_matrix = None
    focal_x = None
    focal_y = None
    cx = None
    cy = None
    use_json_camera = False
    
    if transforms_json_path and os.path.exists(transforms_json_path):
        print(f"  Loading JSON file: {transforms_json_path}")
        
        import json
        with open(transforms_json_path, 'r') as f:
            transforms_data = json.load(f)
        
        # 直接使用第一个相机参数（MPC不需要frame信息，只需要相机位姿）
        cameras_meta = transforms_data.get('cameras', [])
        
        if cameras_meta:
            camera_meta = cameras_meta[0]  # 使用第一个相机
            
            # 直接使用transform_matrix，不在这里应用R_x_180
            # R_x_180会在_create_camera方法中统一处理
            transform_matrix = np.array(camera_meta['transform_matrix'], dtype=np.float32)
            
            # 提取焦距和主点
            focal_x = camera_meta.get('fl_x') or camera_meta.get('focal_length')
            focal_y = camera_meta.get('fl_y') or camera_meta.get('focal_length')
            cx = camera_meta.get('cx', image_width / 2.0)
            cy = camera_meta.get('cy', image_height / 2.0)
            
            use_json_camera = True
            print(f"  ✓ Loaded camera from JSON:")
            print(f"    - Using first camera (index 0)")
            print(f"    - Focal length: fx={focal_x:.2f}, fy={focal_y:.2f}")
            print(f"    - Principal point: cx={cx:.2f}, cy={cy:.2f}")
            print(f"    - Transform matrix (c2w):")
            for row in transform_matrix:
                print(f"      [{', '.join([f'{x:8.4f}' for x in row])}]")
        else:
            print(f"  ⚠ No cameras found in JSON, using manual parameters")
    else:
        print(f"  ⚠ No transforms.json provided, using manual camera parameters")
    
    # 4. 加载初始控制（从transforms.json的frames中读取）
    print(f"\n[4/9] Loading initial joint position from transforms.json...")
    import json
    initial_control = None

    if transforms_json_path and os.path.exists(transforms_json_path):
        with open(transforms_json_path, 'r') as f:
            transforms_data = json.load(f)

        frames = transforms_data.get('frames', [])
        image_filename = os.path.basename(initial_image_path)

        # 查找匹配的frame
        matched_frame = None
        for frame in frames:
            frame_file_path = frame.get('file_path', '')
            frame_filename = os.path.basename(frame_file_path)

            # 匹配文件名（去掉扩展名比较）
            if os.path.splitext(frame_filename)[0] == os.path.splitext(image_filename)[0]:
                matched_frame = frame
                break
            # 或者完整路径包含匹配
            if image_filename in frame_file_path or frame_file_path in image_filename:
                matched_frame = frame
                break

        if matched_frame and 'joint_pos' in matched_frame:
            initial_control = np.array(matched_frame['joint_pos'], dtype=np.float32)
            print(f"  ✓ Loaded initial control from frame: {matched_frame.get('file_path', 'unknown')}")
            print(f"    Joint pos (first 6): [{', '.join([f'{x:.4f}' for x in initial_control[:6]])}]")
            if len(initial_control) > 6:
                print(f"    Control dim: {len(initial_control)}")
        else:
            print(f"  ⚠ No matching frame or joint_pos found for '{image_filename}', using zero control")
            initial_control = np.zeros(control_dim, dtype=np.float32)
    else:
        print(f"  ⚠ No transforms.json found, using zero control")
        initial_control = np.zeros(control_dim, dtype=np.float32)
    
    # 5. 初始化模型
    print(f"\n[5/9] Initializing flow-guided Gaussian model...")
    if use_json_camera:
        print(f"  Using camera from JSON with transform matrix")
    else:
        print(f"  Camera setup: distance={camera_distance}, elevation={camera_elevation}°, azimuth={camera_azimuth}°")
    
    # 准备render_based方法需要的参数
    target_image_tensor = torch.from_numpy(target_image).permute(2, 0, 1).float()  # (3, H, W)
    sample_coords_tensor = torch.from_numpy(goal_flow_dict['source_points']).float()  # (N, 2)
    
    model = FlowGuidedGaussianDynamicsModel(
        model_path=model_path,
        iteration=iteration,
        control_dim=control_dim,
        image_height=image_height,
        image_width=image_width,
        num_context=2,
        use_sparse_rendering=use_sparse_render,
        sparse_ratio=sparse_ratio,
        enable_flow_prediction=True,
        flow_prediction_method="render_based",  # 使用render_based方法
        num_flow_points=len(goal_flow_dict['source_points']),
        device=device,
        camera_distance=camera_distance if not use_json_camera else None,
        camera_elevation=camera_elevation if not use_json_camera else None,
        camera_azimuth=camera_azimuth if not use_json_camera else None,
        fov_degrees=fov_degrees if not use_json_camera else None,
        transform_matrix=transform_matrix,
        focal_x=focal_x,
        focal_y=focal_y,
        cx=cx,
        cy=cy,
        target_image=target_image_tensor,  # 传入目标图像
        sample_coords=sample_coords_tensor,  # 传入采样坐标
    )
    
    cost_stats = model.compute_rendering_cost()
    if cost_stats:
        print(f"    - Total Gaussians: {cost_stats.get('total_gaussians', 'N/A')}")
        print(f"    - Selected Gaussians: {cost_stats.get('selected_gaussians', 'N/A')}")
    
    # 6. 设置MPC优化器
    print(f"\n[6/9] Setting up MPC optimizer...")
    optimizer, objective_components = setup_flow_guided_cem(
        model=model,
        control_dim=control_dim,
        horizon=horizon,
        num_samples=num_samples,
        opt_iters=opt_iters,
        flow_weight=flow_weight,
        image_weight=image_weight,
        vgg_weight=vgg_weight,
        vgg_layer=vgg_layer,
        verbose=verbose,
        direction_weight=direction_weight,
        direction_loss_type=direction_loss_type,
        action_regularization_weight=action_regularization_weight,
        max_action_delta=max_action_delta,
    )
    
    print(f"  ✓ CEM config: {num_samples} samples x {opt_iters} iterations")
    print(f"  ✓ Planning horizon: {horizon} steps")
    print(f"  ✓ Objective functions:")
    for name, obj in objective_components.items():
        print(f"    - {name}: weight={obj.weight}")
    
    # 7. 设置Planning Agent
    print(f"\n[7/9] Setting up planning agent...")
    # 新逻辑：初始状态已知（从joint_pos读取），基于此规划所有动作（包括第一个）
    # 设置use_initial_action=False，让MPC从t=0开始规划，而不是使用预设的第一个动作
    agent = SimplePlanningAgent(
        a_dim=control_dim,
        optimizer=optimizer,
        replan_interval=1,
        initial_action=None,  # 不使用预设的第一个动作
        num_context=model.num_context,
        use_initial_action=False,  # 从初始状态开始规划所有动作
    )
    agent.set_log_dir(None)  # 设置为None以禁用CEM的可视化输出

    # 8. 准备阶段性光流目标
    print(f"\n[8/9] Preparing step-wise flow targets...")
    # 策略：将完整的运动轨迹分成num_steps个阶段

    # 从initial_image到target_image的完整光流
    full_source_points = goal_flow_dict['source_points']  # (N, 2) 初始位置
    full_target_points = goal_flow_dict['target_points']  # (N, 2) 最终目标位置

    # 创建阶段性目标序列：从初始位置到最终目标的线性插值
    step_targets = []  # 每个step的目标点位置
    step_flows = []    # 每个step的光流向量（从起点到阶段性目标的位移）

    for step in range(num_steps):
        alpha = (step + 1) / num_steps  # 线性插值系数
        # 第step步的目标位置：从source到target的线性插值
        target_at_step = full_source_points * (1 - alpha) + full_target_points * alpha
        # 第step步的光流：从初始位置到阶段性目标的位移
        flow_at_step = target_at_step - full_source_points
        step_targets.append(target_at_step)
        step_flows.append(flow_at_step)

    step_targets = np.array(step_targets)  # (num_steps, N, 2)
    step_flows = np.array(step_flows)  # (num_steps, N, 2)

    # 创建阶段性光流目标（带visibility）
    step_flow_targets = []
    for t in range(num_steps):
        flow_target = np.concatenate([
            step_flows[t],
            np.ones((len(step_flows[t]), 1))  # visibility
        ], axis=-1)  # (N, 3)
        step_flow_targets.append(flow_target)

    step_flow_targets = np.array(step_flow_targets)  # (num_steps, N, 3)

    print(f"  Created {num_steps} step-wise flow targets")
    print(f"  Step 0: flow_magnitude = {np.linalg.norm(step_flows[0], axis=-1).mean():.4f}")
    print(f"  Step {num_steps-1}: flow_magnitude = {np.linalg.norm(step_flows[-1], axis=-1).mean():.4f}")

    # 为CEM准备目标序列
    pred_steps = horizon + model.num_context - 1
    goal_flow_sequence = []
    for t in range(pred_steps):
        step_idx = min(t, num_steps - 1)  # 映射到实际step
        alpha = (t + 1) / pred_steps
        step_idx = min(int(alpha * num_steps), num_steps - 1)
        goal_flow_sequence.append(step_flow_targets[step_idx])

    goal_flow_sequence = np.array(goal_flow_sequence)  # (T, N, 3)

    # 🎯 只使用光流作为目标（移除RGB目标）
    # 注意：这个初始goal会在MPC循环中被每步更新的goal覆盖
    goal_obs = ObservationList(
        data_dict={
            'flow': goal_flow_sequence[None],  # (1, T, N, 3)
        },
        image_shape=(image_height, image_width)
    )
    agent.set_goal(goal_obs)
    
    # 9. 执行MPC控制
    print("\n[9/9] Executing MPC control...")
    print("-"*70)
    print(f"  Mode: MPC planning from known initial state")
    print(f"  Initial joint pos (from transforms.json): [{', '.join([f'{x:.4f}' for x in initial_control[:6]])}]...")
    print(f"  Logic: Initial state known → MPC plans all actions from t=0")
    print("-"*70)

    observations = []
    actions = []
    rewards = []
    flow_history = []
    rgb_history = []

    # 准备目标图像tensor（用于GMFlow）
    target_image_tensor = torch.from_numpy(target_image).permute(2, 0, 1).float().to(model.device)

    # 初始化追踪坐标（从光流采样点开始）
    current_tracking_coords = goal_flow_dict['source_points'].copy()

    # 定义GMFlow初始化辅助函数
    def init_gmflow(model):
        """初始化GMFlow模型"""
        if not hasattr(model, 'gmflow_initialized'):
            from gmflow.gmflow import GMFlow
            from gmflow.config import get_cfg as get_gmflow_cfg
            import os as os_module

            gmflow_cfg = get_gmflow_cfg()
            model.flownet = GMFlow(
                feature_channels=gmflow_cfg.feature_channels,
                num_scales=gmflow_cfg.num_scales,
                upsample_factor=gmflow_cfg.upsample_factor,
                num_head=gmflow_cfg.num_head,
                attention_type=gmflow_cfg.attention_type,
                ffn_dim_expansion=gmflow_cfg.ffn_dim_expansion,
            ).to(model.device)

            checkpoint_path = "gmflow/checkpoints/gmflow_sintel-0c07dcb3.pth"
            if os_module.path.exists(checkpoint_path):
                checkpoint = torch.load(checkpoint_path, map_location=model.device)
                model.flownet.load_state_dict(checkpoint['model'], strict=False)
            model.flownet.eval()
            model.gmflow_initialized = True
        return model.flownet

    # 定义从图像计算光流的辅助函数
    def compute_flow_from_images(model, img_curr, img_target):
        """从当前图像和目标图像计算光流场"""
        flownet = init_gmflow(model)

        img1 = img_curr.unsqueeze(0)  # (1, 3, H, W)
        img2 = img_target.unsqueeze(0)  # (1, 3, H, W)
        img1_norm = img1 * 2.0 - 1.0
        img2_norm = img2 * 2.0 - 1.0

        with torch.no_grad():
            flow_predictions = flownet(
                img1_norm, img2_norm,
                attn_splits_list=[2],
                corr_radius_list=[-1],
                prop_radius_list=[-1],
            )
            flow_field_full = flow_predictions[-1][0].permute(1, 2, 0)  # (H, W, 2)
        return flow_field_full

    # 定义从光流场采样追踪点
    def sample_flow_points(model, flow_field_full, tracking_coords):
        """从光流场中采样追踪点的光流向量
        
        Args:
            flow_field_full: (H, W, 2) - 完整的光流场 [dx, dy] (像素单位)
            tracking_coords: (N, 2) - 追踪点坐标 [x, y] (归一化到[0,1])
        
        Returns:
            flow_vectors: (N, 3) - 采样的光流向量 [dx_norm, dy_norm, visibility]
            new_tracking_coords: (N, 2) - 更新后的追踪点坐标
            flow_field_full_np: (H, W, 2) - numpy格式的完整光流场
        """
        H, W = flow_field_full.shape[:2]

        # 计算目标点位置（用于更新追踪坐标）
        y_grid, x_grid = torch.meshgrid(
            torch.arange(H, device=model.device),
            torch.arange(W, device=model.device),
            indexing='ij'
        )
        source_x = x_grid.float() / W
        source_y = y_grid.float() / H
        target_x = source_x + flow_field_full[:, :, 0] / W
        target_y = source_y + flow_field_full[:, :, 1] / H
        visibility = ((target_x >= 0) & (target_x <= 1) &
                      (target_y >= 0) & (target_y <= 1)).float()

        tracking_tensor = torch.from_numpy(tracking_coords).float().to(model.device)
        x_coords = torch.clamp(tracking_tensor[:, 0] * W, 0, W - 1).long()
        y_coords = torch.clamp(tracking_tensor[:, 1] * H, 0, H - 1).long()

        # 采样光流向量（归一化到[0,1]范围）
        sampled_flow_x = flow_field_full[y_coords, x_coords, 0].cpu().numpy() / W
        sampled_flow_y = flow_field_full[y_coords, x_coords, 1].cpu().numpy() / H
        sampled_visibility = visibility[y_coords, x_coords].cpu().numpy()

        # 返回光流向量（而非目标位置）
        flow_vectors = np.stack([sampled_flow_x, sampled_flow_y, sampled_visibility], axis=-1)
        
        # 更新追踪坐标（用于下一步）
        sampled_target_x = target_x[y_coords, x_coords].cpu().numpy()
        sampled_target_y = target_y[y_coords, x_coords].cpu().numpy()
        new_tracking_coords = np.stack([
            np.clip(sampled_target_x, 0.0, 1.0),
            np.clip(sampled_target_y, 0.0, 1.0)
        ], axis=-1)
        
        flow_field_full_np = flow_field_full.cpu().numpy()
        return flow_vectors, new_tracking_coords, flow_field_full_np

    # 第一步渲染：从初始joint_pos渲染初始状态图像
    print(f"\n  Rendering initial state from joint_pos...")
    initial_action_tensor = torch.tensor(initial_control, dtype=torch.float32, device=model.device)
    initial_rendered = model.render_with_control(initial_action_tensor)
    initial_rendered_np = initial_rendered.permute(1, 2, 0).cpu().numpy()

    # 保存初始渲染图像
    img_rendered = (np.clip(initial_rendered_np, 0, 1) * 255).astype(np.uint8)
    Image.fromarray(img_rendered).save(os.path.join(log_dir, f"rendered_step_0000.png"))

    # 用GMFlow计算从初始渲染到目标图像的光流
    flow_field_full = compute_flow_from_images(model, initial_rendered, target_image_tensor)
    flow_field_full_np = flow_field_full.cpu().numpy()

    # 采样追踪点的光流向量
    initial_flow_vectors, new_tracking_coords, _ = sample_flow_points(model, flow_field_full, current_tracking_coords)

    # 计算距离：基于光流幅度（运动大小）
    # initial_flow_vectors[:, :2] 是归一化的光流向量 [dx, dy]
    flow_vectors_norm = initial_flow_vectors[:, :2]  # (N, 2) 归一化光流
    
    # 🔍 动态点过滤：基于光流幅度筛选运动点
    if use_motion_mask:
        # 计算完整光流场的幅度
        flow_magnitude_field = np.linalg.norm(flow_field_full_np, axis=-1)  # (H, W) 像素单位
        
        # 根据阈值确定动态区域
        motion_threshold = np.percentile(flow_magnitude_field.flatten(), motion_threshold_percentile)
        
        # 对每个追踪点，检查其位置的光流幅度
        H, W = flow_magnitude_field.shape
        tracking_px = (current_tracking_coords * [W, H]).astype(np.int32)
        tracking_px[:, 0] = np.clip(tracking_px[:, 0], 0, W - 1)
        tracking_px[:, 1] = np.clip(tracking_px[:, 1], 0, H - 1)
        
        # 提取追踪点处的光流幅度（像素单位）
        point_flow_magnitude_px = flow_magnitude_field[tracking_px[:, 1], tracking_px[:, 0]]
        
        # 创建mask：只保留运动显著的点
        motion_mask = point_flow_magnitude_px > motion_threshold
        num_motion_points = motion_mask.sum()
        
        if num_motion_points > 0:
            # 计算运动点的平均光流幅度（归一化单位）
            flow_magnitude_norm = np.linalg.norm(flow_vectors_norm[motion_mask], axis=-1)  # (N_motion,)
            dist_to_final_target = flow_magnitude_norm.mean()  # 当前到最终目标的平均距离
            
            # 对于step target，计算到第一步阶段目标的距离
            # step_flows[0] 是从初始位置到第1步阶段目标的光流向量
            step_flow_magnitude = np.linalg.norm(step_flows[0][motion_mask], axis=-1).mean()
            dist_to_step_target = step_flow_magnitude  # 到第1步阶段目标的距离（应该约为总距离的1/num_steps）
            
            print(f"  🎯 Motion filtering: {num_motion_points}/{len(motion_mask)} points (threshold={motion_threshold:.1f}px)")
            print(f"  📏 Avg motion magnitude to final: {dist_to_final_target * max(image_width, image_height):.2f}px")
            print(f"  📏 Avg motion magnitude to step 1: {dist_to_step_target * max(image_width, image_height):.2f}px")
        else:
            # 如果没有检测到运动点，退回到全部点
            flow_magnitude_norm = np.linalg.norm(flow_vectors_norm, axis=-1)
            step_flow_magnitude_all = np.linalg.norm(step_flows[0], axis=-1).mean()
            dist_to_step_target = step_flow_magnitude_all
            dist_to_final_target = flow_magnitude_norm.mean()
            print(f"  ⚠ No motion points detected, using all points")
    else:
        # 不使用motion mask，计算所有点的平均光流幅度
        flow_magnitude_norm = np.linalg.norm(flow_vectors_norm, axis=-1)
        step_flow_magnitude_all = np.linalg.norm(step_flows[0], axis=-1).mean()
        dist_to_step_target = step_flow_magnitude_all
        dist_to_final_target = flow_magnitude_norm.mean()
    
    print(f"  Initial distance to step 1 target (pixels): {dist_to_step_target * max(image_width, image_height):.2f}")
    print(f"  Initial distance to final target (pixels): {dist_to_final_target * max(image_width, image_height):.2f}")

    # 更新追踪坐标
    current_tracking_coords = new_tracking_coords

    # 初始化obs_history（使用初始渲染图像和光流）
    obs_history = ObservationList(
        data_dict={
            'rgb': initial_rendered_np[None],
            'flow': initial_flow_vectors[None],
        },
        image_shape=(image_height, image_width)
    )
    for _ in range(model.num_context - 1):
        obs_history.append(obs_history[0])

    # 保存初始状态到历史
    observations.append(initial_rendered_np)
    actions.append(initial_control.copy())  # 保存初始状态的joint_pos（非规划动作，仅用于记录）
    rgb_history.append(initial_rendered_np)
    flow_history.append(initial_flow_vectors)
    current_flow = initial_flow_vectors
    rewards.append(-dist_to_step_target)  # 初始奖励（负距离）

    # 生成初始光流可视化
    hsv_flow = visualize_flow_hsv(flow_field_full_np)
    rendered_uint8 = (np.clip(initial_rendered_np, 0, 1) * 255).astype(np.uint8)
    overlay = cv2.addWeighted(rendered_uint8, 0.5, hsv_flow, 0.7, 0)

    H_vis, W_vis = overlay.shape[:2]
    current_px = (initial_flow_vectors[:, :2] * [W_vis, H_vis]).astype(np.int32)
    for i in range(0, len(current_px), max(1, len(current_px) // 100)):
        cur = tuple(current_px[i])
        if 0 <= cur[0] < W_vis and 0 <= cur[1] < H_vis:
            y, x = cur[1], cur[0]
            hsv_color = tuple(hsv_flow[y, x].tolist())
            cv2.circle(overlay, cur, radius=2, color=hsv_color, thickness=-1)

    overlay_path = os.path.join(log_dir, f"rendered_step_0000_with_flow.png")
    Image.fromarray(overlay).save(overlay_path)

    hsv_path = os.path.join(log_dir, f"flow_hsv_step_0000.png")
    Image.fromarray(hsv_flow).save(hsv_path)

    # ========== 执行MPC循环 ==========
    # 从step 1开始（step 0是初始状态，已渲染）
    for step in range(1, num_steps + 1):
        print(f"\n━━━ Step {step}/{num_steps} ━━━")

        # 确定当前步的阶段性目标
        current_step_target_flow = step_flows[step - 1]  # (N, 2) 从初始位置到阶段性目标的光流
        current_step_target_points = step_targets[step - 1]  # (N, 2) 阶段性目标位置

        # 🔍 调试信息：打印step target和final target的详细数据
        print(f"  📊 Target Data Analysis:")
        print(f"    - Current tracking coords (mean): [{current_tracking_coords[:, 0].mean():.4f}, {current_tracking_coords[:, 1].mean():.4f}]")
        print(f"    - Step target coords (mean): [{current_step_target_points[:, 0].mean():.4f}, {current_step_target_points[:, 1].mean():.4f}]")
        print(f"    - Final target coords (mean): [{full_target_points[:, 0].mean():.4f}, {full_target_points[:, 1].mean():.4f}]")
        print(f"    - Step flow magnitude (mean): {np.linalg.norm(current_step_target_flow, axis=-1).mean():.4f}")
        print(f"    - Full flow magnitude (mean): {np.linalg.norm(full_target_points - full_source_points, axis=-1).mean():.4f}")
        print(f"    - Progress to target: {step / num_steps * 100:.1f}%")

        # 为CEM准备horizon长度的目标序列（从当前步到未来horizon步）
        horizon_goal_flow = []
        for h in range(horizon):
            future_step_idx = min(step + h - 1, num_steps - 1)  # 转换为0-based索引
            horizon_goal_flow.append(step_flow_targets[future_step_idx])
        horizon_goal_flow = np.array(horizon_goal_flow)  # (horizon, N, 3)

        # 🎯 只使用光流作为目标
        print(f"    - Goal modality: FLOW ONLY (no RGB target)")

        # 设置光流目标到agent
        agent.set_goal(ObservationList(
            data_dict={
                'flow': horizon_goal_flow[None],  # (1, horizon, N, 3)
            },
            image_shape=(image_height, image_width)
        ))

        # 使用MPC规划动作（基于当前obs_history和目标）
        print(f"  Planning: MPC optimizing action for step {step} (t={step-1})")

        # 规划动作（当step=1时，t=0，会触发第一次MPC规划）
        raw_action = agent.act(
            t=step - 1,  # step=1时t=0，触发初始规划
            obs_history=obs_history,
            state_obs_history=[],
        )

        # 约束控制变化
        if len(actions) > 0:
            previous_action = actions[-1]
            prev_norm = normalize_sincos_control(previous_action)
            raw_norm = normalize_sincos_control(raw_action)
            action = constrain_control_delta(prev_norm, raw_norm, max_delta_deg=30.0)
        else:
            action = raw_action.copy()

        action_magnitude = np.linalg.norm(action - actions[-1]) if len(actions) > 0 else np.linalg.norm(action)
        print(f"  Action: [{', '.join([f'{a:.3f}' for a in action[:5]])}{'...' if len(action) > 5 else ''}]")
        print(f"  Action magnitude: {action_magnitude:.4f}")

        # 执行动作
        action_tensor = torch.tensor(action, dtype=torch.float32, device=model.device)
        next_image = model.render_with_control(action_tensor)
        next_image_np = next_image.permute(1, 2, 0).cpu().numpy()

        # 保存渲染图像
        img_rendered = (np.clip(next_image_np, 0, 1) * 255).astype(np.uint8)
        Image.fromarray(img_rendered).save(os.path.join(log_dir, f"rendered_step_{step:04d}.png"))

        # 计算到阶段性目标和最终目标的距离
        # 阶段性目标：使用预先计算好的 current_step_target_points
        # current_tracking_coords 是当前追踪点位置 (N, 2)
        # current_step_target_points 是当前步的阶段目标位置 (N, 2)
        current_to_step_flow = current_step_target_points - current_tracking_coords  # (N, 2)
        
        # 最终目标：用GMFlow计算到最终目标图像的光流
        flow_field_full = compute_flow_from_images(model, next_image, target_image_tensor)
        flow_field_full_np = flow_field_full.cpu().numpy()
        next_flow_vectors, new_tracking_coords, _ = sample_flow_points(model, flow_field_full, current_tracking_coords)
        flow_to_final = next_flow_vectors[:, :2]  # (N, 2) 归一化光流向量到最终目标
        
        # 🔍 动态点过滤：基于到最终目标的光流幅度筛选运动点
        if use_motion_mask:
            # 计算完整光流场的幅度
            flow_magnitude_field = np.linalg.norm(flow_field_full_np, axis=-1)  # (H, W) 像素单位
            
            # 根据阈值确定动态区域
            motion_threshold = np.percentile(flow_magnitude_field.flatten(), motion_threshold_percentile)
            
            # 对每个追踪点，检查其位置的光流幅度
            H, W = flow_magnitude_field.shape
            tracking_px = (current_tracking_coords * [W, H]).astype(np.int32)
            tracking_px[:, 0] = np.clip(tracking_px[:, 0], 0, W - 1)
            tracking_px[:, 1] = np.clip(tracking_px[:, 1], 0, H - 1)
            
            # 提取追踪点处的光流幅度（像素单位）
            point_flow_magnitude_px = flow_magnitude_field[tracking_px[:, 1], tracking_px[:, 0]]
            
            # 创建mask：只保留运动显著的点
            motion_mask = point_flow_magnitude_px > motion_threshold
            num_motion_points = motion_mask.sum()
            
            if num_motion_points > 0:
                # 分别计算到阶段目标和最终目标的距离
                dist_to_step_target = np.linalg.norm(current_to_step_flow[motion_mask], axis=-1).mean()
                dist_to_final_target = np.linalg.norm(flow_to_final[motion_mask], axis=-1).mean()
                
                print(f"  🎯 Motion filtering: {num_motion_points}/{len(motion_mask)} points (threshold={motion_threshold:.1f}px)")
            else:
                # 如果没有检测到运动点，退回到全部点
                dist_to_step_target = np.linalg.norm(current_to_step_flow, axis=-1).mean()
                dist_to_final_target = np.linalg.norm(flow_to_final, axis=-1).mean()
                print(f"  ⚠ No motion points detected, using all points")
        else:
            # 不使用motion mask
            dist_to_step_target = np.linalg.norm(current_to_step_flow, axis=-1).mean()
            dist_to_final_target = np.linalg.norm(flow_to_final, axis=-1).mean()

        # 计算奖励
        reward = -dist_to_step_target
        rewards.append(reward)

        # 打印信息
        print(f"  Distance to step target (normalized): {dist_to_step_target:.4f}")
        print(f"  Distance to final target (normalized): {dist_to_final_target:.4f}")
        print(f"  Distance to step target (pixels): {dist_to_step_target * max(image_width, image_height):.2f}")
        print(f"  Distance to final target (pixels): {dist_to_final_target * max(image_width, image_height):.2f}")
        print(f"  Reward: {reward:.4f}")

        # 更新历史记录
        actions.append(action)
        observations.append(next_image_np)
        rgb_history.append(next_image_np)
        flow_history.append(next_flow_vectors)
        current_flow = next_flow_vectors
        current_tracking_coords = new_tracking_coords

        # 更新obs_history用于下一步CEM
        obs_history = ObservationList(
            data_dict={
                'rgb': next_image_np[None],
                'flow': next_flow_vectors[None],
            },
            image_shape=(image_height, image_width)
        )
        for _ in range(model.num_context - 1):
            obs_history.append(obs_history[0])

        # 光流可视化
        hsv_flow = visualize_flow_hsv(flow_field_full_np)
        rendered_uint8 = (np.clip(next_image_np, 0, 1) * 255).astype(np.uint8)
        overlay = cv2.addWeighted(rendered_uint8, 0.5, hsv_flow, 0.7, 0)

        H_vis, W_vis = overlay.shape[:2]
        current_px = (next_flow_vectors[:, :2] * [W_vis, H_vis]).astype(np.int32)
        for i in range(0, len(current_px), max(1, len(current_px) // 100)):
            cur = tuple(current_px[i])
            if 0 <= cur[0] < W_vis and 0 <= cur[1] < H_vis:
                y, x = cur[1], cur[0]
                hsv_color = tuple(hsv_flow[y, x].tolist())
                cv2.circle(overlay, cur, radius=2, color=hsv_color, thickness=-1)

        overlay_path = os.path.join(log_dir, f"rendered_step_{step:04d}_with_flow.png")
        Image.fromarray(overlay).save(overlay_path)

        hsv_path = os.path.join(log_dir, f"flow_hsv_step_{step:04d}.png")
        Image.fromarray(hsv_flow).save(hsv_path)

    print("\n" + "="*70)
    print("MPC Control Completed!")
    print("="*70)
    
    # 10. 保存结果
    print(f"\nSaving results...")

    # 保存动作序列和奖励
    np.save(os.path.join(log_dir, "actions.npy"), np.array(actions))
    np.save(os.path.join(log_dir, "rewards.npy"), np.array(rewards))
    np.save(os.path.join(log_dir, "final_flow.npy"), current_flow)
    np.save(os.path.join(log_dir, "step_targets.npy"), step_targets)
    np.save(os.path.join(log_dir, "step_flows.npy"), step_flows)
    print("  ✓ Saved action sequence and rewards")

    # 11. 生成视频（可选）
    if save_video:
        print("\nGenerating videos...")

        # 1. 渲染序列视频（从rendered_step_0000.png开始，包含初始状态）
        rendered_frames = []
        for step in range(0, num_steps + 1):  # 从0开始，包含初始状态
            img_path = os.path.join(log_dir, f"rendered_step_{step:04d}.png")
            if os.path.exists(img_path):
                img = Image.open(img_path)
                rendered_frames.append(np.array(img))

        if rendered_frames:
            gif_path = os.path.join(log_dir, "trajectory.gif")
            write_moviepy_gif(rendered_frames, gif_path, fps=video_fps)
            print(f"  ✓ Saved trajectory video: trajectory.gif ({len(rendered_frames)} frames @ {video_fps}fps)")

        # 2. 带光流叠加的视频（从rendered_step_0000.png开始）
        flow_frames = []
        for step in range(0, num_steps + 1):  # 从0开始
            img_path = os.path.join(log_dir, f"rendered_step_{step:04d}_with_flow.png")
            if os.path.exists(img_path):
                img = Image.open(img_path)
                flow_frames.append(np.array(img))

        if flow_frames:
            gif_path = os.path.join(log_dir, "trajectory_with_flow.gif")
            write_moviepy_gif(flow_frames, gif_path, fps=video_fps)
            print(f"  ✓ Saved flow overlay video: trajectory_with_flow.gif ({len(flow_frames)} frames @ {video_fps}fps)")

        # 3. HSV光流序列视频（从flow_hsv_step_0000.png开始）
        hsv_frames = []
        for step in range(0, num_steps + 1):  # 从0开始
            img_path = os.path.join(log_dir, f"flow_hsv_step_{step:04d}.png")
            if os.path.exists(img_path):
                img = Image.open(img_path)
                hsv_frames.append(np.array(img))

        if hsv_frames:
            gif_path = os.path.join(log_dir, "flow_hsv_sequence.gif")
            write_moviepy_gif(hsv_frames, gif_path, fps=video_fps)
            print(f"  ✓ Saved HSV flow video: flow_hsv_sequence.gif ({len(hsv_frames)} frames @ {video_fps}fps)")
    
    # 打印统计
    print(f"\n━━━ Final Statistics ━━━")
    print(f"  Total steps executed: {len(actions)}")
    print(f"  Mean reward: {np.mean(rewards):.4f}")
    print(f"  Final flow distance: {-rewards[-1]:.4f}")
    print(f"  Initial flow distance: {-rewards[0]:.4f}" if len(rewards) > 0 else "")
    print(f"  Improvement: {((-rewards[0]) - (-rewards[-1])):.4f}" if len(rewards) > 0 else "")
    
    print(f"\n━━━ Output Directory: {log_dir} ━━━")
    print(f"  Input images:")
    print(f"    - initial_image.png: Initial state")
    print(f"    - target_image.png: Target state")
    print(f"    - target_flow_hsv.png: HSV flow visualization")
    print(f"  Rendered trajectory:")
    print(f"    - rendered_step_*.png: Rendered frames after each action")
    if save_video:
        print(f"  Videos:")
        print(f"    - trajectory.gif: Rendered sequence animation")
        print(f"    - trajectory_with_flow.gif: With flow overlay")
        print(f"    - flow_hsv_sequence.gif: HSV flow evolution")
    print(f"  Data:")
    print(f"    - actions.npy: Control sequence ({len(actions)} steps)")
    print(f"    - rewards.npy: Reward trajectory")
    print(f"    - final_flow.npy / target_flow.npy: Flow states")
    print("="*70)
    
    return {
        'observations': observations,
        'actions': actions,
        'rewards': rewards,
        'final_flow': current_flow,
        'target_flow': step_flows[-1],  # 最终目标光流
    }


def main():
    parser = argparse.ArgumentParser(
        description='光流引导的4DGaussians MPC控制 - 从图像到图像'
    )
    parser.add_argument('--model_path', type=str, required=True,
                        help='4DGaussians模型路径')
    parser.add_argument('--iteration', type=int, default=5000,
                        help='模型迭代次数')
    parser.add_argument('--initial_image', type=str, required=True,
                        help='初始图像路径')
    parser.add_argument('--target_image', type=str, required=True,
                        help='目标图像路径')
    parser.add_argument('--control_dim', type=int, required=True,
                        help='控制向量维度')
    parser.add_argument('--num_steps', type=int, default=20,
                        help='执行步数')
    parser.add_argument('--horizon', type=int, default=10,
                        help='规划horizon')
    parser.add_argument('--num_samples', type=int, default=64,
                        help='CEM采样数')
    parser.add_argument('--opt_iters', type=int, default=10,
                        help='CEM优化迭代数')
    parser.add_argument('--flow_weight', type=float, default=0.7,
                        help='光流目标权重（默认0.7用于70%%光流策略）')
    parser.add_argument('--image_weight', type=float, default=0.3,
                        help='图像一致性权重（默认0.3用于30%%图像策略）')
    parser.add_argument('--vgg_weight', type=float, default=0.0,
                        help='VGG感知一致性权重（默认0.0不启用）')
    parser.add_argument('--vgg_layer', type=str, default='relu3_3',
                        help='VGG感知层（默认relu3_3）')
    parser.add_argument('--image_weight_schedule', type=str, default='none',
                        choices=['none', 'linear'],
                        help='图像类损失权重调度方式')
    parser.add_argument('--image_weight_start', type=float, default=0.0,
                        help='图像类损失起始权重（线性调度起点）')
    parser.add_argument('--use_sparse_render', action='store_true',
                        help='使用稀疏渲染')
    parser.add_argument('--sparse_ratio', type=float, default=0.15,
                        help='稀疏渲染比例')
    parser.add_argument('--image_height', type=int, default=480,
                        help='图像高度')
    parser.add_argument('--image_width', type=int, default=480,
                        help='图像宽度')
    parser.add_argument('--log_dir', type=str, default='./outputs/flow_guided_mpc',
                        help='输出目录')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='计算设备 (例如: cuda:0, cuda:3, cpu)')
    parser.add_argument('--sampling_strategy', type=str, default='adaptive',
                        choices=['uniform', 'adaptive', 'motion_only'],
                        help='光流采样策略: uniform=均匀, adaptive=自适应(推荐), motion_only=纯运动')
    parser.add_argument('--motion_focus_ratio', type=float, default=0.7,
                        help='自适应采样中聚焦运动区域的比例 (0.0-1.0)')

    # 方向指引损失参数
    parser.add_argument('--direction_weight', type=float, default=0.0,
                        help='光流方向指引损失的权重（默认0.0不启用）')
    parser.add_argument('--direction_loss_type', type=str, default='cosine',
                        choices=['cosine', 'angle'],
                        help='光流方向损失类型: cosine=余弦相似度, angle=角度差')

    # 动作正则化参数
    parser.add_argument('--action_regularization_weight', type=float, default=0.0,
                        help='动作正则化损失权重（惩罚过大的动作变化，默认0.0不启用）')
    parser.add_argument('--max_action_delta', type=float, default=0.5,
                        help='允许的最大动作变化（弧度），超出此值将受到惩罚（默认0.5）')

    # 动态点过滤参数
    parser.add_argument('--use_motion_mask', action='store_true',
                        help='启用基于光流幅度的动态点过滤（忽略静态场景点）')
    parser.add_argument('--motion_threshold_percentile', type=float, default=50.0,
                        help='运动阈值百分位数（0-100，默认50表示中位数）')

    # 相机参数
    parser.add_argument('--camera_distance', type=float, default=2.0,
                        help='相机到原点的距离')
    parser.add_argument('--camera_elevation', type=float, default=0.0,
                        help='相机仰角（度）: 0=水平, 90=正上方俯视, -90=正下方仰视')
    parser.add_argument('--camera_azimuth', type=float, default=0.0,
                        help='相机方位角（度）: 在XZ平面上的旋转, 0=沿+Z轴')
    parser.add_argument('--fov_degrees', type=float, default=45.0,
                        help='相机视野角度（度）')
    parser.add_argument('--transforms_json', type=str, required=True,
                        help='transforms.json路径（用于读取joint_pos和相机参数）')
    parser.add_argument('--save_video', action='store_true',
                        help='保存渲染序列为视频(GIF/MP4)')
    parser.add_argument('--video_fps', type=int, default=10,
                        help='视频帧率')
    
    args = parser.parse_args()
    
    # 使用提前设置好的actual_device
    if args.device == 'cpu':
        device_to_use = 'cpu'
        print(f"✓ Using device: CPU")
    else:
        device_to_use = actual_device
        print(f"✓ Using GPU: {actual_device_id} (mapped to cuda)")
    
    # 运行MPC
    results = run_mpc_from_images(
        model_path=args.model_path,
        iteration=args.iteration,
        initial_image_path=args.initial_image,
        target_image_path=args.target_image,
        control_dim=args.control_dim,
        num_steps=args.num_steps,
        horizon=args.horizon,
        num_samples=args.num_samples,
        opt_iters=args.opt_iters,
        flow_weight=args.flow_weight,
        image_weight=args.image_weight,
        vgg_weight=args.vgg_weight,
        vgg_layer=args.vgg_layer,
        image_weight_schedule=args.image_weight_schedule,
        image_weight_start=args.image_weight_start,
        use_sparse_render=args.use_sparse_render,
        sparse_ratio=args.sparse_ratio,
        image_height=args.image_height,
        image_width=args.image_width,
        device=device_to_use,
        log_dir=args.log_dir,
        sampling_strategy=args.sampling_strategy,
        motion_focus_ratio=args.motion_focus_ratio,
        camera_distance=args.camera_distance,
        camera_elevation=args.camera_elevation,
        camera_azimuth=args.camera_azimuth,
        fov_degrees=args.fov_degrees,
        transforms_json_path=args.transforms_json,
        save_video=args.save_video,
        video_fps=args.video_fps,
        direction_weight=args.direction_weight,
        direction_loss_type=args.direction_loss_type,
        action_regularization_weight=args.action_regularization_weight,
        max_action_delta=args.max_action_delta,
        use_motion_mask=args.use_motion_mask,
        motion_threshold_percentile=args.motion_threshold_percentile,
    )


if __name__ == "__main__":
    main()
