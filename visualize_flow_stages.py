#!/usr/bin/env python3
"""
可视化MPC分步光流目标 - 独立版本

直接从初始/目标图像计算光流并分段，不依赖已有的.npy文件
复现demo_flow_guided_mpc.py中的分段逻辑
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import argparse
import os
import sys
from pathlib import Path
import torch

# 添加GMFlow路径
sys.path.insert(0, str(Path(__file__).parent))

try:
    from gmflow.gmflow import GMFlow
    from gmflow.config import get_cfg as get_gmflow_cfg
    GMFLOW_AVAILABLE = True
    print("✓ GMFlow loaded successfully")
except ImportError as e:
    print(f"Warning: GMFlow not available - {e}")
    GMFLOW_AVAILABLE = False


def load_image(image_path, target_size=None):
    """加载图像并归一化到[0, 1]"""
    img = Image.open(image_path).convert('RGB')
    if target_size is not None:
        img = img.resize((target_size[1], target_size[0]), Image.LANCZOS)
    img_np = np.array(img).astype(np.float32) / 255.0
    return img_np


def compute_flow_with_gmflow(initial_image, target_image, device='cuda'):
    """
    使用GMFlow计算从initial到target的光流
    
    Args:
        initial_image: (H, W, 3) numpy array [0, 1]
        target_image: (H, W, 3) numpy array [0, 1]
        device: 计算设备
    
    Returns:
        flow_field: (H, W, 2) numpy array - 光流场
        source_points: (N, 2) - 采样的源点
        target_points: (N, 2) - 对应的目标点
        flow_vectors: (N, 2) - 光流向量
    """
    if not GMFLOW_AVAILABLE:
        raise ImportError("GMFlow is required but not available")
    
    H, W = initial_image.shape[:2]
    
    # 初始化GMFlow
    print("  Loading GMFlow model...")
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
    
    checkpoint_path = "gmflow/checkpoints/gmflow_sintel-0c07dcb3.pth"
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        flownet.load_state_dict(checkpoint['model'], strict=False)
        print("  ✓ Loaded GMFlow checkpoint")
    else:
        raise FileNotFoundError(f"GMFlow checkpoint not found: {checkpoint_path}")
    
    flownet.eval()
    
    # 准备输入
    img1 = torch.from_numpy(initial_image).permute(2, 0, 1).float().unsqueeze(0).to(device)
    img2 = torch.from_numpy(target_image).permute(2, 0, 1).float().unsqueeze(0).to(device)
    
    # 归一化到[-1, 1]
    img1_norm = img1 * 2.0 - 1.0
    img2_norm = img2 * 2.0 - 1.0
    
    # 计算光流
    print("  Computing optical flow...")
    with torch.no_grad():
        flow_predictions = flownet(
            img1_norm, img2_norm,
            attn_splits_list=[2],
            corr_radius_list=[-1],
            prop_radius_list=[-1],
        )
        flow_field = flow_predictions[-1]  # (1, 2, H, W)
    
    flow_field_np = flow_field[0].permute(1, 2, 0).cpu().numpy()  # (H, W, 2)
    
    # 采样光流点 - 使用adaptive策略
    print("  Sampling flow points (adaptive strategy)...")
    num_flow_points = 512
    motion_focus_ratio = 0.7
    
    flow_magnitude = np.linalg.norm(flow_field_np, axis=-1)
    motion_threshold = np.percentile(flow_magnitude, 50)
    motion_mask = flow_magnitude > motion_threshold
    
    # 70%从运动区域采样
    num_motion = int(num_flow_points * motion_focus_ratio)
    motion_coords = np.argwhere(motion_mask)
    if len(motion_coords) > num_motion:
        motion_indices = np.random.choice(len(motion_coords), num_motion, replace=False)
        motion_samples = motion_coords[motion_indices]
    else:
        motion_samples = motion_coords
    
    # 30%均匀采样
    num_uniform = num_flow_points - len(motion_samples)
    grid_step = int(np.sqrt(H * W / num_uniform))
    yy, xx = np.meshgrid(
        np.arange(grid_step // 2, H, grid_step),
        np.arange(grid_step // 2, W, grid_step),
        indexing='ij'
    )
    uniform_samples = np.stack([yy.ravel(), xx.ravel()], axis=-1)
    
    # 合并采样点
    all_samples = np.vstack([motion_samples, uniform_samples])
    if len(all_samples) > num_flow_points:
        indices = np.random.choice(len(all_samples), num_flow_points, replace=False)
        all_samples = all_samples[indices]
    
    # 提取源点和光流
    source_points = all_samples[:, [1, 0]].astype(np.float32)  # (N, 2) [x, y]
    flow_vectors = flow_field_np[all_samples[:, 0], all_samples[:, 1]]  # (N, 2)
    target_points = source_points + flow_vectors  # (N, 2)
    
    print(f"  ✓ Sampled {len(source_points)} flow points")
    print(f"    Mean flow magnitude: {np.linalg.norm(flow_vectors, axis=1).mean():.2f} pixels")
    
    return flow_field_np, source_points, target_points, flow_vectors


def smooth_interpolation(t, smoothing='ease_in_out'):
    """
    平滑插值函数，用于生成更自然的运动曲线
    
    Args:
        t: 插值系数 [0, 1]
        smoothing: 平滑类型
            - 'linear': 线性插值（无平滑）
            - 'ease_in_out': 二次ease-in-out（慢-快-慢）
            - 'cubic': 三次平滑
            - 'quintic': 五次平滑（最平滑）
    
    Returns:
        平滑后的插值系数 [0, 1]
    """
    if smoothing == 'linear':
        return t
    elif smoothing == 'ease_in_out':
        # 二次ease-in-out: 开始和结束时较慢，中间较快
        return t * t * (3.0 - 2.0 * t)
    elif smoothing == 'cubic':
        # 三次平滑
        return t * t * t * (t * (6.0 * t - 15.0) + 10.0)
    elif smoothing == 'quintic':
        # 五次平滑（最平滑）
        return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)
    else:
        return t


def filter_points_by_region(source_points, flow_vectors, target_points, image_shape, 
                           focus_region='gripper'):
    """
    根据区域过滤光流点，只保留感兴趣区域的点
    
    Args:
        source_points: (N, 2) - 源点坐标
        flow_vectors: (N, 2) - 光流向量
        target_points: (N, 2) - 目标点坐标
        image_shape: (H, W) - 图像尺寸
        focus_region: 关注区域类型
            - 'all': 保留所有点
            - 'gripper': 只保留机械爪区域（图像右侧）
            - 'upper_joints': 只保留上半部分关节
            - 'gripper_and_joints': 机械爪+前几个关节
    
    Returns:
        filtered_source: (M, 2) - 过滤后的源点
        filtered_flow: (M, 2) - 过滤后的光流
        filtered_target: (M, 2) - 过滤后的目标点
        mask: (N,) bool - 过滤掩码
    """
    H, W = image_shape
    N = len(source_points)
    
    if focus_region == 'all':
        mask = np.ones(N, dtype=bool)
    
    elif focus_region == 'gripper':
        # 机械爪通常在图像右侧区域
        # 保留x > 0.6*W 的点
        mask = source_points[:, 0] > (0.6 * W)
    
    elif focus_region == 'upper_joints':
        # 上半部分的关节
        # 保留y < 0.5*H 的点
        mask = source_points[:, 1] < (0.5 * H)
    
    elif focus_region == 'gripper_and_joints':
        # 机械爪（右侧）或上半部分关节
        gripper_mask = source_points[:, 0] > (0.5 * W)
        upper_mask = source_points[:, 1] < (0.6 * H)
        mask = gripper_mask | upper_mask
    
    elif focus_region == 'high_motion':
        # 只保留运动幅度较大的点
        flow_magnitude = np.linalg.norm(flow_vectors, axis=1)
        threshold = np.percentile(flow_magnitude, 70)
        mask = flow_magnitude > threshold
    
    else:
        mask = np.ones(N, dtype=bool)
    
    filtered_source = source_points[mask]
    filtered_flow = flow_vectors[mask]
    filtered_target = target_points[mask]
    
    print(f"  Region filter '{focus_region}': {np.sum(mask)}/{N} points retained ({100*np.sum(mask)/N:.1f}%)")
    
    return filtered_source, filtered_flow, filtered_target, mask


def create_step_targets(source_points, target_points, num_steps=25, smoothing='linear'):
    """
    创建分步目标序列 - 支持多种插值平滑方式
    
    Args:
        source_points: (N, 2) - 初始点位置
        target_points: (N, 2) - 最终目标位置
        num_steps: 分步数量
        smoothing: 平滑类型 ('linear', 'ease_in_out', 'cubic', 'quintic')
    
    Returns:
        step_targets: (num_steps, N, 2) - 每步的目标位置
        step_flows: (num_steps, N, 2) - 每步的光流向量（从起点到阶段性目标）
        alphas: (num_steps,) - 每步的插值系数
    """
    step_targets = []
    step_flows = []
    alphas = []
    
    for step in range(num_steps):
        t = (step + 1) / num_steps  # 原始时间参数 [0, 1]
        alpha = smooth_interpolation(t, smoothing)  # 平滑后的插值系数
        alphas.append(alpha)
        
        # 第step步的目标位置：从source到target的平滑插值
        target_at_step = source_points * (1 - alpha) + target_points * alpha
        # 第step步的光流：从初始位置到阶段性目标的位移
        flow_at_step = target_at_step - source_points
        step_targets.append(target_at_step)
        step_flows.append(flow_at_step)
    
    step_targets = np.array(step_targets)  # (num_steps, N, 2)
    step_flows = np.array(step_flows)  # (num_steps, N, 2)
    alphas = np.array(alphas)
    
    print(f"\n  Created {num_steps} step-wise flow targets (smoothing: {smoothing})")
    print(f"  Step 0: mean flow magnitude = {np.linalg.norm(step_flows[0], axis=-1).mean():.4f} pixels")
    print(f"  Step {num_steps-1}: mean flow magnitude = {np.linalg.norm(step_flows[-1], axis=-1).mean():.4f} pixels")
    
    return step_targets, step_flows, alphas


def visualize_step_targets_overlay(initial_image, step_targets, step_flows, output_path):
    """在初始图像上叠加所有步骤的目标点"""
    num_steps, num_points, _ = step_targets.shape
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 12), dpi=150)
    ax.imshow(initial_image)
    
    cmap = plt.cm.get_cmap('coolwarm')
    
    for step_idx in range(num_steps):
        targets = step_targets[step_idx]
        color = cmap(step_idx / max(num_steps - 1, 1))
        
        ax.scatter(
            targets[:, 0], targets[:, 1],
            c=[color], s=3, alpha=0.6,
            label=f'Step {step_idx}' if step_idx % 5 == 0 else None
        )
    
    ax.set_title(f'Step-by-Step Flow Targets Overlay ({num_steps} steps, {num_points} points)', 
                 fontsize=14, fontweight='bold')
    ax.axis('off')
    
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=num_steps-1))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Step Number', fontsize=12)
    
    if num_steps > 5:
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(handles, labels, loc='upper right', fontsize=8, markerscale=2)
    
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"✓ Saved: {output_path}")


def visualize_comparison_grid(initial_image, target_image, step_targets_dict, output_path):
    """创建对比网格 - 对比不同插值方法"""
    num_methods = len(step_targets_dict)
    
    fig = plt.figure(figsize=(6 * (num_methods + 2), 6))
    gs = fig.add_gridspec(1, num_methods + 2, hspace=0.3, wspace=0.1)
    
    # 初始图像
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(initial_image)
    ax1.set_title('Initial State', fontsize=14, fontweight='bold')
    ax1.axis('off')
    
    # 每种插值方法
    cmap = plt.cm.get_cmap('coolwarm')
    for idx, (method_name, step_targets) in enumerate(step_targets_dict.items()):
        ax = fig.add_subplot(gs[0, idx + 1])
        ax.imshow(initial_image)
        
        num_steps = step_targets.shape[0]
        for step_idx in range(0, num_steps, max(1, num_steps // 10)):
            targets = step_targets[step_idx]
            color = cmap(step_idx / max(num_steps - 1, 1))
            ax.scatter(targets[:, 0], targets[:, 1], c=[color], s=5, alpha=0.6)
        
        ax.set_title(f'{method_name}\n({num_steps} steps)', fontsize=12, fontweight='bold')
        ax.axis('off')
    
    # 目标图像
    ax_last = fig.add_subplot(gs[0, -1])
    ax_last.imshow(target_image)
    ax_last.set_title('Target State', fontsize=14, fontweight='bold')
    ax_last.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"✓ Saved: {output_path}")


def visualize_flow_magnitude_progression(step_flows, output_path):
    """可视化光流幅度随步骤的变化"""
    num_steps = step_flows.shape[0]
    
    flow_magnitudes = []
    for step_idx in range(num_steps):
        magnitudes = np.linalg.norm(step_flows[step_idx], axis=1)
        flow_magnitudes.append({
            'mean': np.mean(magnitudes),
            'std': np.std(magnitudes),
            'max': np.max(magnitudes),
            'min': np.min(magnitudes)
        })
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    steps = np.arange(num_steps)
    means = [m['mean'] for m in flow_magnitudes]
    stds = [m['std'] for m in flow_magnitudes]
    maxs = [m['max'] for m in flow_magnitudes]
    mins = [m['min'] for m in flow_magnitudes]
    
    # 平均光流幅度
    ax1.plot(steps, means, 'b-', linewidth=2, label='Mean')
    ax1.fill_between(steps, 
                     np.array(means) - np.array(stds),
                     np.array(means) + np.array(stds),
                     alpha=0.3, color='blue', label='±1 Std')
    ax1.set_xlabel('Step', fontsize=12)
    ax1.set_ylabel('Flow Magnitude (pixels)', fontsize=12)
    ax1.set_title('Average Flow Magnitude per Step', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # 最大和最小光流幅度
    ax2.plot(steps, maxs, 'r-', linewidth=2, label='Max')
    ax2.plot(steps, mins, 'g-', linewidth=2, label='Min')
    ax2.fill_between(steps, mins, maxs, alpha=0.2, color='gray')
    ax2.set_xlabel('Step', fontsize=12)
    ax2.set_ylabel('Flow Magnitude (pixels)', fontsize=12)
    ax2.set_title('Min/Max Flow Magnitude per Step', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"✓ Saved: {output_path}")


def visualize_target_heatmap(initial_image, step_targets, output_path):
    """创建目标点密度热力图"""
    H, W = initial_image.shape[:2]
    num_steps = step_targets.shape[0]
    
    heatmap = np.zeros((H, W), dtype=np.float32)
    
    for step_idx in range(num_steps):
        targets = step_targets[step_idx]
        for point in targets:
            x, y = int(point[0]), int(point[1])
            if 0 <= x < W and 0 <= y < H:
                for dy in range(-2, 3):
                    for dx in range(-2, 3):
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < W and 0 <= ny < H:
                            weight = np.exp(-(dx**2 + dy**2) / 2.0)
                            heatmap[ny, nx] += weight
    
    if heatmap.max() > 0:
        heatmap = heatmap / heatmap.max()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # 叠加热力图
    ax1.imshow(initial_image)
    im1 = ax1.imshow(heatmap, cmap='hot', alpha=0.5)
    ax1.set_title('Target Density Heatmap Overlay', fontsize=14, fontweight='bold')
    ax1.axis('off')
    plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    
    # 纯热力图
    im2 = ax2.imshow(heatmap, cmap='hot')
    ax2.set_title('Target Density Heatmap', fontsize=14, fontweight='bold')
    ax2.axis('off')
    plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"✓ Saved: {output_path}")


def visualize_animation_frames(initial_image, target_image, source_points, step_targets, output_dir):
    """创建动画帧序列"""
    num_steps = step_targets.shape[0]
    animation_dir = os.path.join(output_dir, "step_animation")
    os.makedirs(animation_dir, exist_ok=True)
    
    cmap = plt.cm.get_cmap('coolwarm')
    
    for step_idx in range(num_steps):
        fig, ax = plt.subplots(1, 1, figsize=(10, 10), dpi=120)
        ax.imshow(initial_image)
        
        # 源点（灰色）
        ax.scatter(source_points[:, 0], source_points[:, 1], 
                  c='gray', s=15, alpha=0.3, label='Source')
        
        # 当前步骤之前的轨迹
        for prev_step in range(step_idx + 1):
            targets = step_targets[prev_step]
            color = cmap(prev_step / max(num_steps - 1, 1))
            alpha = 0.3 if prev_step < step_idx else 0.9
            size = 8 if prev_step < step_idx else 20
            
            ax.scatter(targets[:, 0], targets[:, 1],
                      c=[color], s=size, alpha=alpha,
                      label=f'Step {prev_step}' if prev_step == step_idx else None)
        
        # 部分点的箭头
        current_targets = step_targets[step_idx]
        arrow_indices = np.arange(0, len(source_points), 10)
        for idx in arrow_indices:
            src = source_points[idx]
            tgt = current_targets[idx]
            dx, dy = tgt[0] - src[0], tgt[1] - src[1]
            ax.arrow(src[0], src[1], dx, dy, 
                    color=cmap(step_idx / max(num_steps - 1, 1)),
                    alpha=0.4, width=0.5, head_width=3, head_length=3)
        
        ax.set_title(f'Progress to Step {step_idx}/{num_steps-1}', 
                    fontsize=14, fontweight='bold')
        ax.axis('off')
        ax.legend(loc='upper right', fontsize=10, markerscale=1.5)
        
        plt.tight_layout()
        frame_path = os.path.join(animation_dir, f"frame_{step_idx:04d}.png")
        plt.savefig(frame_path, bbox_inches='tight', dpi=120)
        plt.close()
    
    print(f"✓ Saved {num_steps} animation frames to: {animation_dir}")


def main():
    parser = argparse.ArgumentParser(description="Visualize MPC step-wise flow targets (standalone)")
    parser.add_argument("--initial_image", type=str, 
                       default="/home/ubuntu/yyf/4DGaussians/assets/start-end/cam5_sample1_frame_00001.jpg",
                       help="Path to initial image")
    parser.add_argument("--target_image", type=str,
                       default="/home/ubuntu/yyf/4DGaussians/assets/start-end/cam5_sample1_frame_00018.jpg",
                       help="Path to target image")
    parser.add_argument("--num_steps", type=int, default=25,
                       help="Number of steps to divide the trajectory")
    parser.add_argument("--image_size", type=int, nargs=2, default=[480, 480],
                       help="Image size (H W)")
    parser.add_argument("--output_dir", type=str, default="outputs/flow_stage_visualization",
                       help="Output directory for visualizations")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device for computation")
    parser.add_argument("--create_animation", action="store_true",
                       help="Create animation frames")
    parser.add_argument("--smoothing", type=str, default="ease_in_out",
                       choices=['linear', 'ease_in_out', 'cubic', 'quintic'],
                       help="Interpolation smoothing type")
    parser.add_argument("--focus_region", type=str, default="gripper_and_joints",
                       choices=['all', 'gripper', 'upper_joints', 'gripper_and_joints', 'high_motion'],
                       help="Focus on specific region")
    parser.add_argument("--compare_methods", action="store_true",
                       help="Compare different interpolation methods")
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"\n{'='*70}")
    print(f"MPC Flow Stage Visualization (Standalone)")
    print(f"{'='*70}")
    print(f"Initial image: {args.initial_image}")
    print(f"Target image: {args.target_image}")
    print(f"Num steps: {args.num_steps}")
    print(f"Output dir: {args.output_dir}")
    
    # 1. 加载图像
    print(f"\n[1/7] Loading images...")
    initial_image = load_image(args.initial_image, tuple(args.image_size))
    target_image = load_image(args.target_image, tuple(args.image_size))
    print(f"  ✓ Loaded images: {initial_image.shape}")
    
    # 2. 计算光流
    print(f"\n[2/8] Computing optical flow with GMFlow...")
    flow_field, source_points, target_points, flow_vectors = compute_flow_with_gmflow(
        initial_image, target_image, device=args.device
    )
    
    # 2.5. 区域过滤
    print(f"\n[3/8] Filtering points by region...")
    filtered_source, filtered_flow, filtered_target, region_mask = filter_points_by_region(
        source_points, flow_vectors, target_points,
        image_shape=initial_image.shape[:2],
        focus_region=args.focus_region
    )
    
    # 3. 创建分步目标
    if args.compare_methods:
        print(f"\n[4/8] Creating step-wise targets (comparing methods)...")
        methods = ['linear', 'ease_in_out', 'cubic', 'quintic']
        step_targets_dict = {}
        step_flows_dict = {}
        alphas_dict = {}
        
        for method in methods:
            targets, flows, alphas = create_step_targets(
                filtered_source, filtered_target, 
                num_steps=args.num_steps,
                smoothing=method
            )
            step_targets_dict[method] = targets
            step_flows_dict[method] = flows
            alphas_dict[method] = alphas
        
        # 使用主要方法进行后续可视化
        step_targets = step_targets_dict[args.smoothing]
        step_flows = step_flows_dict[args.smoothing]
        alphas = alphas_dict[args.smoothing]
    else:
        print(f"\n[4/8] Creating step-wise targets...")
        step_targets, step_flows, alphas = create_step_targets(
            filtered_source, filtered_target,
            num_steps=args.num_steps,
            smoothing=args.smoothing
        )
    
    # 4. 叠加可视化
    print(f"\n[5/8] Creating overlay visualization...")
    overlay_path = os.path.join(args.output_dir, f"step_targets_overlay_{args.smoothing}_{args.focus_region}.png")
    visualize_step_targets_overlay(initial_image, step_targets, step_flows, overlay_path)
    
    # 5. 对比网格
    print(f"\n[6/8] Creating comparison grid...")
    if args.compare_methods:
        grid_path = os.path.join(args.output_dir, f"comparison_methods_{args.focus_region}.png")
        visualize_comparison_grid(initial_image, target_image, step_targets_dict, grid_path)
    else:
        grid_path = os.path.join(args.output_dir, f"comparison_grid_{args.smoothing}_{args.focus_region}.png")
        visualize_comparison_grid(initial_image, target_image, {args.smoothing: step_targets}, grid_path)
    
    # 6. 光流幅度分析（对比不同方法）
    print(f"\n[7/8] Analyzing flow magnitudes...")
    if args.compare_methods:
        # 对比所有方法
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()
        
        for idx, (method, flows) in enumerate(step_flows_dict.items()):
            ax = axes[idx]
            num_steps = flows.shape[0]
            steps = np.arange(num_steps)
            means = [np.linalg.norm(flows[s], axis=1).mean() for s in range(num_steps)]
            stds = [np.linalg.norm(flows[s], axis=1).std() for s in range(num_steps)]
            
            ax.plot(steps, means, linewidth=2, label='Mean')
            ax.fill_between(steps, 
                           np.array(means) - np.array(stds),
                           np.array(means) + np.array(stds),
                           alpha=0.3)
            ax.set_xlabel('Step', fontsize=11)
            ax.set_ylabel('Flow Magnitude (pixels)', fontsize=11)
            ax.set_title(f'{method.upper()}', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        magnitude_path = os.path.join(args.output_dir, f"flow_magnitude_comparison_{args.focus_region}.png")
        plt.savefig(magnitude_path, bbox_inches='tight', dpi=150)
        plt.close()
        print(f"✓ Saved: {magnitude_path}")
    else:
        magnitude_path = os.path.join(args.output_dir, f"flow_magnitude_{args.smoothing}_{args.focus_region}.png")
        visualize_flow_magnitude_progression(step_flows, magnitude_path)
    
    # 7. 热力图
    print(f"\n[8/8] Creating density heatmap...")
    heatmap_path = os.path.join(args.output_dir, f"target_heatmap_{args.smoothing}_{args.focus_region}.png")
    visualize_target_heatmap(initial_image, step_targets, heatmap_path)
    
    # 动画帧（可选）
    if args.create_animation:
        print(f"\n[Extra] Creating animation frames...")
        visualize_animation_frames(initial_image, target_image, filtered_source, 
                                   step_targets, args.output_dir)
    
    print(f"\n{'='*70}")
    print(f"Visualization Complete!")
    print(f"{'='*70}")
    print(f"\nSettings:")
    print(f"  Smoothing: {args.smoothing}")
    print(f"  Focus region: {args.focus_region}")
    print(f"  Compare methods: {args.compare_methods}")
    print(f"\nOutput files in: {args.output_dir}")
    print(f"  - step_targets_overlay_{args.smoothing}_{args.focus_region}.png")
    if args.compare_methods:
        print(f"  - comparison_methods_{args.focus_region}.png")
        print(f"  - flow_magnitude_comparison_{args.focus_region}.png")
    else:
        print(f"  - comparison_grid_{args.smoothing}_{args.focus_region}.png")
        print(f"  - flow_magnitude_{args.smoothing}_{args.focus_region}.png")
    print(f"  - target_heatmap_{args.smoothing}_{args.focus_region}.png")
    if args.create_animation:
        print(f"  - step_animation/ (animation frames)")
    print(f"\nTips:")
    print(f"  - Use --compare_methods to compare all interpolation methods")
    print(f"  - Use --smoothing [linear|ease_in_out|cubic|quintic] to change smoothing")
    print(f"  - Use --focus_region [all|gripper|upper_joints|gripper_and_joints|high_motion]")
    print(f"  - Add --create_animation for frame-by-frame animation")


if __name__ == "__main__":
    main()
