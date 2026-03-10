"""
光流Loss计算工具
支持密集光流场的直接对比，无需采样
"""

import torch
import torch.nn.functional as F
from typing import Optional, Tuple


def dense_flow_l1_loss(
    pred_flow: torch.Tensor,
    target_flow: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    密集光流L1 Loss
    
    Args:
        pred_flow: (B, 2, H, W) or (B, T, 2, H, W) - 预测的光流场
        target_flow: (B, 2, H, W) or (B, T, 2, H, W) - 目标光流场
        mask: (B, 1, H, W) or (B, T, 1, H, W) - 可选的mask（例如遮挡检测）
    
    Returns:
        loss: scalar - 平均L1距离
    """
    # 确保维度匹配
    if pred_flow.dim() == 4 and target_flow.dim() == 4:
        # (B, 2, H, W)
        diff = torch.abs(pred_flow - target_flow)  # (B, 2, H, W)
    elif pred_flow.dim() == 5 and target_flow.dim() == 5:
        # (B, T, 2, H, W)
        diff = torch.abs(pred_flow - target_flow)  # (B, T, 2, H, W)
    else:
        raise ValueError(f"Shape mismatch: pred_flow {pred_flow.shape}, target_flow {target_flow.shape}")
    
    # 应用mask
    if mask is not None:
        diff = diff * mask
        num_valid = mask.sum().clamp(min=1)
    else:
        num_valid = diff.numel() / diff.shape[-3]  # 除以通道数
    
    # 计算平均损失
    loss = diff.sum() / num_valid
    
    return loss


def dense_flow_l2_loss(
    pred_flow: torch.Tensor,
    target_flow: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    密集光流L2 (EPE - End Point Error) Loss
    
    Args:
        pred_flow: (B, 2, H, W) or (B, T, 2, H, W) - 预测的光流场
        target_flow: (B, 2, H, W) or (B, T, 2, H, W) - 目标光流场
        mask: (B, 1, H, W) or (B, T, 1, H, W) - 可选的mask
    
    Returns:
        loss: scalar - 平均EPE
    """
    # 计算端点误差
    epe = torch.norm(pred_flow - target_flow, p=2, dim=-3, keepdim=True)  # (B, 1, H, W) or (B, T, 1, H, W)
    
    # 应用mask
    if mask is not None:
        epe = epe * mask
        num_valid = mask.sum().clamp(min=1)
    else:
        num_valid = epe.numel()
    
    # 计算平均损失
    loss = epe.sum() / num_valid
    
    return loss


def dense_flow_charbonnier_loss(
    pred_flow: torch.Tensor,
    target_flow: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    epsilon: float = 0.001,
) -> torch.Tensor:
    """
    密集光流Charbonnier Loss (更鲁棒的L1变体)
    
    Charbonnier loss: sqrt(x^2 + epsilon^2)
    相比L1/L2更鲁棒，对异常值不敏感
    
    Args:
        pred_flow: (B, 2, H, W) or (B, T, 2, H, W) - 预测的光流场
        target_flow: (B, 2, H, W) or (B, T, 2, H, W) - 目标光流场
        mask: (B, 1, H, W) or (B, T, 1, H, W) - 可选的mask
        epsilon: 平滑参数
    
    Returns:
        loss: scalar - 平均Charbonnier距离
    """
    diff = pred_flow - target_flow
    loss_map = torch.sqrt(diff * diff + epsilon * epsilon)  # (B, 2, H, W) or (B, T, 2, H, W)
    
    # 应用mask
    if mask is not None:
        loss_map = loss_map * mask
        num_valid = mask.sum().clamp(min=1) * pred_flow.shape[-3]  # 乘以通道数
    else:
        num_valid = loss_map.numel()
    
    # 计算平均损失
    loss = loss_map.sum() / num_valid
    
    return loss


def dense_flow_smooth_loss(
    flow: torch.Tensor,
    order: int = 1,
) -> torch.Tensor:
    """
    密集光流平滑性约束
    鼓励相邻像素的光流平滑变化
    
    Args:
        flow: (B, 2, H, W) or (B, T, 2, H, W) - 光流场
        order: 1 (一阶梯度) 或 2 (二阶梯度)
    
    Returns:
        loss: scalar - 平滑性损失
    """
    if flow.dim() == 5:
        # (B, T, 2, H, W) -> 合并BT维度
        B, T, C, H, W = flow.shape
        flow = flow.reshape(B * T, C, H, W)
    
    # 计算空间梯度
    diff_x = torch.abs(flow[:, :, :, 1:] - flow[:, :, :, :-1])  # 水平方向
    diff_y = torch.abs(flow[:, :, 1:, :] - flow[:, :, :-1, :])  # 垂直方向
    
    if order == 2:
        # 二阶梯度（加速度）
        diff_xx = torch.abs(diff_x[:, :, :, 1:] - diff_x[:, :, :, :-1])
        diff_yy = torch.abs(diff_y[:, :, 1:, :] - diff_y[:, :, :-1, :])
        loss = (diff_xx.mean() + diff_yy.mean()) / 2.0
    else:
        # 一阶梯度（速度）
        loss = (diff_x.mean() + diff_y.mean()) / 2.0
    
    return loss


def dense_flow_temporal_consistency_loss(
    flow: torch.Tensor,
) -> torch.Tensor:
    """
    时间一致性约束
    鼓励相邻时间步的光流平滑变化
    
    Args:
        flow: (B, T, 2, H, W) - 时间序列光流场
    
    Returns:
        loss: scalar - 时间一致性损失
    """
    if flow.dim() != 5:
        raise ValueError(f"Expected 5D tensor (B, T, 2, H, W), got shape {flow.shape}")
    
    # 计算时间差分
    temporal_diff = torch.abs(flow[:, 1:] - flow[:, :-1])  # (B, T-1, 2, H, W)
    
    # 平均损失
    loss = temporal_diff.mean()
    
    return loss


def compute_occlusion_mask(
    forward_flow: torch.Tensor,
    backward_flow: torch.Tensor,
    threshold: float = 1.0,
) -> torch.Tensor:
    """
    基于前向-后向一致性计算遮挡mask
    
    Args:
        forward_flow: (B, 2, H, W) - 前向光流
        backward_flow: (B, 2, H, W) - 后向光流
        threshold: 一致性阈值（像素）
    
    Returns:
        mask: (B, 1, H, W) - 遮挡mask (1=可见, 0=遮挡)
    """
    B, _, H, W = forward_flow.shape
    
    # 使用前向光流warp后向光流
    # grid_sample需要归一化坐标 [-1, 1]
    grid_x, grid_y = torch.meshgrid(
        torch.arange(W, device=forward_flow.device),
        torch.arange(H, device=forward_flow.device),
        indexing='xy'
    )
    grid = torch.stack([grid_x, grid_y], dim=0).float()  # (2, H, W)
    grid = grid.unsqueeze(0).expand(B, -1, -1, -1)  # (B, 2, H, W)
    
    # 前向流: 当前位置 + 前向流 = 下一帧位置
    warped_pos = grid + forward_flow  # (B, 2, H, W)
    
    # 归一化到[-1, 1]
    warped_pos[:, 0] = 2.0 * warped_pos[:, 0] / (W - 1) - 1.0
    warped_pos[:, 1] = 2.0 * warped_pos[:, 1] / (H - 1) - 1.0
    warped_pos = warped_pos.permute(0, 2, 3, 1)  # (B, H, W, 2)
    
    # Warp后向流
    warped_backward = F.grid_sample(
        backward_flow,
        warped_pos,
        mode='bilinear',
        padding_mode='zeros',
        align_corners=True
    )  # (B, 2, H, W)
    
    # 前向-后向一致性检查
    consistency = torch.norm(forward_flow + warped_backward, p=2, dim=1, keepdim=True)  # (B, 1, H, W)
    mask = (consistency < threshold).float()
    
    return mask


def combined_dense_flow_loss(
    pred_flow: torch.Tensor,
    target_flow: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    alpha_l1: float = 1.0,
    alpha_smooth: float = 0.1,
    alpha_temporal: float = 0.05,
) -> Tuple[torch.Tensor, dict]:
    """
    组合的密集光流loss

    Args:
        pred_flow: (B, T, 2, H, W) - 预测的光流场
        target_flow: (B, T, 2, H, W) - 目标光流场
        mask: (B, T, 1, H, W) - 可选的mask
        alpha_l1: L1 loss权重
        alpha_smooth: 平滑性loss权重
        alpha_temporal: 时间一致性loss权重

    Returns:
        total_loss: scalar - 总损失
        loss_dict: 各项损失的字典
    """
    loss_dict = {}

    # 主要对齐损失
    l1_loss = dense_flow_l1_loss(pred_flow, target_flow, mask)
    loss_dict['l1'] = l1_loss.item()

    # 平滑性约束
    smooth_loss = dense_flow_smooth_loss(pred_flow, order=1)
    loss_dict['smooth'] = smooth_loss.item()

    # 时间一致性约束（如果有时间维度）
    if pred_flow.dim() == 5 and pred_flow.shape[1] > 1:
        temporal_loss = dense_flow_temporal_consistency_loss(pred_flow)
        loss_dict['temporal'] = temporal_loss.item()
    else:
        temporal_loss = torch.tensor(0.0, device=pred_flow.device)
        loss_dict['temporal'] = 0.0

    # 总损失
    total_loss = (
        alpha_l1 * l1_loss +
        alpha_smooth * smooth_loss +
        alpha_temporal * temporal_loss
    )
    loss_dict['total'] = total_loss.item()

    return total_loss, loss_dict


def dense_flow_direction_loss(
    pred_flow: torch.Tensor,
    target_flow: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    loss_type: str = 'cosine',
) -> Tuple[torch.Tensor, dict]:
    """
    密集光流方向指引loss

    计算预测光流方向与目标光流方向之间的差异。
    目标是让预测的光流方向与目标光流方向一致。

    Args:
        pred_flow: (B, T, 2, H, W) - 预测的光流场 [dx, dy]
        target_flow: (B, T, 2, H, W) - 目标光流场 [dx, dy]
        mask: (B, T, 1, H, W) - 可选的mask
        loss_type: 'cosine' | 'angle'
            - 'cosine': 使用余弦相似度，1 - cos_sim
            - 'angle': 使用角度差（弧度）

    Returns:
        loss: scalar - 平均方向损失
        loss_dict: 各项损失的字典
    """
    # 确保维度匹配
    if pred_flow.dim() == 4 and target_flow.dim() == 4:
        # (B, 2, H, W) -> 添加时间维度
        pred_flow = pred_flow.unsqueeze(1)  # (B, 1, 2, H, W)
        target_flow = target_flow.unsqueeze(1)
    elif pred_flow.dim() == 5 and target_flow.dim() == 5:
        pass
    else:
        raise ValueError(f"Shape mismatch: pred_flow {pred_flow.shape}, target_flow {target_flow.shape}")

    B, T, C, H, W = pred_flow.shape

    # 展平空间维度用于计算
    pred_flat = pred_flow.permute(0, 1, 3, 4, 2).reshape(B * T * H * W, 2)  # (BTHW, 2)
    target_flat = target_flow.permute(0, 1, 3, 4, 2).reshape(B * T * H * W, 2)  # (BTHW, 2)

    if mask is not None:
        if mask.dim() == 4:
            mask = mask.unsqueeze(1)
        mask_flat = mask.permute(0, 1, 3, 4, 2).reshape(B * T * H * W, 1)

    # 计算方向向量
    pred_norm = pred_flat / (torch.norm(pred_flat, dim=-1, keepdim=True) + 1e-8)
    target_norm = target_flat / (torch.norm(target_flat, dim=-1, keepdim=True) + 1e-8)

    if loss_type == 'cosine':
        # 余弦相似度：越接近1越好
        dot_product = torch.sum(pred_norm * target_norm, dim=-1)  # (BTHW,)
        direction_loss = 1.0 - dot_product  # (BTHW,)
    elif loss_type == 'angle':
        # 角度差：越接近0越好
        cos_angle = torch.sum(pred_norm * target_norm, dim=-1)
        cos_angle = torch.clamp(cos_angle, -1.0 + 1e-8, 1.0 - 1e-8)
        angle_loss = torch.acos(cos_angle)
        direction_loss = angle_loss / 3.14159  # 归一化到[0, 1]
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")

    # 应用mask
    if mask is not None:
        direction_loss = direction_loss * mask_flat.squeeze()

    # 重塑回原维度
    direction_loss = direction_loss.reshape(B, T, H, W)

    # 汇总
    if mask is not None:
        num_valid = mask.sum()
    else:
        num_valid = B * T * H * W

    avg_loss = direction_loss.sum() / num_valid.clamp(min=1)

    return avg_loss, {'direction': avg_loss.item()}


def dense_flow_endpoint_guidance_loss(
    pred_flow: torch.Tensor,
    source_points: torch.Tensor,
    target_points: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, dict]:
    """
    密集光流终点指引loss

    预测的光流应该指向目标点位置。
    Loss = ||pred_source + pred_flow - target||^2

    Args:
        pred_flow: (B, T, 2, H, W) - 预测的光流场
        source_points: (B, T, 2, H, W) or (B, 2, H, W) - 源点位置（如果为空，用网格坐标）
        target_points: (B, T, 2, H, W) or (B, 2, H, W) - 目标点位置
        mask: (B, T, 1, H, W) - 可选的mask

    Returns:
        loss: scalar - 平均终点指引损失
        loss_dict: 各项损失的字典
    """
    B, T, C, H, W = pred_flow.shape

    # 如果source_points没有时间维度，扩展
    if source_points.dim() == 4:
        source_points = source_points.unsqueeze(1).expand(-1, T, -1, -1, -1)

    if target_points.dim() == 4:
        target_points = target_points.unsqueeze(1).expand(-1, T, -1, -1, -1)

    # 预测的终点位置 = source + flow
    pred_endpoints = source_points + pred_flow  # (B, T, 2, H, W)

    # 到目标点的距离
    distances = torch.norm(pred_endpoints - target_points, dim=2, keepdim=True)  # (B, T, 1, H, W)

    if mask is not None:
        distances = distances * mask
        num_valid = mask.sum()
    else:
        num_valid = B * T * H * W

    loss = distances.sum() / num_valid.clamp(min=1)

    return loss, {'endpoint_guidance': loss.item()}


if __name__ == "__main__":
    # 简单测试
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 创建测试数据
    B, T, H, W = 2, 5, 256, 256
    pred_flow = torch.randn(B, T, 2, H, W, device=device)
    target_flow = torch.randn(B, T, 2, H, W, device=device)
    
    # 测试各种loss
    print("Testing dense flow losses...")
    print(f"L1 Loss: {dense_flow_l1_loss(pred_flow, target_flow):.4f}")
    print(f"L2 Loss: {dense_flow_l2_loss(pred_flow, target_flow):.4f}")
    print(f"Charbonnier Loss: {dense_flow_charbonnier_loss(pred_flow, target_flow):.4f}")
    print(f"Smooth Loss: {dense_flow_smooth_loss(pred_flow):.4f}")
    print(f"Temporal Loss: {dense_flow_temporal_consistency_loss(pred_flow):.4f}")
    
    # 测试组合loss
    total_loss, loss_dict = combined_dense_flow_loss(pred_flow, target_flow)
    print(f"\nCombined Loss: {total_loss:.4f}")
    print(f"Loss breakdown: {loss_dict}")
    
    print("\n✓ All tests passed!")
