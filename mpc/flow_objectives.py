"""
光流目标函数 - 用于CEM优化器

将光流作为高层次的运动规划目标，
替代或补充像素级的图像损失。
"""

import torch
import numpy as np
from mpc.objectives import Objective
from typing import Dict, Optional, Tuple


class FlowAlignmentObjective(Objective):
    """
    光流对齐目标函数
    
    核心思想：最小化当前光流与目标光流之间的差异
    相比LPIPS/MSE，光流损失更关注运动而非外观
    """
    
    def __init__(
        self,
        weight: float = 1.0,
        flow_key: str = 'flow',
        distance_metric: str = 'l2',  # 'l2', 'chamfer', 'emd'
        use_visibility_mask: bool = True,
        temporal_weight_decay: float = 0.9,  # 未来帧权重衰减
    ):
        super().__init__(weight)
        self.flow_key = flow_key
        self.distance_metric = distance_metric
        self.use_visibility_mask = use_visibility_mask
        self.temporal_weight_decay = temporal_weight_decay
    
    def compute_reward(
        self,
        prediction: Dict[str, torch.Tensor],
        goal: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        计算光流对齐奖励
        
        Args:
            prediction: 包含预测光流的字典
                - 'flow': (B, T, N, 3) - 预测的光流轨迹
            goal: 包含目标光流的字典
                - 'flow': (T_goal, N, 3) - 目标光流轨迹
        
        Returns:
            reward: (B, 1, 1) - 每个样本的奖励
        """
        pred_flow = prediction[self.flow_key]  # (B, T, N, 3)
        goal_flow = goal[self.flow_key]  # (T_goal, N, 3) or (B, T_goal, N, 3)
        
        import torch
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        if isinstance(pred_flow, np.ndarray):
            pred_flow = torch.from_numpy(pred_flow).float().to(device)
        elif pred_flow.device.type == 'cpu':
            pred_flow = pred_flow.to(device)
            
        if isinstance(goal_flow, np.ndarray):
            goal_flow = torch.from_numpy(goal_flow).float().to(device)
        elif goal_flow.device.type == 'cpu':
            goal_flow = goal_flow.to(device)
        
        # 确保goal有batch维度
        if len(goal_flow.shape) == 3:
            goal_flow = goal_flow.unsqueeze(0).expand(pred_flow.shape[0], -1, -1, -1)
        
        B, T_pred, N, _ = pred_flow.shape
        _, T_goal, _, _ = goal_flow.shape
        
        # 时间对齐：只比较overlapping的时间步
        T_compare = min(T_pred, T_goal)
        pred_flow = pred_flow[:, :T_compare]
        goal_flow = goal_flow[:, :T_compare]
        
        # 提取坐标和可见性
        pred_coords = pred_flow[..., :2]  # (B, T, N, 2)
        goal_coords = goal_flow[..., :2]  # (B, T, N, 2)
        
        if self.use_visibility_mask:
            pred_vis = pred_flow[..., 2:3]  # (B, T, N, 1)
            goal_vis = goal_flow[..., 2:3]  # (B, T, N, 1)
            # 只在两者都可见的点上计算损失
            visibility_mask = (pred_vis > 0.5) & (goal_vis > 0.5)  # (B, T, N, 1)
        else:
            visibility_mask = torch.ones_like(pred_flow[..., 2:3])
        
        # 计算距离
        if self.distance_metric == 'l2':
            # L2距离（最简单）
            distances = torch.norm(pred_coords - goal_coords, dim=-1, keepdim=True)  # (B, T, N, 1)
            distances = distances * visibility_mask
        
        elif self.distance_metric == 'chamfer':
            # Chamfer距离（处理点集不对应的情况）
            distances = self._chamfer_distance(pred_coords, goal_coords, visibility_mask)
        
        elif self.distance_metric == 'emd':
            # Earth Mover's Distance（考虑点的分布）
            distances = self._earth_movers_distance(pred_coords, goal_coords, visibility_mask)
        
        # 时间加权：离当前越远的帧权重越小
        time_weights = torch.tensor(
            [self.temporal_weight_decay ** t for t in range(T_compare)],
            device=pred_flow.device
        ).view(1, -1, 1, 1)  # (1, T, 1, 1)
        
        weighted_distances = distances * time_weights
        
        # 汇总为奖励（距离越小，奖励越高）
        total_distance = weighted_distances.sum(dim=(1, 2, 3))  # (B,)
        num_valid_points = visibility_mask.sum(dim=(1, 2, 3)).clamp(min=1)  # (B,)
        avg_distance = total_distance / num_valid_points
        
        # 转换为奖励（使用负指数）
        reward = -avg_distance  # 或 torch.exp(-avg_distance / temperature)
        
        # 转换为numpy以与其他Objective保持一致
        reward = reward.cpu().numpy() if isinstance(reward, torch.Tensor) else reward
        return reward[:, None, None]  # (B, 1, 1)
    
    def _chamfer_distance(
        self,
        pred_coords: torch.Tensor,
        goal_coords: torch.Tensor,
        visibility_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Chamfer距离：pred->goal + goal->pred
        """
        B, T, N, _ = pred_coords.shape
        
        # Reshape for batch computation
        pred_flat = pred_coords.reshape(B * T, N, 2)
        goal_flat = goal_coords.reshape(B * T, N, 2)
        vis_flat = visibility_mask.reshape(B * T, N, 1)
        
        # pred -> goal
        dist_matrix = torch.cdist(pred_flat, goal_flat)  # (BT, N, N)
        dist_pred_to_goal, _ = torch.min(dist_matrix, dim=2)  # (BT, N)
        
        # goal -> pred
        dist_goal_to_pred, _ = torch.min(dist_matrix, dim=1)  # (BT, N)
        
        # 平均
        chamfer = (dist_pred_to_goal + dist_goal_to_pred) / 2  # (BT, N)
        chamfer = chamfer.reshape(B, T, N, 1) * vis_flat.reshape(B, T, N, 1)
        
        return chamfer
    
    def _earth_movers_distance(
        self,
        pred_coords: torch.Tensor,
        goal_coords: torch.Tensor,
        visibility_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        简化的EMD：基于最优传输
        （完整实现需要Sinkhorn算法，这里用近似）
        """
        # 简化：使用Chamfer距离作为近似
        return self._chamfer_distance(pred_coords, goal_coords, visibility_mask)


class FlowConsistencyObjective(Objective):
    """
    光流一致性目标
    
    确保预测的光流轨迹在时间上平滑连续
    """
    
    def __init__(
        self,
        weight: float = 0.1,
        flow_key: str = 'flow',
        order: int = 1,  # 1: 速度平滑, 2: 加速度平滑
    ):
        super().__init__(weight)
        self.flow_key = flow_key
        self.order = order
    
    def compute_reward(
        self,
        prediction: Dict[str, torch.Tensor],
        goal: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        计算时间平滑性奖励
        """
        pred_flow = prediction[self.flow_key]  # (B, T, N, 3)
        
        # 转换为torch tensor（如果不是）并移到0CUDA
        import torch
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        if isinstance(pred_flow, np.ndarray):
            pred_flow = torch.from_numpy(pred_flow).float().to(device)
        elif pred_flow.device.type == 'cpu':
            pred_flow = pred_flow.to(device)
        
        # 计算时间差分
        if self.order == 1:
            # 一阶差分（速度）
            velocity = pred_flow[:, 1:] - pred_flow[:, :-1]  # (B, T-1, N, 3)
            smoothness = torch.norm(velocity[..., :2], dim=-1)  # (B, T-1, N)
        elif self.order == 2:
            # 二阶差分（加速度）
            velocity = pred_flow[:, 1:] - pred_flow[:, :-1]
            acceleration = velocity[:, 1:] - velocity[:, :-1]  # (B, T-2, N, 3)
            smoothness = torch.norm(acceleration[..., :2], dim=-1)  # (B, T-2, N)
        
        # 平滑度越小，奖励越高
        avg_smoothness = smoothness.mean(dim=(1, 2))  # (B,)
        reward = -avg_smoothness

        # 转换为numpy以与其他Objective保持一致
        reward = reward.cpu().numpy() if isinstance(reward, torch.Tensor) else reward
        return reward[:, None, None]


class FlowDirectionGuidanceObjective(Objective):
    """
    光流方向指引目标

    核心思想：计算当前预测光流的方向与目标光流指向位置之间的夹角余弦。
    预测光流应该指向目标光流想去的位置（从source_points到target_points的方向）。

    输入格式：
        - prediction['flow']: (B, T, N, 3) 预测的光流 [x, y, visibility]
        - goal['flow']: (T_goal, N, 3) 目标光流 [x, y, visibility]

    原理：
        1. 预测光流向量 v_pred = pred_target - pred_source
        2. 期望方向向量 v_goal = goal_target - goal_source
        3. 计算方向一致性：cos_angle = dot(v_pred, v_goal) / (|v_pred| * |v_goal|)
        4. 奖励 = cos_angle（越接近1越好）
    """

    def __init__(
        self,
        weight: float = 1.0,
        flow_key: str = 'flow',
        use_visibility_mask: bool = True,
        temporal_weight_decay: float = 0.95,
        epsilon: float = 1e-8,
    ):
        super().__init__(weight)
        self.flow_key = flow_key
        self.use_visibility_mask = use_visibility_mask
        self.temporal_weight_decay = temporal_weight_decay
        self.epsilon = epsilon

    def compute_reward(
        self,
        prediction: Dict[str, torch.Tensor],
        goal: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        计算光流方向指引奖励

        Args:
            prediction: 包含预测光流的字典
                - 'flow': (B, T, N, 3) 预测的光流轨迹
            goal: 包含目标光流的字典
                - 'flow': (T_goal, N, 3) 目标光流轨迹

        Returns:
            reward: (B, 1, 1) - 每个样本的奖励（方向一致性，1为完全一致）
        """
        pred_flow = prediction[self.flow_key]  # (B, T, N, 3)
        goal_flow = goal[self.flow_key]  # (T_goal, N, 3) or (B, T_goal, N, 3)

        import torch
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if isinstance(pred_flow, np.ndarray):
            pred_flow = torch.from_numpy(pred_flow).float().to(device)
        elif pred_flow.device.type == 'cpu':
            pred_flow = pred_flow.to(device)

        if isinstance(goal_flow, np.ndarray):
            goal_flow = torch.from_numpy(goal_flow).float().to(device)
        elif goal_flow.device.type == 'cpu':
            goal_flow = goal_flow.to(device)

        # 确保goal有batch维度
        if len(goal_flow.shape) == 3:
            goal_flow = goal_flow.unsqueeze(0).expand(pred_flow.shape[0], -1, -1, -1)

        B, T_pred, N, _ = pred_flow.shape
        _, T_goal, _, _ = goal_flow.shape

        # 时间对齐
        T_compare = min(T_pred, T_goal)
        pred_flow = pred_flow[:, :T_compare]
        goal_flow = goal_flow[:, :T_compare]

        # 提取坐标和可见性
        pred_coords = pred_flow[..., :2]  # (B, T, N, 2)
        goal_coords = goal_flow[..., :2]  # (B, T, N, 2)

        if self.use_visibility_mask:
            pred_vis = pred_flow[..., 2:3]  # (B, T, N, 1)
            goal_vis = goal_flow[..., 2:3]  # (B, T, N, 1)
            visibility_mask = (pred_vis > 0.5) & (goal_vis > 0.5)  # (B, T, N, 1)
        else:
            visibility_mask = torch.ones_like(pred_flow[..., :1])

        # ========== 计算光流方向向量 ==========
        # pred_flow向量 = pred_target - pred_source = 预测的光流本身
        # goal_flow向量 = goal_target - goal_source = 期望的光流方向

        # 归一化坐标到[-1, 1]范围以计算正确的向量
        # 假设输入坐标是[0, 1]范围的归一化坐标
        # 需要将归一化坐标转换为实际的位移向量
        # 假设图像宽高比约为1:1，如果是其他比例需要额外处理

        # 预测的光流方向向量（已经是位移向量）
        pred_direction = pred_coords  # (B, T, N, 2)

        # 期望的光流方向向量 = 目标位置 - 起始位置
        # goal_coords表示光流向量本身，即从source到target的位移
        goal_direction = goal_coords  # (B, T, N, 2)

        # 计算方向余弦相似度
        # cos_angle = dot(v1, v2) / (|v1| * |v2|)
        dot_product = torch.sum(pred_direction * goal_direction, dim=-1, keepdim=True)  # (B, T, N, 1)
        pred_magnitude = torch.norm(pred_direction + self.epsilon, dim=-1, keepdim=True)  # (B, T, N, 1)
        goal_magnitude = torch.norm(goal_direction + self.epsilon, dim=-1, keepdim=True)  # (B, T, N, 1)

        # 方向余弦相似度
        cosine_similarity = dot_product / (pred_magnitude * goal_magnitude + self.epsilon)

        # 只在可见点上计算
        cosine_similarity = cosine_similarity * visibility_mask

        # ========== 额外奖励：预测光流的大小也应该与目标一致 ==========
        pred_mag = torch.norm(pred_direction + self.epsilon, dim=-1, keepdim=True)  # (B, T, N, 1)
        goal_mag = torch.norm(goal_direction + self.epsilon, dim=-1, keepdim=True)  # (B, T, N, 1)

        # 大小一致性：目标越大，预测也应该越大
        mag_ratio = torch.min(pred_mag / (goal_mag + self.epsilon),
                              goal_mag / (pred_mag + self.epsilon))  # (B, T, N, 1)
        mag_ratio = mag_ratio * visibility_mask

        # 组合奖励：方向一致性 + 大小一致性
        direction_reward = cosine_similarity
        magnitude_reward = mag_ratio

        # 时间加权
        time_weights = torch.tensor(
            [self.temporal_weight_decay ** t for t in range(T_compare)],
            device=pred_flow.device
        ).view(1, -1, 1, 1)  # (1, T, 1, 1)

        weighted_direction = direction_reward * time_weights
        weighted_magnitude = magnitude_reward * time_weights

        # 汇总
        total_direction = weighted_direction.sum(dim=(1, 2, 3))
        total_magnitude = weighted_magnitude.sum(dim=(1, 2, 3))
        num_valid = visibility_mask.sum(dim=(1, 2, 3)).clamp(min=1)

        avg_direction = total_direction / num_valid
        avg_magnitude = total_magnitude / num_valid

        # 最终奖励：方向一致性优先，大小一致性作为辅助
        reward = avg_direction * 0.7 + avg_magnitude * 0.3

        # 转换为numpy
        reward = reward.cpu().numpy() if isinstance(reward, torch.Tensor) else reward
        return reward[:, None, None]


class FlowDirectionLoss(Objective):
    """
    光流方向Loss
    直接计算预测光流方向与目标光流方向之间的角度误差。
    Loss越小越好，等价于负的奖励。

    输入格式：
        - prediction['flow']: (B, T, N, 3) 预测的光流 [x, y, visibility]
        - goal['flow']: (T_goal, N, 3) 目标光流 [x, y, visibility]

    输出：
        - loss: (B, 1, 1) - 每条轨迹的方向损失
    """

    def __init__(
        self,
        weight: float = 1.0,
        flow_key: str = 'flow',
        use_visibility_mask: bool = True,
        temporal_weight_decay: float = 0.95,
        loss_type: str = 'cosine',  # 'cosine': 余弦相似度, 'angle': 角度差
        epsilon: float = 1e-8,
    ):
        super().__init__(weight)
        self.flow_key = flow_key
        self.use_visibility_mask = use_visibility_mask
        self.temporal_weight_decay = temporal_weight_decay
        self.loss_type = loss_type
        self.epsilon = epsilon

    def compute_reward(
        self,
        prediction: Dict[str, torch.Tensor],
        goal: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        计算光流方向Loss

        Args:
            prediction: 包含预测光流的字典
                - 'flow': (B, T, N, 3) 预测的光流轨迹
            goal: 包含目标光流的字典
                - 'flow': (T_goal, N, 3) 目标光流轨迹

        Returns:
            loss: (B, 1, 1) - 每条轨迹的方向损失
        """
        pred_flow = prediction[self.flow_key]  # (B, T, N, 3)
        goal_flow = goal[self.flow_key]  # (T_goal, N, 3) or (B, T_goal, N, 3)

        import torch
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if isinstance(pred_flow, np.ndarray):
            pred_flow = torch.from_numpy(pred_flow).float().to(device)
        elif pred_flow.device.type == 'cpu':
            pred_flow = pred_flow.to(device)

        if isinstance(goal_flow, np.ndarray):
            goal_flow = torch.from_numpy(goal_flow).float().to(device)
        elif goal_flow.device.type == 'cpu':
            goal_flow = goal_flow.to(device)

        # 确保goal有batch维度
        if len(goal_flow.shape) == 3:
            goal_flow = goal_flow.unsqueeze(0).expand(pred_flow.shape[0], -1, -1, -1)

        B, T_pred, N, _ = pred_flow.shape
        _, T_goal, _, _ = goal_flow.shape

        # 时间对齐
        T_compare = min(T_pred, T_goal)
        pred_flow = pred_flow[:, :T_compare]
        goal_flow = goal_flow[:, :T_compare]

        # 提取坐标和可见性
        pred_coords = pred_flow[..., :2]  # (B, T, N, 2)
        goal_coords = goal_flow[..., :2]  # (B, T, N, 2)

        if self.use_visibility_mask:
            pred_vis = pred_flow[..., 2:3]  # (B, T, N, 1)
            goal_vis = goal_flow[..., 2:3]  # (B, T, N, 1)
            visibility_mask = (pred_vis > 0.5) & (goal_vis > 0.5)  # (B, T, N, 1)
        else:
            visibility_mask = torch.ones_like(pred_flow[..., :1])

        # 预测光流方向向量
        pred_direction = pred_coords  # (B, T, N, 2)
        # 目标光流方向向量
        goal_direction = goal_coords  # (B, T, N, 2)

        if self.loss_type == 'cosine':
            # 余弦相似度：越接近1越好
            dot_product = torch.sum(pred_direction * goal_direction, dim=-1, keepdim=True)
            pred_mag = torch.norm(pred_direction + self.epsilon, dim=-1, keepdim=True)
            goal_mag = torch.norm(goal_direction + self.epsilon, dim=-1, keepdim=True)

            cosine_sim = dot_product / (pred_mag * goal_mag + self.epsilon)

            # 转换为loss：1 - cos_sim（越接近0越好）
            direction_loss = 1.0 - cosine_sim

        elif self.loss_type == 'angle':
            # 直接计算角度差（弧度）
            dot_product = torch.sum(pred_direction * goal_direction, dim=-1, keepdim=True)
            pred_mag = torch.norm(pred_direction + self.epsilon, dim=-1, keepdim=True)
            goal_mag = torch.norm(goal_direction + self.epsilon, dim=-1, keepdim=True)

            # 夹角余弦值
            cos_angle = dot_product / (pred_mag * goal_mag + self.epsilon)
            # 限制在[-1, 1]范围内
            cos_angle = torch.clamp(cos_angle, -1.0 + self.epsilon, 1.0 - self.epsilon)
            # 角度差（弧度）
            angle_loss = torch.acos(cos_angle)

            direction_loss = angle_loss / (3.14159)  # 归一化到[0, 1]

        else:
            raise ValueError(f"Unknown loss_type: {self.loss_type}")

        # 只在可见点上计算
        direction_loss = direction_loss * visibility_mask

        # 时间加权
        time_weights = torch.tensor(
            [self.temporal_weight_decay ** t for t in range(T_compare)],
            device=pred_flow.device
        ).view(1, -1, 1, 1)  # (1, T, 1, 1)

        weighted_loss = direction_loss * time_weights

        # 汇总为loss
        total_loss = weighted_loss.sum(dim=(1, 2, 3))
        num_valid = visibility_mask.sum(dim=(1, 2, 3)).clamp(min=1)
        avg_loss = total_loss / num_valid

        # 返回loss（越小越好，奖励为负loss）
        reward = -avg_loss

        # 转换为numpy
        reward = reward.cpu().numpy() if isinstance(reward, torch.Tensor) else reward
        return reward[:, None, None]


class FlowGuidanceWithTargetObjective(Objective):
    """
    光流指向目标位置指引目标

    核心思想：预测的光流应该指向目标光流的目标位置。
    即：pred_source + pred_flow ≈ goal_target

    与FlowDirectionGuidanceObjective的区别：
        - DirectionGuidance: 让pred_flow的方向与goal_flow的方向一致
        - GuidanceWithTarget: 让pred_source + pred_flow的位置接近goal_target

    输入格式：
        - prediction['source_points']: (B, T, N, 2) 预测的源点位置
        - prediction['flow']: (B, T, N, 2) 预测的光流向量
        - goal['target_points']: (T_goal, N, 2) 目标点位置（可选，默认用goal_flow）

    输出：
        - reward: (B, 1, 1)
    """

    def __init__(
        self,
        weight: float = 1.0,
        source_key: str = 'source_points',
        flow_key: str = 'flow',
        target_key: str = 'target_points',
        use_visibility_mask: bool = True,
        temporal_weight_decay: float = 0.95,
    ):
        super().__init__(weight)
        self.source_key = source_key
        self.flow_key = flow_key
        self.target_key = target_key
        self.use_visibility_mask = use_visibility_mask
        self.temporal_weight_decay = temporal_weight_decay

    def compute_reward(
        self,
        prediction: Dict[str, torch.Tensor],
        goal: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        计算指向目标位置的指引奖励
        """
        import torch
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 提取数据
        if self.source_key in prediction:
            source_points = prediction[self.source_key]  # (B, T, N, 2)
        else:
            # 如果没有source_points，从flow反推
            flow = prediction[self.flow_key]  # (B, T, N, 3)
            source_points = flow[..., :2]  # 假设flow存储的是从source到target的向量

        flow = prediction[self.flow_key]  # (B, T, N, 2 or 3)
        if flow.shape[-1] == 3:
            flow_vectors = flow[..., :2]  # (B, T, N, 2)
        else:
            flow_vectors = flow

        # 目标点位置
        if self.target_key in goal:
            target_points = goal[self.target_key]  # (T_goal, N, 2) or (B, T_goal, N, 2)
        else:
            # 从goal['flow']提取目标点
            goal_flow = goal[self.flow_key]  # (T_goal, N, 2)
            if len(goal_flow.shape) == 3:
                goal_flow = goal_flow.unsqueeze(0)
            target_points = goal_flow[..., :2]  # (B, T, N, 2)

        # 确保维度
        if len(target_points.shape) == 3:
            target_points = target_points.unsqueeze(0)

        B, T, N, _ = source_points.shape
        _, T_goal, _, _ = target_points.shape

        # 时间对齐
        T_compare = min(T, T_goal)
        source_points = source_points[:, :T_compare]
        flow_vectors = flow_vectors[:, :T_compare]
        target_points = target_points[:, :T_compare]

        # 预测的终点位置 = source + flow
        pred_endpoints = source_points + flow_vectors  # (B, T, N, 2)
        # 目标终点位置
        goal_endpoints = target_points  # (B, T, N, 2)

        # 计算到目标点的距离
        distances = torch.norm(pred_endpoints - goal_endpoints, dim=-1, keepdim=True)

        # 可见性mask
        if self.use_visibility_mask:
            pred_vis = prediction[self.flow_key][:, :T_compare, :, 2:3] if prediction[self.flow_key].shape[-1] == 3 else None
            goal_vis = goal[self.flow_key][:, :T_compare, :, 2:3] if goal[self.flow_key].shape[-1] == 3 else None

            if pred_vis is not None and goal_vis is not None:
                visibility_mask = (pred_vis > 0.5) & (goal_vis > 0.5)
            else:
                visibility_mask = torch.ones_like(distances)
        else:
            visibility_mask = torch.ones_like(distances)

        distances = distances * visibility_mask

        # 时间加权
        time_weights = torch.tensor(
            [self.temporal_weight_decay ** t for t in range(T_compare)],
            device=source_points.device
        ).view(1, -1, 1, 1)

        weighted_distances = distances * time_weights

        # 汇总
        total_distance = weighted_distances.sum(dim=(1, 2, 3))
        num_valid = visibility_mask.sum(dim=(1, 2, 3)).clamp(min=1)
        avg_distance = total_distance / num_valid

        # 奖励 = -距离
        reward = -avg_distance

        reward = reward.cpu().numpy() if isinstance(reward, torch.Tensor) else reward
        return reward[:, None, None]


class FlowSparseRenderObjective(Objective):
    """
    光流引导的稀疏渲染目标
    
    只在光流点附近评估渲染质量，大幅减少计算
    """
    
    def __init__(
        self,
        weight: float = 1.0,
        rgb_key: str = 'rgb',
        flow_key: str = 'flow',
        patch_size: int = 16,  # 每个光流点周围的patch大小
        use_lpips: bool = True,
    ):
        super().__init__(weight)
        self.rgb_key = rgb_key
        self.flow_key = flow_key
        self.patch_size = patch_size
        self.use_lpips = use_lpips
        
        if use_lpips:
            import piq
            self.lpips = piq.LPIPS(reduction="none")
    
    def compute_reward(
        self,
        prediction: Dict[str, torch.Tensor],
        goal: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        只在光流点周围计算渲染损失
        """
        pred_rgb = prediction[self.rgb_key]  # (B, T, H, W, 3)
        goal_rgb = goal[self.rgb_key]  # (T, H, W, 3)
        pred_flow = prediction[self.flow_key]  # (B, T, N, 3)
        
        if len(goal_rgb.shape) == 4:
            goal_rgb = goal_rgb.unsqueeze(0).expand(pred_rgb.shape[0], -1, -1, -1, -1)
        
        B, T, H, W, C = pred_rgb.shape
        
        # 提取光流点位置
        flow_coords = pred_flow[..., :2]  # (B, T, N, 2)
        
        # 在每个光流点周围提取patch
        patches_pred = []
        patches_goal = []
        
        for b in range(B):
            for t in range(T):
                coords = flow_coords[b, t]  # (N, 2)
                # 转换为像素坐标
                coords_px = coords.clone()
                coords_px[:, 0] *= W
                coords_px[:, 1] *= H
                coords_px = coords_px.long()
                
                # 提取patches
                for n in range(len(coords_px)):
                    x, y = coords_px[n]
                    x_min = max(0, x - self.patch_size // 2)
                    x_max = min(W, x + self.patch_size // 2)
                    y_min = max(0, y - self.patch_size // 2)
                    y_max = min(H, y + self.patch_size // 2)
                    
                    patch_pred = pred_rgb[b, t, y_min:y_max, x_min:x_max]
                    patch_goal = goal_rgb[b, t, y_min:y_max, x_min:x_max]
                    
                    patches_pred.append(patch_pred)
                    patches_goal.append(patch_goal)
        
        # 计算patch-level损失
        if self.use_lpips and len(patches_pred) > 0:
            # TODO: 实现patch-level LPIPS
            pass
        
        # 简化：使用L2
        total_loss = 0
        for p_pred, p_goal in zip(patches_pred, patches_goal):
            if p_pred.numel() > 0 and p_goal.numel() > 0:
                total_loss += torch.nn.functional.mse_loss(p_pred, p_goal)
        
        reward = -total_loss / max(len(patches_pred), 1)
        
        return torch.tensor([reward], device=pred_rgb.device).view(1, 1, 1)


class HybridFlowImageObjective(Objective):
    """
    混合目标：光流对齐 + 稀疏图像渲染
    
    在im2flow2act的光流监督基础上，
    增加4DGaussians的物理约束
    """
    
    def __init__(
        self,
        flow_objective: FlowAlignmentObjective,
        image_objective: Optional[Objective] = None,
        flow_weight: float = 0.7,
        image_weight: float = 0.3,
    ):
        super().__init__(weight=1.0)
        self.flow_objective = flow_objective
        self.image_objective = image_objective
        self.flow_weight = flow_weight
        self.image_weight = image_weight
    
    def compute_reward(
        self,
        prediction: Dict[str, torch.Tensor],
        goal: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        组合光流和图像奖励
        """
        # 光流奖励
        flow_reward = self.flow_objective(prediction, goal)
        
        # 图像奖励（如果有且goal包含所需的key）
        if self.image_objective is not None:
            # 检查goal是否包含image objective需要的key
            # 对于MSEError，需要检查'rgb'键
            if hasattr(self.image_objective, 'key') and self.image_objective.key in goal:
                image_reward = self.image_objective(prediction, goal)
            elif hasattr(self.image_objective, 'key'):
                # goal中没有需要的key，跳过image reward
                image_reward = 0
            else:
                # 尝试调用，如果失败则跳过
                try:
                    image_reward = self.image_objective(prediction, goal)
                except (KeyError, AttributeError):
                    image_reward = 0
        else:
            image_reward = 0
        
        # 加权组合
        total_reward = (
            self.flow_weight * flow_reward +
            self.image_weight * image_reward
        )
        
        return total_reward


# 使用示例
if __name__ == "__main__":
    # 创建光流对齐目标
    flow_objective = FlowAlignmentObjective(
        weight=1.0,
        distance_metric='l2',
        use_visibility_mask=True,
        temporal_weight_decay=0.9,
    )
    
    # 创建平滑性目标
    smoothness_objective = FlowConsistencyObjective(
        weight=0.1,
        order=2,  # 加速度平滑
    )
    
    # 创建混合目标
    from mpc.objectives import CombinedObjective
    combined = CombinedObjective(
        objectives={
            'flow_alignment': flow_objective,
            'flow_smoothness': smoothness_objective,
        },
        combine_method='sum'
    )
    
    print("光流目标函数初始化完成")
    print(f"主要目标: 光流对齐")
    print(f"辅助目标: 时间平滑性")


class ActionRegularizationObjective(Objective):
    """
    动作正则化目标函数
    
    惩罚过大的动作变化，避免在优化过程中产生不合理的控制信号。
    这比在输出时进行硬约束更加柔和，允许优化器找到平滑的控制轨迹。
    
    支持两种惩罚方式：
    1. 'delta': 惩罚相邻动作之间的变化（速度）
       - 如果提供了 current_joint_pos，会约束从当前位置到第一个预测动作的变化
       - 然后约束预测序列内部相邻动作之间的变化
       - 例如：current_pos=0° → action[0]=15° → action[1]=30° → action[2]=45°
       - 每一步的变化都会被约束（delta[0]=15°, delta[1]=15°, delta[2]=15°）
    2. 'magnitude': 惩罚动作的绝对大小
    
    注意：需要在 prediction 或 goal 中提供 'current_joint_pos' 才能约束第一个动作的变化。
    """
    
    def __init__(
        self,
        weight: float = 0.1,
        penalty_type: str = 'delta',  # 'delta', 'magnitude', 'both'
        max_delta: float = 0.5,  # 最大允许的动作变化（归一化）
        max_magnitude: float = 1.0,  # 最大允许的动作幅度
        penalty_scale: str = 'quadratic',  # 'linear', 'quadratic', 'exponential'
        apply_to_joints_only: bool = True,  # 是否只对关节角度应用（不包括gripper）
        num_joints: int = 6,  # 关节数量（每个关节用sin/cos表示，共12维）
        current_pos_key: str = 'current_joint_pos',  # 当前关节位置的键名
    ):
        super().__init__(weight)
        self.penalty_type = penalty_type
        self.max_delta = max_delta
        self.max_magnitude = max_magnitude
        self.penalty_scale = penalty_scale
        self.apply_to_joints_only = apply_to_joints_only
        self.num_joints = num_joints
        self.current_pos_key = current_pos_key
    
    def compute_reward(
        self,
        prediction: Dict[str, torch.Tensor],
        goal: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        计算动作正则化奖励
        
        Args:
            prediction: 包含动作序列的字典
                - 'actions': (B, T, A) - 预测的动作序列
                - 'current_joint_pos': (B, A) - 当前关节位置（可选）
            goal: 包含当前状态的字典
                - 'current_joint_pos': (A,) 或 (B, A) - 当前关节位置（可选）
        
        Returns:
            reward: (B, 1, 1) - 每个样本的奖励（负的惩罚）
        """
        if 'actions' not in prediction:
            # 如果没有actions键，返回零奖励
            print(f"      [DEBUG] ActionRegularization: 'actions' key not found in prediction. Keys: {list(prediction.keys())}")
            return np.zeros((prediction[list(prediction.keys())[0]].shape[0], 1, 1))
        
        actions = prediction['actions']  # (B, T, A)
        
        import torch
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        if isinstance(actions, np.ndarray):
            actions = torch.from_numpy(actions).float().to(device)
        elif actions.device.type == 'cpu':
            actions = actions.to(device)
        
        B, T, A = actions.shape
        
        # 获取当前关节位置（如果提供）
        current_joint_pos = None
        if self.current_pos_key in prediction:
            current_joint_pos = prediction[self.current_pos_key]
        elif self.current_pos_key in goal:
            current_joint_pos = goal[self.current_pos_key]
        
        # 处理 current_joint_pos 的维度
        if current_joint_pos is not None:
            if isinstance(current_joint_pos, np.ndarray):
                current_joint_pos = torch.from_numpy(current_joint_pos).float().to(device)
            elif current_joint_pos.device.type == 'cpu':
                current_joint_pos = current_joint_pos.to(device)
            
            # 确保是 (B, A) 的形状
            if current_joint_pos.ndim == 1:
                current_joint_pos = current_joint_pos.unsqueeze(0).expand(B, -1)
            elif current_joint_pos.shape[0] == 1 and B > 1:
                current_joint_pos = current_joint_pos.expand(B, -1)
        
        # 确定要正则化的维度
        if self.apply_to_joints_only:
            # 只对前12维（6个关节的sin/cos）进行正则化
            joint_dims = min(self.num_joints * 2, A)
            actions_to_regularize = actions[..., :joint_dims]
            if current_joint_pos is not None:
                current_pos_to_regularize = current_joint_pos[..., :joint_dims]
        else:
            actions_to_regularize = actions
            if current_joint_pos is not None:
                current_pos_to_regularize = current_joint_pos
        
        total_penalty = torch.zeros(B, device=device)
        
        # 1. Delta惩罚：惩罚相邻动作之间的大变化（包括从当前位置到第一个动作的变化）
        if self.penalty_type in ['delta', 'both']:
            # 构建完整的动作序列：[current_pos, action[0], action[1], ..., action[T-1]]
            if current_joint_pos is not None:
                # 将当前位置添加到序列开头 (B, 1, A') + (B, T, A') -> (B, T+1, A')
                full_sequence = torch.cat([
                    current_pos_to_regularize.unsqueeze(1),  # (B, 1, A')
                    actions_to_regularize  # (B, T, A')
                ], dim=1)  # (B, T+1, A')
            else:
                # 如果没有提供当前位置，只使用预测序列
                full_sequence = actions_to_regularize  # (B, T, A')
            
            if full_sequence.shape[1] > 1:
                # 计算动作变化（一阶差分）
                # 如果有 current_pos: delta[0] = action[0] - current_pos, delta[1] = action[1] - action[0], ...
                # 如果没有: delta[0] = action[1] - action[0], delta[1] = action[2] - action[1], ...
                action_deltas = full_sequence[:, 1:] - full_sequence[:, :-1]  # (B, T, A') 或 (B, T-1, A')
                
                # 对于sin/cos表示，需要计算角度变化
                if self.apply_to_joints_only:
                    # 将sin/cos对转换为角度差
                    angle_deltas = []
                    for i in range(self.num_joints):
                        sin_idx = 2 * i
                        cos_idx = 2 * i + 1
                        if sin_idx < full_sequence.shape[-1]:
                            # 前一步的角度（从 full_sequence 中取）
                            prev_sin = full_sequence[:, :-1, sin_idx]
                            prev_cos = full_sequence[:, :-1, cos_idx]
                            prev_angle = torch.atan2(prev_sin, prev_cos)
                            
                            # 当前步的角度（从 full_sequence 中取）
                            curr_sin = full_sequence[:, 1:, sin_idx]
                            curr_cos = full_sequence[:, 1:, cos_idx]
                            curr_angle = torch.atan2(curr_sin, curr_cos)
                            
                            # 角度差（处理周期性）
                            angle_delta = curr_angle - prev_angle
                            angle_delta = torch.atan2(torch.sin(angle_delta), torch.cos(angle_delta))
                            angle_deltas.append(torch.abs(angle_delta))
                    
                    if angle_deltas:
                        angle_deltas = torch.stack(angle_deltas, dim=-1)  # (B, T-1, num_joints)
                        # 计算超出阈值的惩罚
                        excess_delta = torch.clamp(angle_deltas - self.max_delta, min=0)
                    else:
                        excess_delta = torch.zeros(B, T-1, 1, device=device)
                else:
                    # 对于非关节动作，直接使用L2范数
                    delta_magnitude = torch.norm(action_deltas, dim=-1)  # (B, T-1)
                    excess_delta = torch.clamp(delta_magnitude - self.max_delta, min=0)
                
                # 应用惩罚尺度
                if self.penalty_scale == 'linear':
                    delta_penalty = excess_delta
                elif self.penalty_scale == 'quadratic':
                    delta_penalty = excess_delta ** 2
                elif self.penalty_scale == 'exponential':
                    # 限制指数避免溢出
                    delta_penalty = torch.exp(torch.clamp(excess_delta, max=10.0)) - 1
                else:
                    delta_penalty = excess_delta
                
                total_penalty += delta_penalty.sum(dim=(1, 2)) if len(delta_penalty.shape) > 2 else delta_penalty.sum(dim=1)
        
        # 2. Magnitude惩罚：惩罚过大的动作幅度
        if self.penalty_type in ['magnitude', 'both']:
            if self.apply_to_joints_only:
                # 将sin/cos转换为角度幅度
                angle_mags = []
                for i in range(self.num_joints):
                    sin_idx = 2 * i
                    cos_idx = 2 * i + 1
                    if sin_idx < actions_to_regularize.shape[-1]:
                        sin_val = actions_to_regularize[..., sin_idx]
                        cos_val = actions_to_regularize[..., cos_idx]
                        angle = torch.atan2(sin_val, cos_val)
                        angle_mags.append(torch.abs(angle))
                
                if angle_mags:
                    angle_mags = torch.stack(angle_mags, dim=-1)  # (B, T, num_joints)
                    excess_mag = torch.clamp(angle_mags - self.max_magnitude, min=0)
                else:
                    excess_mag = torch.zeros(B, T, 1, device=device)
            else:
                action_magnitude = torch.norm(actions_to_regularize, dim=-1)  # (B, T)
                excess_mag = torch.clamp(action_magnitude - self.max_magnitude, min=0)
            
            # 应用惩罚尺度
            if self.penalty_scale == 'linear':
                mag_penalty = excess_mag
            elif self.penalty_scale == 'quadratic':
                mag_penalty = excess_mag ** 2
            elif self.penalty_scale == 'exponential':
                # 限制指数避免溢出
                mag_penalty = torch.exp(torch.clamp(excess_mag, max=10.0)) - 1
            else:
                mag_penalty = excess_mag
            
            total_penalty += mag_penalty.sum(dim=(1, 2)) if len(mag_penalty.shape) > 2 else mag_penalty.sum(dim=1)
        
        # 归一化：除以时间步数
        if T > 0:
            avg_penalty = total_penalty / T
        else:
            avg_penalty = torch.zeros(B, device=device)
        
        # 转换为奖励（惩罚越大，奖励越小）
        reward = -avg_penalty
        
        # 检查并处理 NaN/Inf
        if torch.isnan(reward).any() or torch.isinf(reward).any():
            print(f"      [WARNING] ActionRegularization: NaN or Inf detected in reward, replacing with zeros")
            reward = torch.nan_to_num(reward, nan=0.0, posinf=0.0, neginf=0.0)
        
        # 转换为numpy
        reward = reward.cpu().numpy() if isinstance(reward, torch.Tensor) else reward
        return reward[:, None, None]  # (B, 1, 1)
