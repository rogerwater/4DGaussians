"""
重要高斯选择器 - 用于稀疏渲染优化

基于多种启发式方法选择与任务相关的重要高斯点，
大幅减少渲染开销同时保持规划质量。
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional


class ImportantGaussianSelector:
    """选择对任务重要的高斯子集进行渲染"""
    
    def __init__(
        self,
        selection_method: str = "hybrid",  # "flow", "attention", "physics", "hybrid"
        top_k_ratio: float = 0.1,  # 保留前10%的高斯
        min_gaussians: int = 1000,
        max_gaussians: int = 10000,
        update_frequency: int = 5,  # 每5步更新一次选择
    ):
        self.selection_method = selection_method
        self.top_k_ratio = top_k_ratio
        self.min_gaussians = min_gaussians
        self.max_gaussians = max_gaussians
        self.update_frequency = update_frequency
        self.cache = {}
        self.step_counter = 0
        
    def select_important_gaussians(
        self,
        gaussian_model,
        control_vector: torch.Tensor,
        initial_flow: Optional[torch.Tensor] = None,
        target_flow: Optional[torch.Tensor] = None,
        roi_mask: Optional[torch.Tensor] = None,
        camera_viewpoint = None,
    ) -> Dict[str, torch.Tensor]:
        """
        选择重要高斯
        
        Args:
            gaussian_model: 4DGaussians模型
            control_vector: 当前控制向量
            initial_flow: 初始光流点 (N, 3)
            target_flow: 目标光流点 (N, 3)
            roi_mask: 感兴趣区域mask
            camera_viewpoint: 相机视角
            
        Returns:
            selection_indices: 选中的高斯索引
            importance_scores: 重要性分数
        """
        self.step_counter += 1
        
        # 检查缓存
        if self.step_counter % self.update_frequency != 0 and "indices" in self.cache:
            return self.cache
        
        with torch.no_grad():
            importance_scores = torch.zeros(
                len(gaussian_model.get_xyz), device=gaussian_model.get_xyz.device
            )
            
            # 方法1: 基于光流的选择
            if self.selection_method in ["flow", "hybrid"]:
                flow_scores = self._compute_flow_based_importance(
                    gaussian_model, initial_flow, target_flow, camera_viewpoint
                )
                importance_scores += flow_scores
            
            # 方法2: 基于控制敏感度的选择
            if self.selection_method in ["physics", "hybrid"]:
                control_scores = self._compute_control_sensitivity(
                    gaussian_model, control_vector
                )
                importance_scores += control_scores
            
            # 方法3: 基于视觉显著性的选择
            if self.selection_method in ["attention", "hybrid"]:
                visual_scores = self._compute_visual_saliency(
                    gaussian_model, camera_viewpoint, roi_mask
                )
                importance_scores += visual_scores
            
            # Top-K选择
            num_select = int(len(importance_scores) * self.top_k_ratio)
            num_select = max(self.min_gaussians, min(num_select, self.max_gaussians))
            
            _, selection_indices = torch.topk(importance_scores, num_select)
            
            # 缓存结果
            self.cache = {
                "indices": selection_indices,
                "scores": importance_scores,
                "num_selected": num_select,
            }
            
            return self.cache
    
    def _compute_flow_based_importance(
        self,
        gaussian_model,
        initial_flow: torch.Tensor,
        target_flow: torch.Tensor,
        camera_viewpoint,
    ) -> torch.Tensor:
        """
        基于光流计算高斯重要性
        
        核心思想: 距离光流路径近的高斯更重要
        """
        if initial_flow is None or target_flow is None:
            return torch.zeros(len(gaussian_model.get_xyz), device=gaussian_model.get_xyz.device)
        
        gaussian_xyz = gaussian_model.get_xyz  # (M, 3)
        
        # 将光流点投影到3D空间（假设已有深度信息）
        # 这里简化处理，实际需要根据相机参数反投影
        flow_points_3d = initial_flow  # (N, 3) - 假设已经是3D坐标
        
        # 计算每个高斯到光流点的最小距离
        # gaussian_xyz: (M, 3), flow_points_3d: (N, 3)
        distances = torch.cdist(gaussian_xyz, flow_points_3d)  # (M, N)
        min_distances, _ = torch.min(distances, dim=1)  # (M,)
        
        # 计算目标光流的运动方向
        if target_flow is not None and len(target_flow.shape) > 2:
            # target_flow: (T, N, 3)
            motion_magnitude = torch.norm(
                target_flow[-1] - target_flow[0], dim=-1
            )  # (N,)
            # 使用运动幅度加权距离
            weighted_distances = distances * motion_magnitude[None, :]  # (M, N)
            min_weighted_dist, _ = torch.min(weighted_distances, dim=1)
            min_distances = min_distances * 0.7 + min_weighted_dist * 0.3
        
        # 距离越近，重要性越高（使用高斯核）
        importance = torch.exp(-min_distances / 0.1)
        
        return importance
    
    def _compute_control_sensitivity(
        self,
        gaussian_model,
        control_vector: torch.Tensor,
    ) -> torch.Tensor:
        """
        基于控制敏感度计算重要性
        
        核心思想: 对控制输入变化敏感的高斯更重要
        """
        # 获取高斯的变形参数（如果有）
        if not hasattr(gaussian_model, 'get_deformation'):
            return torch.zeros(len(gaussian_model.get_xyz), device=gaussian_model.get_xyz.device)
        
        # 计算高斯位置对控制向量的梯度（需要启用梯度）
        # 这里简化：使用高斯的速度/加速度特征
        xyz = gaussian_model.get_xyz
        
        # 假设有时间相关的变形网络
        # deformation = gaussian_model.get_deformation(control_vector)
        # sensitivity = torch.norm(deformation, dim=-1)
        
        # 简化版本：基于位置的方差
        xyz_var = torch.var(xyz, dim=-1) if len(xyz.shape) > 2 else torch.ones(len(xyz), device=xyz.device)
        
        return xyz_var
    
    def _compute_visual_saliency(
        self,
        gaussian_model,
        camera_viewpoint,
        roi_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        基于视觉显著性计算重要性
        
        核心思想: 在视野中心、颜色鲜艳、不透明度高的高斯更重要
        """
        xyz = gaussian_model.get_xyz
        opacity = gaussian_model.get_opacity
        
        # 1. 视野中心性
        if camera_viewpoint is not None:
            camera_center = camera_viewpoint.camera_center
            # 确保camera_center在同一设备上
            if not isinstance(camera_center, torch.Tensor):
                camera_center = torch.tensor(camera_center, device=xyz.device, dtype=xyz.dtype)
            elif camera_center.device != xyz.device:
                camera_center = camera_center.to(xyz.device)
            distances_to_camera = torch.norm(xyz - camera_center, dim=-1)
            centrality = 1.0 / (1.0 + distances_to_camera)
        else:
            centrality = torch.ones(len(xyz), device=xyz.device)
        
        # 2. 不透明度（更不透明 = 更重要）
        opacity_score = opacity.squeeze() if len(opacity.shape) > 1 else opacity
        
        # 3. ROI mask（如果提供）
        roi_score = torch.ones_like(centrality)
        if roi_mask is not None:
            # 假设roi_mask是投影到3D的mask
            roi_score = roi_mask
        
        # 综合分数
        saliency = centrality * 0.3 + opacity_score * 0.5 + roi_score * 0.2
        
        return saliency
    
    def create_sparse_gaussian_model(
        self,
        gaussian_model,
        selection_indices: torch.Tensor,
    ):
        """
        创建稀疏高斯模型用于快速渲染
        
        Returns:
            sparse_model: 只包含选中高斯的模型副本
        """
        # 这里需要根据实际的GaussianModel实现
        # 创建一个轻量级的渲染用模型
        sparse_data = {
            "xyz": gaussian_model.get_xyz[selection_indices],
            "opacity": gaussian_model.get_opacity[selection_indices],
            "scaling": gaussian_model.get_scaling[selection_indices],
            "rotation": gaussian_model.get_rotation[selection_indices],
            "features": gaussian_model.get_features[selection_indices],
        }
        return sparse_data


class FlowGuidedRenderer:
    """基于光流引导的稀疏渲染器"""
    
    def __init__(
        self,
        gaussian_selector: ImportantGaussianSelector,
        render_resolution: Tuple[int, int] = (256, 256),  # 降低分辨率加速
        use_flow_mask: bool = True,
    ):
        self.selector = gaussian_selector
        self.render_resolution = render_resolution
        self.use_flow_mask = use_flow_mask
    
    def render_sparse(
        self,
        gaussian_model,
        camera_viewpoint,
        control_vector: torch.Tensor,
        initial_flow: Optional[torch.Tensor] = None,
        target_flow: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        执行稀疏渲染
        
        Returns:
            rendered_image: 渲染的RGB图像
            flow_projection: 光流点的投影位置
            importance_map: 重要性热图
        """
        # 1. 选择重要高斯
        selection = self.selector.select_important_gaussians(
            gaussian_model, control_vector, initial_flow, target_flow, None, camera_viewpoint
        )
        
        # 2. 创建稀疏模型
        sparse_data = self.selector.create_sparse_gaussian_model(
            gaussian_model, selection["indices"]
        )
        
        # 3. 执行渲染（调用4DGaussians的渲染函数）
        # 注意：需要修改原始渲染函数支持稀疏高斯
        # rendered_image = render_sparse_gaussians(sparse_data, camera_viewpoint, ...)
        
        # 4. 如果使用flow mask，只关注光流区域
        if self.use_flow_mask and initial_flow is not None:
            # 创建光流区域的mask
            flow_mask = self._create_flow_mask(initial_flow, self.render_resolution)
        else:
            flow_mask = None
        
        return {
            "image": None,  # 需要实际渲染
            "flow_mask": flow_mask,
            "num_gaussians": len(selection["indices"]),
            "selection_indices": selection["indices"],
        }
    
    def _create_flow_mask(
        self,
        flow_points: torch.Tensor,
        resolution: Tuple[int, int],
        dilation_radius: int = 20,
    ) -> torch.Tensor:
        """创建光流点周围的mask"""
        mask = torch.zeros(resolution, device=flow_points.device)
        
        # 将光流点转换为像素坐标
        flow_pixels = flow_points[:, :2]  # (N, 2)
        flow_pixels[:, 0] *= resolution[1]
        flow_pixels[:, 1] *= resolution[0]
        flow_pixels = flow_pixels.long()
        
        # 在每个点周围创建圆形区域
        for px, py in flow_pixels:
            y_min = max(0, py - dilation_radius)
            y_max = min(resolution[0], py + dilation_radius)
            x_min = max(0, px - dilation_radius)
            x_max = min(resolution[1], px + dilation_radius)
            mask[y_min:y_max, x_min:x_max] = 1.0
        
        return mask


# 使用示例
if __name__ == "__main__":
    # 初始化选择器
    selector = ImportantGaussianSelector(
        selection_method="hybrid",
        top_k_ratio=0.15,  # 只渲染15%的高斯
        min_gaussians=2000,
        max_gaussians=20000,
    )
    
    # 初始化渲染器
    renderer = FlowGuidedRenderer(
        gaussian_selector=selector,
        render_resolution=(256, 256),
        use_flow_mask=True,
    )
    
    print("重要高斯选择器初始化完成")
    print(f"选择方法: {selector.selection_method}")
    print(f"目标比例: {selector.top_k_ratio * 100}%")
