"""
光流引导的4D Gaussian动力学模型
flow loss
"""

import os
import torch
import numpy as np
from typing import Dict, Optional, Tuple
from mpc.gaussian_dynamics_model import GaussianDynamicsModel
from mpc.important_gaussian_selector import ImportantGaussianSelector, FlowGuidedRenderer
from utils.graphics_utils import getWorld2View2, getProjectionMatrix
from utils.general_utils import build_rotation, build_scaling_rotation


class FlowGuidedGaussianDynamicsModel(GaussianDynamicsModel):
    
    def __init__(
        self,
        model_path: str,
        iteration: int,
        control_dim: int,
        image_height: int = 256,
        image_width: int = 256,
        num_context: int = 2,
        # 新增参数
        use_sparse_rendering: bool = True,
        sparse_ratio: float = 0.15,
        enable_flow_prediction: bool = True,
        flow_prediction_method: str = "render_based",  # "tapnet", "raft", "learned", "render_based"
        num_flow_points: int = 512,
        # 相机参数（父类支持）
        device: str = "cuda",
        camera_distance: float = 2.0,
        camera_elevation: float = 0.0,
        camera_azimuth: float = 0.0,
        fov_degrees: float = 45.0,
        transform_matrix = None,
        focal_x = None,
        focal_y = None,
        cx = None,
        cy = None,
        # render_based方法需要的参数
        target_image = None,
        sample_coords = None,
    ):
        # 只传递父类支持的参数
        super().__init__(
            model_path=model_path,
            iteration=iteration,
            control_dim=control_dim,
            image_height=image_height,
            image_width=image_width,
            device=device,
            camera_distance=camera_distance,
            camera_elevation=camera_elevation,
            camera_azimuth=camera_azimuth,
            fov_degrees=fov_degrees,
            transform_matrix=transform_matrix,
            focal_x=focal_x,
            focal_y=focal_y,
            cx=cx,
            cy=cy,
        )
        
        # 保存子类特有的参数
        self.num_context = num_context
        self.use_sparse_rendering = use_sparse_rendering
        self.enable_flow_prediction = enable_flow_prediction
        self.num_flow_points = num_flow_points
        self.flow_prediction_method = flow_prediction_method
        
        # 保存render_based方法需要的参数
        self.target_image = target_image
        self.sample_coords = sample_coords
        
        # 保存子类特有的参数
        self.num_context = num_context
        self.use_sparse_rendering = use_sparse_rendering
        self.enable_flow_prediction = enable_flow_prediction
        self.num_flow_points = num_flow_points
        self.flow_prediction_method = flow_prediction_method
        
        # 保存render_based方法需要的参数
        self.target_image = target_image
        self.sample_coords = sample_coords
        
        # 初始化重要高斯选择器
        if use_sparse_rendering:
            self.gaussian_selector = ImportantGaussianSelector(
                selection_method="hybrid",
                top_k_ratio=sparse_ratio,
                min_gaussians=1000,
                max_gaussians=20000,
            )
            self.sparse_renderer = FlowGuidedRenderer(
                gaussian_selector=self.gaussian_selector,
                render_resolution=(image_height, image_width),
                use_flow_mask=True,
            )
        
        # 初始化光流预测器
        if enable_flow_prediction:
            if flow_prediction_method == "tapnet":
                self._init_tapnet()
            elif flow_prediction_method == "learned":
                self._init_learned_flow_predictor()
    
    def _init_tapnet(self):
        """初始化TAP-Net光流跟踪器"""
        try:
            from im2flow2act.tapnet.online_point_tracking import build
            self.tapnet_predict, self.tapnet_init, _, _ = build(
                num_points=self.num_flow_points,
                img_size=[self.image_height, self.image_width],
            )
            print("✓ TAP-Net光流跟踪器初始化成功")
        except ImportError:
            print("⚠ TAP-Net未安装，使用简化的光流估计")
            self.tapnet_predict = None
            self.tapnet_init = None
    
    def _init_learned_flow_predictor(self):
        """初始化学习式光流预测网络"""
        # TODO: 实现一个轻量级的光流预测网络
        # 可以是一个简单的MLP: control_vector -> flow_delta
        import torch.nn as nn
        
        self.flow_predictor = nn.Sequential(
            nn.Linear(self.control_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, self.num_flow_points * 2),  # (N*2) for x,y
        )
        print("✓ 学习式光流预测器初始化成功")
    
    def predict_flow_from_control(
        self,
        control_sequence: torch.Tensor,
        initial_flow: torch.Tensor,
        method: str = "gs_flow",
        target_image: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        从控制序列预测光流轨迹
        
        Args:
            control_sequence: (B, T, control_dim) - 控制序列
            initial_flow: (B, N, 3) - 初始光流点 [x, y, visibility]
            method: "gs_flow" (使用calculate_gs_flow) 或 "render_based" (使用GMFlow对渲染图像)
            target_image: (3, H, W) - 目标图像（仅在render_based模式需要）
        
        Returns:
            predicted_flow: (B, T, N, 3) - 预测的光流轨迹
        """
        from utils.flow_utils import calculate_gs_flow
        from gaussian_renderer import render
        
        B, T, _ = control_sequence.shape
        N = initial_flow.shape[1]
        
        flow_trajectory = []
        background = torch.tensor([1, 1, 1], dtype=torch.float32, device=self.device)
        
        print(f"\\n[Flow Prediction] Starting prediction for {T} steps with {N} points")
        
        for t in range(T):
            control_vec = control_sequence[0, t, :]  # (control_dim,)
            
            # 打印控制向量信息
            control_norm = torch.norm(control_vec).item()
            control_nonzero = (control_vec.abs() > 0.01).sum().item()
            print(f"  Step {t}: control norm={control_norm:.4f}, nonzero={control_nonzero}/{control_vec.numel()}, range=[{control_vec.min():.3f}, {control_vec.max():.3f}]")
            
            with torch.no_grad():
                # 渲染当前状态
                if t == 0:
                    # 初始状态：使用15D零控制向量（而不是None，避免默认6D）
                    zero_control = torch.zeros(1, self.control_dim, device=initial_flow.device)
                    render_curr = render(
                        self.camera, self.gaussians, self.pipe_params, background,
                        stage="fine", cam_type="PerspectiveCameras",
                        is_training=False,
                        override_control_vec=zero_control
                    )
                else:
                    render_curr = render_next  # 复用上一步的next作为当前
                
                # 渲染下一状态 (应用control)
                render_next = render(
                    self.camera, self.gaussians, self.pipe_params, background,
                    stage="fine", cam_type="PerspectiveCameras",
                    is_training=False,
                    override_control_vec=control_vec
                )
                
                # 计算光流场 - 使用与训练时完全相同的函数
                gs_flow = calculate_gs_flow(
                    render_curr["gs_per_pixel"],
                    render_curr["weight_per_gs_pixel"],
                    render_next["conic_2D"],
                    render_curr["conic_2D_inv"],
                    render_curr["proj_2D"],
                    render_next["proj_2D"],
                    render_curr["x_mu"]
                )  # (2, H, W)
                
                #打印光流统计
                flow_magnitude = torch.sqrt(gs_flow[0]**2 + gs_flow[1]**2)
                flow_nonzero = (flow_magnitude > 0.01).sum().item()
                flow_total = flow_magnitude.numel()
                flow_nonzero_pct = 100.0 * flow_nonzero / flow_total
                print(f"    GS Flow: mean={flow_magnitude.mean():.4f}, max={flow_magnitude.max():.4f}, median={flow_magnitude.median():.4f}")
                print(f"             non-zero: {flow_nonzero}/{flow_total} ({flow_nonzero_pct:.1f}%)")
                
                # 从光流场采样得到flow points
                flow_points = torch.zeros(B, N, 3, device=self.device)
                
                # 使用当前位置采样（第一步用initial_flow，之后用更新后的位置）
                if t == 0:
                    current_positions = initial_flow[0, :, :2]  # (N, 2) 初始位置
                else:
                    current_positions = flow_trajectory[-1][0, :, :2]  # (N, 2) 上一步的位置
                
                # 限制在有效范围内
                x_coords = torch.clamp(current_positions[:, 0], 0, self.image_width - 1).long()
                y_coords = torch.clamp(current_positions[:, 1], 0, self.image_height - 1).long()
                
                # 从光流场采样（得到位移增量）
                flow_x = gs_flow[0, y_coords, x_coords]  # (N,) 位移增量
                flow_y = gs_flow[1, y_coords, x_coords]  # (N,) 位移增量
                
                # 新位置 = 当前位置 + 位移增量
                flow_points[0, :, 0] = current_positions[:, 0] + flow_x
                flow_points[0, :, 1] = current_positions[:, 1] + flow_y
                flow_points[0, :, 2] = 1.0  # visibility
                
                # 打印采样的flow points统计
                sampled_magnitude = torch.sqrt(flow_x**2 + flow_y**2)
                sampled_nonzero = (sampled_magnitude > 0.01).sum().item()
                print(f"    Sampled flow: mean={sampled_magnitude.mean():.4f}, max={sampled_magnitude.max():.4f}, median={sampled_magnitude.median():.4f}")
                print(f"                  non-zero: {sampled_nonzero}/{N} ({100.0*sampled_nonzero/N:.1f}%)")
                print(f"    Position update: from [{current_positions[:3, 0].mean():.1f}, {current_positions[:3, 1].mean():.1f}] -> [{flow_points[0, :3, 0].mean():.1f}, {flow_points[0, :3, 1].mean():.1f}]")
                
                flow_trajectory.append(flow_points)
        
        print(f"[Flow Prediction] Completed\n")
        
        return torch.stack(flow_trajectory, dim=1)  # (B, T, N, 3)
    
    def predict_dense_flow_field(
        self,
        control_sequence: torch.Tensor,
        initial_image: torch.Tensor,
        target_image: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        基于渲染图像使用GMFlow计算密集光流场（不进行采样）
        
        这个方法直接返回GMFlow的完整输出，适合直接计算密集loss。
        相比predict_flow_render_based，这个方法：
        1. 不进行采样，保留全分辨率光流
        2. 返回 (B, T, 2, H, W) 而非 (B, T, N, 3)
        3. 更适合用于密集对比和loss计算
        
        Args:
            control_sequence: (B, T, control_dim) - 控制序列
            initial_image: (3, H, W) - 初始渲染图像
            target_image: (3, H, W) - 目标图像（可选）
        
        Returns:
            flow_fields: (B, T, 2, H, W) - 密集光流场 (u, v 分量)
        """
        try:
            from gmflow.gmflow import GMFlow
            from gmflow.config import get_cfg as get_gmflow_cfg
        except ImportError:
            print("⚠ GMFlow not available, cannot predict dense flow field")
            B, T, _ = control_sequence.shape
            # 返回零光流场
            return torch.zeros(B, T, 2, self.image_height, self.image_width, device=self.device)
        
        B, T, _ = control_sequence.shape
        
        # 初始化GMFlow
        gmflow_cfg = get_gmflow_cfg()
        flownet = GMFlow(
            feature_channels=gmflow_cfg.feature_channels,
            num_scales=gmflow_cfg.num_scales,
            upsample_factor=gmflow_cfg.upsample_factor,
            num_head=gmflow_cfg.num_head,
            attention_type=gmflow_cfg.attention_type,
            ffn_dim_expansion=gmflow_cfg.ffn_dim_expansion,
        ).to(self.device)
        
        # 加载预训练权重
        checkpoint_path = "gmflow/checkpoints/gmflow_sintel-0c07dcb3.pth"
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            flownet.load_state_dict(checkpoint['model'], strict=False)
            flownet.eval()
        else:
            print(f"⚠ GMFlow checkpoint not found: {checkpoint_path}")
            return torch.zeros(B, T, 2, self.image_height, self.image_width, device=self.device)
        
        flow_fields = []
        
        if not hasattr(self, '_dense_flow_initialized'):
            print(f"\n[Dense Flow Field] Initialized for full resolution ({self.image_height}x{self.image_width})")
            self._dense_flow_initialized = True
        
        for t in range(T):
            control_vec = control_sequence[0, t, :]  # (control_dim,)
            
            with torch.no_grad():
                # 渲染下一状态（应用当前control）
                next_render = self.render_with_control(control_vec)
                
                # 准备GMFlow输入：(1, 3, H, W)
                # 固定初始帧作为参考帧（初始帧 → 预测未来帧）
                img1 = initial_image.unsqueeze(0) if initial_image.dim() == 3 else initial_image
                img2 = next_render.unsqueeze(0) if next_render.dim() == 3 else next_render
                
                # 归一化到[-1, 1]
                img1 = img1 * 2.0 - 1.0
                img2 = img2 * 2.0 - 1.0
                
                # 计算密集光流场
                flow_predictions = flownet(
                    img1, img2,
                    attn_splits_list=[2],
                    corr_radius_list=[-1],
                    prop_radius_list=[-1],
                )
                # GMFlow返回的是列表，取最后一个
                flow_field = flow_predictions[-1]  # (1, 2, H, W)
                flow_fields.append(flow_field)
        
        # 堆叠为batch和时间维度
        flow_fields = torch.cat(flow_fields, dim=0)  # (T, 2, H, W)
        flow_fields = flow_fields.unsqueeze(0).expand(B, -1, -1, -1, -1)  # (B, T, 2, H, W)
        
        return flow_fields
    
    def predict_flow_render_based(
        self,
        control_sequence: torch.Tensor,
        initial_image: torch.Tensor,
        target_image: torch.Tensor,
        sample_coords: torch.Tensor,
    ) -> torch.Tensor:
        """
        基于渲染图像使用GMFlow计算光流
        
        Args:
            control_sequence: (B, T, control_dim) - 控制序列
            initial_image: (3, H, W) - 初始渲染图像
            target_image: (3, H, W) - 目标图像
            sample_coords: (N, 2) - 采样坐标 [x, y] 归一化到[0, 1]
        
        Returns:
            predicted_flow: (B, T, N, 3) - 预测的光流轨迹
        """
        import os  # Move import to the top
        try:
            from gmflow.gmflow import GMFlow
            from gmflow.config import get_cfg as get_gmflow_cfg
        except ImportError:
            print("⚠ GMFlow not available, falling back to gs_flow method")
            # 回退到gs_flow方法
            initial_flow = torch.cat([
                sample_coords * torch.tensor([self.image_width, self.image_height], device=sample_coords.device),
                torch.ones(sample_coords.shape[0], 1, device=sample_coords.device)
            ], dim=-1).unsqueeze(0)  # (1, N, 3)
            return self.predict_flow_from_control(control_sequence, initial_flow, method="gs_flow")
        
        B, T, _ = control_sequence.shape
        N = sample_coords.shape[0]
        
        # 初始化GMFlow
        gmflow_cfg = get_gmflow_cfg()
        flownet = GMFlow(
            feature_channels=gmflow_cfg.feature_channels,
            num_scales=gmflow_cfg.num_scales,
            upsample_factor=gmflow_cfg.upsample_factor,
            num_head=gmflow_cfg.num_head,
            attention_type=gmflow_cfg.attention_type,
            ffn_dim_expansion=gmflow_cfg.ffn_dim_expansion,
        ).to(self.device)
        
        # 加载预训练权重
        checkpoint_path = "gmflow/checkpoints/gmflow_sintel-0c07dcb3.pth"
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            flownet.load_state_dict(checkpoint['model'], strict=False)
            flownet.eval()
        else:
            print(f"⚠ GMFlow checkpoint not found: {checkpoint_path}")
            print("  Falling back to gs_flow method")
            initial_flow = torch.cat([
                sample_coords * torch.tensor([self.image_width, self.image_height], device=sample_coords.device),
                torch.ones(sample_coords.shape[0], 1, device=sample_coords.device)
            ], dim=-1).unsqueeze(0)
            return self.predict_flow_from_control(control_sequence, initial_flow, method="gs_flow")
        
        flow_trajectory = []
        background = torch.tensor([1, 1, 1], dtype=torch.float32, device=self.device)
        
        # 减少调试输出，只在第一次调用时显示简要信息
        if not hasattr(self, '_flow_render_initialized'):
            print(f"\n[Render-Based Flow] Initialized for {N} tracking points")
            self._flow_render_initialized = True
        
        # 初始位置：使用sample_coords作为起始点（固定为初始帧位置）
        current_coords = sample_coords.clone()  # (N, 2) 归一化坐标
        
        # 固定初始帧作为参考帧
        current_render = initial_image  # 从真实初始图像开始
        
        for t in range(T):
            control_vec = control_sequence[0, t, :]  # (control_dim,) - 当前步的独立控制
            
            with torch.no_grad():
                # 【关键修复2】渲染下一状态：应用当前步的控制向量
                # 注意：deformation网络训练时使用的是每一步的独立控制向量
                # 不应该累积或平均，应该直接使用当前控制
                next_render = self.render_with_control(control_vec)
                
                # 准备GMFlow输入：(1, 3, H, W)
                # 固定初始帧作为参考帧（初始帧 → 预测未来帧）
                img1 = current_render.unsqueeze(0) if current_render.dim() == 3 else current_render
                img2 = next_render.unsqueeze(0) if next_render.dim() == 3 else next_render
                
                # 归一化到[-1, 1]
                img1 = img1 * 2.0 - 1.0
                img2 = img2 * 2.0 - 1.0
                
                # 计算光流场
                # GMFlow的forward返回flow_preds列表，直接调用即可
                flow_predictions = flownet(
                    img1, img2,
                    attn_splits_list=[2],  # 使用固定参数
                    corr_radius_list=[-1],
                    prop_radius_list=[-1],
                )
                # GMFlow返回的是列表，取最后一个
                flow_field_tensor = flow_predictions[-1]  # (1, 2, H, W)
                
                flow_field = flow_field_tensor[0].permute(1, 2, 0).cpu().numpy()  # (H, W, 2)
                
                # 【关键修改】从当前追踪的位置采样光流（而不是固定的sample_coords）
                # 这样可以追踪移动的物体，即使它移动到了新位置
                H, W = flow_field.shape[:2]
                x_coords = torch.clamp(current_coords[:, 0] * W, 0, W - 1).long()
                y_coords = torch.clamp(current_coords[:, 1] * H, 0, H - 1).long()
                
                # 采样光流增量（从追踪的当前位置）
                flow_x = torch.from_numpy(flow_field[y_coords.cpu(), x_coords.cpu(), 0]).to(self.device)
                flow_y = torch.from_numpy(flow_field[y_coords.cpu(), x_coords.cpu(), 1]).to(self.device)
                
                # 【关键】更新位置：new_position = current_position + flow_delta
                # 这实现了对移动点的追踪
                new_x = current_coords[:, 0] + flow_x / W
                new_y = current_coords[:, 1] + flow_y / H
                
                # 检查可见性（点是否仍在画面内）
                visibility = ((new_x >= 0) & (new_x <= 1) & (new_y >= 0) & (new_y <= 1)).float()
                
                # 保存光流点
                flow_points = torch.zeros(B, N, 3, device=self.device)
                flow_points[0, :, 0] = new_x
                flow_points[0, :, 1] = new_y
                flow_points[0, :, 2] = visibility
                
                flow_trajectory.append(flow_points)
                
                # 固定初始帧参考，不更新current_render与current_coords
        
        return torch.stack(flow_trajectory, dim=1)  # (B, T, N, 3)

    def __call__(self, batch, grad_enabled=False):
        """
        MPC接口：调用forward进行预测
        覆盖父类的__call__以使用光流引导的预测
        """
        return self.forward(batch, grad_enabled=grad_enabled)
    
    def forward(
        self,
        batch: Dict[str, np.ndarray],
        grad_enabled: bool = False,
    ) -> Dict[str, np.ndarray]:
        """
        前向传播：预测未来状态
        
        Args:
            batch: 包含以下键的字典
                - 'actions': (B, T, control_dim) - 动作序列
                - 'video': (B, n_context, H, W, 3) - 历史观察
                - 'flow': (B, n_context, N, 3) - 历史光流（可选）
        
        Returns:
            predictions: 包含预测的字典
                - 'rgb': (B, T, H, W, 3) - 如果启用完整渲染
                - 'flow': (B, T, N, 3) - 预测的光流
                - 'sparse_rgb': (B, T, H, W, 3) - 稀疏渲染（如果启用）
        """
        actions = batch['actions']
        if isinstance(actions, np.ndarray):
            actions = torch.from_numpy(actions)
        if isinstance(actions, torch.Tensor):
            actions = actions.to(self.device).float()
        else:
            raise TypeError(f"Unsupported actions type: {type(actions)}")
        B, T_total, _ = actions.shape
        T = T_total - self.num_context + 1  # 实际预测的时间步数
        
        # 提取上下文和预测部分
        context_actions = actions[:, :self.num_context]
        pred_actions = actions[:, self.num_context - 1:]
        
        predictions = {
            'rgb': [],
            'flow': [],
            'sparse_rgb': [],
        }
        
        # 获取初始光流
        if 'flow' in batch:
            flow_data = batch['flow']
            # 处理两种格式: (B, N, 3) 或 (B, T_context, N, 3)
            if len(flow_data.shape) == 3:
                # (B, N, 3) - 直接使用
                current_flow = torch.tensor(
                    flow_data,
                    dtype=torch.float32,
                    device=self.device
                )
            else:
                # (B, T_context, N, 3) - 取最后一帧
                current_flow = torch.tensor(
                    flow_data[:, -1],
                    dtype=torch.float32,
                    device=self.device
                )
        else:
            # 初始化光流点（均匀采样或从关键点检测）
            current_flow = self._initialize_flow_points(
                batch['video'][:, -1]
            )  # (B, N, 3)
        
        # 渲染初始帧（用于“初始帧 → 预测未来帧”的光流）
        initial_rendered = torch.tensor(
            batch['video'][0, 0],
            dtype=torch.float32,
            device=self.device
        ).permute(2, 0, 1)

        # 逐步预测
        for t in range(T):
            # 保留梯度：直接转换而非创建新tensor
            control_vec = pred_actions[:, t].to(self.device).float()
            
            # 1. 光流预测
            if self.enable_flow_prediction:
                if self.flow_prediction_method == "render_based":
                    # 使用render_based方法：渲染当前状态并用GMFlow计算光流
                    if t > 0:
                        # 固定使用初始帧作为参考帧（初始帧 → 预测未来帧）
                        current_rendered = initial_rendered  # (3, H, W)
                        
                        # 使用predict_flow_render_based
                        # control_vec 已经是 (B, control_dim)，只需添加时间维度
                        next_flow_tensor = self.predict_flow_render_based(
                            control_vec.unsqueeze(1),  # (B, 1, control_dim)
                            current_rendered,  # (3, H, W)
                            self.target_image.to(self.device) if self.target_image is not None else current_rendered,
                            self.sample_coords.to(self.device) if self.sample_coords is not None else current_flow[:, :, :2],
                        )  # (B, 1, N, 3)
                        next_flow = next_flow_tensor[:, 0]  # (B, N, 3)
                    else:
                        next_flow = current_flow  # 第一步保持不变
                
                # 使用已训练的deformation网络预测真实光流
                elif t > 0:
                    with torch.no_grad():
                        # 获取3D高斯点
                        xyz_3d = self.gaussians.get_xyz  # (M, 3)
                        
                        # 构建投影矩阵
                        world_view_transform = torch.tensor(
                            getWorld2View2(self.camera.R, self.camera.T),
                            dtype=torch.float32
                        ).transpose(0, 1).to(self.device)
                        
                        projection_matrix = torch.tensor(
                            getProjectionMatrix(
                                znear=self.camera.znear,
                                zfar=self.camera.zfar,
                                fovX=self.camera.FoVx,
                                fovY=self.camera.FoVy
                            ),
                            dtype=torch.float32
                        ).transpose(0, 1).to(self.device)
                        
                        full_proj_transform = world_view_transform @ projection_matrix
                        
                        # 投影所有3D点到2D
                        xyz_homo = torch.cat([xyz_3d, torch.ones(xyz_3d.shape[0], 1, device=self.device)], dim=1)
                        xyz_proj = xyz_homo @ full_proj_transform.T
                        xyz_ndc = xyz_proj[:, :2] / (xyz_proj[:, 3:4] + 1e-7)
                        xyz_2d = (xyz_ndc + 1.0) / 2.0  # [0, 1]
                        
                        # 找到每个光流点最近的高斯点
                        # 处理batch: 对每个batch element分别处理
                        all_next_flows = []
                        for b in range(B):
                            flow_2d_b = current_flow[b, :, :2]  # (N, 2)
                            
                            # 为当前batch计算最近邻索引
                            distances = torch.cdist(flow_2d_b, xyz_2d)  # (N, M)
                            nearest_indices = distances.argmin(dim=1)  # (N,)
                            
                            # 获取对应的高斯参数
                            means3D = xyz_3d[nearest_indices]
                            scales = self.gaussians._scaling[nearest_indices]
                            rotations = self.gaussians._rotation[nearest_indices]
                            opacity = self.gaussians._opacity[nearest_indices]
                            shs = self.gaussians.get_features[nearest_indices]
                            
                            # 准备control_vec - 需要为每个点复制
                            # deformation网络期望: (N, control_dim)
                            control_batch = control_vec[b].unsqueeze(0).repeat(means3D.shape[0], 1)  # (N, control_dim)
                            
                            # 准备deformation网络的输入（triplane+film结构）
                            # 需要rays_pts_emb (位置+PE), scales_emb, rotations_emb等
                            # 为简化，我们只传入基础参数，不使用PE（deformation内部会处理）
                            rays_pts_emb = means3D  # (N, 3) - deformation会在内部处理PE
                            scales_emb = scales  # (N, 3)
                            rotations_emb = rotations  # (N, 4)
                            opacity_emb = opacity  # (N, 1)
                            shs_emb = shs  # (N, 16, 3)
                            
                            # 应用deformation网络（triplane+film版本）
                            # forward_dynamic返回: (pts, scales, rotations, opacity, shs)
                            means3D_deformed, _, _, _, _ = self.gaussians._deformation.forward_dynamic(
                                rays_pts_emb, scales_emb, rotations_emb, opacity_emb, shs_emb, control_batch
                            )
                            
                            # 投影变形后的3D点到2D
                            xyz_deformed_homo = torch.cat(
                                [means3D_deformed, torch.ones(means3D_deformed.shape[0], 1, device=self.device)],
                                dim=1
                            )
                            xyz_deformed_proj = xyz_deformed_homo @ full_proj_transform.T
                            
                            # 检查深度有效性
                            valid_depth = xyz_deformed_proj[:, 2] > 0  # (N,)
                            
                            xyz_deformed_ndc = xyz_deformed_proj[:, :2] / (xyz_deformed_proj[:, 3:4] + 1e-7)
                            
                            # 裁剪到有效NDC范围 [-1, 1]，防止异常值
                            xyz_deformed_ndc = torch.clamp(xyz_deformed_ndc, -1.0, 1.0)
                            xyz_deformed_2d = (xyz_deformed_ndc + 1.0) / 2.0
                            
                            # 检查可见性（深度 > 0 且在屏幕内）
                            in_screen = (xyz_deformed_2d[:, 0] >= 0) & (xyz_deformed_2d[:, 0] <= 1) & \
                                       (xyz_deformed_2d[:, 1] >= 0) & (xyz_deformed_2d[:, 1] <= 1)
                            visibility = (valid_depth & in_screen).float().unsqueeze(-1)
                            
                            # 对于不可见的点，保持原位置
                            next_flow_2d = torch.where(
                                visibility.expand(-1, 2) > 0,
                                xyz_deformed_2d,
                                flow_2d_b  # 保持原位置
                            )
                            
                            # 组合成光流
                            next_flow_b = torch.cat([next_flow_2d, visibility], dim=-1)  # (N, 3)
                            all_next_flows.append(next_flow_b)
                        
                        # 合并所有batch
                        next_flow = torch.stack(all_next_flows, dim=0)  # (B, N, 3)
                else:
                    next_flow = current_flow  # 第一步保持不变
            else:
                next_flow = current_flow  # 不预测
            
            if grad_enabled:
                predictions['flow'].append(next_flow)  # 保留 tensor
            else:
                predictions['flow'].append(next_flow.cpu().numpy())  # numpy
            
            # 2. 稀疏渲染（用于验证和可解释性）
            if self.use_sparse_rendering:
                # 选择重要高斯并渲染
                sparse_rgb = self._render_sparse(
                    control_vec,
                    current_flow,
                    next_flow,
                )
                predictions['sparse_rgb'].append(sparse_rgb)
            
            # 3. 完整渲染（可选，更慢）
            if not self.use_sparse_rendering:
                # render_with_control返回 (3, H, W)，需要转换为 (H, W, 3)
                timestep_rgbs = []
                for b in range(B):
                    full_rgb = self.render_with_control(control_vec[b], grad_enabled=grad_enabled)
                    
                    if grad_enabled:
                        # 保留tensor用于梯度计算
                        full_rgb_hwc = full_rgb.permute(1, 2, 0)  # (H, W, 3) tensor
                    else:
                        # 转换为numpy（原始行为）
                        full_rgb_hwc = full_rgb.permute(1, 2, 0).cpu().numpy()  # (H, W, 3)
                    timestep_rgbs.append(full_rgb_hwc)
                
                if grad_enabled:
                    # 返回torch.stack保留梯度
                    predictions['rgb'].append(torch.stack(timestep_rgbs, dim=0))  # (B, H, W, 3) tensor
                else:
                    # 返回numpy（原始行为）
                    predictions['rgb'].append(np.stack(timestep_rgbs, axis=0))  # (B, H, W, 3)
            
            # 更新当前光流
            current_flow = next_flow
        
        # 转换为numpy数组（仅在非梯度模式下）
        if not grad_enabled:
            predictions['flow'] = np.stack(predictions['flow'], axis=1)  # (B, T, N, 3)
        else:
            # 保留torch tensor for gradients
            predictions['flow'] = torch.stack(predictions['flow'], dim=1)  # (B, T, N, 3)
        
        # 处理RGB输出
        if self.use_sparse_rendering:
            if len(predictions['sparse_rgb']) > 0:
                # 用sparse_rgb作为rgb输出
                if not grad_enabled:
                    predictions['rgb'] = np.stack(predictions['sparse_rgb'], axis=1)
                else:
                    predictions['rgb'] = torch.stack(predictions['sparse_rgb'], dim=1)
            else:
                # Sparse rendering没有生成RGB，使用fallback
                print("[Warning] Sparse rendering failed, rendering full images...")
                predictions['rgb'] = self._render_fallback_rgb(pred_actions, B, T, grad_enabled=grad_enabled)
        else:
            if len(predictions['rgb']) > 0:
                # 使用完整渲染的RGB
                if not grad_enabled:
                    predictions['rgb'] = np.stack(predictions['rgb'], axis=1)
                else:
                    predictions['rgb'] = torch.stack(predictions['rgb'], dim=1)
            else:
                # 完整渲染失败，使用fallback
                print("[Warning] No RGB predictions generated, rendering fallback images...")
                predictions['rgb'] = self._render_fallback_rgb(pred_actions, B, T, grad_enabled=grad_enabled)
        
        # 确保删除sparse_rgb键
        if 'sparse_rgb' in predictions:
            del predictions['sparse_rgb']
        
        # 添加actions到predictions，供ActionRegularizationObjective使用
        predictions['actions'] = actions  # (B, T_total, control_dim)
        
        return predictions
    
    def _render_fallback_rgb(self, pred_actions, B, T, grad_enabled=False):
        """生成fallback RGB图像"""
        fallback_rgbs = []
        for t in range(T):
            control_vec = torch.tensor(
                pred_actions[:, t], 
                dtype=torch.float32, 
                device=self.device
            )
            timestep_rgbs = []
            for b in range(B):
                full_rgb = self.render_with_control(control_vec[b], grad_enabled=grad_enabled)
                
                if grad_enabled:
                    full_rgb_hwc = full_rgb.permute(1, 2, 0)  # (H, W, 3) tensor
                else:
                    full_rgb_hwc = full_rgb.permute(1, 2, 0).cpu().numpy()  # (H, W, 3)
                timestep_rgbs.append(full_rgb_hwc)
            
            if grad_enabled:
                fallback_rgbs.append(torch.stack(timestep_rgbs, dim=0))
            else:
                fallback_rgbs.append(np.stack(timestep_rgbs, axis=0))
        
        if grad_enabled:
            return torch.stack(fallback_rgbs, dim=1)  # (B, T, H, W, 3)
        else:
            return np.stack(fallback_rgbs, axis=1)  # (B, T, H, W, 3)
    
    def _initialize_flow_points(
        self,
        images: np.ndarray,
        method: str = "grid",
    ) -> torch.Tensor:
        """
        初始化光流跟踪点
        
        改进：直接从投影的高斯点采样，而不是使用任意网格
        这确保追踪的是真实的物理点
        
        Args:
            images: (B, H, W, 3) - 初始图像
            method: "grid", "gaussian_projection", "keypoint", "random"
        
        Returns:
            flow_points: (B, N, 3) - 初始化的点 [x, y, visibility]
        """
        B, H, W, _ = images.shape
        
        if method == "gaussian_projection":
            # 方法1: 从投影的高斯点采样（推荐）
            with torch.no_grad():
                xyz_3d = self.gaussians.get_xyz  # (M, 3)
                
                # 投影3D高斯到2D
                from utils.graphics_utils import getWorld2View2, getProjectionMatrix
                world_view_transform = torch.tensor(
                    getWorld2View2(self.camera.R, self.camera.T),
                    dtype=torch.float32
                ).transpose(0, 1).to(self.device)
                
                projection_matrix = torch.tensor(
                    getProjectionMatrix(
                        znear=self.camera.znear,
                        zfar=self.camera.zfar,
                        fovX=self.camera.FoVx,
                        fovY=self.camera.FoVy
                    ),
                    dtype=torch.float32
                ).transpose(0, 1).to(self.device)
                
                full_proj_transform = world_view_transform @ projection_matrix
                
                # 投影所有3D点到2D
                xyz_homo = torch.cat([xyz_3d, torch.ones(xyz_3d.shape[0], 1, device=self.device)], dim=1)
                xyz_proj = xyz_homo @ full_proj_transform.T
                
                # 检查深度有效性
                valid_depth = xyz_proj[:, 2] > 0
                
                xyz_ndc = xyz_proj[:, :2] / (xyz_proj[:, 3:4] + 1e-7)
                xyz_2d = (xyz_ndc + 1.0) / 2.0  # [0, 1]
                
                # 检查在屏幕内
                in_screen = (xyz_2d[:, 0] >= 0) & (xyz_2d[:, 0] <= 1) & \
                           (xyz_2d[:, 1] >= 0) & (xyz_2d[:, 1] <= 1) & valid_depth
                
                # 只保留可见的点
                valid_indices = torch.where(in_screen)[0]
                
                if len(valid_indices) > self.num_flow_points:
                    # 随机采样到目标数量
                    perm = torch.randperm(len(valid_indices))[:self.num_flow_points]
                    selected_indices = valid_indices[perm]
                else:
                    # 不够就全选，然后补充
                    selected_indices = valid_indices
                
                xyz_2d_selected = xyz_2d[selected_indices]
                visibility = torch.ones(len(selected_indices), 1, device=self.device)
                
                # 如果不够，用网格点补充
                if len(selected_indices) < self.num_flow_points:
                    num_needed = self.num_flow_points - len(selected_indices)
                    grid_size = int(np.sqrt(num_needed)) + 1
                    y_coords = torch.linspace(0, 1, grid_size, device=self.device)
                    x_coords = torch.linspace(0, 1, grid_size, device=self.device)
                    yy, xx = torch.meshgrid(y_coords, x_coords, indexing='ij')
                    grid_points = torch.stack([xx.flatten(), yy.flatten()], dim=-1)[:num_needed]
                    grid_vis = torch.ones(num_needed, 1, device=self.device)
                    
                    xyz_2d_selected = torch.cat([xyz_2d_selected, grid_points], dim=0)
                    visibility = torch.cat([visibility, grid_vis], dim=0)
                
                points = torch.cat([xyz_2d_selected, visibility], dim=-1)  # (N, 3)
                # 复制到batch
                points = points.unsqueeze(0).repeat(B, 1, 1)  # (B, N, 3)
                
                # 保存高斯索引映射，用于后续追踪
                self.flow_to_gaussian_indices = selected_indices
                
                return points
        
        elif method == "grid":
            # 方法2: 均匀网格采样（原方法）
            grid_size = int(np.ceil(np.sqrt(self.num_flow_points)))
            y_coords = np.linspace(0, 1, grid_size)
            x_coords = np.linspace(0, 1, grid_size)
            yy, xx = np.meshgrid(y_coords, x_coords)
            
            points = np.stack([xx.flatten(), yy.flatten()], axis=-1)[:self.num_flow_points]  # (N, 2)
            points = points[:self.num_flow_points]
            
            # 添加可见性
            visibility = np.ones((points.shape[0], 1))
            points = np.concatenate([points, visibility], axis=-1)  # (N, 3)
            
            # 复制到batch
            points = np.tile(points[None], (B, 1, 1))  # (B, N, 3)
            
            return torch.tensor(points, dtype=torch.float32, device=self.device)
        
        elif method == "keypoint":
            # 使用关键点检测器（如SIFT, ORB）
            # TODO: 实现
            print("  ⚠ keypoint方法未实现，fallback到grid方法")
            return self._initialize_flow_points(images, method="grid")
        
        elif method == "random":
            # 随机采样
            points = np.random.rand(B, self.num_flow_points, 2)
            visibility = np.ones((B, self.num_flow_points, 1))
            points = np.concatenate([points, visibility], axis=-1)
            
            return torch.tensor(points, dtype=torch.float32, device=self.device)
        
        else:
            raise ValueError(f"Unknown initialization method: {method}")
    
    def _render_sparse(
        self,
        control_vec: torch.Tensor,
        current_flow: torch.Tensor,
        target_flow: torch.Tensor,
    ) -> np.ndarray:
        """
        执行稀疏渲染
        
        Returns:
            rendered_image: (B, H, W, 3)
        """
        B = control_vec.shape[0]
        
        # 调用稀疏渲染器
        render_results = []
        for b in range(B):
            result = self.sparse_renderer.render_sparse(
                self.gaussians,
                self.camera,
                control_vec[b],
                current_flow[b],
                target_flow[b],
            )
            render_results.append(result)
        
        # TODO: 实际渲染图像
        # 这里需要修改4DGaussians的渲染函数，支持稀疏高斯
        # rendered_images = ...
        
        # 暂时返回dummy
        rendered_images = np.zeros((B, self.image_height, self.image_width, 3))
        
        return rendered_images
    
    def compute_rendering_cost(self) -> Dict[str, float]:
        """
        计算渲染开销统计
        """
        if not hasattr(self, 'gaussian_selector'):
            return {"full_gaussians": len(self.gaussians.get_xyz)}
        
        selection = self.gaussian_selector.cache
        if 'num_selected' not in selection:
            return {}
        
        total_gaussians = len(self.gaussians.get_xyz)
        selected_gaussians = selection['num_selected']
        reduction_ratio = 1 - (selected_gaussians / total_gaussians)
        
        return {
            "total_gaussians": total_gaussians,
            "selected_gaussians": selected_gaussians,
            "reduction_ratio": reduction_ratio,
            "speedup_estimate": 1 / (1 - reduction_ratio + 0.1),  # 粗略估计
        }


# 使用示例
if __name__ == "__main__":
    # 初始化光流引导的Gaussian模型
    model = FlowGuidedGaussianDynamicsModel(
        model_path="/path/to/4dgs/model",
        iteration=5000,
        control_dim=15,
        image_height=256,
        image_width=256,
        num_context=2,
        # 新参数
        use_sparse_rendering=True,
        sparse_ratio=0.15,  # 只渲染15%的高斯
        enable_flow_prediction=True,
        flow_prediction_method="learned",
        num_flow_points=512,
    )
    
    print("✓ 光流引导Gaussian模型初始化完成")
    
    # 测试预测
    batch = {
        'actions': np.random.randn(2, 12, 15),  # (B=2, T=12, D=15)
        'video': np.random.randn(2, 2, 256, 256, 3),  # (B, context=2, H, W, 3)
    }
    
    predictions = model(batch)
    print(f"预测光流形状: {predictions['flow'].shape}")
    print(f"预测图像形状: {predictions['rgb'].shape}")
    
    # 查看渲染开销
    cost_stats = model.compute_rendering_cost()
    print(f"\n渲染开销统计:")
    for k, v in cost_stats.items():
        print(f"  {k}: {v}")
