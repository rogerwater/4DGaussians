"""
简化版的光流预测函数 - 直接使用calculate_gs_flow
替换mpc/flow_guided_gaussian_model.py中的predict_flow_from_control方法
"""

def predict_flow_from_control_simplified(
    self,
    control_sequence: torch.Tensor,
    initial_flow: torch.Tensor,
    method: str = "gs_flow",
) -> torch.Tensor:
    """
    从控制序列预测光流 - 使用calculate_gs_flow
    
    Args:
        control_sequence: (B, T, control_dim)
        initial_flow: (B, N, 3) [x, y, visibility]
        
    Returns:
        predicted_flow: (B, T, N, 3)
    """
    from utils.flow_utils import calculate_gs_flow
    from gaussian_renderer import render
    
    B, T, _ = control_sequence.shape
    N = initial_flow.shape[1]
    
    flow_trajectory = []
    background = torch.tensor([1, 1, 1], dtype=torch.float32, device=self.device)
    
    for t in range(T):
        control_vec = control_sequence[0, t, :]  # (control_dim,)
        
        with torch.no_grad():
            # 渲染当前状态
            if t == 0:
                render_curr = render(
                    self.camera, self.gaussians, self.pipe_params, background,
                    stage="fine", cam_type="PerspectiveCameras",
                    is_training=False,
                    override_control_vec=None
                )
            else:
                render_curr = render_next
            
            # 渲染下一状态
            render_next = render(
                self.camera, self.gaussians, self.pipe_params, background,
                stage="fine", cam_type="PerspectiveCameras",
                is_training=False,
                override_control_vec=control_vec
            )
            
            # 计算光流场
            gs_flow = calculate_gs_flow(
                render_curr["gs_per_pixel"],
                render_curr["weight_per_gs_pixel"],
                render_next["conic_2D"],
                render_curr["conic_2D_inv"],
                render_curr["proj_2D"],
                render_next["proj_2D"],
                render_curr["x_mu"]
            )  # (2, H, W)
            
            # 从光流场采样得到flow points
            flow_points = torch.zeros(B, N, 3, device=self.device)
            xy_coords = initial_flow[0, :, :2]  # (N, 2)
            
            # 限制在有效范围
            x_coords = torch.clamp(xy_coords[:, 0], 0, self.image_width - 1).long()
            y_coords = torch.clamp(xy_coords[:, 1], 0, self.image_height - 1).long()
            
            # 采样光流
            flow_x = gs_flow[0, y_coords, x_coords]
            flow_y = gs_flow[1, y_coords, x_coords]
            
            # 累积光流
            if t == 0:
                flow_points[0, :, 0] = flow_x
                flow_points[0, :, 1] = flow_y
            else:
                prev_flow = flow_trajectory[-1]
                flow_points[0, :, 0] = prev_flow[0, :, 0] + flow_x
                flow_points[0, :, 1] = prev_flow[0, :, 1] + flow_y
            
            flow_points[0, :, 2] = 1.0
            flow_trajectory.append(flow_points)
    
    return torch.stack(flow_trajectory, dim=1)  # (B, T, N, 3)
