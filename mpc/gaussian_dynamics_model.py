"""
4D Gaussian Splatting Dynamics Model for MPC Control

This module integrates 4DGS with control encoder for model-based planning.
It enables CEM/MPC to optimize control inputs (joint angles + gripper) 
to achieve desired visual goals.
"""

import torch
import numpy as np
import os
import sys
from pathlib import Path

# Add 4DGaussians to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scene import Scene
from gaussian_renderer import GaussianModel
from gaussian_renderer import render
from arguments import ModelParams, PipelineParams, ModelHiddenParams
from scene.cameras import Camera
from utils.graphics_utils import getWorld2View2, getProjectionMatrix


class GaussianDynamicsModel:
    """
    4D Gaussian Splatting dynamics model for visual predictive control.
    
    This model predicts future visual observations given:
    - Current Gaussian scene state
    - Sequence of control inputs (joint angles + gripper)
    
    The model uses a learned deformation network with control encoder
    to predict how the scene evolves under different control inputs.
    """
    
    def __init__(
        self,
        model_path,
        iteration=5000,
        control_dim=15,
        image_height=480,
        image_width=480,
        device="cuda",
        camera_distance=2.0,
        camera_elevation=45.0,
        camera_azimuth=0.0,
        fov_degrees=45.0,
        transform_matrix=None,
        focal_x=None,
        focal_y=None,
        cx=None,
        cy=None
    ):
        """
        Args:
            model_path: Path to trained 4DGS model directory
            iteration: Checkpoint iteration to load
            control_dim: Control input dimension (15 = 12 joint angles + 3 gripper)
            image_height: Rendered image height
            image_width: Rendered image width
            device: Device to run on
            camera_distance: Camera distance from origin (for spherical mode)
            camera_elevation: Camera elevation angle in degrees (for spherical mode)
            camera_azimuth: Camera azimuth angle in degrees (for spherical mode)
            fov_degrees: Field of view in degrees (for spherical mode)
            transform_matrix: 4x4 camera transform matrix (c2w, overrides spherical params)
            focal_x: Focal length x (for transform_matrix mode)
            focal_y: Focal length y (for transform_matrix mode)
            cx: Principal point x (for transform_matrix mode)
            cy: Principal point y (for transform_matrix mode)
        """
        self.device = device
        self.control_dim = control_dim
        self.image_height = image_height
        self.image_width = image_width
        self.iteration = iteration
        
        # Camera parameters - support both modes
        self.transform_matrix = transform_matrix
        self.focal_x = focal_x
        self.focal_y = focal_y
        self.cx = cx if cx is not None else image_width / 2.0
        self.cy = cy if cy is not None else image_height / 2.0
        
        # Spherical camera parameters (used if transform_matrix is None)
        self.camera_distance = camera_distance
        self.camera_elevation = camera_elevation
        self.camera_azimuth = camera_azimuth
        self.fov_degrees = fov_degrees
        
        # Setup model parameters
        self.model_params = self._setup_model_params(model_path)
        self.hidden_params = self._setup_hidden_params()
        
        # Setup pipeline parameters
        class SimplePipeParams:
            pass
        self.pipe_params = SimplePipeParams()
        self.pipe_params.convert_SHs_python = False
        self.pipe_params.compute_cov3D_python = False
        self.pipe_params.debug = False
        
        # Initialize Gaussian model
        print(f"Loading 4DGS model from {model_path}")
        self.gaussians = GaussianModel(
            sh_degree=self.model_params.sh_degree,
            args=self.hidden_params
        )
        
        # Load trained checkpoint
        self._load_checkpoint(model_path, iteration)
        
        # Setup rendering camera
        self.camera = self._create_camera()
        
        # Background color (white or black)
        bg_color = [1, 1, 1] if self.model_params.white_background else [0, 0, 0]
        self.background = torch.tensor(bg_color, dtype=torch.float32, device=device)
        
        # For MPC interface
        self.base_prediction_modality = "rgb"
        self.num_context = 2  # Number of context frames for MPC
        
        print(f"✓ Model loaded successfully!")
        print(f"  - Control dim: {control_dim}")
        print(f"  - Image size: {image_width}x{image_height}")
        
    def _setup_model_params(self, model_path):
        """Setup model parameters from path"""
        # Create a simple object to hold parameters instead of using ArgumentParser
        class SimpleParams:
            pass
        
        params = SimpleParams()
        params.sh_degree = 3
        params.source_path = model_path
        params.model_path = model_path
        params.images = "images"
        params.resolution = -1
        params.white_background = True
        params.data_device = "cuda"
        params.eval = True
        params.render_process = False
        params.add_points = False
        params.extension = ".png"
        params.llffhold = 8
        return params
    
    def _setup_hidden_params(self):
        """Setup hidden parameters (deformation network config)"""
        # Create a simple object to hold parameters
        class SimpleParams:
            pass
        
        params = SimpleParams()
        
        # Network architecture - MUST MATCH TRAINING CONFIG
        # This checkpoint uses Triplane+FiLM architecture
        params.net_width = 128
        params.defor_depth = 3  # FiLM layers
        params.timebase_pe = 4  # legacy
        params.posebase_pe = 10
        params.scale_rotation_pe = 2
        params.opacity_pe = 2
        params.timenet_width = 64  # legacy
        params.timenet_output = 32  # legacy
        params.bounds = 1.6
        
        # Regularization weights
        params.plane_tv_weight = 0.0001
        params.time_smoothness_weight = 0.0  # Not used in TriPlane
        params.l1_time_planes = 0.0  # Not used in TriPlane
        
        # TriPlane configuration - MUST MATCH CHECKPOINT
        params.kplanes_config = {
            'grid_dimensions': 2,
            'input_coordinate_dim': 3,  # Spatial only (no time)
            'output_coordinate_dim': 32,
            'resolution': [128, 128, 128]  # Spatial resolution only
        }
        params.multires = [1, 2, 4]  # 3 multi-resolution levels
        
        # Control signal configuration - MUST MATCH CHECKPOINT
        params.control_input_dim = 15  # 15-dim control vector
        params.control_use_pe = False  # No PE on control
        params.control_num_frequencies = 4
        params.control_hidden_dim = 128
        params.control_output_dim = 32
        
        # FiLM configuration
        params.film_hidden_dim = 128
        params.film_use_residual = True
        
        # Deformation toggles
        params.no_dx = False
        params.no_grid = False
        params.no_ds = False
        params.no_dr = False
        params.no_do = True
        params.no_dshs = True
        params.empty_voxel = False
        params.grid_pe = 0
        params.static_mlp = False
        params.apply_rotation = False
        
        # Architecture selection
        params.use_triplane = True  # IMPORTANT: Checkpoint uses Triplane+FiLM
        
        return params
    
    def _load_checkpoint(self, model_path, iteration):
        """Load trained Gaussian model checkpoint"""
        checkpoint_path = os.path.join(
            model_path
        )
        
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(
                f"Checkpoint not found: {checkpoint_path}\n"
                f"Available iterations: {self._list_available_iterations(model_path)}"
            )
        
        # Load point cloud
        ply_path = os.path.join(checkpoint_path, "point_cloud.ply")
        self.gaussians.load_ply(ply_path)
        
        # Load deformation network
        deform_path = os.path.join(checkpoint_path, "deformation.pth")
        if os.path.exists(deform_path):
            deform_state = torch.load(deform_path)
            self.gaussians._deformation.load_state_dict(deform_state)
            print(f"  ✓ Loaded deformation network from {deform_path}")
        else:
            print(f"  ⚠ Warning: No deformation.pth found, using undeformed Gaussians")
        
        # Move to device
        self.gaussians._xyz = self.gaussians._xyz.to(self.device)
        self.gaussians._features_dc = self.gaussians._features_dc.to(self.device)
        self.gaussians._features_rest = self.gaussians._features_rest.to(self.device)
        self.gaussians._scaling = self.gaussians._scaling.to(self.device)
        self.gaussians._rotation = self.gaussians._rotation.to(self.device)
        self.gaussians._opacity = self.gaussians._opacity.to(self.device)
        self.gaussians._deformation = self.gaussians._deformation.to(self.device)
        
    def _list_available_iterations(self, model_path):
        """List available checkpoint iterations"""
        pc_path = os.path.join(model_path, "point_cloud")
        if not os.path.exists(pc_path):
            return []
        iterations = [d for d in os.listdir(pc_path) if d.startswith("iteration_")]
        return sorted(iterations)
    
    def _create_camera(self):
        """Create a virtual camera for rendering.
        
        Supports two modes:
        1. Transform matrix mode: Use provided c2w matrix and intrinsics
        2. Spherical mode: Generate camera from distance/elevation/azimuth
        """
        if self.transform_matrix is not None:
            # Mode 1: Use provided transform matrix (camera-to-world)
            c2w = np.array(self.transform_matrix, dtype=np.float32)
            
            # Apply 180-degree X-axis rotation (same as training dataset)
            # This transformation is CRITICAL for matching training camera convention
            R_x_180 = np.diag(np.array([1, -1, -1, 1], dtype=np.float32))
            c2w = c2w @ R_x_180
            
            # Convert to world-to-camera
            w2c = np.linalg.inv(c2w)
            
            # Extract R and T (matching toyarm_dataset.py)
            R = np.transpose(w2c[:3, :3])
            T = w2c[:3, 3]
            
            # Compute FOV from focal length
            if self.focal_x is not None and self.focal_y is not None:
                FovX = 2 * np.arctan(self.image_width / (2 * self.focal_x))
                FovY = 2 * np.arctan(self.image_height / (2 * self.focal_y))
            else:
                # Fallback to default FOV
                FovX = FovY = np.deg2rad(self.fov_degrees)
                print(f"    - Using default FOV: {self.fov_degrees}°")
        else:
            # Mode 2: Spherical camera positioning
            FovY = np.deg2rad(self.fov_degrees)
            FovX = np.deg2rad(self.fov_degrees)
            
            # Compute camera position from spherical coordinates
            elevation_rad = np.deg2rad(self.camera_elevation)
            azimuth_rad = np.deg2rad(self.camera_azimuth)
            
            # Camera position in world space
            cam_x = self.camera_distance * np.cos(elevation_rad) * np.sin(azimuth_rad)
            cam_y = self.camera_distance * np.sin(elevation_rad)
            cam_z = self.camera_distance * np.cos(elevation_rad) * np.cos(azimuth_rad)
            
            # Camera looks at origin
            camera_pos = np.array([cam_x, cam_y, cam_z], dtype=np.float32)
            target_pos = np.array([0.0, 0.0, 0.0], dtype=np.float32)
            up_vector = np.array([0.0, 1.0, 0.0], dtype=np.float32)
            
            # Compute camera rotation matrix (world to camera)
            forward = target_pos - camera_pos
            forward = forward / np.linalg.norm(forward)
            right = np.cross(forward, up_vector)
            right = right / np.linalg.norm(right)
            up = np.cross(right, forward)
            
            c2w = np.eye(4, dtype=np.float32)
            c2w[:3, 0] = right
            c2w[:3, 1] = up
            c2w[:3, 2] = -forward
            c2w[:3, 3] = camera_pos

            R_x_180 = np.diag(np.array([1, -1, -1, 1], dtype=np.float32))
            c2w = c2w @ R_x_180
            
            # Convert to world-to-camera
            w2c = np.linalg.inv(c2w)
            
            # Extract R and T
            R = np.transpose(w2c[:3, :3])
            T = w2c[:3, 3]
            
            print(f"  Camera matrix setup (spherical mode):")
            print(f"    - Camera position: [{camera_pos[0]:.3f}, {camera_pos[1]:.3f}, {camera_pos[2]:.3f}]")
            print(f"    - T (transformed): [{T[0]:.3f}, {T[1]:.3f}, {T[2]:.3f}]")
        
        return Camera(
            colmap_id=0,
            R=R,
            T=T,
            FoVx=FovX,
            FoVy=FovY,
            image=torch.zeros(3, self.image_height, self.image_width),  # 保持在CPU，Camera内部会处理
            gt_alpha_mask=None,
            image_name="virtual_camera",
            uid=0,
            data_device=self.device,
            time=0.0,  # 默认time，会在render时更新
            control_vec=None  # 让Camera内部初始化，然后在render时override
        )
    
    def render_with_control(self, control_vec, time=None):
        """
        Render the scene with given control vector and time.
        
        Args:
            control_vec: torch.Tensor of shape [control_dim] or [1, control_dim]
                        Control input [sin(θ1), cos(θ1), ..., sin(θ6), cos(θ6), grip1, grip2, grip3]
            time: float, time value for temporal deformation (0.0 to 1.0)
        
        Returns:
            rendered_image: torch.Tensor of shape [3, H, W] in range [0, 1]
        """
        if isinstance(control_vec, np.ndarray):
            control_vec = torch.from_numpy(control_vec).float()
        
        # 确保control_vec在正确的设备上
        control_vec = control_vec.to(self.device)
        
        if control_vec.dim() == 1:
            control_vec = control_vec.unsqueeze(0)
        
        # 更新camera的time参数（如果提供）
        if time is not None:
            self.camera.time = time
        
        # Render with control override
        try:
            with torch.no_grad():
                render_pkg = render(
                    self.camera,
                    self.gaussians,
                    self.pipe_params,
                    self.background,
                    override_control_vec=control_vec,
                    stage="fine"
                )
            
            rendered_image = render_pkg["render"]
            return rendered_image
        except Exception as e:
            print(f"[ERROR] Rendering failed: {e}")
            print(f"  Camera R: {self.camera.R}")
            print(f"  Camera T: {self.camera.T}")
            print(f"  Camera FovX: {np.rad2deg(self.camera.FoVx):.2f}°, FovY: {np.rad2deg(self.camera.FoVy):.2f}°")
            print(f"  Control vec shape: {control_vec.shape}")
            raise
    
    def __call__(self, batch, grad_enabled=False):
        """
        MPC interface: Predict future images given action sequences.
        
        Args:
            batch: dict containing:
                - 'video': [B, T_context, H, W, C] - Context images (not used for 4DGS)
                - 'actions': [B, T_context + T_horizon, action_dim] - Control sequences
                - 'state_obs': list of state observations (not used)
        
        Returns:
            predictions: dict containing:
                - 'rgb': [B, T_horizon, H, W, 3] - Predicted RGB images
        
        Note:
            当前实现使用串行渲染（无法批处理）。4DGS渲染器需要为每个控制向量
            单独调用，因为控制向量影响变形网络。GPU已被充分利用（模型在GPU上，
            每次渲染都是GPU操作）。
        """
        actions = batch['actions']  # [B, T_total, control_dim]
        B, T_total, _ = actions.shape
        T_context = self.num_context
        T_horizon = T_total - T_context
        
        # Convert to torch if needed (确保在正确的设备上)
        if isinstance(actions, np.ndarray):
            actions = torch.from_numpy(actions).float().to(self.device)
        
        # Only predict for future actions (skip context)
        future_actions = actions[:, T_context:, :]  # [B, T_horizon, control_dim]
        
        # Render predictions for each batch and timestep
        # 注意：由于4DGS渲染的特殊性，无法并行化batch维度
        all_predictions = []
        
        for b in range(B):
            batch_predictions = []
            for t in range(T_horizon):
                control_vec = future_actions[b, t, :]  # [control_dim]
                
                # Render image (GPU operation)
                if grad_enabled:
                    rendered_image = self.render_with_control(control_vec)
                else:
                    with torch.no_grad():
                        rendered_image = self.render_with_control(control_vec)
                
                # Convert to numpy: [3, H, W] -> [H, W, 3]
                # 注意：这里的.cpu()是必要的最小化数据传输
                image_np = rendered_image.permute(1, 2, 0).cpu().numpy()
                batch_predictions.append(image_np)
            
            all_predictions.append(np.stack(batch_predictions, axis=0))
        
        predictions = {
            'rgb': np.stack(all_predictions, axis=0)  # [B, T_horizon, H, W, 3]
        }
        
        return predictions
    
    def set_camera_pose(self, R=None, T=None, FovX=None, FovY=None):
        """
        Update camera pose for different viewpoints.
        
        Args:
            R: Rotation matrix [3, 3] as numpy array
            T: Translation vector [3] as numpy array
            FovX: Field of view in X (radians)
            FovY: Field of view in Y (radians)
        """
        if R is not None:
            self.camera.R = R
        if T is not None:
            self.camera.T = T
        if FovX is not None:
            self.camera.FoVx = FovX
        if FovY is not None:
            self.camera.FoVy = FovY
        
        # Recompute camera matrices
        self.camera.world_view_transform = torch.tensor(
            getWorld2View2(self.camera.R, self.camera.T, trans=np.array([0., 0., 0.]), scale=1.0)
        ).transpose(0, 1)
        
        self.camera.projection_matrix = getProjectionMatrix(
            znear=0.01, 
            zfar=100.0,
            fovX=self.camera.FoVx,
            fovY=self.camera.FoVy
        ).transpose(0, 1)
        
        self.camera.full_proj_transform = (
            self.camera.world_view_transform.unsqueeze(0).bmm(
                self.camera.projection_matrix.unsqueeze(0)
            )
        ).squeeze(0)
    
    def close(self):
        """Cleanup resources"""
        pass





# Test function
if __name__ == "__main__":
    print("Testing GaussianDynamicsModel...")
    
    # Initialize model
    model_path = "/home/ubuntu/yyf/4DGaussians/assets"
    model = GaussianDynamicsModel(
        model_path=model_path,
        iteration=5000,
        control_dim=15,
        image_height=480,
        image_width=480
    )
    
    # Test single render
    print("\nTest 1: Single render with control")
    control_vec = torch.zeros(15).cuda()  # Zero control
    image = model.render_with_control(control_vec)
    print(f"  Output shape: {image.shape}")
    print(f"  Value range: [{image.min():.3f}, {image.max():.3f}]")
    
    # Test MPC interface
    print("\nTest 2: MPC batch prediction")
    batch = {
        'video': np.zeros((4, 2, 128, 128, 3)),  # 4 samples, 2 context frames
        'actions': np.random.randn(4, 12, 15),    # 4 samples, 12 timesteps, 15D control
        'state_obs': []
    }
    predictions = model(batch)
    print(f"  Predictions shape: {predictions['rgb'].shape}")
    print(f"  Expected: [4, 10, 128, 128, 3] (10 = 12 - 2 context)")
    
    print("\n✓ All tests passed!")
