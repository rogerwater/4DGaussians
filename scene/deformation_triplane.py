import math
import torch
import torch.nn as nn
import torch.nn.init as init
from typing import Optional, List, Tuple

from scene.triplane import TriPlaneField, ActionProcessor
from scene.grid import DenseGrid
from utils.graphics_utils import batch_quaternion_multiply


# ============================================================================
# FiLM Layers
# ============================================================================

class FiLMLayer(nn.Module):
    def __init__(self, feature_dim: int, condition_dim: int, hidden_dim: int = 64):
        super(FiLMLayer, self).__init__()
        
        self.feature_dim = feature_dim
        
        # Generate γ (scale) and β (shift) from conditioning signal
        self.film_generator = nn.Sequential(
            nn.Linear(condition_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, feature_dim * 2)  # γ and β
        )
        
        self._init_weights()
    
    def _init_weights(self):
        nn.init.xavier_uniform_(self.film_generator[0].weight, gain=1.0)
        nn.init.zeros_(self.film_generator[0].bias)
        
        nn.init.normal_(self.film_generator[-1].weight, std=0.01)
        nn.init.normal_(self.film_generator[-1].bias, std=0.01)
        
        self.film_generator[-1].bias.data[:self.feature_dim] += 1.0
    
    def forward(self, features: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        # Generate γ and β
        film_params = self.film_generator(condition)
        gamma = film_params[:, :self.feature_dim]
        beta = film_params[:, self.feature_dim:]
        
        # Apply affine transformation
        modulated = gamma * features + beta
        
        return modulated


class FiLMBlock(nn.Module):
    def __init__(
        self, 
        in_dim: int, 
        out_dim: int, 
        condition_dim: int,
        hidden_dim: int = 64,
        activation: str = 'relu',
        use_layer_norm: bool = True
    ):
        super(FiLMBlock, self).__init__()
        
        self.linear = nn.Linear(in_dim, out_dim)
        
        if use_layer_norm:
            self.norm = nn.LayerNorm(out_dim)
        else:
            self.norm = nn.Identity()
        
        self.film = FiLMLayer(out_dim, condition_dim, hidden_dim)
        
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'leaky_relu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'silu':
            self.activation = nn.SiLU(inplace=True)
        else:
            self.activation = nn.ReLU(inplace=True)
            
        if in_dim == out_dim:
            self.skip = nn.Identity()
        else:
            self.skip = nn.Linear(in_dim, out_dim)
    
    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        identity = self.skip(x)
        
        h = self.linear(x)
        h = self.norm(h)
        h = self.film(h, condition)
        h = self.activation(h)
        
        return h + identity


class MultiHeadFiLMDecoder(nn.Module):
    def __init__(
        self,
        spatial_dim: int,
        control_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 3,
        film_hidden: int = 64,
        use_layer_norm: bool = True
    ):
        super(MultiHeadFiLMDecoder, self).__init__()
        
        self.spatial_dim = spatial_dim
        self.control_dim = control_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Build FiLM blocks
        self.film_blocks = nn.ModuleList()
        
        # First block: spatial_dim -> hidden_dim
        self.film_blocks.append(
            FiLMBlock(spatial_dim, hidden_dim, control_dim, film_hidden, use_layer_norm=use_layer_norm)
        )
        
        # Middle blocks: hidden_dim -> hidden_dim
        for _ in range(num_layers - 1):
            self.film_blocks.append(
                FiLMBlock(hidden_dim, hidden_dim, control_dim, film_hidden, use_layer_norm=use_layer_norm)
            )
        
        print(f"[MultiHeadFiLMDecoder] Created with {num_layers} FiLM blocks (each with internal residual)")
        print(f"[MultiHeadFiLMDecoder]   Spatial: {spatial_dim} → Control: {control_dim} → Hidden: {hidden_dim}")
        print(f"[MultiHeadFiLMDecoder]   FiLM hidden dim: {film_hidden}")
    
    def forward(
        self, 
        spatial_feat: torch.Tensor, 
        control_feat: torch.Tensor
    ) -> torch.Tensor:
        
        # Pass through FiLM blocks
        h = spatial_feat
        for film_block in self.film_blocks:
            h = film_block(h, control_feat)
        
        return h


# ============================================================================
# Deformation Network with FiLM Fusion
# ============================================================================

class DeformationTriPlane(nn.Module):
    def __init__(self, D=2, W=128, grid_pe=0, args=None):
        super(DeformationTriPlane, self).__init__()
        
        self.D = D
        self.W = W
        self.grid_pe = grid_pe
        self.args = args
        self.no_grid = getattr(args, 'no_grid', False)
        
        # TriPlane configuration
        # 处理kplanes_config可能是dict或对象的情况
        kplanes_cfg = getattr(args, 'kplanes_config', {})
        if isinstance(kplanes_cfg, dict):
            resolution = kplanes_cfg.get('resolution', [64, 64, 64])[:3]
            output_dim = kplanes_cfg.get('output_coordinate_dim', 32)
        else:
            # 如果是对象，使用getattr
            resolution = getattr(kplanes_cfg, 'resolution', [64, 64, 64])[:3]
            output_dim = getattr(kplanes_cfg, 'output_coordinate_dim', 32)
        
        triplane_config = {
            'resolution': resolution,
            'output_coordinate_dim': output_dim
        }
        
        # 1. TriPlane for spatial feature encoding
        self.triplane = TriPlaneField(
            bounds=getattr(args, 'bounds', 1.6),
            planeconfig=triplane_config,
            multires=getattr(args, 'multires', [1, 2, 4])
        )
        
        # 2. Control signal processor
        # 默认值与ActionProcessor保持一致
        action_use_pe = getattr(args, 'action_use_pe', True)
        action_num_freq = getattr(args, 'action_num_frequencies', 4)
        action_input_dim = getattr(args, 'action_input_dim', 6)
        action_hidden = getattr(args, 'action_hidden_dim', 128)  # 与ActionProcessor默认值一致
        # TriPlane+FiLM必须指定output_dim，提供合理默认值
        action_output_dim = getattr(args, 'action_output_dim', 64)
        
        self.action_processor = ActionProcessor(
            input_dim=action_input_dim,
            use_pe=action_use_pe,
            num_frequencies=action_num_freq,
            hidden_dim=action_hidden,  # 总是使用hidden_dim
            output_dim=action_output_dim  # 保证非None
        )
        
        # 3. Compute dimensions
        self.spatial_dim = self.triplane.feat_dim
        if grid_pe > 0:
            self.spatial_dim = self.spatial_dim * (1 + 2 * grid_pe)
        self.action_dim = self.action_processor.output_dim
        
        # 4. FiLM fusion configuration
        film_hidden = getattr(args, 'film_hidden_dim', 64)
        
        # 5. Multi-head FiLM decoder
        self.film_decoder = MultiHeadFiLMDecoder(
            spatial_dim=self.spatial_dim,
            control_dim=self.action_dim,
            hidden_dim=W,
            num_layers=D,
            film_hidden=film_hidden
        )
        
        # 6. Optional modules
        if getattr(args, 'empty_voxel', False):
            self.empty_voxel = DenseGrid(channels=1, world_size=[64, 64, 64])
        else:
            self.empty_voxel = None
        
        if getattr(args, 'static_mlp', False):
            self.static_mlp = nn.Sequential(
                nn.ReLU(),
                nn.Linear(W, W),
                nn.ReLU(),
                nn.Linear(W, 1)
            )
        else:
            self.static_mlp = None
        
        self.ratio = 0
        
        # 7. Deformation prediction heads
        self._create_deform_heads()
        
        print(f"[DeformationTriPlane] Using Multi-head FiLM fusion")
        print(f"[DeformationTriPlane]   - Spatial dim: {self.spatial_dim}")
        print(f"[DeformationTriPlane]   - Action dim: {self.action_dim}")
        print(f"[DeformationTriPlane]   - FiLM layers: {D}")
        print(f"[DeformationTriPlane]   - Hidden dim: {W}")
    
    def _create_deform_heads(self):
        """Create deformation prediction heads."""
        
        self.pos_deform = nn.Sequential(
            nn.ReLU(),
            nn.Linear(self.W, self.W),
            nn.ReLU(),
            nn.Linear(self.W, 3)
        )
        
        self.scales_deform = nn.Sequential(
            nn.ReLU(),
            nn.Linear(self.W, self.W),
            nn.ReLU(),
            nn.Linear(self.W, 3)
        )
        
        self.rotations_deform = nn.Sequential(
            nn.ReLU(),
            nn.Linear(self.W, self.W),
            nn.ReLU(),
            nn.Linear(self.W, 4)
        )
        
        self.opacity_deform = nn.Sequential(
            nn.ReLU(),
            nn.Linear(self.W, self.W),
            nn.ReLU(),
            nn.Linear(self.W, 1)
        )
        
        self.shs_deform = nn.Sequential(
            nn.ReLU(),
            nn.Linear(self.W, self.W),
            nn.ReLU(),
            nn.Linear(self.W, 16 * 3)
        )
    
    @property
    def get_aabb(self):
        return self.triplane.get_aabb
    
    def set_aabb(self, xyz_max, xyz_min):
        print(f"[DeformationTriPlane] Setting AABB: max={xyz_max}, min={xyz_min}")
        self.triplane.set_aabb(xyz_max, xyz_min)
        if self.empty_voxel is not None:
            self.empty_voxel.set_aabb(xyz_max, xyz_min)
    
    @property
    def get_empty_ratio(self):
        return self.ratio
    
    def _apply_grid_pe(self, features: torch.Tensor) -> torch.Tensor:
        """Apply positional encoding to grid features."""
        if self.grid_pe <= 0:
            return features
        
        freq_bands = 2.0 ** torch.arange(
            self.grid_pe, device=features.device, dtype=features.dtype
        )
        encoded = [features]
        for freq in freq_bands:
            encoded.append(torch.sin(features * freq))
            encoded.append(torch.cos(features * freq))
        return torch.cat(encoded, dim=-1)
    
    def query_features(
        self, 
        pts: torch.Tensor, 
        action_vec: torch.Tensor
    ) -> torch.Tensor:
        """
        Query fused features using Multi-head FiLM.
        
        Args:
            pts: [N, 3] - 3D positions
            action_vec: [N, action_dim] - Action vectors
            
        Returns:
            hidden: [N, W] - Fused hidden features
        """
        if self.no_grid:
            # Fallback mode without grid
            action_feat = self.action_processor(action_vec)
            # Simple MLP fallback
            combined = torch.cat([pts, action_feat], dim=-1)
            hidden = nn.functional.relu(
                nn.functional.linear(combined, torch.randn(self.W, combined.shape[-1], device=pts.device))
            )
            return hidden
        
        # 1. Get spatial features from TriPlane
        spatial_feat = self.triplane(pts)
        
        # Optional: apply PE to spatial features
        if self.grid_pe > 0:
            spatial_feat = self._apply_grid_pe(spatial_feat)
        
        # 2. Process action vector
        action_feat = self.action_processor(action_vec)
        
        # 3. Multi-head FiLM fusion
        hidden = self.film_decoder(spatial_feat, action_feat)
        
        return hidden
    
    def forward(
        self, 
        rays_pts_emb: torch.Tensor, 
        scales_emb: Optional[torch.Tensor] = None, 
        rotations_emb: Optional[torch.Tensor] = None, 
        opacity: Optional[torch.Tensor] = None, 
        shs_emb: Optional[torch.Tensor] = None, 
        time_feature: Optional[torch.Tensor] = None, 
        action_vec: Optional[torch.Tensor] = None
    ):
        """
        Forward pass.
        
        Args:
            rays_pts_emb: [N, 3+PE] - Position with PE
            scales_emb, rotations_emb: Attribute embeddings
            opacity, shs_emb: Gaussian attributes
            time_feature: [UNUSED] Legacy parameter
            action_vec: [N, action_dim] - Action vectors
            
        Returns:
            Tuple of deformed attributes
        """
        if action_vec is None:
            return self.forward_static(rays_pts_emb[:, :3])
        else:
            return self.forward_dynamic(
                rays_pts_emb, scales_emb, rotations_emb,
                opacity, shs_emb, action_vec
            )
    
    def forward_static(self, pts: torch.Tensor):
        """Static forward (no control-based deformation)."""
        if self.static_mlp is not None:
            spatial_feat = self.triplane(pts)
            dx = self.static_mlp(spatial_feat)
            return pts + dx
        return pts
    
    def forward_dynamic(
        self, 
        rays_pts_emb: torch.Tensor, 
        scales_emb: torch.Tensor, 
        rotations_emb: torch.Tensor, 
        opacity_emb: torch.Tensor, 
        shs_emb: torch.Tensor, 
        action_vec: torch.Tensor
    ):
        """
        Dynamic forward with Multi-head FiLM control modulation.
        
        Args:
            rays_pts_emb: [N, 3+PE] - Position with positional encoding
            scales_emb: [N, 3+PE] - Scales with PE
            rotations_emb: [N, 4+PE] - Rotations with PE
            opacity_emb: [N, 1] - Opacity
            shs_emb: [N, 16, 3] - SH coefficients
            action_vec: [N, action_dim] - Full action vector
            
        Returns:
            Tuple of (pts, scales, rotations, opacity, shs)
        """
        # Extract raw position
        pts = rays_pts_emb[:, :3]
        
        # Query fused features with FiLM modulation
        hidden = self.query_features(pts, action_vec)
        
        # Compute deformation mask
        if self.static_mlp is not None:
            mask = self.static_mlp(hidden)
        elif self.empty_voxel is not None:
            mask = self.empty_voxel(pts)
        else:
            mask = torch.ones_like(opacity_emb[:, 0]).unsqueeze(-1)
        
        # Predict deformations
        # Position
        if getattr(self.args, 'no_dx', False):
            pts_out = pts
        else:
            dx = self.pos_deform(hidden)
            pts_out = pts * mask + dx
        
        # Scale
        if getattr(self.args, 'no_ds', False):
            scales_out = scales_emb[:, :3]
        else:
            ds = self.scales_deform(hidden)
            scales_out = scales_emb[:, :3] * mask + ds
        
        # Rotation
        if getattr(self.args, 'no_dr', False):
            rotations_out = rotations_emb[:, :4]
        else:
            dr = self.rotations_deform(hidden)
            if getattr(self.args, 'apply_rotation', False):
                rotations_out = batch_quaternion_multiply(rotations_emb[:, :4], dr)
            else:
                rotations_out = rotations_emb[:, :4] + dr
        
        # Opacity
        if getattr(self.args, 'no_do', True):
            opacity_out = opacity_emb[:, :1]
        else:
            do = self.opacity_deform(hidden)
            opacity_out = opacity_emb[:, :1] * mask + do
        
        # SH coefficients
        if getattr(self.args, 'no_dshs', True):
            shs_out = shs_emb
        else:
            dshs = self.shs_deform(hidden).reshape([shs_emb.shape[0], 16, 3])
            shs_out = shs_emb * mask.unsqueeze(-1) + dshs
        
        return pts_out, scales_out, rotations_out, opacity_out, shs_out
    
    def get_mlp_parameters(self) -> List[torch.nn.Parameter]:
        """Get MLP parameters (excluding grid)."""
        params = []
        for name, param in self.named_parameters():
            if 'triplane' not in name:
                params.append(param)
        return params
    
    def get_grid_parameters(self) -> List[torch.nn.Parameter]:
        """Get grid (TriPlane) parameters."""
        params = []
        for name, param in self.named_parameters():
            if 'triplane' in name:
                params.append(param)
        return params


# ============================================================================
# Top-level Network
# ============================================================================

class deform_network_triplane(nn.Module):
    def __init__(self, args):
        super(deform_network_triplane, self).__init__()
        
        net_width = args.net_width
        defor_depth = args.defor_depth
        posbase_pe = args.posebase_pe
        scale_rotation_pe = args.scale_rotation_pe
        opacity_pe = args.opacity_pe
        grid_pe = args.grid_pe
        
        # Create TriPlane-based deformation network with FiLM fusion
        self.deformation_net = DeformationTriPlane(
            W=net_width,
            D=defor_depth,
            grid_pe=grid_pe,
            args=args
        )
        
        # Positional encoding buffers
        self.register_buffer('pos_poc', torch.FloatTensor([(2**i) for i in range(posbase_pe)]))
        self.register_buffer('rotation_scaling_poc', torch.FloatTensor([(2**i) for i in range(scale_rotation_pe)]))
        self.register_buffer('opacity_poc', torch.FloatTensor([(2**i) for i in range(opacity_pe)]))
        
        # Initialize weights
        self.apply(initialize_weights)
        
        print(f"[deform_network_triplane] Initialized with TriPlane + Multi-head FiLM Fusion")
    
    def forward(
        self, 
        point: torch.Tensor, 
        scales: Optional[torch.Tensor] = None, 
        rotations: Optional[torch.Tensor] = None, 
        opacity: Optional[torch.Tensor] = None, 
        shs: Optional[torch.Tensor] = None, 
        action_vec: Optional[torch.Tensor] = None
    ):
        """
        Forward pass.
        
        Args:
            point: [N, 3] - Positions
            scales, rotations, opacity, shs: Gaussian attributes
            action_vec: [N, action_dim] - Full action vector
            
        Returns:
            Deformed attributes
        """
        if action_vec is None:
            return self.forward_static(point)
        else:
            return self.forward_dynamic(point, scales, rotations, opacity, shs, action_vec)
    
    @property
    def get_aabb(self):
        return self.deformation_net.get_aabb
    
    @property
    def get_empty_ratio(self):
        return self.deformation_net.get_empty_ratio
    
    def forward_static(self, points: torch.Tensor):
        return self.deformation_net(points)
    
    def forward_dynamic(
        self, 
        point: torch.Tensor, 
        scales: torch.Tensor, 
        rotations: torch.Tensor, 
        opacity: torch.Tensor, 
        shs: torch.Tensor, 
        action_vec: torch.Tensor
    ):
        """
        Dynamic deformation with Multi-head FiLM fusion.
        
        The action vector modulates spatial features through FiLM layers,
        allowing for expressive action-dependent deformations.
        """
        # Apply positional encoding
        point_emb = poc_fre(point, self.pos_poc)
        scales_emb = poc_fre(scales, self.rotation_scaling_poc)
        rotations_emb = poc_fre(rotations, self.rotation_scaling_poc)
        
        # Forward with FiLM fusion
        means3D, scales, rotations, opacity, shs = self.deformation_net(
            point_emb,
            scales_emb,
            rotations_emb,
            opacity,
            shs,
            None,  # time_feature (unused)
            action_vec
        )
        
        return means3D, scales, rotations, opacity, shs
    
    def get_mlp_parameters(self):
        return self.deformation_net.get_mlp_parameters()
    
    def get_grid_parameters(self):
        return self.deformation_net.get_grid_parameters()


# ============================================================================
# Utilities
# ============================================================================

def initialize_weights(m):
    """Xavier initialization for linear layers."""
    if isinstance(m, nn.Linear):
        init.xavier_uniform_(m.weight, gain=1)
        if m.bias is not None:
            init.zeros_(m.bias)


def poc_fre(input_data: torch.Tensor, poc_buf: torch.Tensor) -> torch.Tensor:
    """Positional encoding using frequency buffer."""
    input_data_emb = (input_data.unsqueeze(-1) * poc_buf).flatten(-2)
    input_data_sin = input_data_emb.sin()
    input_data_cos = input_data_emb.cos()
    input_data_emb = torch.cat([input_data, input_data_sin, input_data_cos], -1)
    return input_data_emb
