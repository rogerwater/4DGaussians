#
# TriPlane Field for Controllable 4D Gaussian Splatting
# 
# This module implements a TriPlane (3 planes: XY, XZ, YZ) for spatial feature encoding.
# Unlike HexPlane which includes time-space planes, TriPlane only encodes spatial geometry.
# Action signals are injected directly into the decoder for better expressiveness.
#

import itertools
from typing import Optional, Sequence, Iterable, Collection

import torch
import torch.nn as nn
import torch.nn.functional as F


def normalize_aabb(pts, aabb):
    """Normalize points to [-1, 1] based on AABB bounds."""
    return (pts - aabb[0]) * (2.0 / (aabb[1] - aabb[0])) - 1.0


def grid_sample_wrapper(grid: torch.Tensor, coords: torch.Tensor, align_corners: bool = True) -> torch.Tensor:
    """
    Wrapper for grid sampling with proper dimension handling.
    
    Args:
        grid: Feature grid [B, C, H, W] for 2D or [B, C, D, H, W] for 3D
        coords: Query coordinates [N, 2] or [N, 3]
        align_corners: Whether to align corners in grid_sample
        
    Returns:
        Interpolated features [N, C]
    """
    grid_dim = coords.shape[-1]

    if grid.dim() == grid_dim + 1:
        grid = grid.unsqueeze(0)
    if coords.dim() == 2:
        coords = coords.unsqueeze(0)

    if grid_dim == 2 or grid_dim == 3:
        grid_sampler = F.grid_sample
    else:
        raise NotImplementedError(f"Grid-sample was called with {grid_dim}D data but is only "
                                  f"implemented for 2 and 3D data.")

    coords = coords.view([coords.shape[0]] + [1] * (grid_dim - 1) + list(coords.shape[1:]))
    B, feature_dim = grid.shape[:2]
    n = coords.shape[-2]
    interp = grid_sampler(
        grid,
        coords,
        align_corners=align_corners,
        mode='bilinear', 
        padding_mode='border'
    )
    interp = interp.view(B, feature_dim, n).transpose(-1, -2)
    interp = interp.squeeze()
    return interp


def init_triplane_param(
        out_dim: int,
        reso: Sequence[int],
        a: float = 0.1,
        b: float = 0.5):
    """
    Initialize TriPlane parameters (3 planes: XY, XZ, YZ).
    
    Args:
        out_dim: Output feature dimension per plane
        reso: Resolution [res_x, res_y, res_z]
        a, b: Initialization range for uniform distribution
        
    Returns:
        nn.ParameterList containing 3 2D plane parameters
    """
    assert len(reso) == 3, "Resolution must have 3 elements (x, y, z)"
    
    # TriPlane: XY, XZ, YZ
    plane_configs = [
        (0, 1, [reso[1], reso[0]]),  # XY plane: shape [out_dim, res_y, res_x]
        (0, 2, [reso[2], reso[0]]),  # XZ plane: shape [out_dim, res_z, res_x]
        (1, 2, [reso[2], reso[1]]),  # YZ plane: shape [out_dim, res_z, res_y]
    ]
    
    grid_coefs = nn.ParameterList()
    for (dim_a, dim_b, plane_reso) in plane_configs:
        new_grid_coef = nn.Parameter(torch.empty([1, out_dim] + plane_reso))
        nn.init.uniform_(new_grid_coef, a=a, b=b)
        grid_coefs.append(new_grid_coef)
    
    return grid_coefs


def interpolate_triplane_features(
        pts: torch.Tensor,
        ms_grids: Collection[Iterable[nn.Module]],
        concat_features: bool = True,
) -> torch.Tensor:
    """
    Interpolate multi-scale TriPlane features.
    
    Args:
        pts: Query points [N, 3] in normalized coordinates [-1, 1]
        ms_grids: Multi-scale grid list, each containing 3 planes
        concat_features: Whether to concatenate multi-scale features
        
    Returns:
        Interpolated features [N, C*num_scales] if concat, else [N, C]
    """
    # TriPlane coordinate combinations
    coo_combs = [(0, 1), (0, 2), (1, 2)]  # XY, XZ, YZ
    
    multi_scale_interp = [] if concat_features else 0.
    
    for scale_id, grid in enumerate(ms_grids):
        interp_space = 1.0
        
        for ci, coo_comb in enumerate(coo_combs):
            feature_dim = grid[ci].shape[1]
            interp_out_plane = (
                grid_sample_wrapper(grid[ci], pts[..., coo_comb])
                .view(-1, feature_dim)
            )
            # Hadamard product over planes
            interp_space = interp_space * interp_out_plane

        if concat_features:
            multi_scale_interp.append(interp_space)
        else:
            multi_scale_interp = multi_scale_interp + interp_space

    if concat_features:
        multi_scale_interp = torch.cat(multi_scale_interp, dim=-1)
    
    return multi_scale_interp


class TriPlaneField(nn.Module):
    """
    TriPlane Field for spatial feature encoding.
    
    Unlike HexPlane which includes 6 planes (XY, XZ, YZ, XT, YT, ZT),
    TriPlane only uses 3 spatial planes (XY, XZ, YZ) for geometry encoding.
    Action signals are handled separately by the decoder.
    
    Architecture:
        Input: 3D position (x, y, z)
        Output: Spatial feature vector
        
    Args:
        bounds: Scene bounds for AABB normalization
        planeconfig: Configuration dict with 'resolution' and 'output_coordinate_dim'
        multires: List of multi-resolution multipliers, e.g., [1, 2, 4]
    """
    
    def __init__(
        self,
        bounds: float,
        planeconfig: dict,
        multires: list
    ) -> None:
        super().__init__()
        
        # AABB for coordinate normalization
        aabb = torch.tensor([
            [bounds, bounds, bounds],
            [-bounds, -bounds, -bounds]
        ])
        self.aabb = nn.Parameter(aabb, requires_grad=False)
        
        self.grid_config = planeconfig
        self.multiscale_res_multipliers = multires
        self.concat_features = True
        
        # Initialize multi-scale TriPlanes
        self.grids = nn.ModuleList()
        self.feat_dim = 0
        
        base_resolution = planeconfig.get('resolution', [64, 64, 64])
        out_dim = planeconfig.get('output_coordinate_dim', 32)
        
        for res_mult in self.multiscale_res_multipliers:
            # Scale resolution by multiplier
            scaled_reso = [r * res_mult for r in base_resolution[:3]]
            
            gp = init_triplane_param(
                out_dim=out_dim,
                reso=scaled_reso,
            )
            
            if self.concat_features:
                self.feat_dim += out_dim
            else:
                self.feat_dim = out_dim
                
            self.grids.append(gp)
        
        print(f"[TriPlaneField] Initialized with {len(self.grids)} scales")
        print(f"[TriPlaneField] Feature dimension: {self.feat_dim}")
        print(f"[TriPlaneField] Resolutions: {[r * m for m in multires for r in base_resolution[:3]]}")
    
    @property
    def get_aabb(self):
        return self.aabb[0], self.aabb[1]
    
    def set_aabb(self, xyz_max, xyz_min):
        aabb = torch.tensor([xyz_max, xyz_min], dtype=torch.float32)
        self.aabb = nn.Parameter(aabb, requires_grad=False)
        print(f"[TriPlaneField] Set AABB = {self.aabb}")
    
    def get_spatial_features(self, pts: torch.Tensor) -> torch.Tensor:
        """
        Query spatial features from TriPlane.
        
        Args:
            pts: 3D positions [N, 3]
            
        Returns:
            Spatial features [N, feat_dim]
        """
        # Normalize to [-1, 1]
        pts_normalized = normalize_aabb(pts, self.aabb)
        pts_normalized = pts_normalized.reshape(-1, 3)
        
        # Interpolate from all scales
        features = interpolate_triplane_features(
            pts_normalized,
            ms_grids=self.grids,
            concat_features=self.concat_features
        )
        
        if len(features) < 1:
            features = torch.zeros((0, self.feat_dim), device=pts.device)
        
        return features
    
    def forward(self, pts: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: extract spatial features.
        
        Args:
            pts: 3D positions [N, 3]
            
        Returns:
            Spatial features [N, feat_dim]
        """
        return self.get_spatial_features(pts)


class ActionProcessor(nn.Module):
    """
    Process action signals with optional positional encoding.
    
    This module prepares action vectors for injection into the decoder.
    Unlike ActionEncoder which compresses to 1D, this preserves dimensionality.
    
    Args:
        input_dim: Action vector dimension (e.g., 6 for 6-DOF)
        use_pe: Whether to use positional encoding
        num_frequencies: Number of frequency bands for PE
        hidden_dim: Optional MLP hidden dimension for feature extraction
        output_dim: Output dimension (None = auto based on PE)
    """
    
    def __init__(
        self,
        input_dim: int = 6,
        use_pe: bool = True,
        num_frequencies: int = 4,
        hidden_dim: int = 128,
        output_dim: int = 64,
        use_layer_norm: bool = True
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.use_pe = use_pe
        self.num_frequencies = num_frequencies
        
        # Compute PE output dimension
        if use_pe:
            # PE: [x, sin(2^0*x), cos(2^0*x), ..., sin(2^(F-1)*x), cos(2^(F-1)*x)]
            pe_dim = input_dim * (1 + 2 * num_frequencies)
        else:
            pe_dim = input_dim
        
        self.pe_dim = pe_dim
        
        assert output_dim is not None, "output_dim must be specified for TriPlane+FiLM"

        self.mlp = nn.Sequential(
            nn.Linear(pe_dim, hidden_dim),
            nn.LayerNorm(hidden_dim) if use_layer_norm else nn.Identity(),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim) if use_layer_norm else nn.Identity(),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim)
        )
        self.output_dim = output_dim
        
        # Register frequency bands as buffer
        if use_pe:
            freq_bands = 2.0 ** torch.linspace(0.0, num_frequencies - 1, num_frequencies)
            self.register_buffer('freq_bands', freq_bands)
            
        self._init_weights()
        
        print(f"[ActionProcessor] Input: {input_dim} → PE: {pe_dim} → MLP → Output: {output_dim}")
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=1.0)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def positional_encoding(self, x: torch.Tensor) -> torch.Tensor:
        """Apply positional encoding to input."""
        if not self.use_pe:
            return x
        
        encoded = [x]
        for freq in self.freq_bands:
            encoded.append(torch.sin(x * freq))
            encoded.append(torch.cos(x * freq))
        
        return torch.cat(encoded, dim=-1)
    
    def forward(self, action_vec: torch.Tensor) -> torch.Tensor:
        """
        Process action vector.
        
        Args:
            action_vec: [N, input_dim]
            
        Returns:
            Processed features [N, output_dim]
        """
        # Apply positional encoding
        x = self.positional_encoding(action_vec)
        x = self.mlp(x)
        
        return x
