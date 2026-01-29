#
# Configuration for TriPlane-based Controllable 4D Gaussian Splatting
#
# This configuration uses the new TriPlane architecture with Multi-head FiLM fusion.
# Key features:
# 1. use_triplane = True to enable TriPlane architecture
# 2. Multi-head FiLM for control-spatial feature fusion
# 3. Residual connections for stable training
#

ModelHiddenParams = dict(
    # ========== Architecture Selection ==========
    use_triplane = True,  # Use TriPlane instead of HexPlane
    
    # ========== TriPlane Configuration ==========
    # Note: Only first 3 values (spatial) are used for TriPlane
    # The 4th value (time) is ignored when use_triplane=True
    kplanes_config = {
        'grid_dimensions': 2,
        'input_coordinate_dim': 3,  # Changed from 4 to 3 (no time dim)
        'output_coordinate_dim': 32,
        'resolution': [128, 128, 128]  # Only spatial resolution
    },
    multires = [1, 2, 4],  # Multi-scale factors
    
    # ========== Deformation MLP Configuration ==========
    defor_depth = 3,      # Number of FiLM layers
    net_width = 128,      # Hidden dimension
    grid_pe = 0,          # Positional encoding on grid features (0 = disabled)
    
    # ========== Control Signal Configuration ==========
    control_input_dim = 15,         # Input control dimension (e.g., 6-DOF)
    control_use_pe = False,         # Use positional encoding for control
    control_num_frequencies = 4,   # PE frequency bands
    control_hidden_dim = 128,       # Hidden dim for control MLP (if used)
    control_output_dim = 32,     # None = use PE output directly (recommended)
                                   # Set to a value (e.g., 32) to project control features
    
    # ========== FiLM Fusion Configuration (NEW) ==========
    film_hidden_dim = 128,          # Hidden dim for FiLM γ/β generation
    film_use_residual = True,      # Use residual connection (recommended)
    
    # ========== Regularization ==========
    plane_tv_weight = 0.01,       # Total variation on spatial planes
    time_smoothness_weight = 0.0,  # Not used in TriPlane
    l1_time_planes = 0.0,          # Not used in TriPlane
    
    # ========== Deformation Flags ==========
    no_dx = False,    # Position deformation
    no_ds = False,    # Scale deformation (disabled by default)
    no_dr = False,    # Rotation deformation
    no_do = True,     # Opacity deformation (disabled by default)
    no_dshs = True,   # SH deformation (disabled by default)
    
    # ========== Other Options ==========
    no_grid = False,       # Disable grid entirely (fallback to pure MLP)
    empty_voxel = False,   # Empty voxel masking
    static_mlp = False,    # Static scene MLP
    render_process = False,
    apply_rotation = False,  # Use quaternion multiplication for rotation
    
    # ========== Legacy Parameters (kept for compatibility) ==========
    posebase_pe = 10,
    scale_rotation_pe = 2,
    opacity_pe = 2,
    timebase_pe = 4,       # Not used in TriPlane
    timenet_width = 64,    # Not used in TriPlane
    timenet_output = 32,   # Not used in TriPlane
    bounds = 1.6,
)

OptimizationParams = dict(
    dataloader = True,
    iterations = 20000,
    zerostamp_init = False,
    batch_size = 4,
    coarse_iterations = 3000,
    densify_until_iter = 5000,
    densification_interval = 100,
    opacity_reset_interval = 2000,

    lambda_dssim = 0.2,
    lambda_lpips = 0.0,
    use_depth_loss = True,
    lambda_depth = 0.2,
    depth_scale = 1000.0,

    use_gmflow = True,
    flow_loss_weight = 1.0,
)
