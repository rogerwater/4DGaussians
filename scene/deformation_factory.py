#
# Deformation Network Factory
#
# This module provides a unified interface to create deformation networks.
# It automatically selects between HexPlane-based and TriPlane-based architectures
# based on configuration.
#

import torch.nn as nn


def create_deform_network(args) -> nn.Module:
    """
    Factory function to create deformation network.
    
    Automatically selects between:
    - HexPlane-based: Original 4DGaussians with 6 space-time planes
    - TriPlane-based: New architecture with 3 spatial planes + direct control injection
    
    Args:
        args: Configuration arguments
              - use_triplane (bool): If True, use TriPlane architecture
              
    Returns:
        Deformation network module
    """
    use_triplane = getattr(args, 'use_triplane', False)
    
    if use_triplane:
        # New TriPlane-based architecture
        from scene.deformation_triplane import deform_network_triplane
        print("[Factory] Creating TriPlane-based deformation network")
        print("[Factory]   - Spatial encoding: 3 planes (XY, XZ, YZ)")
        print("[Factory]   - Control injection: Direct to decoder")
        return deform_network_triplane(args)
    else:
        # Original HexPlane-based architecture
        from scene.deformation import deform_network
        print("[Factory] Creating HexPlane-based deformation network")
        print("[Factory]   - Spatial-temporal encoding: 6 planes")
        print("[Factory]   - Control injection: Compressed to 1D")
        return deform_network(args)


def get_network_info(args) -> dict:
    """
    Get information about the network architecture based on config.
    
    Returns:
        dict with network architecture details
    """
    use_triplane = getattr(args, 'use_triplane', False)
    
    if use_triplane:
        action_dim = getattr(args, 'action_input_dim', 6)
        use_pe = getattr(args, 'action_use_pe', True)
        num_freq = getattr(args, 'action_num_frequencies', 4)
        action_output = getattr(args, 'action_output_dim', None)
        
        if use_pe:
            pe_dim = action_dim * (1 + 2 * num_freq)
        else:
            pe_dim = action_dim
        
        if action_output is not None:
            action_feat_dim = action_output
        else:
            action_feat_dim = pe_dim
        
        resolution = args.kplanes_config.get('resolution', [64, 64, 64])[:3]
        out_dim = args.kplanes_config.get('output_coordinate_dim', 32)
        num_scales = len(args.multires)
        spatial_feat_dim = out_dim * num_scales
        
        return {
            'architecture': 'TriPlane',
            'num_planes': 3,
            'plane_names': ['XY', 'XZ', 'YZ'],
            'resolution': resolution,
            'num_scales': num_scales,
            'spatial_feat_dim': spatial_feat_dim,
            'action_input_dim': action_dim,
            'action_feat_dim': action_feat_dim,
            'total_decoder_input': spatial_feat_dim + action_feat_dim,
            'action_compression': 'None (direct injection)',
        }
    else:
        resolution = args.kplanes_config.get('resolution', [64, 64, 64, 25])
        out_dim = args.kplanes_config.get('output_coordinate_dim', 32)
        num_scales = len(args.multires)
        
        return {
            'architecture': 'HexPlane',
            'num_planes': 6,
            'plane_names': ['XY', 'XZ', 'YZ', 'XT', 'YT', 'ZT'],
            'resolution': resolution,
            'num_scales': num_scales,
            'spatial_feat_dim': out_dim * num_scales,
            'action_input_dim': getattr(args, 'action_input_dim', 6),
            'action_feat_dim': 1,
            'total_decoder_input': out_dim * num_scales,
            'action_compression': '6D -> 1D (potential information loss)',
        }


def print_network_comparison():
    """Print comparison between HexPlane and TriPlane architectures."""
    print("=" * 80)
    print("Deformation Network Architecture Comparison")
    print("=" * 80)
    
    print("""
    ┌─────────────────────────┬─────────────────────────────────────────────────┐
    │        Feature          │    HexPlane (Original)    │    TriPlane (New)   │
    ├─────────────────────────┼─────────────────────────────────────────────────┤
    │ Spatial Planes          │ 3 (XY, XZ, YZ)            │ 3 (XY, XZ, YZ)      │
    │ Temporal Planes         │ 3 (XT, YT, ZT)            │ 0 (None)            │
    │ Total Planes            │ 6                         │ 3                   │
    │ Memory Usage            │ Higher                    │ ~50% reduction      │
    ├─────────────────────────┼─────────────────────────────────────────────────┤
    │ Control Input           │ 6D joint angles           │ 6D joint angles     │
    │ Control Processing      │ MLP -> 1D latent          │ PE -> Direct inject │
    │ Control Dimension       │ Compressed to 1D          │ Preserved (54D w/PE)│
    │ Information Loss        │ High (6:1 compression)    │ None                │
    ├─────────────────────────┼─────────────────────────────────────────────────┤
    │ Decoder Input           │ grid_feat                 │ grid_feat + ctrl    │
    │ Decoder Input Dim       │ ~96 (32×3 scales)         │ ~150 (96 + 54)      │
    │ Control Expressiveness  │ Limited                   │ Full                │
    ├─────────────────────────┼─────────────────────────────────────────────────┤
    │ Use Case                │ Video/Time sequences      │ Controllable motion │
    │ Best For                │ Temporal interpolation    │ Articulated objects │
    └─────────────────────────┴─────────────────────────────────────────────────┘
    """)
    
    print("\nRecommendation:")
    print("  - For robot/articulated arm control: Use TriPlane (use_triplane=True)")
    print("  - For video reconstruction/interpolation: Use HexPlane (use_triplane=False)")
    print("=" * 80)


if __name__ == "__main__":
    print_network_comparison()
