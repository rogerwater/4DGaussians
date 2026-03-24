import torch
import torch.nn as nn
import torch.nn.init as init


class ActionEncoder(nn.Module):
    """
    Action Encoder for Controllable 4D Gaussian Splatting

    Architecture:
        Input: action_vector [B, input_dim] - N-dimensional action parameters
        Optional: Positional Encoding (Fourier Features)
        MLP: [input_dim] -> [hidden] -> [hidden] -> [hidden/2] -> [1]
        Output: action_latent [B, 1] - Normalized action latent in [-1, 1]
        
    Args:
        input_dim (int): Dimensionality of input control vector (default: 6)
        hidden_dim (int): Hidden layer dimension of MLP encoder (default: 64)
        output_dim (int): Output dimension, fixed at 1 for HexPlane compatibility
        use_positional_encoding (bool): Whether to use positional encoding
        num_frequencies (int): Number of frequency bands for positional encoding
        activation (str): Activation function type - 'relu', 'elu', or 'silu'
    """
    
    def __init__(
        self,
        input_dim=6,
        hidden_dim=64,
        output_dim=1,
        use_positional_encoding=False,
        num_frequencies=4,
        activation='relu'
    ):
        super(ActionEncoder, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.use_pe = use_positional_encoding
        self.num_frequencies = num_frequencies
        
        if use_positional_encoding:
            # Each dimension: [x, sin(2^0*x), cos(2^0*x), ..., sin(2^(F-1)*x), cos(2^(F-1)*x)]
            # Total: input_dim * (1 + 2 * num_frequencies)
            pe_dim = input_dim * (1 + 2 * num_frequencies)
        else:
            pe_dim = input_dim
            
        if activation == 'relu':
            act_fn = nn.ReLU(inplace=True)
        elif activation == 'elu':
            act_fn = nn.ELU(inplace=True)
        elif activation == 'silu':
            act_fn = nn.SiLU(inplace=True)
        else:
            raise ValueError(f"Unsupported activation function: {activation}")
        
        self.encoder = nn.Sequential(
            nn.Linear(pe_dim, hidden_dim),
            act_fn,
            nn.Linear(hidden_dim, hidden_dim),
            act_fn,
            nn.Linear(hidden_dim, hidden_dim // 2),
            act_fn,
            nn.Linear(hidden_dim // 2, output_dim),
            nn.Tanh()  # Normalize output to [-1, 1]
        )
        
        if use_positional_encoding:
            freq_bands = 2.0 ** torch.linspace(0.0, num_frequencies - 1, num_frequencies)
            self.register_buffer('freq_bands', freq_bands)
            
        self._initialize_weights()
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.xavier_uniform_(m.weight, gain=1.0)
                if m.bias is not None:
                    init.zeros_(m.bias)
                    
    def position_encoding(self, action_vec):
        """
        Apply positional encoding to action vector.
        
        Args:
            action_vec: [B, input_dim] - Raw action vector
            
        Returns:
            encoded: [B, input_dim * (1 + 2*num_frequencies)] - Encoded features
        """
        
        if not self.use_pe:
            return action_vec
        
        encoded = [action_vec]
        
        for freq in self.freq_bands:
            encoded.append(torch.sin(action_vec * freq))
            encoded.append(torch.cos(action_vec * freq))
            
        return torch.cat(encoded, dim=-1)
    
    def forward(self, action_vec):
        """
        Forward pass: Encode action vector to action latent
        
        Args:
            action_vec: [B, input_dim] - Action vector
                      Recommended to normalize to [-1, 1] or [-pi, pi]
        
        Returns:
            action_latent: [B, 1] - Encoded action latent in range [-1, 1]
        """
        
        assert action_vec.shape[-1] == self.input_dim, \
            f"Expected {self.input_dim}D action vector, got {action_vec.shape[-1]}D"
            
        encoded_input = self.position_encoding(action_vec)
        
        action_latent = self.encoder(encoded_input)
        
        return action_latent
    
    def encode_batch(self, action_vec_batch):
        """
        Batch encoding interface for compatibility
        
        Args:
            action_vec_batch: [B, N, input_dim] - Batched action vectors
            
        Returns:
            action_latent: [B, N, 1] - Batched encoded results
        """
        
        batch_size = action_vec_batch.shape[0]
        num_points = action_vec_batch.shape[1]
        
        # Reshape : [B, N, D] -> [B*N, D]
        action_vec_flat = action_vec_batch.reshape(-1, self.input_dim)
        
        action_latent_flat = self.forward(action_vec_flat)
        
        action_latent = action_latent_flat.reshape(batch_size, num_points, -1)
        
        return action_latent
    

def create_action_encoder(config):
    encoder_args = {
        'input_dim': getattr(config, 'action_input_dim', 6),
        'hidden_dim': getattr(config, 'action_hidden_dim', 64),
        'output_dim': 1,
        'use_positional_encoding': getattr(config, 'action_use_pe', False),
        'num_frequencies': getattr(config, 'action_num_frequencies', 4),
        'activation': getattr(config, 'action_activation', 'relu')
    }
    
    return ActionEncoder(**encoder_args)


# testing functions
def test_action_encoder():
    print("=" * 70)
    print("Testing ActionEncoder")
    print("=" * 70)

    # Create encoder
    encoder = ActionEncoder(
        input_dim=6,
        hidden_dim=64,
        output_dim=1,
        use_positional_encoding=True,
        num_frequencies=4,
        activation='relu'
    )
    
    print(f"\nEncoder Configuration:")
    print(f"  Input dim: 6")
    print(f"  Hidden dim: 64")
    print(f"  Output dim: 1")
    print(f"  Positional Encoding: True (4 frequencies)")
    
    # Count parameters
    num_params = sum(p.numel() for p in encoder.parameters())
    print(f"  Total parameters: {num_params:,}")
    
    # Test data
    batch_size = 10
    action_vec = torch.randn(batch_size, 6) * 3.14  # Random angles in [-pi, pi]
    
    print(f"\nInput:")
    print(f"  Shape: {action_vec.shape}")
    print(f"  Range: [{action_vec.min():.3f}, {action_vec.max():.3f}]")
    
    # Forward pass
    with torch.no_grad():
        action_latent = encoder(action_vec)
    
    print(f"\nOutput:")
    print(f"  Shape: {action_latent.shape}")
    print(f"  Range: [{action_latent.min():.3f}, {action_latent.max():.3f}]")
    print(f"  Expected range: [-1.0, 1.0]")
    
    # Test gradient flow
    action_vec.requires_grad = True
    action_latent = encoder(action_vec)
    loss = action_latent.sum()
    loss.backward()
    
    print(f"\nGradient Check:")
    print(f"  Input gradient exists: {action_vec.grad is not None}")
    print(f"  Input gradient norm: {action_vec.grad.norm():.6f}")
    
    # Test interpolation smoothness
    vec_a = torch.zeros(1, 6)
    vec_b = torch.ones(1, 6) * 3.14159
    
    print(f"\nSmoothness Test (Interpolation):")
    print(f"  Interpolating between zero vector and π vector")
    with torch.no_grad():
        prev_val = None
        for alpha in torch.linspace(0, 1, 6):
            vec_interp = (1 - alpha) * vec_a + alpha * vec_b
            latent = encoder(vec_interp)
            val = latent.item()
            
            if prev_val is not None:
                diff = abs(val - prev_val)
                print(f"    α={alpha:.2f}: latent={val:+.4f}, Δ={diff:.4f}")
            else:
                print(f"    α={alpha:.2f}: latent={val:+.4f}")
            prev_val = val
    
    print("\n" + "=" * 70)
    print("✓ All tests passed!")
    print("=" * 70)
    
    
def visualize_action_space(encoder, num_samples=100, save_path=None):
    """
    Visualize the learned action space by sampling
    
    Args:
        encoder: ActionEncoder instance
        num_samples: Number of samples to generate
        save_path: Optional path to save visualization
    """
    import numpy as np
    import matplotlib.pyplot as plt
    
    print(f"\nVisualizing action space with {num_samples} samples...")
    
    encoder.eval()
    with torch.no_grad():
        # Sample random action vectors
        action_vecs = torch.randn(num_samples, encoder.input_dim) * 2.0
        latents = encoder(action_vecs).squeeze().numpy()
        
        # Plot histogram
        plt.figure(figsize=(10, 6))
        plt.hist(latents, bins=50, alpha=0.7, edgecolor='black')
        plt.xlabel('Action Latent Value')
        plt.ylabel('Frequency')
        plt.title(f'Distribution of Action Latents ({num_samples} samples)')
        plt.axvline(x=-1, color='r', linestyle='--', label='Lower bound')
        plt.axvline(x=1, color='r', linestyle='--', label='Upper bound')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"  Saved to {save_path}")
        else:
            plt.show()
        
        # Print statistics
        print(f"\nStatistics:")
        print(f"  Mean: {latents.mean():.4f}")
        print(f"  Std: {latents.std():.4f}")
        print(f"  Min: {latents.min():.4f}")
        print(f"  Max: {latents.max():.4f}")
        print(f"  Coverage: {(latents.min(), latents.max())}")


if __name__ == "__main__":
    # Run basic tests
    test_action_encoder()
    
    # Optional: Visualize action space
    try:
        encoder = ActionEncoder(input_dim=6, hidden_dim=64)
        visualize_action_space(encoder, num_samples=1000)
    except ImportError:
        print("\nNote: matplotlib not available, skipping visualization")   
