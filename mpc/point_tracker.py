
import os
import sys
import torch
import numpy as np

# Add submodules/tapir_pytorch to path
TAPIR_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "submodules", "tapir_pytorch")
if TAPIR_PATH not in sys.path:
    sys.path.append(TAPIR_PATH)

try:
    from tapnet.tapir_inference import TapirInference
    TAPNET_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import tapnet.tapir_inference: {e}")
    TAPNET_AVAILABLE = False
except Exception as e:
    print(f"Warning: Error importing tapnet.tapir_inference: {e}")
    TAPNET_AVAILABLE = False

class PointTracker:
    def __init__(self, device="cuda", checkpoint_path=None, input_resolution=(512, 512)):
        """
        Initialize TAPIR point tracker.
        
        Args:
            device: Device to run on ('cuda' or 'cpu')
            checkpoint_path: Path to TAPIR checkpoint
            input_resolution: Target resolution for tracking (H, W). TAPIR will resize all 
                            input frames to this resolution. Point coordinates should match
                            this resolution. Default (512, 512) recommended by official BootsTAPIR
                            (achieves 69.2% on RoboTAP vs 59.6% for standard TAPIR).
        """
        self.device = device
        if not TAPNET_AVAILABLE:
            print("Warning: PointTracker is disabled because tapnet is not available.")
            return

        if checkpoint_path is None:
            checkpoint_path = os.path.join(TAPIR_PATH, "causal_bootstapir_checkpoint.pt")
        
        self.checkpoint_path = checkpoint_path
        if not os.path.exists(self.checkpoint_path):
            print(f"Warning: Checkpoint not found at {self.checkpoint_path}")
            print("Please download it: wget -O submodules/tapir_pytorch/causal_bootstapir_checkpoint.pt https://storage.googleapis.com/dm-tapnet/causal_bootstapir_checkpoint.pt")
            self.model = None
            return

        # Initialize model with specified resolution
        # IMPORTANT: All input images will be resized to this resolution by TAPIR internally.
        # Point coordinates MUST be in this resolution's coordinate system.
        # Research finding: BootsTAPIR performs best at 512x512 resolution.
        self.resolution = input_resolution
        print(f"[PointTracker] Initializing TAPIR with input_resolution={self.resolution}")
        self.model_wrapper = TapirInference(
            model_path=self.checkpoint_path,
            input_resolution=self.resolution,
            num_pips_iter=6,  # Increased from 4 for better refinement (research recommendation)
            device=torch.device(device)
        )

    def track(self, video_tensor, initial_points, return_confidence=False):
        """
        Track points in a video sequence.

        Args:
            video_tensor: (B, T, C, H, W) or (T, C, H, W) tensor, normalized [0, 1]
            initial_points: (N, 2) tensor or numpy array, [x, y] coordinates (pixels)
            return_confidence: If True, also return tracking confidence scores

        Returns:
            If return_confidence=False:
                tracks: (B, N, T, 2) or (N, T, 2) coordinates [x, y] (pixels)
                visibles: (B, N, T) or (N, T) boolean
            If return_confidence=True:
                tracks: same as above
                visibles: same as above  
                confidence: (B, N, T) or (N, T) confidence scores [0, 1] (higher = more reliable)
        """
        if not TAPNET_AVAILABLE or not hasattr(self, 'model_wrapper') or self.model_wrapper is None:
            if return_confidence:
                return None, None, None
            return None, None

        # Handle inputs
        is_batched = video_tensor.ndim == 5
        if not is_batched:
            video_tensor = video_tensor.unsqueeze(0) # (1, T, C, H, W)

        B, T, C, H, W = video_tensor.shape
        
        # Ensure points are numpy
        if isinstance(initial_points, torch.Tensor):
            initial_points_np = initial_points.cpu().numpy()
        else:
            initial_points_np = initial_points
        
        # Initialize rescale variables
        needs_rescale = False
        scale_back_x = 1.0
        scale_back_y = 1.0
            
        # CRITICAL: Check if input resolution matches what TAPIR expects
        # TAPIR internally resizes all frames to self.resolution, so coordinates
        # MUST be in that coordinate system. If mismatch, we need to scale coords.
        if (H, W) != self.resolution:
            # print(f"[PointTracker] WARNING: Input video resolution ({H}x{W}) differs from "
                  # f"TAPIR resolution {self.resolution}. Scaling coordinates accordingly.")
            # Scale coordinates to match TAPIR's expected resolution
            scale_x = self.resolution[1] / W  # width scale
            scale_y = self.resolution[0] / H  # height scale
            initial_points_np = initial_points_np.copy()
            initial_points_np[:, 0] *= scale_x  # x coordinates
            initial_points_np[:, 1] *= scale_y  # y coordinates
            
            # We will need to scale back the output tracks later
            needs_rescale = True
            scale_back_x = W / self.resolution[1]
            scale_back_y = H / self.resolution[0]

        all_tracks = []
        all_visibles = []
        all_confidences = [] if return_confidence else None

        # Loop over batch
        for b in range(B):
            # Extract video for this batch
            # (T, C, H, W) -> (T, H, W, C) numpy uint8 for TapirInference?
            # TapirInference set_points expects float [0, 1] or uint8?
            # example_video_tracking.py uses cv2.read() which is uint8.
            # tapir_inference.py preprocess_frame:
            #   frame = torch.tensor(frame).to(device)
            #   if frame.ndim == 3: frame = frame.unsqueeze(0)
            #   frame = frame.permute(0, 3, 1, 2)
            #   frame = F.interpolate(frame, size=resize, mode='bilinear', align_corners=False)
            #   frame = frame / 255.0
            # So it expects [0, 255] input!
            
            video_b = video_tensor[b].permute(0, 2, 3, 1).cpu().detach().numpy() # (T, H, W, C)
            video_b = (np.clip(video_b, 0, 1) * 255).astype(np.uint8)
            
            # 1. Initialize with first frame
            # set_points resets the causal state
            self.model_wrapper.set_points(video_b[0], initial_points_np)
            
            batch_tracks = []
            batch_visibles = []
            batch_confidences = [] if return_confidence else None
            
            # Store t=0 (initial points)
            # set_points doesn't return them, but we know them.
            # But we need consistent format.
            # We can run inference on frame 0? 
            # TapirInference.forward updates state.
            # set_points initializes state.
            # If we call forward(frame[0]) immediately after set_points, it might advance state.
            # Let's just append initial points for t=0.
            batch_tracks.append(initial_points_np) # (N, 2)
            batch_visibles.append(np.ones(initial_points_np.shape[0], dtype=bool)) # (N,)
            if return_confidence:
                # At t=0, confidence is 1.0 (known positions)
                batch_confidences.append(np.ones(initial_points_np.shape[0], dtype=np.float32))
            
            # 2. Track subsequent frames
            for t in range(1, T):
                if return_confidence:
                    points, visibles, occlusions, expected_dist = self.model_wrapper(video_b[t], return_uncertainty=True)
                    # Compute confidence from TAPIR's uncertainty metrics
                    # Research finding: expected_dist threshold = 8px @ 256x256, scale to current resolution
                    expected_dist_thresh = 8.0 * (self.resolution[0] / 256.0)
                    # Combined confidence: (1 - sigmoid(occlusion)) * (1 - sigmoid(expected_dist))
                    confidence = (1 - 1/(1 + np.exp(-occlusions))) * (1 - 1/(1 + np.exp(-expected_dist)))
                    batch_confidences.append(confidence)
                else:
                    points, visibles = self.model_wrapper(video_b[t])
                
                # points: (N, 2), visibles: (N,)
                batch_tracks.append(points)
                batch_visibles.append(visibles)
                
            # Stack time
            batch_tracks = np.stack(batch_tracks, axis=1) # (N, T, 2)
            batch_visibles = np.stack(batch_visibles, axis=1) # (N, T)
            if return_confidence:
                batch_confidences_stacked = np.stack(batch_confidences, axis=1) # (N, T)
            
            all_tracks.append(batch_tracks)
            all_visibles.append(batch_visibles)
            if return_confidence:
                all_confidences.append(batch_confidences_stacked)

        all_tracks = np.stack(all_tracks) # (B, N, T, 2)
        all_visibles = np.stack(all_visibles) # (B, N, T)
        if return_confidence:
            all_confidences = np.stack(all_confidences) # (B, N, T)
        
        # Scale tracks back to original video resolution if needed
        if needs_rescale:
            all_tracks[:, :, :, 0] *= scale_back_x  # x coordinates
            all_tracks[:, :, :, 1] *= scale_back_y  # y coordinates
            #print(f"[PointTracker] Scaled output coordinates back to {H}x{W}")

        # Convert to torch
        tracks_tensor = torch.from_numpy(all_tracks).to(self.device).float()
        visibles_tensor = torch.from_numpy(all_visibles).to(self.device).float() # boolean -> float
        
        if return_confidence:
            confidence_tensor = torch.from_numpy(all_confidences).to(self.device).float()

        if not is_batched:
            tracks_tensor = tracks_tensor.squeeze(0)
            visibles_tensor = visibles_tensor.squeeze(0)
            if return_confidence:
                confidence_tensor = confidence_tensor.squeeze(0)

        if return_confidence:
            return tracks_tensor, visibles_tensor, confidence_tensor
        else:
            return tracks_tensor, visibles_tensor
